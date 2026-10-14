"""체크포인트 하나(보통 상위 10개 평균 모델)의 2단계 측정 보고서(docs/experiments.md 2단계, 규칙 변경 11).

dev 세트마다 다음을 잰다: full context greedy CER_ns/CER_ws(전체 + 한글 전용, 부트스트랩 구간),
simulated 청크 마스크 {4, 8, 16, 32}프레임(160~1,280 ms) × 좌측 128의 CER과 full 대비 비율, 중간 자기조건
헤드별 CER(full context). 여기에 학습셋 표본 CER(과적합 점검), offline 청크 마스크 대비 fp16 캐시 스트리밍
점검(max |dlog p|, max |dp|, 프레임 argmax 일치율), QK-norm·어텐션 로짓 안정성 통계를 더한다.

사용법:
  uv run python scripts/stage2_report.py --checkpoint outputs/stage2/runA/avg10.pt --name runA-avg10
결과는 reports/stage2/<name>.json과 <name>.md, 발화별 가설은 hyps/*.jsonl에 쓴다. --limit N을 주면 세트마다
앞 N발화만 평가한다(연기 시험).
"""

# [왜 필요한가] 2단계 가설 H2.2(청크별 CER ÷ full), H2.3(9층 헤드 vs 최종 헤드), 서비스 청크 판정, 규칙 변경
# 11의 fp16 스트리밍 추가 보고, 3단계 (c)의 과적합 조건(학습셋 CER < dev_H × 0.5)이 모두 이 보고서의 수치로
# 판정됐다(docs/experiments.md 2단계 결과). 3·4단계 후보 보고서도 같은 스크립트로 만들어 수치 정의가 같다.
# [연결] 2단계는 README 명령으로 직접, 3·4단계는 scripts/stage3.py·scripts/stage4.py가 하위 프로세스로 부른다.
# 저장한 가설 파일은 scripts/compare_runs.py의 짝지은 비교 입력이다. 디코딩·채점은 stt_scratch.evaluate의
# 함수를 그대로 쓴다.

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# torch: 모델 forward(no_grad)와 텐서 비교.
import torch

# 프로젝트 내부 모듈: 평가 경로 그대로(evaluate의 체크포인트 로더, autocast, 길이 버킷 배치, 배치 특징 계산,
# offline 디코딩, 채점), fp32 log_softmax, greedy 디코딩, 캐시 스트리밍 인코더. 평가 CLI와 같은 함수를 써야 이
# 보고서의 수치가 다른 단계 수치와 같은 정의가 된다.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, batch_features, decode_offline, duration_batches, load_model, score
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.text import Vocab

# ROOT: 저장소 루트(체크포인트·매니페스트·보고서 경로의 기준). FRAME_MS: 인코더 프레임 하나의 길이(4배
# 서브샘플링 뒤 25 fps = 40 ms). 청크 프레임 수를 ms로 바꿔 모드 이름(chunk8_320ms 등)에 쓴다.
ROOT = Path(__file__).resolve().parents[1]
FRAME_MS = 40


# 채점 결과에서 주지표(한글 전용 부분집합의 CER_ns)만 꺼낸다.
def rate(scored: dict) -> float:
    return scored["hangul_only"]["cer_ns"]["rate"]


# 중간 CTC 헤드는 자기조건/InterCTC 층(예: 3·6·9층)이다. 반환: {층 번호(1부터): {발화 id: 가설}}.
# 왜: H2.3(자기조건화가 정상이면 9층 헤드 CER > 최종 CER)을 판정하려고 각 헤드를 따로 디코딩한다. 결과는 H2.3
# 위반(중간 헤드가 더 나음)으로 학습 문장 암기 가설의 근거가 됐다. 모델 forward가 중간 로짓을 층 순서대로
# 돌려주므로 정렬한 층 목록과 짝지을 수 있다.
@torch.no_grad()
def decode_heads(model, vocab, utts, store, device, logmel, amp: bool) -> dict[int, dict[str, str]]:
    """중간 헤드마다 full context greedy 가설을 만든다. 키는 1부터 센 층 번호다."""
    layers = sorted(model.sc_layers)
    hyps: dict[int, dict[str, str]] = {layer: {} for layer in layers}
    for batch in duration_batches(utts, 400.0):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, amp):
            _, inter, out_lengths = model(mel, frames, ChunkSpec(0, -1))
        for layer, logits in zip(layers, inter, strict=True):
            for j, ids in zip(batch, greedy_batch(logits.float(), out_lengths), strict=True):
                hyps[layer][utts[j].id] = vocab.decode(ids)
    return hyps


# 비교 대상은 둘 다 fp16이다. 반환: log-prob 최대 절대차, 확률 최대 절대차, 프레임 argmax 일치율.
# 왜: 0단계 게이트(fp16 log-prob 차 ≤ 1e-2)는 무작위 초기화 모델에서 쟀고, 학습된 모델은 거의 0인 확률의
# log-prob 절댓값이 커서 fp16 반올림만으로 1e-2를 넘을 수 있다. 그래서 확률 차이와 argmax 일치율을 함께
# 보고하기로 했다(규칙 변경 11). 2단계 결과: log-prob 차 2.27, 확률 차 0.172, argmax 99.991 % 일치(다른
# 4프레임은 fp16 분해능 안의 동점).
@torch.no_grad()
def streaming_check(model, utts, store, device, logmel, spec: ChunkSpec, amp: bool) -> dict:
    """발화 전체에서 fp16 캐시 스트리밍과 offline 청크 마스크 forward를 비교한다(log-prob·확률 차, argmax)."""
    enc = StreamingEncoder(model, spec)
    max_dlp, max_dp, agree, frames = 0.0, 0.0, 0, 0
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, n = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        with autocast(device, amp):
            # offline: 전체 발화를 청크 마스크로 한 번에 forward. streamed: 같은 mel을 인코더 청크(mel 4 ×
            # chunk_size프레임)씩 잘라 캐시 스트리밍 step으로 흘린다. 둘 다 같은 autocast 아래라 차이는 캐시
            # 경로와 fp16 반올림에서만 나온다.
            offline = log_probs_fp32(model(mel, n, spec)[0])[0]
            state = enc.new_state()
            parts = []
            step = 4 * spec.chunk_size
            for i in range(0, mel.shape[1], step):
                parts.append(enc.step(mel[0, i : i + step], state))
            streamed = torch.cat(parts, dim=0)
        # 두 출력의 프레임 수를 맞춘 뒤 프레임별로 비교해 최대 차이와 argmax 일치 수를 누적한다.
        t = min(offline.shape[0], streamed.shape[0])
        offline, streamed = offline[:t], streamed[:t]
        max_dlp = max(max_dlp, float((offline - streamed).abs().max()))
        max_dp = max(max_dp, float((offline.exp() - streamed.exp()).abs().max()))
        agree += int((offline.argmax(-1) == streamed.argmax(-1)).sum())
        frames += t
    return {
        "n_utterances": len(utts),
        "frames": frames,
        "max_abs_dlogprob": max_dlp,
        "max_abs_dprob": max_dp,
        "argmax_agreement": agree / max(1, frames),
    }


# 통계: 블록별 QK-norm LayerNorm 가중치의 최대 |값|(q·k)과 블록별 어텐션 로짓(q·k/√d) 표준편차.
# 왜: 규칙 변경 13의 주의대로, QK-norm의 LayerNorm은 affine 가중치·편향이 1·0 근처에 있는 동안만 로짓을 묶어
# 두므로 가중치가 커지는지 감시하기로 했다(docs/survey/README.md "결정"의 위치 항목). 2단계 결과: 로짓
# 표준편차 1.4~2.3, q 이득 최대 1.20~1.89.
# 주의: 실제 forward와 다른 진단 경로다. 자기조건 피드백 없이 블록을 차례로 통과하고, q·k는 RoPE 전 값이며,
# 마스크는 모두 True(전체 문맥, 패딩 프레임 포함)다. 입력은 호출자가 넘긴 발화의 앞 8개(main에서는 스트리밍
# 점검용 dev_h 목록)다.
@torch.no_grad()
def stability_stats(model, utts, store, device, logmel, amp: bool) -> dict:
    """dev 배치 하나에서 학습 안정성 진단 통계(QK-norm affine 크기, 블록별 어텐션 로짓 표준편차)를 잰다."""
    blocks = model.blocks
    # QK-norm이 꺼진 모델(q_norm None)에서는 빈 목록이 된다.
    gains = [float(b.attn.q_norm.weight.abs().max()) for b in blocks if b.attn.q_norm is not None]
    kgains = [float(b.attn.k_norm.weight.abs().max()) for b in blocks if b.attn.k_norm is not None]
    batch = list(range(min(8, len(utts))))
    mel, frames = batch_features(utts, batch, store, device, logmel)
    stds = []
    with autocast(device, amp):
        x = model.normalize_mel(mel)
        x, lengths = model.subsampling(x, frames)
        t = x.shape[1]
        mask = torch.ones(x.shape[0], 1, t, t, dtype=torch.bool, device=device)
        pos = torch.arange(t, device=device)
        valid = pos[None, :] < lengths[:, None]
        # 블록 내부를 Conformer 블록 forward와 같은 순서로 따라가 어텐션 입력(ffn1 반 잔차 → attn_norm)을
        # 만들고, qkv 투영에서 q·k를 헤드별로 잘라 QK-norm까지 적용한 뒤 로짓 표준편차를 잰다. 다음 블록
        # 입력은 블록 전체 forward로 구한다(청크 0 = 전체 문맥 conv).
        for blk in blocks:
            h = blk.attn_norm(x + 0.5 * blk.ffn1(x))
            d = h.shape[-1]
            qkv = blk.attn.qkv(h)
            hd = blk.attn.head_dim
            q = qkv[..., :d].view(h.shape[0], t, -1, hd).transpose(1, 2)
            k = qkv[..., d : 2 * d].view(h.shape[0], t, -1, hd).transpose(1, 2)
            if blk.attn.q_norm is not None:
                q, k = blk.attn.q_norm(q), blk.attn.k_norm(k)
            stds.append(float(((q.float() @ k.float().transpose(-1, -2)) / hd**0.5).std()))
            x = blk(x, mask, pos, 0, valid)
    return {
        "qk_norm_max_gain_q": gains,
        "qk_norm_max_gain_k": kgains,
        "attention_logit_std_per_block": stds,
    }


# scripts/compare_runs.py가 두 모델의 이 파일을 같은 발화끼리 짝지어 비교한다.
def save_hyps(path: Path, utts, hyps: dict[str, str]) -> None:
    """발화별 가설을 JSONL로 저장한다. 한 줄에 id, 정답, 가설, hangul_only 표시(compare_runs.py 입력)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for u in utts:
            row = {"id": u.id, "ref": u.text, "hyp": hyps[u.id], "hangul_only": u.hangul_only}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 보고서 전체를 만든다.
# 주요 인자: --checkpoint, --name(보고서 파일 이름), --report-dir(기본 reports/stage2), --devs(dev_mix,dev_h),
# --chunks(4,8,16,32), --left(128), --streaming-check(점검할 dev_h 발화 수, 기본 200), --limit(세트마다 앞
# N발화만; 연기 시험용), --train-sample(학습셋 표본 수, 기본 1,000), --fp32(fp32로 학습한 모델용), --device.
# 정밀도: GPU에서는 학습과 같은 fp16 autocast로 디코딩한다(규칙 변경 8).
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True, help="--report-dir 아래에 쓸 보고서 파일 이름(확장자 제외)")
    p.add_argument("--report-dir", default="reports/stage2")
    p.add_argument("--devs", default="dev_mix,dev_h")
    p.add_argument("--chunks", default="4,8,16,32")
    p.add_argument("--left", type=int, default=128)
    p.add_argument(
        "--streaming-check", type=int, default=200, help="fp16 스트리밍 점검에 쓸 dev_h 발화 수(앞에서부터)"
    )
    p.add_argument("--limit", type=int, default=0, help="세트마다 앞 N발화만 평가한다(연기 시험)")
    p.add_argument(
        "--train-sample",
        type=int,
        default=1000,
        help="과적합 점검에 쓸 학습 발화 수(시드 고정 표본, full context, 증강 없음)",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    amp = not args.fp32 and device.type == "cuda"
    model, state = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    t0 = time.perf_counter()
    # 보고서 머리: 체크포인트 경로, 평균에 쓴 epoch 목록(average_checkpoints.py가 저장한 "averaged"), 정밀도,
    # 좌측 문맥. 모든 수치에 체크포인트와 디코딩 모드를 함께 적으라는 규칙(docs/experiments.md "지표")을
    # 따른다.
    report: dict = {
        "checkpoint": args.checkpoint,
        "averaged_epochs": state.get("averaged"),
        "precision": "fp16-autocast" if amp else "fp32",
        "left_context": args.left,
        "sets": {},
    }
    for name in args.devs.split(","):
        utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
        if args.limit:
            utts = utts[: args.limit]
        store = PcmStore(ROOT / "data" / "pcm", name)
        entry: dict = {"n": len(utts), "modes": {}}
        hyp_dir = ROOT / args.report_dir / "hyps"
        # full context greedy 디코딩 → 가설 저장 → 채점(전체·한글 전용, CER_ns·CER_ws, 부트스트랩 구간).
        hyps = decode_offline(model, vocab, utts, store, ChunkSpec(0, -1), device, logmel, amp=amp)
        save_hyps(hyp_dir / f"{args.name}-{name}-full.jsonl", utts, hyps)
        full = score(utts, hyps)
        entry["modes"]["full"] = full
        # simulated 청크: 전체 발화를 한 번에 넣되 청크 경계 마스크(청크 크기 + 좌측 128)를 건다. 캐시
        # 스트리밍과 같은 결과를 내도록 설계된 경로라 스트리밍 CER을 빠르게 잴 수 있다. full 대비 비율이
        # H2.2의 판정량이다(640 ms ≤ 1.25, 320 ms ≤ 1.40).
        for chunk in (int(c) for c in args.chunks.split(",")):
            spec = ChunkSpec(chunk, args.left)
            hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
            mode = f"chunk{chunk}_{chunk * FRAME_MS}ms"
            save_hyps(hyp_dir / f"{args.name}-{name}-{mode}.jsonl", utts, hyps)
            scored = score(utts, hyps)
            scored["ratio_to_full"] = rate(scored) / rate(full) if rate(full) else None
            entry["modes"][mode] = scored
        # 중간 헤드별 full context CER(H2.3).
        heads = decode_heads(model, vocab, utts, store, device, logmel, amp)
        entry["heads"] = {f"layer{layer}": score(utts, hyps) for layer, hyps in heads.items()}
        report["sets"][name] = entry
        print(f"{name}: full {rate(full) * 100:.2f} %", flush=True)
    # 과적합 점검: 학습셋에서 고정 시드로 뽑은 발화(full context, 증강 없음)의 CER과 dev_H full CER에 대한 비.
    # 비가 0.5보다 작으면 과적합 징후로 보고 3단계 (c) 폭 384 후보를 빼기로 등록했다(실측 0.008배로 빠짐).
    # 표본은 random.Random(0).sample로 뽑았다. 프로젝트 규칙의 numpy Generator·SHA-256 방식이 아니라는 점은
    # 문서에 그대로 밝혀 두었고, 이후 단계도 같은 표본을 쓴다(docs/experiments.md 2단계 결과의 과적합 항목).
    if args.train_sample:
        train = read_manifest(ROOT / "manifests" / "train.jsonl")
        picked = sorted(random.Random(0).sample(range(len(train)), min(args.train_sample, len(train))))
        utts = [train[i] for i in picked]
        store = PcmStore(ROOT / "data" / "pcm", "train")
        scored = score(
            utts, decode_offline(model, vocab, utts, store, ChunkSpec(0, -1), device, logmel, amp=amp)
        )
        report["train_sample"] = {"n": len(utts), "seed": 0, "full": scored}
        dev_h = report["sets"].get("dev_h", {}).get("modes", {}).get("full")
        if dev_h:
            report["train_sample"]["ratio_to_dev_h"] = rate(scored) / rate(dev_h) if rate(dev_h) else None
    # fp16 스트리밍 점검(서비스 청크 8 = 320 ms, 좌측 128)과 안정성 통계. 둘 다 dev_h 앞 N발화를 쓴다.
    if args.streaming_check:
        utts = read_manifest(ROOT / "manifests" / "dev_h.jsonl")[: args.streaming_check]
        store = PcmStore(ROOT / "data" / "pcm", "dev_h")
        report["streaming_check"] = {
            "spec": "chunk8 left128",
            **streaming_check(model, utts, store, device, logmel, ChunkSpec(8, args.left), amp),
        }
        report["stability"] = stability_stats(model, utts, store, device, logmel, amp)
    # JSON 보고서를 쓰고, 사람이 읽는 Markdown 표(세트·모드별 CER_ns 한글 전용과 구간, 전체 CER_ns, CER_ws,
    # full 대비 비율, 헤드별 CER, 학습셋 표본, 스트리밍 점검, 안정성)를 만든다.
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out_dir = ROOT / args.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lines = [f"# {args.name} ({report['precision']}, left {args.left}, checkpoint {args.checkpoint})", ""]
    lines.append(
        "| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, entry in report["sets"].items():
        for mode, s in entry["modes"].items():
            h = s["hangul_only"]["cer_ns"]
            ratio = s.get("ratio_to_full")
            lines.append(
                f"| {name} | {mode} | {h['rate'] * 100:.2f} | "
                f"{h['ci'][0] * 100:.2f}-{h['ci'][1] * 100:.2f} | {s['all']['cer_ns']['rate'] * 100:.2f} | "
                f"{s['hangul_only']['cer_ws']['rate'] * 100:.2f} | "
                f"{'' if ratio is None else f'{ratio:.3f}'} |"
            )
        for head, s in entry["heads"].items():
            lines.append(
                f"| {name} | head {head} (full) | {rate(s) * 100:.2f} | | "
                f"{s['all']['cer_ns']['rate'] * 100:.2f} | | |"
            )
    if "train_sample" in report:
        ts = report["train_sample"]
        ratio = ts.get("ratio_to_dev_h")
        lines += [
            "",
            f"train sample ({ts['n']} utts, seed 0, full context, no augmentation): CER_ns "
            f"{rate(ts['full']) * 100:.2f} %"
            + ("" if ratio is None else f", ratio to dev_h full {ratio:.3f} (overfitting flag if < 0.5)"),
        ]
    if "streaming_check" in report:
        c = report["streaming_check"]
        lines += [
            "",
            f"fp16 streaming check ({c['spec']}, {c['n_utterances']} dev_h utts, {c['frames']} frames): "
            f"max |dlog p| {c['max_abs_dlogprob']:.2e}, max |dp| {c['max_abs_dprob']:.2e}, "
            f"argmax agreement {c['argmax_agreement'] * 100:.3f} %",
        ]
    if "stability" in report:
        s = report["stability"]
        lines += [
            "",
            "stability: attention logit std per block "
            f"{[round(v, 1) for v in s['attention_logit_std_per_block']]}, "
            f"QK-norm max |gain| q {[round(v, 2) for v in s['qk_norm_max_gain_q']]}",
        ]
    (out_dir / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
