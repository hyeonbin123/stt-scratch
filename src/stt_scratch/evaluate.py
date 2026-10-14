"""모델 하나로 매니페스트 하나를 디코딩하고 채점하는 평가 모듈.

디코딩 방식은 세 가지다: full(전체 문맥), simulated(전체 발화 + 청크 마스크), real
streaming(StreamingEncoder로 패킷을 흘려 넣는 실제 캐시 스트리밍). 전체 발화와 한글 전용 부분집합 각각의
CER_ns(whisper-ko-ft와 같은 정의)와 CER_ws를 발화 단위 부트스트랩 구간과 함께 보고하고, 체크포인트와
디코딩 모드도 함께 남긴다(docs/experiments.md "지표").
"""

# [왜 세 가지인가] simulated는 배치로 빠르게 스트리밍 성능을 재고, real streaming은 서비스와 같은 경로라 둘의
# 차이로 캐시 버그를 잡는다(6단계 H6.1: 차이 ≤ 0.1포인트, 실측 0.03 안). 세 경로 모두 학습과 같은 fp16
# autocast로 돌려 차이가 정밀도가 아니라 디코딩 방식에서만 나오게 한다(규칙 변경 8).
# [보고] 전체 발화와 한글 전용 부분집합 각각의 CER_ns·CER_ws, 부트스트랩 구간, 체크포인트·디코딩 모드·정밀도를
# 함께 남긴다(docs/experiments.md "지표").
# [연결] train.py(매 epoch dev CER·dev 손실), scripts/stage2_report.py·stage4_latency.py·stage6.py·stage7.py,
# service.server(load_model)가 이 함수들을 가져다 쓴다. 단독 실행 명령은 README "학습·평가".

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 모듈: 매니페스트·PCM 저장소(data), 채점(metrics), 모델·손실·디코딩·특징·마스크·스트리밍
# 추론(model/), 어휘(text).
import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.metrics import score_ns, score_ws, summarize
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import ctc_loss
from stt_scratch.model.decode import GreedyState, greedy_batch
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel, StreamingFbank
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.text import Vocab

# 저장소 루트(src/stt_scratch/evaluate.py에서 두 단계 위). manifests/, data/pcm/, 체크포인트 경로를 모두 이
# 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[2]


# 체크포인트를 읽어 저장된 config로 모델을 다시 만들고 가중치를 넣어 eval 모드로 돌려준다(체크포인트 dict도
# 함께). weights_only=True는 텐서·기본 자료형만 복원해 임의 코드 실행을 막는다(보안 경고를 받고 바꿈,
# docs/ai-verification.md).
# 호출 위치: main, service.server.build_app, scripts의 단계별 도구(stage2_report, stage4_latency, stage6,
# stage7 등).
def load_model(path: Path, device: torch.device) -> tuple[ChunkConformerCTC, dict]:
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig(**state["config"])
    model = ChunkConformerCTC(cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


# 그 밖(CPU, --fp32 모델)에서는 autocast가 꺼진다. 학습과 같은 정밀도로 재야 dev 수치가 학습 조건과 맞는다.
def autocast(device, amp: bool):
    """모든 디코딩·손실 경로가 공유하는 정밀도 문맥: CUDA이고 amp면(학습 기본값) fp16 autocast."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")


# 평가용 배치 생성기다. 호출자는 보통 batch_seconds 400 s를 쓴다. 학습 샘플러와 달리 섞지도, 속도 교란 여유도
# 두지 않는다.
def duration_batches(utts, batch_seconds: float):
    """길이순으로 정렬한 인덱스를, 패딩 포함 길이(가장 긴 발화 × 개수)가 예산을 넘지 않게 묶어 내준다."""
    order = sorted(range(len(utts)), key=lambda i: utts[i].duration_s)
    i = 0
    while i < len(order):
        batch, longest = [], 0.0
        while i < len(order):
            d = utts[order[i]].duration_s
            if batch and max(longest, d) * (len(batch) + 1) > batch_seconds:
                break
            batch.append(order[i])
            longest = max(longest, d)
            i += 1
        yield batch


# 배치의 파형을 PCM 저장소에서 읽어 0으로 패딩하고 LogMel로 (mel, 유효 프레임 수)를 만든다.
def batch_features(utts, batch, store, device, logmel):
    wavs = [store.waveform(utts[j].id).to(device) for j in batch]
    lengths = torch.tensor([w.numel() for w in wavs], device=device)
    padded = torch.zeros(len(wavs), int(lengths.max()), device=device)
    for k, w in enumerate(wavs):
        padded[k, : w.numel()] = w
    return logmel(padded, lengths)


# full·simulated 디코딩: spec이 전체 문맥이면 full, 청크 설정이면 같은 forward에 청크 마스크만 걸린
# simulated다. 로짓은 fp32로 바꾼 뒤 greedy를 한다. 결과는 발화 id → 가설 문자열.
@torch.no_grad()
def decode_offline(
    model, vocab, utts, store, spec: ChunkSpec, device, logmel, batch_seconds: float = 400.0, amp: bool = True
):
    hyps: dict[str, str] = {}
    for batch in duration_batches(utts, batch_seconds):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, amp):
            logits, _, out_lengths = model(mel, frames, spec)
        for j, ids in zip(batch, greedy_batch(logits.float(), out_lengths), strict=True):
            hyps[utts[j].id] = vocab.decode(ids)
    return hyps


# 발화마다 서비스와 같은 패킷 규약(청크 8 기준 첫 패킷 5,360표본, 이후 5,120표본)으로 파형을 잘라
# StreamingFbank → 청크 단위 StreamingEncoder.step → GreedyState로 흘린다. 발화 하나씩 순서대로 돌리므로
# simulated보다 느리다. 봉인 test의 스트리밍 수치와 6단계 H6.1이 이 경로로 쟀다.
@torch.no_grad()
def decode_streaming(model, vocab, utts, store, spec: ChunkSpec, device, logmel, amp: bool = True):
    """실제 캐시 스트리밍 디코딩.

    음성을 4 * chunk_size * 160표본(청크 8이면 320 ms) 패킷으로 넣는다. 첫 패킷에는 FIRST_PACKET_EXTRA 표본을
    더 실어 모든 패킷이 인코더 청크 하나를 채우게 한다. offline 경로와 같은 autocast 아래에서 돌려, 차이가
    dtype이 아니라 캐시에서만 나오게 한다.
    """
    enc = StreamingEncoder(model, spec)
    samples_per_chunk = spec.chunk_size * 4 * 160
    hyps: dict[str, str] = {}
    with autocast(device, amp):
        for u in utts:
            wav = store.waveform(u.id).to(device)
            fb = StreamingFbank(logmel)
            state = enc.new_state()
            greedy = GreedyState()
            pending = torch.zeros(0, 80, device=device)
            n_frames = 0
            pos, packet = 0, samples_per_chunk + FIRST_PACKET_EXTRA
            while pos < wav.numel():
                piece = wav[pos : pos + packet]
                pos += packet
                packet = samples_per_chunk
                pending = torch.cat([pending, fb.push(piece)], dim=0)
                # 모인 mel이 청크 하나(4 × chunk_size프레임) 이상이면 청크 단위로 인코더에 넣는다. 패킷 규약을
                # 지키면 패킷마다 정확히 청크 하나가 나온다.
                while pending.shape[0] >= 4 * spec.chunk_size:
                    mel_chunk, pending = pending[: 4 * spec.chunk_size], pending[4 * spec.chunk_size :]
                    lp = enc.step(mel_chunk, state)
                    greedy.push(lp, n_frames)
                    n_frames += lp.shape[0]
            # 발화 끝의 남은 프레임(청크보다 짧은 마지막 부분 청크)은 한 번 더 step해 마저 디코딩한다. 인과
            # 전단부는 어떤 길이든 정확히 처리한다. 서비스의 flush와 같은 처리다.
            if pending.shape[0] > 0:
                lp = enc.step(pending, state)
                greedy.push(lp, n_frames)
            hyps[u.id] = vocab.decode(greedy.tokens)
    return hyps


# 학습과 같은 정밀도로 잰다. train.py가 매 epoch 기록하고 'epoch 3 끝 값 ≥ epoch 1 끝 값이면 발산' 판정에
# 쓴다(규칙 변경 3). 처음부터 학습하는 런에서 정렬이 시작되지 않은 경우를 이 규칙이 잡았다(docs/experiments.md
# 1단계 2차 시도, Run B 2e-3, 3단계 (d)).
@torch.no_grad()
def ctc_loss_manifest(
    model, vocab, name: str, device, logmel=None, amp: bool = True, batch_seconds: float = 400.0
) -> float:
    """dev CTC 손실: 최종 헤드, 전체 문맥에서 발화당 평균 CTC 손실(dev 손실 발산 검사용)."""
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", name)
    logmel = logmel or LogMel().to(device)
    total = 0.0
    for batch in duration_batches(utts, batch_seconds):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        tokens = [vocab.encode(utts[j].text) for j in batch]
        targets = torch.tensor([t for seq in tokens for t in seq], device=device, dtype=torch.int32)
        target_lengths = torch.tensor([len(seq) for seq in tokens], device=device, dtype=torch.int32)
        with autocast(device, amp):
            logits, _, out_lengths = model(mel, frames, ChunkSpec(0, -1))
            total += float(ctc_loss(logits, out_lengths, targets, target_lengths))
    return total / len(utts)


# 전체 발화(all)와 한글 전용(hangul_only) 두 부분집합에 대해 CER_ns·CER_ws를 요약한다. 정답이 채점 정규화 뒤
# 비는 발화는 뺀다. 숫자·라틴이 든 발화는 all에만 들어가며, 모델이 그 글자를 낼 수 없어 오류로 센다.
def score(utts, hyps: dict[str, str]) -> dict:
    out = {}
    for subset in ("all", "hangul_only"):
        rows = [u for u in utts if subset == "all" or u.hangul_only]
        ns = [s for s in (score_ns(u.text, hyps[u.id]) for u in rows) if s is not None]
        ws = [s for s in (score_ws(u.text, hyps[u.id]) for u in rows) if s is not None]
        out[subset] = {"cer_ns": summarize(ns), "cer_ws": summarize(ws)}
    return out


# 'cer_ns'(최상위)는 판정에 쓰는 한글 전용 CER_ns다. out_path를 주면 결과 JSON과 발화별 가설(.hyps.jsonl:
# id, 정답, 가설)을 저장하며, 이 가설 파일로 짝지은 비교(scripts/compare_runs.py)와 봉인 test 사후
# 분할(scripts/sealed_breakdown.py)을 새 디코딩 없이 한다.
# extra는 출처 기록용(7단계는 체크포인트 SHA-256을 넣는다).
# 호출 위치: train.py(매 epoch dev), main, scripts/stage7.py.
def evaluate_manifest(
    model,
    vocab,
    name: str,
    spec: ChunkSpec,
    device,
    logmel=None,
    streaming: bool = False,
    out_path: Path | None = None,
    checkpoint: str = "",
    amp: bool = True,
    extra: dict | None = None,
) -> dict:
    """매니페스트 하나를 디코딩·채점해 결과 dict를 돌려준다. `extra` 필드(출처 등)도 결과 파일에 저장한다."""
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", name)
    logmel = logmel or LogMel().to(device)
    # mode 문자열에 디코딩 방식과 청크·좌측 문맥을 남겨, 모든 수치에 디코딩 모드를 적는다는 규칙을 결과
    # 파일에서도 지킨다.
    if streaming:
        hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = f"streaming chunk{spec.chunk_size} left{spec.left_context}"
    elif spec.full:
        hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = "full"
    else:
        hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = f"simulated chunk{spec.chunk_size} left{spec.left_context}"
    scored = score(utts, hyps)
    result = {
        "manifest": name,
        "mode": mode,
        "precision": "fp16-autocast" if amp and device.type == "cuda" else "fp32",
        "checkpoint": checkpoint,
        "n": len(utts),
        **scored,
        "cer_ns": scored["hangul_only"]["cer_ns"]["rate"],
        **(extra or {}),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(out_path.with_suffix(".hyps.jsonl"), "w", encoding="utf-8") as f:
            for u in utts:
                f.write(json.dumps({"id": u.id, "ref": u.text, "hyp": hyps[u.id]}, ensure_ascii=False) + "\n")
    return result


# 명령행 평가. --chunk 0이면 full, 양수면 simulated(--streaming이면 실제 캐시 스트리밍), 좌측 문맥 기본
# 128프레임(5.12 s). --fp32는 fp32로 학습한 모델용이다. 결과 요약을 표준 출력에 쓰고 --out이면 파일로도
# 저장한다.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--chunk", type=int, default=0, help="청크 크기(인코더 프레임). 0이면 전체 문맥")
    p.add_argument("--left", type=int, default=128)
    p.add_argument(
        "--streaming", action="store_true", help="simulated 마스크 대신 실제 캐시 스트리밍으로 디코딩한다"
    )
    p.add_argument("--out", default="")
    p.add_argument("--fp32", action="store_true", help="fp16 autocast를 쓰지 않는다(--fp32로 학습한 모델용)")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    spec = ChunkSpec(args.chunk, args.left if args.chunk else -1)
    result = evaluate_manifest(
        model,
        vocab,
        args.manifest,
        spec,
        device,
        streaming=args.streaming,
        out_path=ROOT / args.out if args.out else None,
        checkpoint=args.checkpoint,
        amp=not args.fp32,
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "all" and k != "hangul_only"}, ensure_ascii=False)
    )
    for subset in ("all", "hangul_only"):
        print(subset, json.dumps(result[subset], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
