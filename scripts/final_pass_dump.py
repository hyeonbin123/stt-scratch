"""끝점 뒤 2차 패스의 프레임별 log-prob 통계 덤프(docs/experiments.md 규칙 변경 28, GPU).

후속 측정 P1과 같은 세션 조건으로 dev 매니페스트를 흘리고, 세그먼트가 닫힐 때 2차 패스가 계산한
전체 문맥 로짓에서 프레임마다 아래 값을 저장한다. 세션 조건: 발화마다 새 세션, 발화 뒤 −60 dBFS
가우시안 잡음 3 s(seed = 발화 인덱스, scripts/followup.py session과 같음), 서버 명령행 기본 구성
(320 ms 청크·좌측 128, rule2 0.8 s, 끝점 뒤 캐시 유지, step 경로 auto = GPU graph·CPU fast,
2차 패스 full, GPU fp16 autocast). 서비스 코드는 바꾸지 않는다: StreamSession을 상속한 DumpSession이
2차 패스 forward의 출력만 받아 적는다.

  argmax         프레임 argmax(int16). 이 열의 greedy 축약이 그 세그먼트 확정 텍스트다
  max_logprob    프레임 최대 log-prob(float32, = argmax 토큰의 log-prob)
  entropy        전체 어휘 분포의 엔트로피(nats, float32). top-k가 아니라 전체 분포에서 계산한다
  blank_logprob  blank(id 0)의 log-prob(float32). CTC prefix beam은 blank·비blank를 나눠 쓴다
  topk_ids       상위 k개(기본 10) 토큰 id(int16, log-prob 내림차순; 같은 값끼리 순서는 정하지 않음)
  topk_logprobs  그 log-prob(float32)

출력(세트마다, 커밋하지 않는 outputs/ 아래): <out-dir>/<세트>.npz = 위 배열들(모든 세그먼트의 프레임을
이은 것)과 <세트>.json = 출처(체크포인트 SHA-256·코드 커밋·장치), 구성, 어휘 SHA-256, npz SHA-256,
세그먼트 표(발화 id, 세그먼트 번호, final·endpoint, 끝점 규칙, 내보냈는지, 확정 텍스트, stream_text,
프레임 위치·수, mel 프레임 수), 발화 표(id, 출처, 한글 전용, 잡음 seed, 확정·1-pass 가설, 세그먼트
번호들). 전체 분포를 저장하면 약 1 GB라 top-k만 남긴다.

사용법:
  uv run --no-sync python scripts/final_pass_dump.py --manifests dev_mix,dev_h   # GPU, 약 35분
  uv run --no-sync python scripts/final_pass_dump.py --device cpu --limit 3 --out-dir work/<폴더>  # 연기 시험
"""

# [왜 P1 조건인가] 신뢰도는 서비스 확정 자막(2차 패스)의 어절에 붙일 값이므로 서비스와 같은 세션 경로에서
# 얻어야 하고, 같은 조건의 확정 가설이 P1 측정으로 이미 커밋돼
# 있어(reports/followup/hyps/*-session-rule2_0.8-final_pass.jsonl) 덤프가 측정된 구성을 그대로 재현했는지
# 확인할 수 있다(scripts/confidence_eval.py의 재현 조건).
# [왜 지연을 재지 않는가] 정확도·신뢰도용 덤프라 쉬는 상태를 기다리지 않고 예열도 하지 않는다. 기록되는
# pass_ms에는 forward 훅의 시간이 들어 있어 지연 수치로 쓰지 않는다.
# [재사용] 매니페스트(봉인 세트 거부)·출처·결과 경로 검사는 scripts/followup.py, 잡음·패킷 규약은
# scripts/stage6.py를 그대로 쓴다. 다른 평가(신뢰도, 확정 자막 디코딩 실험)는 load_dump로 읽는다.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# numpy: 덤프 배열 저장. torch: 모델 실행과 프레임 통계 계산.
import numpy as np
import torch

# 프로젝트 모듈: 모델 로드·어휘 검사, CUDA Graph 캡처 상태, 특징, 청크 설정, 끝점, 세션, 어휘.
from stt_scratch.evaluate import check_vocab, load_model
from stt_scratch.model import streaming
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import Event, StreamSession
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import followup  # noqa: E402
from stage6 import noise, packets  # noqa: E402

FORMAT_VERSION = 1  # 덤프 형식 번호(배열·표의 뜻을 바꾸면 올린다)
FRAME_KEYS = ("argmax", "max_logprob", "entropy", "blank_logprob", "topk_ids", "topk_logprobs")


# 2차 패스 로짓(유효 프레임만, (T, V))에서 프레임 통계를 계산한다. 로짓은 greedy와 같게 float로 바꿔
# 쓰므로(fp16 autocast 출력 → fp32) argmax가 greedy_batch의 argmax와 같고, log_softmax·엔트로피는 fp32에서
# 계산한다. 확률이 0으로 넘친(underflow) 항은 엔트로피에서 0으로 센다(p·log p → 0). 출력: FRAME_KEYS → CPU
# numpy 배열(argmax·max_logprob·entropy·blank_logprob는 (T,), topk 둘은 (T, k)). 주의: 어휘가 32,767개를
# 넘으면 int16에 담을 수 없어 거부한다(실제 어휘 1,288개).
@torch.no_grad()
def frame_stats(logits: torch.Tensor, top_k: int) -> dict[str, np.ndarray]:
    """2차 패스 로짓 (T, V) → 프레임별 argmax·최대 log-prob·엔트로피·blank log-prob·top-k."""
    if logits.shape[-1] > np.iinfo(np.int16).max:
        raise ValueError(f"vocabulary of {logits.shape[-1]} tokens does not fit int16 ids")
    lf = logits.float()
    argmax = lf.argmax(dim=-1)
    lp = torch.log_softmax(lf, dim=-1)
    p = lp.exp()
    entropy = -torch.where(p > 0, p * lp, torch.zeros_like(lp)).sum(dim=-1)
    top = torch.topk(lp, min(top_k, lp.shape[-1]), dim=-1)
    return {
        "argmax": argmax.cpu().numpy().astype(np.int16),
        "max_logprob": lp.gather(-1, argmax[:, None])[:, 0].cpu().numpy().astype(np.float32),
        "entropy": entropy.cpu().numpy().astype(np.float32),
        "blank_logprob": lp[:, 0].cpu().numpy().astype(np.float32),
        "topk_ids": top.indices.cpu().numpy().astype(np.int16),
        "topk_logprobs": top.values.cpu().numpy().astype(np.float32),
    }


# 서비스 세션과 똑같이 동작하면서 세그먼트를 닫을 때마다 2차 패스의 프레임 통계를 남기는 세션. decode_full에서
# 모델 forward 훅으로 부모의 2차 패스 출력(로짓, 중간 로짓, 출력 길이)을 받아 통계를 만들고, _close_event에서
# 그 세그먼트의 실제 닫힘 이벤트(종류·규칙·확정 텍스트·stream_text)와 함께 closed에 적는다. 부모의 계산을
# 그대로 쓰므로 이벤트는 StreamSession과 같다(tests/test_confidence.py). 인자: StreamSession과 같고
# final_pass는 "full"이어야 한다. top_k: 남길 상위 토큰 수. keep_mel: 세그먼트 mel도 남김(테스트용; 실제
# 덤프에서는 GPU 메모리를 아끼려고 끈다). closed 항목: {"segment", "kind", "rule", "text", "stream_text",
# "capture"}; capture는 2차 패스가 돌았으면 {"stats": frame_stats 결과, "frames": 출력 프레임 수,
# "mel_frames": 입력 mel 프레임 수, "pass_ms"}(+ "mel"), 넣은 mel이 없는 세그먼트(끝점 직후 바로 flush)는
# None.
class DumpSession(StreamSession):
    def __init__(self, *args, top_k: int = 10, keep_mel: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.final_pass != "full":
            raise ValueError("the dump records the final pass; use final_pass='full'")
        self.top_k = top_k
        self.keep_mel = keep_mel
        self.closed: list[dict] = []
        self._captured: dict | None = None

    @torch.no_grad()
    def decode_full(self, mel: torch.Tensor) -> tuple[str, float]:
        outputs = []
        handle = self.model.register_forward_hook(lambda _module, _inputs, out: outputs.append(out))
        try:
            text, ms = super().decode_full(mel)
        finally:
            handle.remove()
        logits, _inter, lengths = outputs[-1]
        frames = int(lengths[0])
        self._captured = {
            "stats": frame_stats(logits[0, :frames], self.top_k),
            "frames": frames,
            "mel_frames": int(mel.shape[0]),
            "pass_ms": ms,
        }
        if self.keep_mel:
            self._captured["mel"] = mel.detach().cpu()
        return text, ms

    def _close_event(self, text: str, rule: str, ms: float) -> Event:
        self._captured = None  # 예열 등에서 남은 값이 다른 세그먼트에 붙지 않게
        event = super()._close_event(text, rule, ms)
        captured, self._captured = self._captured, None
        self.closed.append(
            {
                "segment": event.segment,
                "kind": event.kind,
                "rule": rule,
                "text": event.text,
                "stream_text": event.stream_text,
                "capture": captured,
            }
        )
        return event


# 서버 명령행 기본값(service.server.build_parser)에서 세션 구성을 만든다. 서버 기본값이 P1 판정 구성(rule2 0.8
# s, 캐시 유지, auto, full)과 같으므로 덤프도 같은 구성이 되고, 기본값이 바뀌면 재현 조건이 그 차이를
# 드러낸다.
def session_config(device: torch.device) -> dict:
    from stt_scratch.service import server as service

    defaults = service.build_parser().parse_args([])
    if defaults.final_pass != "full":
        raise SystemExit("the server default final pass is not 'full'; there is no final pass to dump")
    return {
        "spec": ChunkSpec(defaults.chunk, defaults.left),
        "endpoint": EndpointConfig(rule2_s=defaults.rule2),
        "reset_cache_on_endpoint": defaults.reset_cache,
        "step_path": service.resolve_step_path(defaults.step_path, str(device)),
        "final_pass": defaults.final_pass,
    }


def _empty(key: str, top_k: int) -> np.ndarray:
    shape = (0, top_k) if key.startswith("topk") else (0,)
    return np.zeros(shape, dtype=np.int16 if key in ("argmax", "topk_ids") else np.float32)


# 매니페스트 하나를 흘려 덤프 배열과 세그먼트·발화 표를 만든다. 흐름은 scripts/followup.py
# run_sessions_two_pass와 같다(발화마다 새 세션, 같은 잡음 seed, 같은 패킷, flush). 발화 가설: hyp_final =
# final 이벤트 텍스트를 공백으로 이음, hyp_stream = 비지 않은 stream_text를 이음(P1 기록 파일과 같은 정의).
def dump_set(model, vocab, utts, store, session_args: dict, device, tail_s: float, top_k: int):
    logmel = LogMel().to(device)
    frames: dict[str, list[np.ndarray]] = {key: [] for key in FRAME_KEYS}
    segments: list[dict] = []
    utterances: list[dict] = []
    offset = 0
    for k, u in enumerate(utts):
        wav = torch.cat([store.waveform(u.id), noise(tail_s, seed=k)]).to(device)
        session = DumpSession(model, vocab, logmel=logmel, top_k=top_k, **session_args)
        closes = []
        for _pos, piece in packets(wav):
            closes.extend(e for e in session.push_samples(piece) if e.kind != "partial")
        closes.extend(session.flush())
        emitted = {e.segment for e in closes}
        seg_ids = []
        for rec in session.closed:
            cap = rec["capture"]
            n = 0 if cap is None else cap["frames"]
            if cap is not None:
                for key in FRAME_KEYS:
                    frames[key].append(cap["stats"][key])
            seg_ids.append(len(segments))
            segments.append(
                {
                    "utt": u.id,
                    "segment": rec["segment"],
                    "kind": rec["kind"],
                    "rule": rec["rule"],
                    "emitted": rec["segment"] in emitted,
                    "text": rec["text"],
                    "stream_text": rec["stream_text"],
                    "offset": offset,
                    "frames": n,
                    "mel_frames": 0 if cap is None else cap["mel_frames"],
                    "pass_ms": 0.0 if cap is None else cap["pass_ms"],
                }
            )
            offset += n
        utterances.append(
            {
                "id": u.id,
                "corpus": u.corpus,
                "hangul_only": u.hangul_only,
                "noise_seed": k,
                "hyp_final": " ".join(e.text for e in closes if e.kind == "final"),
                "hyp_stream": " ".join(e.stream_text for e in closes if (e.stream_text or "").strip()),
                "segments": seg_ids,
            }
        )
    arrays = {key: np.concatenate(v) if v else _empty(key, top_k) for key, v in frames.items()}
    return arrays, segments, utterances


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 덤프 쓰기·읽기. npz(압축 없음, 읽기 빠름)를 먼저 쓰고 그 SHA-256을 메타에 넣어 json을 쓴다. 읽을 때는 해시를
# 다시 계산해 두 파일이 같은 실행의 것인지 확인하고, 세그먼트 위치가 프레임 배열을 빈틈없이 덮는지
# 본다(어긋나면 SystemExit: 잘못된 덤프로 평가하지 않는다).
def write_dump(directory: Path, name: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    npz_path = directory / f"{name}.npz"
    np.savez(npz_path, **arrays)
    meta = {**meta, "npz_sha256": _sha256(npz_path)}
    (directory / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {npz_path} ({meta['frames']} frames, {len(meta['segments'])} segments)", flush=True)


def load_dump(directory: Path, name: str) -> tuple[dict, dict[str, np.ndarray]]:
    """<directory>/<name>.json·.npz를 읽어 (메타, 배열)을 돌려준다. 해시·위치가 어긋나면 SystemExit."""
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    npz_path = directory / f"{name}.npz"
    if _sha256(npz_path) != meta["npz_sha256"]:
        raise SystemExit(
            f"{npz_path} does not match the SHA-256 in {name}.json; the dump files are from different runs"
        )
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    n = len(arrays["argmax"])
    expected = 0
    for seg in meta["segments"]:
        if seg["offset"] != expected:
            raise SystemExit(f"{name}: segment offsets do not tile the frame arrays")
        expected += seg["frames"]
    if expected != n or any(len(arrays[key]) != n for key in FRAME_KEYS):
        raise SystemExit(f"{name}: frame arrays and the segment table disagree")
    return meta, arrays


def parse_manifests(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in followup.DEV_SETS]
    if not names or bad:
        raise SystemExit(f"--manifests takes dev sets only ({', '.join(followup.DEV_SETS)}), got {text!r}")
    return names


# 덤프 명령: 매니페스트 이름(dev만)과 결과 파일(이미 있으면 --overwrite 없이 거부)을 계산 전에 확인하고,
# 세트마다 흘려 쓴다. 메타의 출처는 followup.provenance(체크포인트 SHA-256, 코드 커밋·미커밋 여부, 장치,
# torch, --limit, 시각).
def cmd_dump(args, model, vocab: Vocab, device: torch.device) -> None:
    names = parse_manifests(args.manifests)
    for name in names:
        followup.out_file(args, f"{name}.npz")
        followup.out_file(args, f"{name}.json")
    session_args = session_config(device)
    # 2차 패스 길이 묶음과 정밀도는 세션이 정하는 값을 그대로 적는다(서버도 이 둘은 세션 기본값을 쓴다).
    probe = DumpSession(model, vocab, logmel=LogMel().to(device), **session_args)
    config = {
        "chunk": session_args["spec"].chunk_size,
        "left": session_args["spec"].left_context,
        **asdict(session_args["endpoint"]),
        "reset_cache_on_endpoint": session_args["reset_cache_on_endpoint"],
        "step_path": session_args["step_path"],
        "final_pass": session_args["final_pass"],
        "final_pass_bucket": probe.final_pass_bucket,
        "precision": "fp16-autocast" if probe.amp else "fp32",
        "tail_s": args.tail_s,
        "tail_noise": "-60 dBFS gaussian, seed = utterance index (as scripts/followup.py session)",
        "top_k": args.top_k,
    }
    vocab_sha = _sha256(followup.ROOT / "manifests" / "vocab.json")
    for name in names:
        utts, store = followup.manifest(name, args.limit)
        t0 = time.perf_counter()
        arrays, segments, utterances = dump_set(
            model, vocab, utts, store, session_args, device, args.tail_s, args.top_k
        )
        meta = {
            "format": FORMAT_VERSION,
            "set": name,
            "provenance": followup.provenance(args, device),
            "config": config,
            "vocab_sha256": vocab_sha,
            "graph": streaming.graph_status(model, session_args["spec"]),
            "seconds": round(time.perf_counter() - t0, 1),
            "frames": int(len(arrays["argmax"])),
            "segments": segments,
            "utterances": utterances,
        }
        write_dump(followup.ROOT / args.out_dir, name, arrays, meta)


# 명령행 인자. 기본값은 등록 측정(docs/experiments.md 규칙 변경 28)의 값이고, 연기 시험은
# --device cpu --limit N --out-dir work/<폴더>로 줄여 돌린다(모듈 docstring의 사용법). 세션 구성
# (청크·끝점·캐시 정책·step 경로·2차 패스)은 인자로 받지 않고 서버 명령행 기본값을 그대로
# 쓴다(session_config): 덤프가 서비스 확정 자막과 같은 구성이어야 평가 도구가 P1 확정 가설과 재현 조건을
# 대조할 수 있기 때문이다.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="끝점 뒤 2차 패스의 프레임별 log-prob 통계 덤프(docs/experiments.md 규칙 변경 28). "
        "P1과 같은 세션 조건으로 dev 세트를 흘려 세그먼트마다 argmax·최대 log-prob·엔트로피·blank·top-k를 "
        "outputs/에 남긴다. 봉인 test 세트는 열지 않는다."
    )
    p.add_argument(
        "--manifests", default="dev_mix,dev_h", help="쉼표로 구분한 dev 세트 이름(등록: dev_mix,dev_h)"
    )
    p.add_argument("--checkpoint", default="outputs/stage4/trimtail/avg10.pt", help="측정 모델(최종 모델)")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="실행 장치(기본: CUDA가 보이면 cuda, 아니면 cpu). 등록 측정은 cuda(서버 기본 구성 그대로 graph "
        "step·fp16 autocast)다. cpu는 step 경로 fast·fp32라 구성이 P1과 달라 평가 도구(confidence_eval·"
        "bias_eval)가 재현 대조를 하지 않고 판정을 남기지 않는다(연기 시험용)",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/followup/final_pass_dump",
        help="결과 폴더(저장소 기준, 커밋하지 않음). 연기 시험은 work/ 아래로",
    )
    p.add_argument("--top-k", type=int, default=10, help="프레임마다 남길 상위 토큰 수(등록 10)")
    p.add_argument(
        "--tail-s", type=float, default=3.0, help="발화 뒤에 붙이는 -60 dBFS 잡음 길이(초, P1과 같은 3)"
    )
    p.add_argument(
        "--limit", type=int, default=0, help="앞 N발화만(연기 시험용; 0 = 전수, 평가는 전수 덤프로만)"
    )
    p.add_argument("--overwrite", action="store_true", help="이미 있는 덤프 파일을 덮어쓴다")
    return p


# 진입점: --top-k를 검사하고 표준 출력을 UTF-8로 바꾼 뒤(Windows 콘솔 기본 cp949에서 한글 메시지가 깨지지
# 않게) 체크포인트를 올리고, 어휘 토큰 수를 체크포인트 출력 크기와 대조한 다음 덤프한다. 계산 전에 멈추는
# 경우(SystemExit, 종료 코드 1): --top-k < 1, 어휘 크기 불일치, dev가 아닌 세트 이름, 결과 파일이 이미 있음
# (--overwrite 없이), 서버 기본 2차 패스가 full이 아님.
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device(args.device)
    model, _ = load_model(followup.ROOT / args.checkpoint, device)
    vocab = Vocab.load(followup.ROOT / "manifests" / "vocab.json")
    check_vocab(model, vocab)  # 어휘 토큰 수 ≠ 체크포인트 출력 크기면 덤프 전에 멈춘다
    cmd_dump(args, model, vocab, device)


if __name__ == "__main__":
    main(sys.argv[1:])
