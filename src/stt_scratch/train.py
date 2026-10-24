"""학습 루프: fp16 autocast, fp32 CTC, 동적 청크 학습, warmup+cosine, epoch마다 dev CER과 dev 손실.

사용법(사전 등록 설정은 docs/experiments.md 참고):
  uv run python -m stt_scratch.train --run outputs/stage2/runA --epochs 100 --lr 1e-3 --seed 1

로그의 카운터: `step` = 옵티마이저 step 시도 횟수(LR 스케줄, 예산, '같은 step 수' 규칙이 이 값을 기준으로
돈다), `updates` = GradScaler가 건너뛰지 않은 시도, `skipped` = 건너뛴 시도 + 비유한 손실. 발산
(docs/experiments.md "fp16 규칙"): 비유한 손실 누적 3회 이상, scale < 1이 100 step 연속, 또는 epoch 3 끝의
dev CTC 손실이 epoch 1 끝 값보다 낮지 않음. 발산한 런은 멈추고 last.pt를 남긴다.
"""

# [파일 역할] 모든 단계(1단계 LR 선택, 2단계 주 모델, 3단계 비교군, 4단계 이어 학습)가 공유하는 학습 진입점.
# 한 번 실행이 런 디렉터리 하나(--run)를 만들고 meta.json, train.jsonl(step 로그), epochs.jsonl(epoch 요약·dev
# 결과), epoch-NNN.pt·last.pt(체크포인트)를 남긴다.
# [흐름] 인자 파싱 → 데이터·모델·옵티마이저 준비 → (--resume면 체크포인트·처방 대조·로그 정리, --init이면
# 가중치만 이어받기) → meta 기록 → epoch 루프[배치마다: GPU 속도 교란·log-mel → SpecAugment 커리큘럼 판단 →
# 동적 청크 설정 → LR 스케줄 → (TrimTail·CR-CTC 변형) → fp16 autocast forward·손실 → GradScaler
# backward·clip·step → 발산 카운터 → 로그] → epoch 끝 dev CER·dev CTC 손실 → dev 손실 발산 규칙 → 체크포인트.
# [설계 이유] 사전 등록 규칙(docs/experiments.md "fp16 규칙"과 규칙 변경 1~3, 10, 13, 15, 16)을 사람이 지키는
# 절차가 아니라 코드로 강제한다: 발산 판정, 재개 조건, 처방 대조, step 정의가 모두 여기 있다.
# [연결] scripts/stage1_sweep.py·stage3.py·stage4.py가 이 모듈을 하위 프로세스로 돌리고, stage3·stage4는
# build_parser·expected_recipe·completed_epochs로 기존 런을 대조한다.

from __future__ import annotations

# 표준 라이브러리 중 subprocess는 meta.json에 git 커밋 해시를 남기려고, os는 PYTORCH_CUDA_ALLOC_CONF 설정을
# 기록하고 체크포인트 임시 파일을 os.replace로 바꿔 넣으려고, contextlib는 실패한 저장의 임시 파일을 지울 때
# 없어도 넘어가려고, random은 속도 교란·동적 청크·TrimTail용 시드 고정 난수에 쓴다.
import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# 프로젝트 모듈: 데이터 입력(data), 매 epoch dev 평가·dev 손실(evaluate), 모델·설정, 손실 함수(ctc),
# 특징·증강(features), 청크 설정(mask), 어휘·CTC 최소 프레임(text).
import torch
from torch import nn

from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.evaluate import ctc_loss_manifest, evaluate_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import cr_consistency, peak_first_regularizer, total_loss
from stt_scratch.model.features import LogMel, speed_perturb, trim_tail
from stt_scratch.model.mask import ChunkSpec, sample_chunk_spec
from stt_scratch.text import Vocab, ctc_min_frames

# ROOT: 저장소 루트(--run, manifests/, data/pcm/의 기준). SPEEDS: 속도 교란 배율 세 가지(발화마다 하나를
# 고른다). COUNTERS: 발산 판정·보고에 쓰는 누적 카운터로, 체크포인트에 저장되어 재개 뒤에도 이어진다.
ROOT = Path(__file__).resolve().parents[2]
SPEEDS = (0.9, 1.0, 1.1)
COUNTERS = ("nan_count", "low_scale_streak", "skipped", "updates")
# 처방(recipe) 키: 학습 결과를 바꾸는 인자 목록. 체크포인트마다 저장해 --resume가 다른 처방으로 이어 학습하는
# 실수를 막고(규칙 변경 15), 3·4단계 실행기가 기존 런을 다시 쓰기 전에 등록 처방과 대조할 때 쓴다(규칙 변경
# 19, 23). --run, --resume, --cuda-memory-fraction, --log-every처럼 결과를 바꾸지 않는 인자는 넣지 않는다.
# --resume는 저장된 처방과 같아야 하며, 일부러 바꿀 때는 --allow-recipe-change를 명시한다.
RECIPE_KEYS = (
    "train",
    "lr",
    "warmup",
    "schedule_steps",
    "epochs",
    "max_duration",
    "seed",
    "d_model",
    "n_layers",
    "ffn_dim",
    "sc_layers",
    "sc_weight",
    "no_sc_feedback",
    "no_bacon",
    "no_qk_norm",
    "offline_only",
    "no_perturb",
    "no_augment",
    "augment_start_epoch",
    "augment_start_step",
    "perturb_start_epoch",
    "dev_loss_rule",
    "dev_loss_set",
    "fp32",
    "init",
    "block",
    "cgmlp_units",
    "cgmlp_kernel",
    "merge_kernel",
    "time_masks",
    "cr_ctc_alpha",
    "trimtail_ms",
    "peak_first_lambda",
)


# meta.json에 남길 현재 코드 버전(짧은 커밋 해시). git이 없거나 저장소 밖이면 "unknown".
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# 명령행 인자와 manifests/mel_stats.json(전역 정규화 통계)으로 ModelConfig를 채워 모델을 만든다.
# --offline-only(2단계 통제군 Run B)면 config의 full_context_prob를 0.0으로 기록하지만, 이 경우 학습 루프가
# sample_chunk_spec을 부르지 않고 항상 ChunkSpec(0, -1)(전체 문맥)을 쓰므로 실제 학습에는 영향이 없다(기록된
# 값의 뜻은 '전체 문맥 확률 0'이라 실제 동작과 반대이니 config를 읽을 때 주의).
def build_model(vocab: Vocab, args: argparse.Namespace) -> tuple[ChunkConformerCTC, ModelConfig]:
    stats = json.loads((ROOT / "manifests" / "mel_stats.json").read_text(encoding="utf-8"))
    cfg = ModelConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        sc_layers=tuple(int(x) for x in args.sc_layers.split(",")) if args.sc_layers else (),
        sc_weight=args.sc_weight,
        sc_feedback=not args.no_sc_feedback,
        bacon=not args.no_bacon,
        qk_norm=not args.no_qk_norm,
        block=args.block,
        cgmlp_units=args.cgmlp_units,
        cgmlp_kernel=args.cgmlp_kernel,
        merge_kernel=args.merge_kernel,
        time_masks=args.time_masks,
        full_context_prob=0.0 if args.offline_only else 0.5,
        mel_mean=stats["mean"],
        mel_std=stats["std"],
    )
    return ChunkConformerCTC(cfg), cfg


# 학습률 스케줄: warmup step 동안 peak까지 선형 증가, 그 뒤 total까지 cosine으로 floor(1e-5)까지 감소.
# warmup은 2,000 step(규칙 변경 1, 원래 4,000). total은 --schedule-steps(1단계는 2단계와 같은 69,700 step을
# 넘겨 5 epoch만 도는 '잘린 복사본'이 되게 함, 규칙 변경 13)이고 없으면 epochs × epoch당 step이다.
def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# collate가 넘긴 파형 목록으로 학습 입력을 만든다: 파형을 GPU로 옮기고, perturb면 발화마다 SPEEDS 중 하나로
# 속도 교란, 0으로 패딩해 LogMel. 정답은 이어 붙인 1차원 int32 토큰과 발화별 길이로 만든다(ctc_loss가 다시 (B,
# S)로 펼친다).
def prepare_batch(batch: dict, logmel: LogMel, device: torch.device, rng: random.Random, perturb: bool):
    wavs = []
    for wav in batch["wavs"]:
        wav = wav.to(device, non_blocking=True)
        if perturb:
            wav = speed_perturb(wav, rng.choice(SPEEDS))
        wavs.append(wav)
    lengths = torch.tensor([w.numel() for w in wavs], device=device)
    padded = torch.zeros(len(wavs), int(lengths.max()), device=device)
    for i, w in enumerate(wavs):
        padded[i, : w.numel()] = w
    mel, frames = logmel(padded, lengths)
    tokens = batch["tokens"]
    targets = torch.tensor([t for seq in tokens for t in seq], device=device, dtype=torch.int32)
    target_lengths = torch.tensor([len(seq) for seq in tokens], device=device, dtype=torch.int32)
    return mel, frames, targets, target_lengths


# 재개용 난수 상태 묶음: Python random(속도 교란·청크·TrimTail), torch CPU(SpecAugment 등), CUDA. 체크포인트에
# 넣어 재개한 런이 멈추지 않은 런과 같은 난수 흐름을 잇게 한다(규칙 변경 10).
def rng_states(rng: random.Random, device: torch.device) -> dict:
    states = {"python": rng.getstate(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["torch_cuda"] = torch.cuda.get_rng_state(device)
    return states


# 저장한 난수 상태를 되돌린다. Python 상태의 내부 값은 random.setstate가 요구하는 튜플로 바꿔 넣고,
# torch.load의 map_location이 상태 바이트 텐서를 GPU로 옮기면 set_rng_state가 실패하므로 .cpu()로
# 되돌린다(docs/ai-verification.md).
def restore_rng(states: dict, rng: random.Random, device: torch.device) -> None:
    version, internal, gauss = states["python"]
    rng.setstate((version, tuple(internal), gauss))
    torch.set_rng_state(states["torch_cpu"].cpu())
    if device.type == "cuda" and "torch_cuda" in states:
        torch.cuda.set_rng_state(states["torch_cuda"].cpu(), device)


# 같은 epoch에서 두 번 재개해도 이전 재개 기록이 남는다.
def unique_path(path: Path) -> Path:
    """path를 돌려주되, 이미 있으면 확장자 앞에 -2, -3, …을 붙인 새 경로를 준다(기록을 덮어쓰지 않는다)."""
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


# 체크포인트 한 개를 쓴다. 가중치뿐 아니라 옵티마이저·GradScaler 상태, epoch·step, config, 카운터, dev 손실
# 이력, 난수 상태, 처방, dev 손실 규칙 적용 여부까지 넣어 재개가 '멈추지 않은 것과 같은 학습'이 되게 한다.
# epoch_complete=False(발산·--max-steps로 epoch 도중 멈춤)인 체크포인트는 재개를 거부하는 표시다.
# 원자적 쓰기: 같은 디렉터리의 임시 파일 <이름>.tmp에 다 쓴 뒤 os.replace로 바꿔 넣는다(같은 볼륨이면 Windows·
# POSIX 모두 원자적 교체). 그래서 저장 도중 멈춰도(epoch 경계에서 멈추고 --resume하는 운영) 이전 last.pt는
# 그대로 남고, 새 파일은 완전하거나 없다. 전에는 대상에 바로 써서 중단되면 last.pt가 깨진 zip으로 남아
# --resume의 torch.load와 3·4단계 실행기의 completed_epochs가 예외로 멈췄다. 임시 파일 이름이 .pt로 끝나지
# 않으므로 실행기의 `*.pt`·`epoch-*.pt` 검색에 잡히지 않는다. 쓰다 실패하면(예외·KeyboardInterrupt) 임시
# 파일을 지우고 예외를 그대로 올린다. epochs.jsonl 요약 줄을 체크포인트보다 먼저 쓰는 순서와 재개 규칙(규칙
# 변경 10·15)은 그대로다(tests/test_train_io.py).
def save_checkpoint(
    path: Path,
    model,
    opt,
    scaler,
    cfg,
    epoch: int,
    step: int,
    counters: dict,
    dev_loss,
    rng,
    device,
    complete,
    divergence=None,
    recipe=None,
    dev_rule_active=True,
):
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(
            {
                "recipe": recipe,
                "dev_loss_rule_active": dev_rule_active,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "config": asdict(cfg),
                "epoch_complete": complete,
                "divergence": divergence,
                "counters": dict(counters),
                "dev_loss": list(dev_loss),
                "rng": rng_states(rng, device),
            },
            tmp,
        )
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


# main뿐 아니라 3·4단계 실행기가 같은 파서로 '이 명령이면 어떤 처방이 기록될지'를 계산하므로(expected_recipe)
# 함수로 분리했다.
def build_parser() -> argparse.ArgumentParser:
    """학습 인자 전체를 정의한다. 실행기는 이 파서로 후보의 기대 처방을 계산한다."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--train", default="train")
    p.add_argument("--devs", default="dev_mix,dev_h")
    p.add_argument("--dev-loss-set", default="dev_mix", help="dev CTC 손실 발산 검사에 쓸 매니페스트")
    # 기본값의 근거: 100 epoch과 max-duration 300 s는 0단계 예산 규칙 판정(epoch 약 7.8분 → 100 epoch ≈ 13 h,
    # 14 h 안), 피크 메모리 4.31 GB(docs/experiments.md 0단계 결과). 실제 채택 peak LR은 1단계 판정의 2e-3이라
    # 실행 때 --lr 2e-3을 넘긴다(README 명령).
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help="warmup+cosine 스케줄의 전체 step 수. 0 = epoch 수 × epoch당 step. 1단계 런은 2단계 전체 "
        "값을 넘겨, 짧은 런이 긴 스케줄의 앞부분을 잘라 낸 복사본이 되게 한다",
    )
    p.add_argument(
        "--dev-loss-rule",
        default="1,3",
        help="A,B(1부터 센 epoch): epoch B 끝의 dev CTC 손실이 epoch A 끝 값보다 낮지 않으면 발산"
        "(docs/experiments.md fp16 규칙, 규칙 변경 3/13). --init이면 검사하지 않는다",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--ffn-dim", type=int, default=1024)
    # 중간 CTC 헤드 기본값은 Run A의 자기조건 CTC. 3단계 (a)와 최종 모델은 --sc-layers 6 --sc-weight 0.3
    # --no-sc-feedback(InterCTC 단독)이다.
    p.add_argument("--sc-layers", default="3,6,9")
    p.add_argument("--sc-weight", type=float, default=0.5)
    p.add_argument(
        "--no-sc-feedback", action="store_true", help="InterCTC 단독: 중간 손실만 두고 피드백은 끈다"
    )
    p.add_argument("--no-bacon", action="store_true")
    p.add_argument("--no-qk-norm", action="store_true", help="q와 k의 헤드별 LayerNorm(QK-norm)을 뺀다")
    p.add_argument("--block", default="conformer", choices=["conformer", "ebranchformer"])
    p.add_argument("--cgmlp-units", type=int, default=1024, help="E-Branchformer cgMLP 폭(둘로 나누기 전)")
    p.add_argument("--cgmlp-kernel", type=int, default=31)
    p.add_argument("--merge-kernel", type=int, default=31)
    p.add_argument("--time-masks", type=int, default=5, help="발화당 SpecAugment 시간 마스크 개수")
    p.add_argument(
        "--trimtail-ms", type=int, default=0, help="TrimTail: 발화 끝 음성을 U{0..N} ms 잘라 낸다(0 = 끔)"
    )
    p.add_argument("--peak-first-lambda", type=float, default=0.0, help="Peak-First 정규화 가중치(0 = 끔)")
    p.add_argument(
        "--cr-ctc-alpha",
        type=float,
        default=0.0,
        help="CR-CTC 일관성 가중치(0 = 끔). 배치마다 같은 배치를 두 번 이어 붙여 증강한 두 뷰를 만든다",
    )
    p.add_argument(
        "--offline-only", action="store_true", help="통제 런: 청크 마스크 없이 항상 전체 문맥으로 학습한다"
    )
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    # SpecAugment·속도 교란 시작 시점. 처음부터 SpecAugment를 걸면 CTC 정렬이 시작되지 않아 기본값을 epoch
    # 3으로 둔다(규칙 변경 13). 배치 규칙이 다른 3단계 (d)는 같은 step(2,091)에 켜도록 --augment-start-step을
    # 쓰고(규칙 변경 15), 이미 정렬된 모델을 이어 학습하는 4단계는 둘 다 0으로 처음부터 켠다(규칙 변경 16).
    p.add_argument(
        "--augment-start-epoch",
        type=int,
        default=3,
        help="이 epoch(0부터 셈) 전에는 SpecAugment를 끈다. CTC 정렬이 먼저 형성되게 한다(규칙 변경 13)",
    )
    p.add_argument("--perturb-start-epoch", type=int, default=0, help="이 epoch 전에는 속도 교란을 끈다")
    p.add_argument(
        "--augment-start-step",
        type=int,
        default=0,
        help="0보다 크면 --augment-start-epoch 대신 이 옵티마이저 step 시도부터 SpecAugment를 켠다"
        "(배치 규칙이 다른 후보를 같은 step 수에 맞추려는 것)",
    )
    p.add_argument(
        "--allow-recipe-change",
        action="store_true",
        help="저장된 처방이 지금 인자와 달라도 재개한다(차이는 meta에 기록된다)",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--eval-chunk", type=int, default=16, help="매 epoch 평가할 simulated 청크 크기(프레임)")
    p.add_argument("--max-steps", type=int, default=0, help="이 step 수에서 일찍 멈춘다(벤치마크용)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default="", help="시작 가중치로 쓸 체크포인트(4단계 이어 학습)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.0,
        help="PyTorch GPU 메모리 풀을 카드 메모리의 이 비율로 제한한다(0 = 제한 없음). 제한이 없으면 캐싱 "
        "할당기가 조각난 블록을 붙잡고(실사용 4.7 GB에 9.4 GB 예약) Windows가 넘친 부분을 페이징한다",
    )
    return p


# scripts/stage3.py·stage4.py가 기존 런의 meta.json과 비교해, 등록 처방과 같은 런만 이어받거나 다시 쓴다.
def expected_recipe(argv: list[str]) -> dict:
    """train.py에 argv를 넘겼을 때 기록될 처방(RECIPE_KEYS)."""
    args = build_parser().parse_args(argv)
    return {k: getattr(args, k) for k in RECIPE_KEYS}


# epochs.jsonl 줄 수로 세면 체크포인트 저장 전에 멈춘 epoch까지 '완료'로 셀 수 있어 체크포인트를 기준으로
# 삼는다(Codex 2단계 검토 반영, docs/ai-verification.md). 3·4단계 실행기가 완료·재개 여부 판단에 쓴다.
def completed_epochs(run: Path) -> int:
    """런의 last.pt 기준으로 끝낸 epoch 수를 센다.

    last.pt가 없거나, epoch 도중에 썼거나, 발산한 런이면 0이다. 두 파일을 쓰는 사이에 멈추면 epoch 로그가
    체크포인트보다 앞설 수 있다.
    """
    path = run / "last.pt"
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not state.get("epoch_complete", False) or state.get("divergence"):
        return 0
    return int(state["epoch"]) + 1


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    # 같은 --run에 이전 시도의 로그가 있으면 --resume 없이는 시작하지 않는다. 서로 다른 시도의 로그가 한
    # 파일에 섞이지 않게 하려는 규칙이다(규칙 변경 10).
    if not args.resume and any((run / f).exists() for f in ("train.jsonl", "epochs.jsonl", "last.pt")):
        raise SystemExit(
            f"{run} already holds a run; pass --resume or use a new --run (logs are never mixed)"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GPU 메모리 풀 상한(선택). 2단계 Run A에서 할당기가 실사용 4.7 GB에 9.4 GB를 붙잡고 Windows 화면 처리와
    # 합쳐 11 GB를 넘으면서 시스템 메모리로 페이징되어 epoch 시간이 두 배 가까이 늘었다. 상한 0.8과
    # PYTORCH_CUDA_ALLOC_CONF 설정으로 해결했다(docs/experiments.md 2단계 진행 기록, README). 할당기 동작만
    # 바뀌고 계산은 같아 처방 키에 넣지 않는다.
    if device.type == "cuda" and args.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.cuda.current_device())
    torch.manual_seed(args.seed)
    # cuDNN 자동 튜닝을 끈다. 배치마다 모양이 달라 켜 두면 step마다 알고리즘 탐색을 다시 돌려 1.36
    # s/step이었고, 끄자 0.66 s/step이 됐다(docs/experiments.md 0단계 결과).
    torch.backends.cudnn.benchmark = (
        False  # 배치마다 입력 모양이 바뀌어 모양별 자동 튜닝이 step 시간을 차지했다(위 설명)
    )

    # 데이터 준비: 어휘, 학습 매니페스트, PCM 저장소, 길이 버킷 샘플러. DataLoader는 num_workers=0(메인
    # 프로세스에서 읽기)이며, 0단계 측정의 데이터 대기 비율은 1.1 %였다.
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    train_utts = read_manifest(ROOT / "manifests" / f"{args.train}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", args.train)
    dataset = SpeechDataset(train_utts, store, vocab)
    sampler = DurationBucketSampler(train_utts, args.max_duration, args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    # step 예산: epoch당 step = 배치 수(시드·max_duration이 같으면 항상 같다). 스케줄 길이는
    # --schedule-steps가 있으면 그 값. dev 손실 규칙 'A,B'는 1부터 센 epoch 번호다(기본 1,3).
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs
    schedule_steps = args.schedule_steps or total_steps
    rule_a, rule_b = (int(x) for x in args.dev_loss_rule.split(","))

    model, cfg = build_model(vocab, args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logmel = LogMel().to(device)
    # 옵티마이저와 정밀도(docs/survey/README.md "결정", docs/experiments.md "fp16 규칙"): AdamW β=(0.9, 0.98),
    # weight decay 1e-6. 파라미터·옵티마이저는 fp32이고 forward만 fp16 autocast, GradScaler는 기본값(시작
    # scale 65,536). bf16_native는 '하드웨어 bf16 지원 여부'를 에뮬레이션을 빼고 기록한다. RTX 2080 Ti에서도
    # 기본 is_bf16_supported()는 True를 돌려주므로 autocast dtype을 자동으로 고르면 느린 bf16 에뮬레이션이 될
    # 수 있어 fp16을 명시한다(조사 검증 C9).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-6)
    use_amp = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bf16_native = torch.cuda.is_bf16_supported(including_emulation=False) if device.type == "cuda" else None
    rng = random.Random(args.seed)
    counters = dict.fromkeys(COUNTERS, 0)
    dev_loss_hist: list[float] = []
    divergence = None

    start_epoch, step = 0, 0
    recipe = {k: getattr(args, k) for k in RECIPE_KEYS}
    # --init(4단계 이어 학습)은 이미 정렬된 모델의 가중치만 이어받고 옵티마이저·스케줄은 새로 시작한다(규칙
    # 변경 16). '정렬이 시작되지 않는 런'을 잡는 dev 손실 규칙은 처음부터 학습하는 런 전용이라 이어 학습에서는
    # 끈다(규칙 변경 3).
    dev_rule_active = not args.init
    recipe_diff: dict = {}
    if args.init and not args.resume:  # --resume면 가중치는 last.pt에서 오고 init은 처방으로만 대조한다
        state = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    # 재개: 완료된 epoch의 last.pt만 받는다. 없거나, epoch 도중에 쓴 것이거나, 발산 판정을 받은 런이면
    # 거부한다(발산한 런은 LR × 0.5로 새 --run에서 다시 시작하는 것이 등록 규칙).
    if args.resume:
        if not (run / "last.pt").exists():
            raise SystemExit(
                f"--resume needs {run / 'last.pt'}; a run without a completed epoch cannot resume"
            )
        state = torch.load(run / "last.pt", map_location=device, weights_only=True)
        if not state.get("epoch_complete", False):
            raise SystemExit(
                "last.pt was written mid-epoch (divergence or --max-steps); resume needs a completed epoch"
            )
        if state.get("divergence"):
            raise SystemExit(
                f"this run diverged ({state['divergence']}); restart with LR x 0.5 under a new --run"
            )
        # 처방 대조. 처방 저장 기능이 생기기 전에 쓴 체크포인트(2단계 Run A 중단분)는 런의 원래 meta.json
        # 인자와 비교하고, 그때 없던 옵션은 기본값으로 본다. 다르면 --allow-recipe-change를 명시해야 하며
        # 차이는 meta-resume-NNN.json에 남는다.
        saved = state.get("recipe")
        recipe_source = "checkpoint"
        if saved is None and (run / "meta.json").exists():
            meta_args = json.loads((run / "meta.json").read_text(encoding="utf-8"))["args"]
            saved = {k: meta_args.get(k, p.get_default(k)) for k in RECIPE_KEYS}
            recipe_source = "meta.json"
        if saved is None:
            recipe_diff = {"recipe": ("missing in checkpoint", "cannot verify")}
        else:
            saved = {k: saved.get(k, p.get_default(k)) for k in RECIPE_KEYS}  # 나중에 추가된 키는 기본값
            recipe_diff = {k: (saved[k], recipe[k]) for k in RECIPE_KEYS if saved[k] != recipe[k]}
        if recipe_diff and not args.allow_recipe_change:
            raise SystemExit(
                f"recipe differs from the checkpoint (pass --allow-recipe-change to override): {recipe_diff}"
            )
        # 가중치·옵티마이저·GradScaler·step·카운터·dev 손실 이력·dev 규칙 적용 여부·난수 상태를 모두 복원해
        # 멈추지 않은 런과 같은 지점에서 잇는다.
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        counters.update(state.get("counters", {}))
        dev_loss_hist = list(state.get("dev_loss", []))
        dev_rule_active = bool(state.get("dev_loss_rule_active", dev_rule_active))
        if "rng" in state:
            restore_rng(state["rng"], rng, device)
        # 중단된 epoch에서 이미 쓴 step 로그(체크포인트 step 이상)는 다시 기록될 것이므로
        # train-discarded-before-resume-NNN.jsonl로 옮겨 train.jsonl에 같은 step이 두 번 나오지 않게 한다.
        # 지우지 않고 옮겨서 기록은 남긴다.
        log_path = run / "train.jsonl"
        if log_path.exists():
            kept, dropped = [], []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line and json.loads(line).get("step", -1) >= step:
                    dropped.append(line)
                elif line:
                    kept.append(line)
            if dropped:
                unique_path(run / f"train-discarded-before-resume-{start_epoch:03d}.jsonl").write_text(
                    "\n".join(dropped) + "\n", encoding="utf-8"
                )
                log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        # last.pt 뒤에 쓴 epoch 요약(두 파일을 쓰는 사이에 멈춘 경우)은 완료로 세면 안 되므로, 재개 시작 epoch
        # 이후의 요약 줄도 epochs-discarded-before-resume-NNN.jsonl로 옮긴다.
        ep_path = run / "epochs.jsonl"
        if ep_path.exists():
            rows = [x for x in ep_path.read_text(encoding="utf-8").splitlines() if x]
            later = [x for x in rows if json.loads(x).get("epoch", -1) >= start_epoch]
            if later:
                unique_path(run / f"epochs-discarded-before-resume-{start_epoch:03d}.jsonl").write_text(
                    "\n".join(later) + "\n", encoding="utf-8"
                )
                kept_rows = [x for x in rows if json.loads(x).get("epoch", -1) < start_epoch]
                ep_path.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")

    # 런 메타데이터: 인자, config, 파라미터 수, step 정의, 발산 규칙 문구, git 커밋, torch·cuDNN 버전, GPU
    # 이름, bf16 지원 여부, 메모리 설정. 결과를 볼 때 어떤 코드·처방·환경에서 나온 수치인지 추적하기 위해
    # 남긴다. 재개면 원래 meta.json은 그대로 두고 meta-resume-NNN.json을 새로 쓴다.
    meta = {
        "args": vars(args),
        "config": asdict(cfg),
        "n_params": n_params,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "schedule_steps": schedule_steps,
        "dev_loss_rule": [rule_a, rule_b],
        "step_definition": "optimizer step attempts; `updates` counts the non-skipped ones",
        "dev_loss_rule_text": f"divergence when dev CTC loss after epoch {rule_b} is not below the value "
        f"after epoch {rule_a} (skipped with --init)",
        "git": git_commit(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bf16_native": bf16_native,
        "amp_fp16": use_amp,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    meta["augment_start_step_effective"] = (
        args.augment_start_step or args.augment_start_epoch * steps_per_epoch
    )
    meta["dev_loss_rule_active"] = dev_rule_active
    if args.resume:
        meta["resumed_from_epoch"] = start_epoch - 1
        meta["recipe_diff"] = {k: list(v) for k, v in recipe_diff.items()}
        meta["recipe_source"] = recipe_source
        meta_path = unique_path(run / f"meta-resume-{start_epoch:03d}.json")
    else:
        meta_path = run / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log = open(run / "train.jsonl", "a", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        # 속도 교란은 --perturb-start-epoch(기본 0)부터 켠다. wait(데이터 대기)·compute(나머지) 시간은 epoch
        # 요약에 남겨 데이터 대기 비율을 감시한다(0단계: 20 % 넘으면 특징 사전 계산으로 바꾸는 규칙).
        perturb = not args.no_perturb and epoch >= args.perturb_start_epoch
        augment = False
        t_epoch = time.perf_counter()
        wait, compute = 0.0, 0.0
        t_wait = time.perf_counter()
        for batch in loader:
            wait += time.perf_counter() - t_wait
            t_step = time.perf_counter()
            mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, perturb)
            # SpecAugment 커리큘럼(규칙 변경 13·15): 기본은 epoch 3(= 300 s 배치에서 step 2,091)부터 켜고,
            # --augment-start-step을 주면 step 기준으로 켠다. 앞 구간에서 CTC 정렬이 먼저 형성되게 하려는
            # 것이다.
            if args.augment_start_step > 0:
                augment = not args.no_augment and step >= args.augment_start_step
            else:
                augment = not args.no_augment and epoch >= args.augment_start_epoch
            # 배치마다 청크 설정을 뽑는다(동적 청크 학습: 50 % 전체 문맥, 50 % 청크 {4, 8, 16, 32} × 좌측 {64,
            # 128, 무제한}). 통제군 Run B(--offline-only)는 항상 전체 문맥.
            spec = (
                ChunkSpec(0, -1)
                if args.offline_only
                else sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
            )
            # LR은 epoch이 아니라 step 시도 번호로 계산한다(규칙 변경 2: 스케줄·예산·동일 비교는 시도 횟수
            # 기준).
            for g in opt.param_groups:
                g["lr"] = lr_at(step, schedule_steps, args.warmup, args.lr)
            # TrimTail(4단계): 절단 뒤에도 CTC 길이 조건(출력 프레임 ≥ U + 반복 수 + 1)을 지키도록 발화별 최소
            # mel 프레임을 4 × (최소 출력 프레임 + 1)로 계산해 넘긴다(서브샘플링 4배). 절단 폭은 ms ÷ 10 = mel
            # 프레임 수.
            if args.trimtail_ms > 0:
                need = torch.tensor([4 * (ctc_min_frames(seq) + 1) for seq in batch["tokens"]], device=device)
                mel, frames = trim_tail(mel, frames, args.trimtail_ms // 10, need, rng)
            # CR-CTC(3단계 (d)): 같은 배치를 배치 축으로 두 번 이어 붙여 한 번의 forward로 두 뷰를 만든다.
            # SpecAugment 마스크는 행마다 따로 뽑히므로 두 뷰의 증강이 다르다. 연산이 두 배라 max-duration을
            # 절반(150 s)으로 줄여 돌렸다(규칙 변경 17).
            if args.cr_ctc_alpha > 0:
                mel, frames = torch.cat([mel, mel]), torch.cat([frames, frames])
                targets, target_lengths = (
                    torch.cat([targets, targets]),
                    torch.cat([target_lengths, target_lengths]),
                )
            # fp16 autocast forward와 손실. 손실 함수 안에서 log_softmax를 fp32로 계산하므로 CTC 입력은 항상
            # fp32다. Peak-First·CR-CTC 항은 옵션일 때만 더하고, 로그용 parts의 loss 값을 갱신한다.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, inter, out_lengths = model(mel, frames, spec, augment=augment)
                loss, parts = total_loss(logits, inter, out_lengths, targets, target_lengths, cfg.sc_weight)
                if args.peak_first_lambda > 0:
                    pfr = peak_first_regularizer(logits, out_lengths)
                    loss = loss + args.peak_first_lambda * pfr
                    parts["pfr"] = float(pfr.detach())
                    parts["loss"] = float(loss.detach())
                if args.cr_ctc_alpha > 0:
                    consistency = cr_consistency(logits, out_lengths)
                    loss = loss + args.cr_ctc_alpha * consistency
                    parts["cr"] = float(consistency.detach())
                    parts["loss"] = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            # 손실이 NaN/Inf면 역전파 없이 건너뛰고 nan_count와 skipped를 올린다(nan_count가 누적 3회면 발산).
            if not torch.isfinite(loss):
                counters["nan_count"] += 1
                counters["skipped"] += 1
            else:
                # fp16 규칙: 스케일된 손실로 역전파 → unscale_로 기울기를 원래 크기로 되돌린 뒤 노름 5.0으로
                # clip → step → scale 갱신. scale이 줄었다면 GradScaler가 inf/NaN 기울기를 발견해 그 step을
                # 건너뛴 것이므로 skipped로, 아니면 updates로 센다(규칙 변경 2: 시도·업데이트·스킵을 따로
                # 기록). 학습 초반 scale이 65,536에서 수백 이하로 내려오며 생기는 스킵 8~10회는 GradScaler의
                # 정상 동작이다(docs/experiments.md 0단계 결과 보강, 7단계 발산 표).
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    counters["skipped"] += 1
                else:
                    counters["updates"] += 1
                parts["grad_norm"] = grad_norm
            # fp16 발산 판정(docs/experiments.md "fp16 규칙"): GradScaler scale < 1.0이 100 step 연속이거나
            # 비유한 손실이 누적 3회면 divergence를 기록하고 이 step 뒤 학습을 멈춘다.
            scale = scaler.get_scale() if use_amp else 1.0
            counters["low_scale_streak"] = (
                counters["low_scale_streak"] + 1 if (use_amp and scale < 1.0) else 0
            )
            if counters["nan_count"] >= 3 or counters["low_scale_streak"] >= 100:
                divergence = {"reason": "nan_or_scale", "epoch": epoch, "step": step, **counters}
            compute += time.perf_counter() - t_step
            # log_every step마다 train.jsonl에 LR, scale, 스킵·업데이트 수, 청크 설정, 배치 크기, 손실 항,
            # 기울기 노름, GPU 메모리(최대 할당·예약·할당기 재시도 횟수)를 쓴다. 할당기 지표는 2단계의 페이징
            # 문제 뒤 추가한 감시 항목이다.
            if step % args.log_every == 0:
                rec = {
                    "epoch": epoch,
                    "step": step,
                    "lr": opt.param_groups[0]["lr"],
                    "scale": scale,
                    "skipped": counters["skipped"],
                    "updates": counters["updates"],
                    "nan": counters["nan_count"],
                    "chunk": spec.chunk_size,
                    "left": spec.left_context,
                    "batch": len(batch["ids"]),
                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                    # 할당기 상태: 예약된 풀 크기와, 캐시 블록을 비우고 할당을 다시 시도한 횟수
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else None,
                    "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0)
                    if device.type == "cuda"
                    else None,
                    **parts,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            step += 1
            # 발산이나 --max-steps(벤치마크용)면 epoch 도중이라도 배치 루프를 빠져나간다.
            if divergence or (args.max_steps and step >= args.max_steps):
                break
            t_wait = time.perf_counter()
        epoch_time = time.perf_counter() - t_epoch
        base = {
            "epoch": epoch,
            "step": step,
            "augment": augment,
            "perturb": perturb,
            "epoch_time_s": epoch_time,
            "wait_s": wait,
            "compute_s": compute,
            "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            **counters,
        }
        # epoch 도중에 멈춘 경우: dev 평가 없이 요약만 쓰고 last.pt를 epoch_complete=False로 저장한 뒤 끝낸다.
        # 이 체크포인트는 원인 분석용이며 재개할 수 없다.
        if divergence or (args.max_steps and step >= args.max_steps):
            summary = {**base, "divergence": divergence, "stopped_early": True}
            (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            save_checkpoint(
                run / "last.pt",
                model,
                opt,
                scaler,
                cfg,
                epoch,
                step,
                counters,
                dev_loss_hist,
                rng,
                device,
                False,
                divergence,
                recipe,
                dev_rule_active,
            )
            break
        # epoch 끝 dev 평가: dev_mix·dev_h 각각 full과 simulated 청크(--eval-chunk 기본 16프레임 = 640 ms,
        # 좌측 128)의 한글 전용 CER_ns. scripts/average_checkpoints.py가 이 기록의 dev_mix full CER로 상위
        # 10개 epoch을 골라 평균 모델을 만든다.
        model.eval()
        dev_results = {}
        for dev in args.devs.split(","):
            dev_results[dev] = {
                "full": evaluate_manifest(
                    model, vocab, dev, ChunkSpec(0, -1), device, logmel=logmel, amp=use_amp
                )["cer_ns"],
                f"chunk{args.eval_chunk}": evaluate_manifest(
                    model, vocab, dev, ChunkSpec(args.eval_chunk, 128), device, logmel=logmel, amp=use_amp
                )["cer_ns"],
            }
        # dev CTC 손실(최종 헤드, 전체 문맥, 학습과 같은 정밀도)을 재고, 평가 배치의 다른 모양 때문에 생긴
        # 캐시 블록을 비운다(다음 epoch 학습의 메모리 조각을 줄인다).
        dev_loss = ctc_loss_manifest(model, vocab, args.dev_loss_set, device, logmel=logmel, amp=use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dev_loss_hist.append(dev_loss)
        # dev 손실 발산 규칙(규칙 변경 3·15): epoch B(기본 3) 끝의 dev 손실이 epoch A(기본 1) 끝 값보다 작지
        # 않으면 발산. 두 값이 모두 유한할 때만 '내려감'으로 인정한다(NaN이면 발산 쪽). 처음부터 학습하는
        # 런에서 epoch B를 마친 시점에만 한 번 검사한다.
        not_decreasing = len(dev_loss_hist) >= rule_b and not (
            math.isfinite(dev_loss_hist[rule_a - 1])
            and math.isfinite(dev_loss_hist[rule_b - 1])
            and dev_loss_hist[rule_b - 1] < dev_loss_hist[rule_a - 1]
        )
        if dev_rule_active and epoch == rule_b - 1 and not_decreasing:
            divergence = {
                "reason": "dev_loss_not_decreasing",
                "epoch": epoch,
                "step": step,
                "rule": [rule_a, rule_b],
                "dev_loss": dev_loss_hist[:rule_b],
            }
        summary = {**base, "dev": dev_results, "dev_loss": dev_loss, "dev_loss_set": args.dev_loss_set}
        if divergence:
            summary["divergence"] = divergence
        (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        # 완료된 epoch마다 epoch-NNN.pt(평균용 보관)와 last.pt(재개용)를 같은 내용으로 쓴다. 발산 판정을
        # 받았으면 기록을 남긴 뒤 멈춘다.
        for path in (run / f"epoch-{epoch:03d}.pt", run / "last.pt"):
            save_checkpoint(
                path,
                model,
                opt,
                scaler,
                cfg,
                epoch,
                step,
                counters,
                dev_loss_hist,
                rng,
                device,
                True,
                divergence,
                recipe,
                dev_rule_active,
            )
        if divergence:
            break
    log.close()


if __name__ == "__main__":
    main(sys.argv[1:])
