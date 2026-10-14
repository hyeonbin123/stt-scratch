"""4단계 Peak-First CTC 변형에 쓸 정규화 가중치 λ를 한 번 계산해 고정한다(docs/experiments.md 규칙 변경 18).

이어 학습을 시작할 모델로, 시드를 고정한 학습 배치 20개에서 학습과 같은 조건(train 모드: dropout,
SpecAugment, 속도 교란, 동적 청크 마스크)으로 평균 CTC 손실과 평균 Peak-First 항을 잰 뒤
λ = 0.1 × 평균 CTC 손실 ÷ 평균 Peak-First 항으로 정한다. 그러면 이어 학습이 보는 CTC 손실의 약 10 %가
정규화 항이 된다. 결과는 reports/stage4/peak_first_lambda.json에 쓴다.

사용법: uv run python scripts/calibrate_peak_first.py --checkpoint outputs/stage2/runA/avg10.pt
"""

# [왜 필요한가] 사전 등록 규칙이 "λ는 정규화 항이 CTC 손실의 약 10 %가 되는 값 하나로
# 고정"이다(docs/experiments.md 4단계, 규칙 변경 18·20). 값을 손으로 고르지 않고 측정 전에 정한 식으로 한 번만
# 계산해 dev 결과를 보고 λ를 고르는 일이 없게 한다.
# [설계 이유] 학습과 같은 조건(train 모드 dropout, SpecAugment, 속도 교란, 동적 청크 마스크)에서 잰다. 증강
# 없이 eval 모드로 재면 학습 문장 암기 때문에 CTC 손실이 발화당 0.27로 작아 λ가 사실상 0이 되는 것을 시험
# 실행에서 확인했다(규칙 변경 18, docs/ai-verification.md). 실제 4단계 값은 λ = 1.62e-3(CTC 15.39, 정규화 항
# 950.1 per utt)이다.
# [연결] scripts/stage4.py가 4단계 시작 모델(3단계 (a)의 상위 10개 평균)로 이 스크립트를 부르고, 나온 λ를
# train.py의 --peak-first-lambda로 넘긴다.

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# torch: 모델 forward, 난수 시드, 장치 선택에 쓴다.
import torch

# 프로젝트 내부 모듈: 학습과 같은 데이터 경로와 배치 준비(prepare_batch, 속도 교란 포함), 평가 모듈의
# 체크포인트 로더와 autocast 설정, 최종 헤드 CTC 손실과 Peak-First 정규화 항(model/ctc.py), 동적 청크 샘플링을
# 가져온다. 학습 코드와 같은 함수를 써야 '이어 학습이 실제로 보는 손실'의 비율이 된다.
from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.evaluate import autocast, load_model
from stt_scratch.model.ctc import ctc_loss, peak_first_regularizer
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import sample_chunk_spec
from stt_scratch.text import Vocab
from stt_scratch.train import prepare_batch

# 저장소 루트. 체크포인트·매니페스트·출력 경로를 이 기준의 상대 경로로 받는다.
ROOT = Path(__file__).resolve().parents[1]


# λ를 계산해 JSON으로 저장하고 출력한다.
# 인자: --checkpoint(이어 학습 시작 모델), --batches(잴 배치 수, 등록값 20), --max-duration(배치 크기 상한,
# 학습과 같은 300 s), --target-ratio(정규화 항이 CTC 손실에서 차지할 비율, 등록값 0.1), --device, --out(결과
# JSON 경로).
# 출력: 체크포인트 경로, 조건, 평균 CTC·정규화 항(발화당), λ가 든 JSON. scripts/stage4.py가 이 파일의
# "lambda"를 읽는다.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--target-ratio", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="reports/stage4/peak_first_lambda.json")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model, _ = load_model(ROOT / args.checkpoint, device)
    # eval 모드가 아니라 train 모드로 둔다. dropout과 (forward의 augment=True와 함께) SpecAugment가 켜져야
    # 이어 학습 중 손실 크기와 같아진다.
    model.train()
    cfg = model.cfg
    # SpecAugment 마스크 위치가 전역 torch 난수를 쓰므로 시드를 고정해 다시 실행해도 같은 λ가 나오게 한다.
    torch.manual_seed(0)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    utts = read_manifest(ROOT / "manifests" / "train.jsonl")
    dataset = SpeechDataset(utts, PcmStore(ROOT / "data" / "pcm", "train"), vocab)
    # 학습과 같은 길이 버킷 배치를 시드 0, epoch 0 순서로 만든다. 고정된 순서의 앞 20배치를 쓰므로 어떤 배치를
    # 쓸지 고르는 여지가 없다.
    sampler = DurationBucketSampler(utts, args.max_duration, 0)
    sampler.set_epoch(0)
    logmel = LogMel().to(device)
    # 속도 교란 배율과 동적 청크 설정을 뽑는 난수(시드 0).
    rng = random.Random(0)
    ctc_sum, pfr_sum = 0.0, 0.0
    # 기울기는 필요 없으므로 끈다. train 모드(dropout)와 no_grad는 서로 독립이라 함께 쓸 수 있다.
    with torch.no_grad():
        for n, idx in enumerate(sampler):
            if n >= args.batches:
                break
            # DataLoader 없이 샘플러가 준 발화 번호로 직접 배치를 만든다.
            batch = collate([dataset[i] for i in idx])
            # 학습 조건 그대로: 속도 교란을 켜고 동적 청크 마스크를 뽑는다.
            mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, perturb=True)
            spec = sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
            # GPU에서는 학습과 같은 fp16 autocast로 forward한다. augment=True로 SpecAugment를 건다.
            with autocast(device, device.type == "cuda"):
                logits, _, lengths = model(mel, frames, spec, augment=True)
            # 두 값을 모두 '발화당'으로 맞춘다. ctc_loss는 배치 합을 돌려주므로 발화 수로 나누고,
            # peak_first_regularizer는 이미 발화 평균을 돌려준다. 둘 다 최종 헤드에만 건다(규칙 변경 20:
            # 정규화 항도 최종 헤드에만).
            b = logits.shape[0]
            ctc_sum += float(ctc_loss(logits, lengths, targets, target_lengths)) / b
            pfr_sum += float(peak_first_regularizer(logits, lengths))
    # 배치 평균을 낸다. 나누는 수가 실제 배치 수가 아니라 args.batches이므로, 샘플러 배치가 그보다 적으면
    # 평균이 작게 나온다(학습 세트는 epoch당 약 700배치라 20배치 설정에서는 해당 없음).
    ctc_mean, pfr_mean = ctc_sum / args.batches, pfr_sum / args.batches
    # λ × 정규화 항 = 0.1 × CTC 손실이 되도록 푼 식이다. 즉 이어 학습 시작 시점에 Peak-First 항이 CTC 손실의
    # 약 10 %를 차지한다.
    lam = args.target_ratio * ctc_mean / pfr_mean
    # 어떤 조건에서 쟀는지(체크포인트, 배치 수, 샘플러 시드, 증강 조건)를 λ와 함께 남겨 나중에 같은 값을
    # 재현·대조할 수 있게 한다.
    result = {
        "checkpoint": args.checkpoint,
        "batches": args.batches,
        "max_duration": args.max_duration,
        "sampler_seed": 0,
        "conditions": "train mode, SpecAugment, speed perturbation, sampled chunk masks",
        "mean_ctc_per_utt": ctc_mean,
        "mean_peak_first_per_utt": pfr_mean,
        "target_ratio": args.target_ratio,
        "lambda": lam,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
