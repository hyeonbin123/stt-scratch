"""0단계 마이크로벤치(docs/experiments.md H0.2).

배치 크기 상한(max-duration, 배치 안 발화 길이 합의 초 단위 상한)을 바꿔 가며 실제 학습 step을
수십 번 돌려, max-duration마다 GPU 피크 메모리, step 시간, 데이터 대기 비율, epoch 시간 추정을 잰다.
이 GPU에서 어떤 SDPA(어텐션) 커널이 쓰이는지와 bf16 하드웨어 지원 여부도 함께 기록한다.

사용법: uv run python scripts/bench_stage0.py --durations 200,300,400,600 --steps 40
결과는 reports/stage0/bench.json에 쓰고 Markdown 표로도 출력한다.
"""

# [왜 필요한가] 사전 등록 가설 H0.2(max-duration 300 s에서 피크 ≤ 9.5 GB, 1 epoch ≤ 8분)와 예산 규칙(epoch
# 시간 × 100 > 14 h이면 2단계 epoch 수를 줄임), 데이터 대기 > 20 %면 fbank 사전 계산으로 전환한다는 판정을 이
# 수치로 내렸다(docs/experiments.md 0단계). 결과는 300 s에서 피크 4.31 GB, epoch 약 7.8분이라 2단계를 100
# epoch로 그대로 돌렸다.
# [흐름] SDPA 커널 탐지 → max-duration마다 새 모델·옵티마이저를 만들어 학습과 같은 step(속도 교란, 동적 청크,
# SpecAugment, fp16 autocast, GradScaler, grad clip)을 반복 → 앞 몇 step은 버리고 평균 →
# reports/stage0/bench.json 저장과 Markdown 표 출력.
# [연결] 학습 코드(stt_scratch.train.prepare_batch, 모델, 손실)를 그대로 불러 쓰므로 측정 대상이 실제 학습
# 경로와 같다. README의 "학습·평가" 첫 명령이다.

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

# torch: GPU 학습 step 실행, CUDA 메모리 통계, SDPA 커널 선택에 쓴다.
import torch

# 프로젝트 내부 모듈: 학습과 같은 데이터 경로(매니페스트, PCM 저장소, 길이 버킷 배치), 같은 모델·설정, 같은
# CTC 손실(최종 + 중간 헤드), 같은 log-mel, 같은 동적 청크 샘플링, 같은 배치 준비 함수(prepare_batch)를
# 가져온다. 벤치가 따로 구현하지 않고 학습 코드를 그대로 써야 잰 시간이 실제 epoch 시간을 대표한다.
from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import total_loss
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import sample_chunk_spec
from stt_scratch.text import Vocab
from stt_scratch.train import prepare_batch

# 저장소 루트. manifests/, data/pcm/, reports/를 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# 왜: 메모리·속도는 어떤 커널이 도는지에 크게 좌우된다. 결과는 flash 커널 불가(sm_75, RTX 2080 Ti),
# memory-efficient 커널 사용이었다(docs/experiments.md 0단계 결과). 미해결 질문 표의 "sm_75에서 SDPA가
# mem-efficient인지 math인지 기록" 항목을 닫는 측정이다(docs/survey/README.md §6).
def sdpa_backend_probe(device: torch.device) -> dict:
    """이 GPU에서 모델과 같은 모양(fp16, bool 마스크)의 입력에 어떤 SDPA 커널이 도는지 확인한다.

    torch.nn.functional.scaled_dot_product_attention이 쓸 수 있는 커널을 하나씩 강제로 골라 실행해 보고,
    커널 이름마다 되면 True, 안 되면 "unavailable: <예외 이름>"을 담은 dict를 돌려준다.
    """
    # 모델과 같은 모양: 배치 8, 헤드 4, 길이 256, 헤드 차원 64(d_model 256 ÷ 4헤드). fp16과 bool 마스크도 학습
    # 때와 같다. 마스크는 모두 True라 계산 결과가 아니라 커널 선택 가능 여부만 본다.
    q = torch.randn(8, 4, 256, 64, device=device, dtype=torch.float16)
    mask = torch.ones(8, 1, 256, 256, dtype=torch.bool, device=device)
    out = {}
    # sdpa_kernel 문맥 관리자는 그 안에서 지정한 커널만 허용한다. 함수 안에서만 쓰므로 여기서 가져온다.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            # 지정한 커널만 허용한 채 한 번 실행한다. 그 커널이 이 입력(GPU 세대, dtype, 마스크 형식)을
            # 지원하지 않으면 RuntimeError가 난다.
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
            out[name] = True
        except Exception as exc:  # noqa: BLE001
            # 어떤 예외든 "이 커널은 못 쓴다"는 결과로 기록하고 다음 커널로 넘어간다(벤치 전체가 멈추지 않게).
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


# max-duration 후보마다 학습 step을 재고 bench.json과 표를 남긴다.
# 인자: --durations(쉼표로 나눈 max-duration 초, 등록값 200·300·400·600), --steps(평균에 넣을 step 수),
# --warmup(버릴 앞 step 수).
# 주의: CUDA 전용이다(device를 cuda로 고정). 같은 프로세스에서 후보를 차례로 재므로 후보 사이에 모델을 지우고
# 캐시를 비운다.
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--durations", default="200,300,400,600")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda")
    # 0단계 전처리 산출물을 읽는다: 어휘(출력 클래스 수), mel 전역 평균·표준편차(모델 입력 정규화), 학습
    # 매니페스트, 학습 PCM 저장소. 모두 scripts/prepare_data.py와 scripts/compute_mel_stats.py가 만든다.
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    stats = json.loads((ROOT / "manifests" / "mel_stats.json").read_text(encoding="utf-8"))
    utts = read_manifest(ROOT / "manifests" / "train.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", "train")
    dataset = SpeechDataset(utts, store, vocab)
    logmel = LogMel().to(device)
    total_hours = sum(u.duration_s for u in utts) / 3600
    # 측정 환경을 결과에 같이 남긴다: GPU 이름, torch 버전, 학습 발화 수·시간, SDPA 탐지 결과.
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "n_train_utts": len(utts),
        "train_hours": total_hours,
        "sdpa": sdpa_backend_probe(device),
        # fp16 규칙이 시작 로그에 남기라고 한 값이다. including_emulation=False여야 에뮬레이션을 빼고 실제
        # 하드웨어 지원만 본다(기본 호출은 2080 Ti에서도 True를 돌려준다는 것이 docs/survey/README.md C9
        # 정정). 2080 Ti는 False여야 하고 실제로 False였다.
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        "runs": [],
    }
    # cuDNN 자동 튜닝을 끈다. 길이 버킷 배치라 입력 모양이 매 step 달라서, 켜 두면 step마다 알고리즘 탐색이
    # 돌아 처음 벤치가 1.36 s/step이었고 끄자 0.66 s/step이 됐다(docs/experiments.md 0단계 결과,
    # docs/ai-verification.md). train.py도 같은 설정이다.
    torch.backends.cudnn.benchmark = (
        False  # 배치마다 입력 모양이 바뀌어 모양별 자동 튜닝이 step 시간을 차지했다(위 설명)
    )
    # max-duration 후보마다 새 모델로 처음부터 잰다. 앞 후보의 메모리 상태가 다음 후보의 피크에 섞이지 않게
    # 하려는 것이다.
    for md in (float(x) for x in args.durations.split(",")):
        # 후보마다 같은 초기 가중치에서 시작하도록 시드를 고정한다.
        torch.manual_seed(0)
        # ModelConfig 기본값이 구조 결정(docs/survey/README.md "결정")의 12층 d=256 Conformer다. 어휘 크기와
        # mel 정규화 통계만 넣는다.
        cfg = ModelConfig(vocab_size=len(vocab), mel_mean=stats["mean"], mel_std=stats["std"])
        model = ChunkConformerCTC(cfg).to(device).train()
        # 옵티마이저 종류·betas·weight decay는 학습과 같다. LR(1e-4 고정)은 시간·메모리에 영향이 없으므로
        # 스케줄 없이 둔다.
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
        scaler = torch.amp.GradScaler("cuda")
        # 학습과 같은 길이 버킷 배치. max-duration이 바뀌면 배치당 발화 수와 epoch당 step 수가 함께 바뀐다.
        sampler = DurationBucketSampler(utts, md, seed=0)
        # num_workers=0: 학습(train.py)과 같이 메인 프로세스에서 PCM을 읽는다. 그래서 아래 '데이터 대기'가 이
        # 방식의 실제 대기 시간이 된다.
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0
        )
        # 속도 교란 배율과 동적 청크 설정을 뽑는 난수. 후보마다 같은 순서로 뽑히게 시드 0으로 만든다.
        rng = random.Random(0)
        # 피크 메모리 기록을 이 후보 시작 시점부터 다시 센다.
        torch.cuda.reset_peak_memory_stats()
        times, waits, batch_sizes = [], [], []
        t_wait = time.perf_counter()
        ok = True
        try:
            for i, batch in enumerate(loader):
                # 데이터 대기: 앞 step이 끝난 뒤 로더가 다음 배치를 내줄 때까지 걸린 시간.
                wait = time.perf_counter() - t_wait
                # GPU 작업은 비동기이므로 step 시작·끝에 동기화해야 벽시계가 이 step의 실제 GPU 시간까지
                # 포함한다.
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                # 학습과 같은 배치 준비: 음성을 GPU로 옮기고 속도 교란(마지막 인자 True; 0.9·1.0·1.1 배 중
                # 하나)을 건 뒤 log-mel을 계산한다.
                mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, True)
                # 동적 청크 학습처럼 배치마다 전체 문맥 또는 (청크, 좌측 문맥)을 뽑아 어텐션 마스크를 바꾼다.
                spec = sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
                # fp16 규칙대로 autocast dtype을 float16으로 명시한다. augment=True라 SpecAugment도 켠 상태로
                # 잰다(가장 무거운 학습 조건). CTC 손실 입력은 total_loss 안에서 fp32 log_softmax로 바뀐다.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, inter, out_lengths = model(mel, frames, spec, augment=True)
                    loss, _ = total_loss(logits, inter, out_lengths, targets, target_lengths, cfg.sc_weight)
                # 학습과 같은 갱신 절차: 손실 스케일 → 역전파 → unscale_ 뒤 기울기 노름 5.0으로 자르기 →
                # GradScaler step·update(docs/experiments.md "fp16 규칙").
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize()
                # 앞 warmup step은 CUDA 초기화·할당기 확장·GradScaler scale 조정이 섞여 느리므로 평균에서
                # 뺀다.
                if i >= args.warmup:
                    times.append(time.perf_counter() - t0)
                    waits.append(wait)
                    batch_sizes.append(len(batch["ids"]))
                # warmup + steps만큼 돌았으면 이 후보를 끝낸다(1 epoch 전체를 돌지 않고 추정한다).
                if i + 1 >= args.steps + args.warmup:
                    break
                t_wait = time.perf_counter()
        except torch.cuda.OutOfMemoryError:
            # 메모리가 넘치면 프로그램을 멈추지 않고 이 후보를 OOM으로 기록한 뒤 다음 후보로 넘어간다.
            ok = False
        peak = torch.cuda.max_memory_allocated() / 1e9
        # epoch당 step 수 = 이 max-duration으로 만든 배치 개수.
        steps_per_epoch = len(sampler)
        step_mean = sum(times) / len(times) if times else None
        wait_mean = sum(waits) / len(waits) if waits else None
        run = {
            "max_duration_s": md,
            "oom": not ok,
            "peak_mem_gb": peak,
            "steps_per_epoch": steps_per_epoch,
            "step_time_s": step_mean,
            "wait_time_s": wait_mean,
            # 데이터 대기 비율 = 대기 ÷ (대기 + step). 20 %를 넘으면 fbank 사전 계산으로 바꾸기로
            # 등록했다(실측 1.1 %라 바꾸지 않음).
            "wait_share": (wait_mean / (wait_mean + step_mean)) if times else None,
            "batch_size_mean": sum(batch_sizes) / len(batch_sizes) if batch_sizes else None,
            # epoch 시간 추정(분) = epoch당 step 수 × (step 평균 + 대기 평균). 예산 규칙(epoch 시간 × 100 ≤ 14
            # h)의 입력이다.
            "epoch_time_min_est": (steps_per_epoch * (step_mean + wait_mean) / 60) if times else None,
        }
        results["runs"].append(run)
        print(json.dumps(run), flush=True)
        # 다음 후보 전에 이 후보의 모델·옵티마이저 상태를 지우고 캐시 블록을 돌려줘 메모리 측정이 서로 섞이지
        # 않게 한다.
        del model, opt, scaler, loader
        torch.cuda.empty_cache()
    out = ROOT / "reports" / "stage0" / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    # 사람이 docs/experiments.md에 옮겨 적기 쉽도록 같은 결과를 Markdown 표로도 찍는다. OOM이라 시간이 없는
    # 후보는 0으로 표시된다.
    print(
        "| max-duration (s) | OOM | peak GB | steps/epoch | step s | wait share | batch | epoch min (est) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results["runs"]:
        print(
            f"| {r['max_duration_s']:.0f} | {r['oom']} | {r['peak_mem_gb']:.2f} | {r['steps_per_epoch']} | "
            f"{r['step_time_s'] or 0:.3f} | {(r['wait_share'] or 0) * 100:.1f} % | "
            f"{r['batch_size_mean'] or 0:.1f} | "
            f"{r['epoch_time_min_est'] or 0:.1f} |"
        )
    print("sdpa:", results["sdpa"], "bf16 native:", results["bf16_native"])


if __name__ == "__main__":
    main()
