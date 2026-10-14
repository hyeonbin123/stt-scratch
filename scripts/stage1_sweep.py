"""1단계 실행기(docs/experiments.md): peak LR {1e-3, 2e-3} × 시드 {1,2,3} × 5 epoch의 발산과 dev CER.

등록된 후보 6개 학습을 GPU 한 장에서 차례로 돌리고, 런마다 발산 여부·카운터·dev CER을 모아
reports/stage1/에 요약(summary.json + summary.md)과 LR별 판정 자료(verdict.json)를 쓴다.
사용법: uv run python scripts/stage1_sweep.py [--epochs 5] [--lrs 1e-3,2e-3] [--seeds 1,2,3]
"""

# [왜 필요한가] 1단계는 fp16 안정성(H1: 2e-3에서 발산 ≤ 1/3)을 보고 peak LR을 고르며, 채택 LR의 3시드 dev_mix
# CER 표준편차 s를 구해 이후 모든 채택 임계 max(0.3, 2s)에 쓴다(docs/experiments.md 1단계, "선택 규칙"). 결과:
# 3차 시도에서 두 LR 모두 발산 0/3, 2e-3 채택, s = 0.93, 임계 1.86포인트.
# [설계 이유] 학습은 stt_scratch.train을 하위 프로세스로 돌린다. 발산 판정 전체(NaN, GradScaler scale, dev
# 손실 규칙)는 train.py가 하고 이 스크립트는 기록을 읽기만 한다. 후보를 정의하는 인자는 이 스크립트만 정하게
# 하고, 런별 meta.json과 대조하며, 등록 설계가 온전히 돌았을 때만 '채택 가능'으로 표시한다(Codex 1단계 검토
# 반영, 규칙 변경 15).
# [연결] README의 1단계 명령(--schedule-steps 69700). 1단계는 세 번 시도했고 --out-dir·--report-prefix로
# 시도마다 결과를 나눴다(reports/stage1/attempt2-*, attempt3-*).

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# 저장소 루트. 하위 프로세스를 이 폴더에서 실행하고 런 폴더·보고서를 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# 후보를 정의하거나 런의 정체를 바꾸는 train.py 인자. --train-args로 이 인자를 다시 넘기면(예: LR을 덮어쓰기)
# 런 이름과 실제 처방이 달라질 수 있으므로 main에서 거부한다(규칙 변경 15: 후보 정의 인자의 중복 전달 거부).
CANDIDATE_ARGS = (
    "--lr",
    "--seed",
    "--epochs",
    "--warmup",
    "--schedule-steps",
    "--dev-loss-rule",
    "--run",
    "--max-duration",
    "--resume",
    "--init",
)


# 후보 하나를 학습시키고 기록을 읽어 결과 사전을 돌려준다.
# 런 폴더 이름은 <out_dir>/lr<LR>_seed<시드>. train.py는 같은 폴더에 로그가 있으면 --resume 없이 시작하지
# 않으므로, 다시 시도할 때는 다른 --out-dir를 쓴다.
# 반환: 처방 대조 결과(recipe_ok), 완료 여부, 발산 사유, dev 손실 이력, 카운터(NaN·skipped·updates), 최저
# GradScaler scale, 마지막 epoch의 dev CER(full, 청크 16 simulated) 등.
def run_one(
    lr: float,
    seed: int,
    epochs: int,
    max_duration: float,
    warmup: int,
    out_dir: str,
    schedule_steps: int,
    dev_loss_rule: str,
    train_args: str = "",
) -> dict:
    run = f"{out_dir}/lr{lr:g}_seed{seed}"
    # train.py 명령: 후보 값(LR, 시드)과 공통 처방(epoch 수, 배치 상한, warmup, 스케줄 길이, dev 손실 규칙)을
    # 넘기고, --train-args의 추가 옵션을 뒤에 붙인다.
    cmd = [
        sys.executable,
        "-m",
        "stt_scratch.train",
        "--run",
        run,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--max-duration",
        str(max_duration),
        "--warmup",
        str(warmup),
        "--schedule-steps",
        str(schedule_steps),
        "--dev-loss-rule",
        dev_loss_rule,
        *train_args.split(),
    ]
    # 학습이 끝날 때까지 기다린다. 발산 판정으로 멈춘 런은 train.py가 정상 종료하므로 sweep이 계속되고, 실행
    # 자체가 실패(예외, 인자 오류)하면 check=True로 sweep도 멈춘다.
    subprocess.run(cmd, cwd=ROOT, check=True)
    # 런이 실제로 쓴 처방(meta.json)을 읽어 의도한 후보와 같은지 대조한다. 다르면 recipe_ok가 False가 되어
    # 채택 가능에서 빠진다.
    meta = json.loads((ROOT / run / "meta.json").read_text(encoding="utf-8"))
    recorded = {
        "lr": meta["args"]["lr"],
        "seed": meta["args"]["seed"],
        "epochs": meta["args"]["epochs"],
        "warmup": meta["args"]["warmup"],
        "schedule_steps": meta["schedule_steps"],
    }
    expected = {"lr": lr, "seed": seed, "epochs": epochs, "warmup": warmup, "schedule_steps": schedule_steps}
    recipe_ok = recorded == expected
    epochs_log = [
        json.loads(ln) for ln in (ROOT / run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if ln
    ]
    train_log = [
        json.loads(ln) for ln in (ROOT / run / "train.jsonl").read_text(encoding="utf-8").splitlines() if ln
    ]
    # 마지막 epoch 요약에 발산 사유·최종 카운터가 있다. 발산이나 --max-steps로 멈춘 epoch 요약에는 "dev"가
    # 없다.
    last = epochs_log[-1]
    first_dev = next((e for e in epochs_log if "dev" in e), None)
    # 발산 정의 전체(NaN 횟수, GradScaler scale, epoch 3까지 dev CTC 손실이 줄지 않음)는 train.py가 적용하고,
    # 런을 멈춘 epoch의 요약에 그 사유를 기록한다. 여기서는 그 기록을 읽기만 한다.
    divergence = last.get("divergence")
    # 완료 = dev 평가까지 끝난 epoch가 등록 epoch 수만큼 있고, 조기 종료·발산이 없다.
    n_complete = sum(1 for e in epochs_log if "dev" in e)
    complete = n_complete == epochs and not last.get("stopped_early") and not divergence
    return {
        "run": run,
        "lr": lr,
        "seed": seed,
        "recipe_ok": recipe_ok,
        "recorded": recorded,
        "complete": complete,
        "n_complete_epochs": n_complete,
        "diverged": bool(divergence),
        "divergence": divergence,
        "dev_loss": [e["dev_loss"] for e in epochs_log if "dev_loss" in e],
        "nan": last.get("nan_count"),
        "skipped": last.get("skipped"),
        "updates": last.get("updates"),
        "step": last.get("step"),
        # 최저 GradScaler scale은 train.jsonl의 로그 간격(--log-every) 표본에서 구한 값이다(규칙 변경 15).
        # 발산 판정 자체는 train.py가 매 step 한다.
        "min_scale": min((r["scale"] for r in train_log), default=None),
        "dev_mix_full": last["dev"]["dev_mix"]["full"] if "dev" in last else None,
        "dev_h_full": last["dev"]["dev_h"]["full"] if "dev" in last else None,
        "dev_mix_chunk16": last["dev"]["dev_mix"].get("chunk16") if "dev" in last else None,
        "epoch_time_s": last.get("epoch_time_s"),
        "first_epoch_dev": first_dev["dev"] if first_dev else None,
    }


# 후보 전부를 돌리고 요약·판정 자료를 쓴다.
# 주요 인자: --epochs(5), --lrs, --seeds, --max-duration(300 s), --warmup(2,000, 규칙 변경 1),
# --schedule-steps(69,700 = 2단계 100 epoch의 스케줄 길이; 규칙 변경 13: 1단계 런을 2단계 스케줄의 앞부분으로
# 돌려 peak LR 구간에서 안정성을 본다), --dev-loss-rule("1,3": epoch 3 끝 dev 손실이 epoch 1 끝보다 낮지
# 않으면 발산), --out-dir, --report-prefix, --train-args.
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lrs", default="1e-3,2e-3")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--out-dir", default="outputs/stage1", help="런 폴더를 이 경로 아래에 만든다")
    p.add_argument(
        "--schedule-steps", type=int, default=69700, help="2단계 스케줄 길이(epoch당 697 step × 100 epoch)"
    )
    p.add_argument("--dev-loss-rule", default="1,3", help="dev 손실 발산 규칙에서 비교할 두 epoch A,B")
    p.add_argument(
        "--report-prefix", default="", help="reports/stage1의 summary·verdict 파일 이름 앞에 붙일 접두어"
    )
    p.add_argument("--train-args", default="", help="stt_scratch.train에 그대로 넘길 추가 인자")
    args = p.parse_args()
    # --train-args 안에 후보 정의 인자가 있으면 시작 전에 거부한다("--lr=…" 형태도 잡도록 '=' 앞만 본다).
    clash = [a for a in args.train_args.split() if a.split("=")[0] in CANDIDATE_ARGS]
    if clash:
        raise SystemExit(f"--train-args may not redefine the candidates: {clash}")
    out_dir = ROOT / "reports" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = args.report_prefix
    results = []
    # LR 순서, 그 안에서 시드 순서로 하나씩 돈다(GPU 한 장이라 동시에 돌리지 않는다). 런이 끝날 때마다
    # summary.json을 다시 써서 중간에 멈춰도 끝난 런의 결과가 남는다.
    for lr in (float(x) for x in args.lrs.split(",")):
        for seed in (int(s) for s in args.seeds.split(",")):
            results.append(
                run_one(
                    lr,
                    seed,
                    args.epochs,
                    args.max_duration,
                    args.warmup,
                    args.out_dir,
                    args.schedule_steps,
                    args.dev_loss_rule,
                    args.train_args,
                )
            )
            (out_dir / f"{pre}summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    # 런별 Markdown 표: 발산, NaN, skipped, updates, 최저 scale, epoch별 dev 손실, epoch 5 dev_mix·dev_h full
    # CER, dev_mix 청크 16 simulated CER, epoch 시간.
    lines = [
        "| LR | seed | 발산 | NaN | skipped | updates | min scale | dev loss | dev_mix | dev_h | "
        "dev_mix c16 | epoch s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        dev_loss = ", ".join(f"{x:.1f}" for x in r["dev_loss"])
        lines.append(
            f"| {r['lr']:g} | {r['seed']} | {r['diverged']} | {r['nan']} | {r['skipped']} | {r['updates']} | "
            f"{r['min_scale']} | {dev_loss} | {r['dev_mix_full']} | {r['dev_h_full']} | "
            f"{r['dev_mix_chunk16']} | {r['epoch_time_s']:.0f} |"
        )
    # LR별 판정 자료. 판정 규칙(docs/experiments.md 1단계): 발산 0/3인 LR 가운데 epoch 5 dev_mix 평균이 낮은
    # 쪽을 채택하고, 채택 LR의 3시드 표준편차를 s로 기록한다. 실제 채택 결정은 이 표를 보고 문서에 적었다.
    verdict = {}
    seeds = [int(s) for s in args.seeds.split(",")]
    for lr in sorted({r["lr"] for r in results}):
        rows = [r for r in results if r["lr"] == lr]
        ok = [r for r in rows if r["complete"] and r["dev_mix_full"] is not None]
        failures = sum(r["diverged"] for r in rows)
        verdict[lr] = {
            "n_seeds": len(rows),
            "n_complete": len(ok),
            "recipe_ok": all(r["recipe_ok"] for r in rows),
            "failures": failures,
            # 완료한 시드만으로 평균과 표준편차를 낸다. 등록된 s는 측정 전 커밋된 정의대로 완료 시드의
            # 모표준편차(pstdev)이고, 표본표준편차는 참고로 함께 적는다(1단계 결과: 2e-3에서 pstdev 0.93,
            # 표본 1.14).
            "dev_mix_mean": statistics.mean(r["dev_mix_full"] for r in ok) if ok else None,
            "dev_mix_std": statistics.pstdev([r["dev_mix_full"] for r in ok]) if len(ok) > 1 else None,
            "dev_mix_sample_std": statistics.stdev([r["dev_mix_full"] for r in ok]) if len(ok) > 1 else None,
            # 채택 가능은 등록 설계가 온전히 돌았을 때만이다: 3시드 × 5 epoch, 발산 0, 처방 대조 통과.
            "adoptable": len(rows) == 3
            and len(seeds) == 3
            and args.epochs == 5
            and len(ok) == 3
            and failures == 0
            and all(r["recipe_ok"] for r in rows),
        }
    # LR별 판정 표를 덧붙여 summary.md와 verdict.json을 쓰고 콘솔에도 출력한다.
    lines.append("")
    lines.append(
        "| LR | 발산 수 | 완료 | dev_mix 평균 | 시드 표준편차 s(pstdev) | 표본 표준편차 | 체택 가능 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for lr, v in verdict.items():
        lines.append(
            f"| {lr:g} | {v['failures']}/{v['n_seeds']} | {v['n_complete']}/{v['n_seeds']} | "
            f"{v['dev_mix_mean']} | {v['dev_mix_std']} | {v['dev_mix_sample_std']} | {v['adoptable']} |"
        )
    (out_dir / f"{pre}summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{pre}verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
