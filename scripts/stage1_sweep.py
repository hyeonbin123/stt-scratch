"""Stage 1 (docs/experiments.md): peak LR {1e-3, 2e-3} x seeds {1,2,3} x 5 epochs, divergence and dev CER.

Runs the six trainings sequentially (one GPU) and writes reports/stage1/summary.json + summary.md.
Usage: uv run python scripts/stage1_sweep.py [--epochs 5] [--lrs 1e-3,2e-3] [--seeds 1,2,3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    subprocess.run(cmd, cwd=ROOT, check=True)
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
    last = epochs_log[-1]
    first_dev = next((e for e in epochs_log if "dev" in e), None)
    # train.py applies the whole divergence definition (NaN count, scale, dev CTC loss not decreasing by
    # epoch 3) and records the reason in the epoch summary that stopped the run
    divergence = last.get("divergence")
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
        "min_scale": min((r["scale"] for r in train_log), default=None),
        "dev_mix_full": last["dev"]["dev_mix"]["full"] if "dev" in last else None,
        "dev_h_full": last["dev"]["dev_h"]["full"] if "dev" in last else None,
        "dev_mix_chunk16": last["dev"]["dev_mix"].get("chunk16") if "dev" in last else None,
        "epoch_time_s": last.get("epoch_time_s"),
        "first_epoch_dev": first_dev["dev"] if first_dev else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lrs", default="1e-3,2e-3")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--out-dir", default="outputs/stage1", help="run directories go under this path")
    p.add_argument(
        "--schedule-steps", type=int, default=69700, help="Stage 2 schedule length (697 steps x 100 epochs)"
    )
    p.add_argument("--dev-loss-rule", default="1,3", help="A,B epochs for the dev-loss divergence rule")
    p.add_argument("--report-prefix", default="", help="prefix for summary/verdict files in reports/stage1")
    p.add_argument("--train-args", default="", help="extra arguments passed through to stt_scratch.train")
    args = p.parse_args()
    clash = [a for a in args.train_args.split() if a.split("=")[0] in CANDIDATE_ARGS]
    if clash:
        raise SystemExit(f"--train-args may not redefine the candidates: {clash}")
    out_dir = ROOT / "reports" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = args.report_prefix
    results = []
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
            "dev_mix_mean": statistics.mean(r["dev_mix_full"] for r in ok) if ok else None,
            # population std over the completed seeds (the registered s; sample std is reported alongside)
            "dev_mix_std": statistics.pstdev([r["dev_mix_full"] for r in ok]) if len(ok) > 1 else None,
            "dev_mix_sample_std": statistics.stdev([r["dev_mix_full"] for r in ok]) if len(ok) > 1 else None,
            # adoptable only when the registered design ran in full: 3 seeds x 5 epochs, 0 divergence
            "adoptable": len(rows) == 3
            and len(seeds) == 3
            and args.epochs == 5
            and len(ok) == 3
            and failures == 0
            and all(r["recipe_ok"] for r in rows),
        }
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
