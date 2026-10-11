"""Stage 3 equal-budget ablations (docs/experiments.md 3단계), run sequentially on one GPU.

Each arm trains from scratch with Run A's LR, seed and step count (rule change 15/17), then is averaged
(top 10 by dev_mix full), evaluated with scripts/stage2_report.py into reports/stage3/, and compared with
Run A's averaged model on the same utterances (scripts/compare_runs.py). Arms already finished are skipped;
an interrupted arm resumes from its last completed epoch.

Usage: uv run python scripts/stage3.py --arms a,b,c,d --budget-hours 45
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--lr", "2e-3", "--seed", "1", "--cuda-memory-fraction", "0.8"]
ARMS = {
    # (a) InterCTC only: one intermediate CTC head at layer 6, weight 0.3, no softmax feedback
    "a": {
        "epochs": 100,
        "args": ["--sc-layers", "6", "--sc-weight", "0.3", "--no-sc-feedback"],
        "factor": 1.0,
    },
    # (b) E-Branchformer blocks, parameters within 5 % of Run A (21.32 M vs 20.79 M)
    "b": {
        "epochs": 100,
        "args": ["--block", "ebranchformer", "--ffn-dim", "768", "--cgmlp-units", "896"],
        "factor": 1.15,
    },
    # (c) width 384 (FFN 4x = 1536, 44.5 M); only when Run A shows no overfitting sign
    "c": {"epochs": 100, "args": ["--d-model", "384", "--ffn-dim", "1536"], "factor": 1.6},
    # (d) CR-CTC: two views, alpha 0.2, time masks 2.5x (5 -> 12), half max-duration, 49 epochs x 1,418 steps
    #     = 69,482 steps (Run A 69,700), SpecAugment from the same step 2,091
    "d": {
        "epochs": 49,
        "args": [
            "--cr-ctc-alpha",
            "0.2",
            "--time-masks",
            "12",
            "--max-duration",
            "150",
            "--augment-start-step",
            "2091",
        ],
        "factor": 1.0,
    },
}
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}


def epochs_done(run: Path) -> list[dict]:
    f = run / "epochs.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]


def run_hours(run: Path) -> float:
    """GPU hours of a run: median interval between consecutive epoch checkpoints (one epoch incl. its dev
    evaluation) times the number of epochs. The median ignores pauses and slow epochs under other GPU load."""
    stamps = sorted(p.stat().st_mtime for p in run.glob("epoch-*.pt"))
    if len(stamps) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:], strict=False))
    return gaps[len(gaps) // 2] * len(stamps) / 3600


DRY_RUN = False


def sh(cmd: list[str], log: Path) -> int:
    if DRY_RUN:
        print("DRY", " ".join(cmd), flush=True)
        return 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def bench_peak_gb(spec: dict, log: Path) -> float | None:
    """300 training steps of an arm in a scratch run directory; returns the peak allocated GPU memory."""
    bench = ROOT / "work" / "stage3_bench_c"
    if bench.exists():
        import shutil

        shutil.rmtree(bench)
    cmd = [
        sys.executable,
        "-m",
        "stt_scratch.train",
        "--run",
        str(bench.relative_to(ROOT)),
        "--epochs",
        "1",
        "--max-steps",
        "300",
        "--log-every",
        "50",
        *BASE,
        *spec["args"],
    ]
    if sh(cmd, log) != 0:
        return None
    if DRY_RUN:
        return 0.0
    rows = [
        json.loads(line) for line in (bench / "train.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    return max((r.get("mem_gb") or 0.0) for r in rows) if rows else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="a,b,c,d")
    p.add_argument("--budget-hours", type=float, default=45.0)
    p.add_argument("--reference", default="runA-avg10", help="Stage 2 report name of the reference model")
    p.add_argument("--dry-run", action="store_true", help="print the commands; write status to work/")
    args = p.parse_args()
    global DRY_RUN
    DRY_RUN = args.dry_run
    out_root, rep = ROOT / "outputs" / "stage3", ROOT / "reports" / "stage3"
    out_root.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    status_path = (ROOT / "work" / "stage3_status_dry.json") if DRY_RUN else rep / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    ref_path = ROOT / "reports" / "stage2" / f"{args.reference}.json"
    ref_report = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    ref_hours = run_hours(ROOT / "outputs" / "stage2" / "runA")
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        run = out_root / arm
        log = ROOT / "work" / f"stage3_{arm}.log"
        if arm == "c" and not epochs_done(run):
            ratio = ref_report.get("train_sample", {}).get("ratio_to_dev_h")
            if ratio is not None and ratio < 0.5:
                status[arm] = {"skipped": f"overfitting sign: train/dev_H CER ratio {ratio:.3f} < 0.5"}
                status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
                continue
            peak = bench_peak_gb(spec, log)
            status["c_bench"] = {"peak_gb_300_steps": peak}
            if peak is None or peak > 8.0:
                status[arm] = {"skipped": f"memory: 300-step bench peak {peak} GB > 8.0 GB"}
                status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
                continue
        used = sum(v.get("hours", 0.0) for v in status.values() if isinstance(v, dict))
        estimate = ref_hours * spec["factor"]
        done = epochs_done(run)
        if (
            len([e for e in done if "dev" in e]) < spec["epochs"]
            and not done
            and used + estimate > args.budget_hours
        ):
            status[arm] = {
                "skipped": f"budget: used {used:.1f} h + estimate {estimate:.1f} h > {args.budget_hours} h"
            }
            status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
            continue
        if len([e for e in done if "dev" in e]) < spec["epochs"]:
            if done and any(e.get("divergence") or e.get("stopped_early") for e in done):
                status[arm] = {"stopped": done[-1]}
                status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
                print(f"arm {arm} stopped early; not continuing", flush=True)
                break
            resume = ["--resume"] if (run / "last.pt").exists() else []
            cmd = [
                sys.executable,
                "-m",
                "stt_scratch.train",
                "--run",
                str(run.relative_to(ROOT)),
                "--epochs",
                str(spec["epochs"]),
                *BASE,
                *spec["args"],
                *resume,
            ]
            code = sh(cmd, log)
            done = epochs_done(run)
            if code != 0 or any(e.get("divergence") or e.get("stopped_early") for e in done):
                status[arm] = {"stopped": done[-1] if done else None, "code": code, "hours": run_hours(run)}
                status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
                print(f"arm {arm} did not finish (code {code}); stopping", flush=True)
                break
        avg = run / "avg10.pt"
        if not avg.exists():
            sh(
                [
                    sys.executable,
                    "scripts/average_checkpoints.py",
                    "--run",
                    str(run.relative_to(ROOT)),
                    "--k",
                    "10",
                    "--out",
                    str(avg.relative_to(ROOT)),
                ],
                log,
            )
        name = f"{arm}-avg10"
        if not (rep / f"{name}.json").exists():
            sh(
                [
                    sys.executable,
                    "scripts/stage2_report.py",
                    "--checkpoint",
                    str(avg.relative_to(ROOT)),
                    "--name",
                    name,
                    "--report-dir",
                    "reports/stage3",
                    "--chunks",
                    "8,16",
                    "--streaming-check",
                    "0",
                    "--train-sample",
                    "1000",
                ],
                log,
            )
        for s in ("dev_mix", "dev_h"):
            sh(
                [
                    sys.executable,
                    "scripts/compare_runs.py",
                    "--a",
                    f"reports/stage3/hyps/{name}-{s}-full.jsonl",
                    "--b",
                    f"reports/stage2/hyps/{args.reference}-{s}-full.jsonl",
                    "--name",
                    f"{arm}-vs-runA-{s}",
                    "--out-dir",
                    "reports/stage3",
                ],
                log,
            )
        status[arm] = {"done": True, "hours": run_hours(run), "epochs": len(epochs_done(run))}
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"arm {arm} done", flush=True)


if __name__ == "__main__":
    main()
