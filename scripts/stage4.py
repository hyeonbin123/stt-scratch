"""Stage 4 emission-latency continuations (docs/experiments.md 4단계, rule changes 16 and 18).

From the starting model (the averaged Stage 2 model or the Stage 3 winner): calibrate the Peak-First lambda,
continue three runs for 15 epochs with the same recipe (control, TrimTail 200 ms, Peak-First), average each
(top 10 of 15 by dev_mix), measure latency and CER with scripts/stage4_latency.py, and write the H4 verdict
(FTD P50 reduction >= 80 ms with CER_ns relative increase <= 3 % on dev_mix, dev_H guard 0.3 absolute).

Usage: uv run python scripts/stage4.py --init outputs/stage2/runA/avg10.pt [--model-args "..."]
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
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}
# rule change 16: weights only, fresh optimizer/schedule, augmentation from the start, warmup 500,
# 15-epoch cosine, peak LR 1e-3 (half of Stage 2's 2e-3), no dev-loss divergence rule (--init)
RECIPE = [
    "--epochs",
    "15",
    "--lr",
    "1e-3",
    "--warmup",
    "500",
    "--seed",
    "1",
    "--augment-start-epoch",
    "0",
    "--perturb-start-epoch",
    "0",
    "--cuda-memory-fraction",
    "0.8",
]


def sh(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def epochs(run: Path) -> list[dict]:
    f = run / "epochs.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x] if f.exists() else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument(
        "--model-args", default="", help="architecture flags of the starting model (Stage 3 winner)"
    )
    args = p.parse_args()
    py = sys.executable
    out_root, rep = ROOT / "outputs" / "stage4", ROOT / "reports" / "stage4"
    out_root.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work" / "stage4.log"
    lam_path = rep / "peak_first_lambda.json"
    if not lam_path.exists():
        if sh([py, "scripts/calibrate_peak_first.py", "--checkpoint", args.init], log) != 0:
            raise SystemExit("lambda calibration failed")
    lam = json.loads(lam_path.read_text(encoding="utf-8"))["lambda"]
    variants = {
        "control": [],
        "trimtail": ["--trimtail-ms", "200"],
        "peakfirst": ["--peak-first-lambda", f"{lam:.6g}"],
    }
    summary = {"init": args.init, "lambda": lam, "variants": {}}
    for name, extra in variants.items():
        run = out_root / name
        done = epochs(run)
        if len([e for e in done if "dev" in e]) < 15:
            resume = ["--resume"] if (run / "last.pt").exists() else ["--init", args.init]
            cmd = [
                py,
                "-m",
                "stt_scratch.train",
                "--run",
                str(run.relative_to(ROOT)),
                *RECIPE,
                *args.model_args.split(),
                *extra,
                *resume,
            ]
            if sh(cmd, log) != 0 or any(e.get("divergence") or e.get("stopped_early") for e in epochs(run)):
                raise SystemExit(f"variant {name} did not finish")
        avg = run / "avg10.pt"
        if not avg.exists():
            sh(
                [
                    py,
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
        if not (rep / f"latency-{name}.json").exists():
            sh(
                [py, "scripts/stage4_latency.py", "--checkpoint", str(avg.relative_to(ROOT)), "--name", name],
                log,
            )
        summary["variants"][name] = json.loads((rep / f"latency-{name}.json").read_text(encoding="utf-8"))

    def pick(variant: str, s: str, chunk_ms: int) -> dict:
        res = summary["variants"][variant]["results"]
        return next(r for r in res if r["set"] == s and r["chunk_ms"] == chunk_ms)

    verdict = {}
    for chunk_ms in (320, 640):
        c_mix, c_h = pick("control", "dev_mix", chunk_ms), pick("control", "dev_h", chunk_ms)
        for name in ("trimtail", "peakfirst"):
            v_mix, v_h = pick(name, "dev_mix", chunk_ms), pick(name, "dev_h", chunk_ms)
            ftd_gain = c_mix["ftd_ms"]["p50"] - v_mix["ftd_ms"]["p50"]
            cer_rel = v_mix["cer_ns"] / c_mix["cer_ns"] - 1
            guard = (v_h["cer_ns"] - c_h["cer_ns"]) * 100 <= 0.3
            verdict[f"{name}@{chunk_ms}ms"] = {
                "ftd_p50_reduction_ms": ftd_gain,
                "cer_relative_change": cer_rel,
                "dev_h_cer_change_pp": (v_h["cer_ns"] - c_h["cer_ns"]) * 100,
                "h4_met": ftd_gain >= 80 and cer_rel <= 0.03 and guard,
                "eligible": cer_rel <= 0.03 and guard,
            }
    summary["verdict"] = verdict
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
