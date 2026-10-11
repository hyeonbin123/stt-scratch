"""Stage 4 emission-latency continuations (docs/experiments.md 4단계, rule changes 16, 18 and 20).

From the starting model (Run A's averaged model, or the averaged model of the Stage 3 winner, with that
arm's architecture and loss options): calibrate the Peak-First lambda, continue three runs for 15 epochs
with the same recipe (control, TrimTail 200 ms, Peak-First), average each (top 10 of 15 by dev_mix),
measure latency and CER with scripts/stage4_latency.py, and write the verdict:
  primary: the service chunk 320 ms on dev_mix; eligible = CER_ns relative increase <= 3 % vs control and
  dev_H not more than 0.3 points worse; the eligible variant with the largest FTD P50 reduction (> 0) is
  adopted, H4 is met when that reduction is >= 80 ms; otherwise the control is kept. 640 ms and the FTD over
  the utterances whose first word both models got right are reported and do not change the verdict.
Files of an earlier starting model are never reused (each output records the starting checkpoint's SHA-256).

Usage: uv run python scripts/stage4.py --start runA      (or --start a|b|d for a Stage 3 winner)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stage3 import ARMS  # noqa: E402

ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}
# rule change 16: weights only, fresh optimizer/schedule, augmentation from the start, warmup 500,
# 15-epoch cosine, peak LR 1e-3 (half of Stage 2's 2e-3), no dev-loss divergence rule (--init)
RECIPE = [
    "--epochs", "15", "--lr", "1e-3", "--warmup", "500", "--seed", "1",
    "--augment-start-epoch", "0", "--perturb-start-epoch", "0", "--cuda-memory-fraction", "0.8",
]  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sh(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def start_model(start: str) -> tuple[Path, list[str]]:
    """Averaged starting checkpoint and the architecture/loss options of that model."""
    if start == "runA":
        return ROOT / "outputs" / "stage2" / "runA" / "avg10.pt", []
    status = json.loads((ROOT / "reports" / "stage3" / "status.json").read_text(encoding="utf-8"))
    if status.get(start, {}).get("state") != "done":
        raise SystemExit(f"Stage 3 arm {start} is not done")
    run = status[start]["run"]
    args = list(ARMS[start]["args"])
    if "--augment-start-step" in args:  # the continuation augments from the start (rule change 16)
        i = args.index("--augment-start-step")
        del args[i : i + 2]
    return ROOT / "outputs" / "stage3" / run / "avg10.pt", args


def fresh(path: Path, init_sha: str) -> bool:
    """True when `path` exists and was made from the same starting checkpoint; stale files are renamed."""
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf-8")).get("init_sha256") == init_sha:
        return True
    path.rename(path.with_suffix(f".stale-{int(time.time())}.json"))
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="runA", help="runA or a finished Stage 3 arm (a, b, d)")
    args = p.parse_args()
    py = sys.executable
    init, model_args = start_model(args.start)
    init_rel, init_sha = str(init.relative_to(ROOT)), sha256(init)
    out_root, rep = ROOT / "outputs" / "stage4", ROOT / "reports" / "stage4"
    out_root.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work" / "stage4.log"

    lam_path = rep / "peak_first_lambda.json"
    if not fresh(lam_path, init_sha):
        cmd = [py, "scripts/calibrate_peak_first.py", "--checkpoint", init_rel]
        if "--max-duration" in model_args:
            cmd += ["--max-duration", model_args[model_args.index("--max-duration") + 1]]
        if sh(cmd, log) != 0:
            raise SystemExit("lambda calibration failed")
        data = json.loads(lam_path.read_text(encoding="utf-8"))
        data["init_sha256"] = init_sha
        lam_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    lam = json.loads(lam_path.read_text(encoding="utf-8"))["lambda"]
    variants = {
        "control": [],
        "trimtail": ["--trimtail-ms", "200"],
        "peakfirst": ["--peak-first-lambda", f"{lam:.6g}"],
    }
    summary = {"start": args.start, "init": init_rel, "init_sha256": init_sha, "lambda": lam, "variants": {}}
    sys.path.insert(0, str(ROOT / "src"))
    from stt_scratch.train import completed_epochs

    for name, extra in variants.items():
        run = out_root / name
        meta = run / "meta.json"
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8"))["args"].get("init") != init_rel:
            raise SystemExit(f"{run} was started from another model; move it away first")
        if completed_epochs(run) < 15:
            resume = ["--resume"] if (run / "last.pt").exists() else []
            cmd = [py, "-m", "stt_scratch.train", "--run", str(run.relative_to(ROOT)), *RECIPE, *model_args,
                   *extra, "--init", init_rel, *resume]  # fmt: skip
            if sh(cmd, log) != 0 or completed_epochs(run) < 15:
                raise SystemExit(f"variant {name} did not finish")
        avg = run / "avg10.pt"
        if not avg.exists() and sh([py, "scripts/average_checkpoints.py", "--run", str(run.relative_to(ROOT)),
                                    "--k", "10", "--out", str(avg.relative_to(ROOT))], log) != 0:  # fmt: skip
            raise SystemExit(f"averaging {name} failed")
        lat = rep / f"latency-{name}.json"
        if not fresh(lat, init_sha):
            if sh(
                [py, "scripts/stage4_latency.py", "--checkpoint", str(avg.relative_to(ROOT)), "--name", name],
                log,
            ):
                raise SystemExit(f"latency measurement of {name} failed")
            data = json.loads(lat.read_text(encoding="utf-8"))
            data["init_sha256"] = init_sha
            lat.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["variants"][name] = json.loads(lat.read_text(encoding="utf-8"))

    def pick(variant: str, s: str, chunk_ms: int) -> dict:
        return next(
            r for r in summary["variants"][variant]["results"] if r["set"] == s and r["chunk_ms"] == chunk_ms
        )

    table = {}
    for chunk_ms in (320, 640):
        c_mix, c_h = pick("control", "dev_mix", chunk_ms), pick("control", "dev_h", chunk_ms)
        for name in ("trimtail", "peakfirst"):
            v_mix, v_h = pick(name, "dev_mix", chunk_ms), pick(name, "dev_h", chunk_ms)
            common = sorted(set(c_mix["ftd_by_utt_ms"]) & set(v_mix["ftd_by_utt_ms"]))
            common_gain = (
                statistics.median(c_mix["ftd_by_utt_ms"][u] for u in common)
                - statistics.median(v_mix["ftd_by_utt_ms"][u] for u in common)
                if common
                else None
            )
            dev_h_change = (v_h["cer_ns"] - c_h["cer_ns"]) * 100
            cer_rel = v_mix["cer_ns"] / c_mix["cer_ns"] - 1
            table[f"{name}@{chunk_ms}ms"] = {
                "ftd_p50_reduction_ms": c_mix["ftd_ms"]["p50"] - v_mix["ftd_ms"]["p50"],
                "ltd_p50_reduction_ms": c_mix["ltd_ms"]["p50"] - v_mix["ltd_ms"]["p50"],
                "ftd_n_control": c_mix["ftd_ms"]["n"],
                "ftd_n_variant": v_mix["ftd_ms"]["n"],
                "ftd_p50_reduction_common_ms": common_gain,
                "ftd_common_n": len(common),
                "cer_relative_change": cer_rel,
                "dev_h_cer_change_pp": dev_h_change,
                "eligible": cer_rel <= 0.03 and dev_h_change <= 0.3,
            }
    candidates = [
        (table[f"{n}@320ms"]["ftd_p50_reduction_ms"], n)
        for n in ("trimtail", "peakfirst")
        if table[f"{n}@320ms"]["eligible"] and table[f"{n}@320ms"]["ftd_p50_reduction_ms"] > 0
    ]
    adopted = max(candidates)[1] if candidates else "control"
    summary["table"] = table
    summary["verdict"] = {
        "adopted": adopted,
        "h4_met": bool(candidates) and max(candidates)[0] >= 80,
        "basis": "service chunk 320 ms, dev_mix FTD P50; eligibility CER +3 % relative and dev_H +0.3 points",
    }
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"table": table, "verdict": summary["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
