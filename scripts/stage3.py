"""Stage 3 equal-budget ablations (docs/experiments.md 3단계, rule changes 17 and 19), run on one GPU.

Each arm trains from scratch with Run A's recipe except its intended difference, is averaged (top 10 by
dev_mix full), evaluated with scripts/stage2_report.py into reports/stage3/, and compared with Run A's
averaged model on the same utterances (scripts/compare_runs.py). Safeguards (Codex Stage 2 review):
- completion is read from the run's last.pt, never from the epoch log; an existing run (finished or
  partial) is reused only if its meta.json recipe equals the registered arm recipe
- a run already being trained by another process is waited for, never started twice
- every evaluation step must succeed and leave its output file, otherwise the arm is marked failed
- budget: actual hours of started arms plus estimates of the pending ones; when they exceed the budget the
  lowest-priority pending arm is dropped first (d, then c, then b, then a), as registered
- arm c runs only when Run A's train/dev_H CER ratio is known and >= 0.5
- a diverged arm restarts with LR x 0.5 under <arm>-lr<value>, at most twice (fp16 rule); the LR is reported

Usage: uv run python scripts/stage3.py --arms a,b,c,d --budget-hours 45 [--dry-run]
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
sys.path.insert(0, str(ROOT / "src"))
from stt_scratch.train import RECIPE_KEYS, build_parser, completed_epochs, expected_recipe  # noqa: E402

PRIORITY = ["a", "b", "c", "d"]  # run order; dropped from the end when over budget
BASE_LR = 2e-3
BASE = ["--seed", "1", "--cuda-memory-fraction", "0.8"]
# epoch_scale: epoch time relative to a Run A epoch (d has 1,418 steps per epoch instead of 697)
ARMS = {
    # (a) InterCTC only: one intermediate CTC head at layer 6, weight 0.3, no softmax feedback
    "a": {
        "epochs": 100,
        "args": ["--sc-layers", "6", "--sc-weight", "0.3", "--no-sc-feedback"],
        "factor": 1.0,
        "epoch_scale": 1.0,
    },
    # (b) E-Branchformer blocks, parameters within 5 % of Run A (21.32 M vs 20.79 M)
    "b": {
        "epochs": 100,
        "args": ["--block", "ebranchformer", "--ffn-dim", "768", "--cgmlp-units", "896"],
        "factor": 1.15,
        "epoch_scale": 1.0,
    },
    # (c) width 384 (FFN 4x = 1536, 44.5 M); only when Run A shows no overfitting sign
    "c": {
        "epochs": 100,
        "args": ["--d-model", "384", "--ffn-dim", "1536"],
        "factor": 1.6,
        "epoch_scale": 1.0,
    },
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
        "epoch_scale": 1418 / 697,
    },
}
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}
DRY_RUN = False


def arm_argv(arm: str, run: Path, lr: float) -> list[str]:
    spec = ARMS[arm]
    return [
        "--run",
        str(run.relative_to(ROOT)),
        "--epochs",
        str(spec["epochs"]),
        "--lr",
        f"{lr:g}",
        *BASE,
        *spec["args"],
    ]


def recipe_matches(run: Path, argv: list[str]) -> tuple[bool, dict]:
    """Compare the run's recorded recipe (meta.json args; keys added later count as defaults) with argv's."""
    meta = run / "meta.json"
    if not meta.exists():
        has_ckpt = run.exists() and any(run.glob("*.pt"))
        return (not has_ckpt), ({"meta.json": ("missing", "required")} if has_ckpt else {})
    recorded = json.loads(meta.read_text(encoding="utf-8"))["args"]
    parser = build_parser()
    want = expected_recipe(argv)
    diff = {}
    for k in RECIPE_KEYS:
        have = recorded.get(k, parser.get_default(k))
        same = have == want[k] or (isinstance(want[k], float) and have is not None and float(have) == want[k])
        if not same:
            diff[k] = (have, want[k])
    return not diff, diff


def training_alive(run: Path) -> bool:
    rel = str(run.relative_to(ROOT)).replace("\\", "/")
    query = (
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
        "$_.CommandLine -match 'stt_scratch.train' -and "
        f"(($_.CommandLine -replace '\\\\','/') -match '{rel}( |$)') }}).Count"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=60
        ).stdout.strip()
        return int(out or "0") > 0
    except Exception:  # noqa: BLE001
        return False


def actual_hours(run: Path) -> float:
    """Wall-clock hours from the run's start (meta.json) to its newest epoch checkpoint (pauses included)."""
    pts = list(run.glob("epoch-*.pt"))
    if not pts or not (run / "meta.json").exists():
        return 0.0
    return (max(p.stat().st_mtime for p in pts) - (run / "meta.json").stat().st_mtime) / 3600


def median_epoch_hours(run: Path) -> float:
    stamps = sorted(p.stat().st_mtime for p in run.glob("epoch-*.pt"))
    if len(stamps) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:], strict=False))
    return gaps[len(gaps) // 2] / 3600


def estimate_hours(arm: str, ref_epoch_h: float) -> float:
    spec = ARMS[arm]
    return ref_epoch_h * spec["epochs"] * spec["epoch_scale"] * spec["factor"]


def sh(cmd: list[str], log: Path) -> int:
    if DRY_RUN:
        print("DRY", " ".join(cmd), flush=True)
        return 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def diverged(run: Path) -> bool:
    f = run / "epochs.jsonl"
    if not f.exists():
        return False
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    return any(r.get("divergence") or r.get("stopped_early") for r in rows)


def bench_peak_gb(spec: dict, log: Path) -> float | None:
    """300 training steps of an arm in a scratch directory; returns the peak allocated GPU memory."""
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
        "--lr",
        f"{BASE_LR:g}",
        *BASE,
        *spec["args"],
    ]
    if sh(cmd, log) != 0:
        return None
    if DRY_RUN:
        return 0.0
    rows = [json.loads(x) for x in (bench / "train.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return max((r.get("mem_gb") or 0.0) for r in rows) if rows else None


def train_arm(arm: str, status: dict, log: Path) -> Path | None:
    """Train (or reuse) the arm; after a divergence restart with LR x 0.5, at most twice. Returns the run dir
    of the finished attempt, or None."""
    lr = BASE_LR
    for attempt in range(3):
        name = arm if attempt == 0 else f"{arm}-lr{lr:g}"
        run = ROOT / "outputs" / "stage3" / name
        argv = arm_argv(arm, run, lr)
        ok, diff = recipe_matches(run, argv)
        if not ok:
            raise SystemExit(f"{run} holds a run with a different recipe: {diff}")
        while not DRY_RUN and training_alive(run):
            time.sleep(120)  # started outside this runner (e.g. arm a launched directly); wait for it
        if completed_epochs(run) >= ARMS[arm]["epochs"]:
            status[arm].update({"run": name, "lr": lr})
            return run
        if not diverged(run):
            resume = ["--resume"] if (run / "last.pt").exists() else []
            code = sh([sys.executable, "-m", "stt_scratch.train", *argv, *resume], log)
            if DRY_RUN:
                status[arm].update({"run": name, "lr": lr})
                return run
            if code == 0 and completed_epochs(run) >= ARMS[arm]["epochs"]:
                status[arm].update({"run": name, "lr": lr})
                return run
        if not diverged(run):
            status[arm]["error"] = f"{name} ended without finishing and without a divergence record"
            return None
        status[arm].setdefault("diverged_attempts", []).append({"run": name, "lr": lr})
        lr /= 2
    return None


def evaluate_arm(run: Path, reference: str, log: Path) -> bool:
    py, rel = sys.executable, str(run.relative_to(ROOT))
    name = f"{run.name}-avg10"
    rep = ROOT / "reports" / "stage3"
    steps = []
    if not (run / "avg10.pt").exists():
        steps.append(
            (
                [py, "scripts/average_checkpoints.py", "--run", rel, "--k", "10", "--out", f"{rel}/avg10.pt"],
                run / "avg10.pt",
            )
        )
    if not (rep / f"{name}.json").exists():
        steps.append(
            (
                [
                    py,
                    "scripts/stage2_report.py",
                    "--checkpoint",
                    f"{rel}/avg10.pt",
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
                rep / f"{name}.json",
            )
        )
    for s in ("dev_mix", "dev_h"):
        steps.append(
            (
                [
                    py,
                    "scripts/compare_runs.py",
                    "--a",
                    f"reports/stage3/hyps/{name}-{s}-full.jsonl",
                    "--b",
                    f"reports/stage2/hyps/{reference}-{s}-full.jsonl",
                    "--name",
                    f"{run.name}-vs-runA-{s}",
                    "--out-dir",
                    "reports/stage3",
                ],
                rep / f"compare-{run.name}-vs-runA-{s}.json",
            )
        )
    for cmd, expected in steps:
        if sh(cmd, log) != 0 or (not DRY_RUN and not expected.exists()):
            return False
    return True


def main() -> None:
    global DRY_RUN
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="a,b,c,d")
    p.add_argument("--budget-hours", type=float, default=45.0)
    p.add_argument("--reference", default="runA-avg10", help="Stage 2 report name of the reference model")
    p.add_argument("--dry-run", action="store_true", help="print the commands; status goes to work/")
    args = p.parse_args()
    DRY_RUN = args.dry_run
    rep = ROOT / "reports" / "stage3"
    (ROOT / "outputs" / "stage3").mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    status_path = (ROOT / "work" / "stage3_status_dry.json") if DRY_RUN else rep / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    ref_path = ROOT / "reports" / "stage2" / f"{args.reference}.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    ref_epoch_h = median_epoch_hours(ROOT / "outputs" / "stage2" / "runA")
    requested = [a for a in PRIORITY if a in args.arms.split(",")]

    def save() -> None:
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")

    for arm in requested:
        status.setdefault(arm, {})
    ratio = ref.get("train_sample", {}).get("ratio_to_dev_h")
    if "c" in requested and status["c"].get("state") != "done" and (ratio is None or ratio < 0.5):
        status["c"] = {"state": "skipped", "reason": f"overfitting condition: train/dev_H CER ratio {ratio}"}
    save()

    def used_hours() -> float:
        """Actual hours of every run this stage started, including the diverged attempts."""
        runs = set()
        for s in status.values():
            if isinstance(s, dict):
                if s.get("run"):
                    runs.add(s["run"])
                runs.update(a["run"] for a in s.get("diverged_attempts", []))
        return sum(actual_hours(ROOT / "outputs" / "stage3" / r) for r in runs)

    for arm in requested:
        if status[arm].get("state") == "done":
            run = ROOT / "outputs" / "stage3" / status[arm]["run"]
            ok, diff = recipe_matches(run, arm_argv(arm, run, status[arm]["lr"]))
            if not ok or completed_epochs(run) < ARMS[arm]["epochs"]:
                raise SystemExit(f"arm {arm} is marked done but {run} does not match: {diff}")
            continue
        if status[arm].get("state") in ("skipped", "failed"):
            continue
        used = used_hours()
        pending = [a for a in requested if status[a].get("state") not in ("done", "skipped", "failed")]
        while pending:
            estimate = sum(estimate_hours(a, ref_epoch_h) for a in pending)
            if used + estimate <= args.budget_hours:
                break
            dropped = pending.pop()  # lowest priority first: d, then c, then b, then a
            status[dropped] = {
                "state": "skipped",
                "reason": f"budget: used {used:.1f} h + estimate {estimate:.1f} h > {args.budget_hours} h",
            }
        status["budget"] = {"limit_h": args.budget_hours, "used_h": used, "ref_epoch_h": ref_epoch_h}
        save()
        if status[arm].get("state") == "skipped":
            continue
        log = ROOT / "work" / f"stage3_{arm}.log"
        if arm == "c" and not (ROOT / "outputs" / "stage3" / "c").exists():
            peak = bench_peak_gb(ARMS["c"], log)
            status["c"]["bench_peak_gb"] = peak
            if peak is None or peak > 8.0:
                status["c"].update(
                    {"state": "skipped", "reason": f"memory: 300-step bench peak {peak} GB > 8.0"}
                )
                save()
                continue
        run = train_arm(arm, status, log)
        if run is None:
            status[arm]["state"] = "failed"
            save()
            continue
        status[arm]["actual_hours"] = actual_hours(run)
        if not evaluate_arm(run, args.reference, log):
            status[arm].update({"state": "failed", "error": "evaluation step failed"})
            save()
            continue
        status[arm]["state"] = "done"
        save()
        print(f"arm {arm} done ({run.name})", flush=True)
    status["budget"] = {"limit_h": args.budget_hours, "used_h": used_hours(), "ref_epoch_h": ref_epoch_h}
    save()


if __name__ == "__main__":
    main()
