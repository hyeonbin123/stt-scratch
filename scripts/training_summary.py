"""fp16 stability and GPU time of every training run under outputs/ (Stage 7 final report).

Per run: completed epochs, summed epoch compute time (training only, dev evaluation and pauses excluded),
GradScaler-skipped steps, NaN/inf loss count, lowest loss scale, longest low-scale streak and the registered
divergence verdict, if any. Reads only the run logs (train.jsonl, epochs.jsonl); no GPU.

    uv run python scripts/training_summary.py [--out reports/final/training_summary]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line.replace("NaN", "null").replace("Infinity", "null")))
    return rows


def summarize(run: Path) -> dict:
    epochs = [r for r in read_jsonl(run / "epochs.jsonl") if "epoch_time_s" in r]
    steps = read_jsonl(run / "train.jsonl")
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    divergence = next((r["divergence"] for r in reversed(epochs) if r.get("divergence")), None)
    if divergence is None:  # a NaN/scale stop is written to the summary line, which has no epoch_time_s
        divergence = next(
            (r["divergence"] for r in reversed(read_jsonl(run / "epochs.jsonl")) if r.get("divergence")), None
        )
    scales = [r["scale"] for r in steps if isinstance(r.get("scale"), (int, float))]
    scales_set = set(scales)
    # skip the first 300 logged steps' scale: GradScaler starts at 2^15 and halves on purpose until it fits
    late_scales = [r["scale"] for r in steps if r.get("step", 0) >= 300 and r.get("scale") in scales_set]
    last = epochs[-1] if epochs else {}
    args = meta.get("args", {})
    return {
        "run": run.relative_to(ROOT / "outputs").as_posix(),
        "lr": args.get("lr"),
        "init": bool(args.get("init")),
        "epochs": len(epochs),
        "compute_h": sum(r.get("epoch_time_s", 0.0) for r in epochs) / 3600,
        "skipped_steps": last.get("skipped"),
        "updates": last.get("updates"),
        "nan_count": max((r.get("nan_count", 0) for r in epochs), default=0),
        "min_scale": min(scales) if scales else None,
        "min_scale_after_300": min(late_scales) if late_scales else None,
        "max_low_scale_streak": max((r.get("low_scale_streak", 0) for r in epochs), default=0),
        "divergence": divergence.get("reason") if isinstance(divergence, dict) else divergence,
        "precision": "fp32" if args.get("fp32") else "fp16",
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/final/training_summary")
    args = p.parse_args(argv)
    runs = sorted(d for d in (ROOT / "outputs").glob("*/*") if (d / "epochs.jsonl").exists())
    rows = [summarize(r) for r in runs]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["compute_h"] for r in rows)
    diverged = [r["run"] for r in rows if r["divergence"]]
    report = {"runs": rows, "compute_h_total": total, "diverged": diverged}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    def fmt(v, spec=""):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        return format(v, spec) if spec else str(v)

    lines = [
        "| run | LR | 이어 학습 | epoch | 학습 계산 h | 건너뛴 step | NaN "
        "| 최저 scale(step 300 이후) | 발산 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {fmt(r['lr'])} | {'예' if r['init'] else ''} | {r['epochs']} | "
            f"{r['compute_h']:.2f} | "
            f"{fmt(r['skipped_steps'])} | {r['nan_count']} | {fmt(r['min_scale_after_300'], 'g')} | "
            f"{fmt(r['divergence'])} |"
        )
    note = "epoch 학습 시간만; dev 평가·일시 중지·측정 제외"
    lines += ["", f"학습 계산 합계 {total:.2f} h ({note}), 발산 {len(diverged)}런"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
