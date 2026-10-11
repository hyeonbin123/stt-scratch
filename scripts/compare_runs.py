"""Paired comparison of two systems on the same utterances (docs/experiments.md "지표": paired bootstrap).

Reads the per-utterance hypothesis files written by scripts/stage2_report.py and reports, for the Hangul-only
subset and for all utterances, both CER_ns rates, the relative ratio a/b and the 95 % paired bootstrap
interval of rate(a) - rate(b) (1,000 resamples, seed 0).

Usage:
  uv run python scripts/compare_runs.py --a reports/stage2/hyps/runA-avg10-dev_mix-full.jsonl \
      --b reports/stage2/hyps/runB-avg10-dev_mix-full.jsonl --name H2.1-dev_mix --out-dir reports/stage2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stt_scratch.metrics import difference_interval, error_rate, score_ns

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


def compare(a: dict[str, dict], b: dict[str, dict], hangul_only: bool) -> dict:
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        raise SystemExit(
            f"the two files cover different utterances ({len(a)} vs {len(b)}, common {len(ids)})"
        )
    edits_a, edits_b, lengths = [], [], []
    for i in ids:
        if a[i]["ref"] != b[i]["ref"]:
            raise SystemExit(f"reference mismatch for {i}")
        if hangul_only and not a[i]["hangul_only"]:
            continue
        sa, sb = score_ns(a[i]["ref"], a[i]["hyp"]), score_ns(b[i]["ref"], b[i]["hyp"])
        if sa is None:
            continue
        edits_a.append(sa.edits)
        edits_b.append(sb.edits)
        lengths.append(sa.length)
    ea, eb, ln = (np.array(x, dtype=np.int64) for x in (edits_a, edits_b, lengths))
    ra, rb = error_rate(ea, ln), error_rate(eb, ln)
    low, high = difference_interval(ea, eb, ln)
    return {
        "n": len(ln),
        "rate_a": ra,
        "rate_b": rb,
        "ratio_a_over_b": ra / rb if rb else None,
        "diff_a_minus_b": ra - rb,
        "diff_ci95": [low, high],
        "ci_excludes_zero": low > 0 or high < 0,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", default="reports/stage2")
    args = p.parse_args(argv)
    a, b = load(ROOT / args.a), load(ROOT / args.b)
    result = {
        "a": args.a,
        "b": args.b,
        "hangul_only": compare(a, b, hangul_only=True),
        "all": compare(a, b, hangul_only=False),
    }
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare-{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for subset in ("hangul_only", "all"):
        r = result[subset]
        print(
            f"{args.name} {subset}: a {r['rate_a'] * 100:.2f} % vs b {r['rate_b'] * 100:.2f} % "
            f"(a/b {r['ratio_a_over_b']:.3f}), a-b {r['diff_a_minus_b'] * 100:+.2f} pp, "
            f"95 % [{r['diff_ci95'][0] * 100:+.2f}, {r['diff_ci95'][1] * 100:+.2f}], n {r['n']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
