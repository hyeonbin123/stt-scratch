"""Post-hoc breakdown of the sealed Zeroth official test (no new decoding, not a verdict).

Splits the saved Stage 7 hypotheses of test_zeroth by whether the reference sentence (scoring normalization)
also occurs in our training manifest, and reports CER_ns with bootstrap intervals for both subsets.

    uv run python scripts/sealed_breakdown.py   # -> reports/final/zeroth_seen_unseen.json
"""

from __future__ import annotations

import json
from pathlib import Path

from stt_scratch.data import read_manifest
from stt_scratch.metrics import score_ns, summarize
from stt_scratch.text import normalize_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train = {normalize_score(u.text) for u in read_manifest(ROOT / "manifests" / "train.jsonl")}
    out: dict = {"note": "post-hoc split of the saved sealed-test hypotheses; no new decoding"}
    for mode in ("full", "streaming"):
        path = ROOT / "reports" / "final" / f"test_zeroth-{mode}.hyps.jsonl"
        rows = [json.loads(x) for x in path.open(encoding="utf-8")]
        groups = {"sentence_in_train": [], "sentence_not_in_train": []}
        for r in rows:
            key = "sentence_in_train" if normalize_score(r["ref"]) in train else "sentence_not_in_train"
            s = score_ns(r["ref"], r["hyp"])
            if s is not None:
                groups[key].append(s)
        out[mode] = {k: summarize(v) for k, v in groups.items()}
    path = ROOT / "reports" / "final" / "zeroth_seen_unseen.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for mode in ("full", "streaming"):
        for k, s in out[mode].items():
            print(mode, k, s["n"], round(s["rate"] * 100, 2), [round(c * 100, 2) for c in s["ci"]])


if __name__ == "__main__":
    main()
