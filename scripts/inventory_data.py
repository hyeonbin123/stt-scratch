"""Inventory the downloaded corpora: hours, utterance counts, text statistics, overlap, syllable coverage.

Reads data/raw/* written by download_data.py and writes data/inventory.json plus a Markdown summary
on stdout. Audio is decoded only to measure duration and sample rate (no features are computed).

Usage: uv run python scripts/inventory_data.py [--raw data/raw] [--out data/inventory.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]")
PUNCT = re.compile(r"[^\w\s가-힣]")


def text_stats(texts: list[str], durations: list[float]) -> dict:
    """Per-split text statistics; durations are seconds aligned with texts (may be empty)."""
    n = len(texts)
    syl = np.array([len(HANGUL.findall(t)) for t in texts])
    chars_no_space = np.array([len(t.replace(" ", "")) for t in texts])
    out = {
        "n_utts": n,
        "n_unique_texts": len(set(texts)),
        "syllables_per_utt_mean": float(syl.mean()) if n else 0.0,
        "syllables_per_utt_max": int(syl.max()) if n else 0,
        "pct_with_digits": float(100 * np.mean([bool(DIGIT.search(t)) for t in texts])) if n else 0.0,
        "pct_with_latin": float(100 * np.mean([bool(LATIN.search(t)) for t in texts])) if n else 0.0,
        "pct_with_punct": float(100 * np.mean([bool(PUNCT.search(t)) for t in texts])) if n else 0.0,
        "syllable_inventory": len(set(ch for t in texts for ch in HANGUL.findall(t))),
    }
    if durations and len(durations) == n:
        d = np.array(durations)
        rate = chars_no_space / np.maximum(d, 1e-6)
        out.update(
            {
                "hours": float(d.sum() / 3600),
                "duration_mean_s": float(d.mean()),
                "duration_p95_s": float(np.percentile(d, 95)),
                "duration_max_s": float(d.max()),
                "chars_per_sec_p50": float(np.percentile(rate, 50)),
                "chars_per_sec_p95": float(np.percentile(rate, 95)),
                "chars_per_sec_max": float(rate.max()),
            }
        )
    return out


def decode_duration(blob: bytes) -> tuple[float, int]:
    with sf.SoundFile(io.BytesIO(blob)) as f:
        return f.frames / f.samplerate, f.samplerate


def read_parquet_split(
    files: list[Path], text_keys: tuple[str, ...]
) -> tuple[list[str], list[float], Counter]:
    texts: list[str] = []
    durations: list[float] = []
    rates: Counter = Counter()
    for file in files:
        table = pq.read_table(file)
        names = table.column_names
        text_col = next(k for k in text_keys if k in names)
        audio = table.column("audio").to_pylist()
        for row_text, row_audio in zip(table.column(text_col).to_pylist(), audio, strict=True):
            texts.append(row_text.strip())
            dur, rate = decode_duration(row_audio["bytes"])
            durations.append(dur)
            rates[rate] += 1
    return texts, durations, rates


def inventory_zeroth(root: Path) -> dict:
    out = {}
    for split in ("train", "test"):
        files = sorted((root / "data").glob(f"{split}-*.parquet"))
        texts, durations, rates = read_parquet_split(files, ("text", "sentence", "transcription"))
        out[split] = {
            "files": len(files),
            "sample_rates": dict(rates),
            "texts": texts,
            **text_stats(texts, durations),
        }
    return out


def inventory_fleurs(root: Path) -> dict:
    out = {}
    for split in ("train", "validation", "test"):
        files = sorted((root / "ko_kr" / split).glob("*.parquet"))
        texts, durations, rates = read_parquet_split(files, ("transcription",))
        raw_texts: list[str] = []
        for file in files:
            raw_texts.extend(
                t.strip() for t in pq.read_table(file, columns=["raw_transcription"]).column(0).to_pylist()
            )
        stats = text_stats(texts, durations)
        stats["raw_transcription_pct_with_digits"] = float(
            100 * np.mean([bool(DIGIT.search(t)) for t in raw_texts])
        )
        out[split] = {"files": len(files), "sample_rates": dict(rates), "texts": texts, **stats}
    return out


def inventory_common_voice(root: Path) -> dict:
    out = {}
    durations_ms = {}
    with open(root / "transcript" / "ko" / "clip_durations.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            durations_ms[row["clip"]] = float(row["duration[ms]"])
    for split in ("train", "dev", "test", "validated", "other", "invalidated"):
        if not (root / "transcript" / "ko" / f"{split}.tsv").exists():
            continue
        with open(root / "transcript" / "ko" / f"{split}.tsv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        texts = [r["sentence"].strip() for r in rows]
        durations = [durations_ms.get(r["path"], 0.0) / 1000 for r in rows]
        stats = text_stats(texts, durations)
        stats["n_speakers"] = len({r["client_id"] for r in rows})
        tars = sorted((root / "audio" / "ko" / split).glob("*.tar"))
        rates: Counter = Counter()
        if tars:
            with tarfile.open(tars[0]) as tf:
                for member in tf.getmembers()[:20]:
                    if member.isfile():
                        blob = tf.extractfile(member).read()
                        try:
                            _, rate = decode_duration(blob)
                            rates[rate] += 1
                        except Exception as exc:  # noqa: BLE001 - report decoder problems, do not abort
                            rates[f"decode_error:{type(exc).__name__}"] += 1
        out[split] = {"files": len(tars), "sample_rates": dict(rates), "texts": texts, **stats}
    return out


def overlap(a: list[str], b: list[str]) -> dict:
    sa, sb = set(a), set(b)
    inter = sa & sb
    return {
        "exact_match_unique": len(inter),
        "pct_of_b_utts_in_a": float(100 * np.mean([t in sa for t in b])) if b else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/inventory.json")
    args = parser.parse_args()
    raw = Path(args.raw)
    inv = {
        "zeroth_korean": inventory_zeroth(raw / "zeroth_korean"),
        "fleurs_ko": inventory_fleurs(raw / "fleurs_ko"),
    }
    cv_names = [n for n in ("common_voice_22_ko", "common_voice_17_ko") if (raw / n).exists()]
    for name in cv_names:
        inv[name] = inventory_common_voice(raw / name)
    z, fl, cv = inv["zeroth_korean"], inv["fleurs_ko"], inv[cv_names[0]]
    inv["overlap"] = {
        "zeroth_test_in_zeroth_train": overlap(z["train"]["texts"], z["test"]["texts"]),
        "fleurs_test_in_fleurs_train": overlap(fl["train"]["texts"], fl["test"]["texts"]),
        "fleurs_validation_in_fleurs_train": overlap(fl["train"]["texts"], fl["validation"]["texts"]),
        "cv_corpus": cv_names[0],
        "cv_test_in_cv_train": overlap(cv["train"]["texts"], cv["test"]["texts"]),
        "cv_dev_in_cv_train": overlap(cv["train"]["texts"], cv["dev"]["texts"]),
    }
    train_syl = set(
        ch
        for t in z["train"]["texts"] + fl["train"]["texts"] + cv["train"]["texts"]
        for ch in HANGUL.findall(t)
    )
    inv["syllable_coverage"] = {"train_inventory": len(train_syl)}
    for name, split in (("zeroth_test", z["test"]), ("fleurs_test", fl["test"]), ("cv_test", cv["test"])):
        toks = [ch for t in split["texts"] for ch in HANGUL.findall(t)]
        inv["syllable_coverage"][name] = {
            "n_syllable_tokens": len(toks),
            "oov_tokens": sum(1 for ch in toks if ch not in train_syl),
            "oov_types": len(set(ch for ch in toks if ch not in train_syl)),
        }
    # drop raw text lists from the JSON (keep the file small); they are recomputable
    for corpus in ["zeroth_korean", "fleurs_ko", *cv_names]:
        for split in inv[corpus].values():
            split.pop("texts", None)
    Path(args.out).write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| corpus | split | utts | hours | unique texts | syl inventory | chars/s p50/p95 | sample rates |")
    print("|---|---|---|---|---|---|---|---|")
    for corpus in ["zeroth_korean", "fleurs_ko", *cv_names]:
        for split, s in inv[corpus].items():
            row = [
                corpus,
                split,
                str(s["n_utts"]),
                f"{s.get('hours', 0):.2f}",
                str(s["n_unique_texts"]),
                str(s["syllable_inventory"]),
                f"{s.get('chars_per_sec_p50', 0):.1f}/{s.get('chars_per_sec_p95', 0):.1f}",
                str(s["sample_rates"]),
            ]
            print("| " + " | ".join(row) + " |")
    print(
        json.dumps(
            {"overlap": inv["overlap"], "syllable_coverage": inv["syllable_coverage"]},
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
