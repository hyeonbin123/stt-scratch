"""Export Hangul-only dev utterances as an MFA corpus (Stage 4 reference alignment, experiments.md 4단계).

Writes <out>/<set>/<safe id>.wav (16 kHz int16 mono) and .lab (normalized transcript) plus ids.json
mapping the file stems back to manifest ids. Usage:
  uv run python scripts/export_mfa_corpus.py --sets dev_mix,dev_h --out work/mfa/corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stt_scratch.data import PcmStore, read_manifest

ROOT = Path(__file__).resolve().parents[1]


def safe(utt_id: str) -> str:
    return utt_id.replace("/", "__").replace(".mp3", "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="dev_mix,dev_h")
    p.add_argument("--out", default="work/mfa/corpus")
    args = p.parse_args()
    for name in args.sets.split(","):
        out = ROOT / args.out / name
        out.mkdir(parents=True, exist_ok=True)
        utts = [u for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl") if u.hangul_only]
        store = PcmStore(ROOT / "data" / "pcm", name)
        mapping = {}
        for u in utts:
            offset, length = store.index[u.id]
            pcm = np.asarray(store.bin[offset : offset + length], dtype=np.int16)
            stem = safe(u.id)
            sf.write(out / f"{stem}.wav", pcm, 16000, subtype="PCM_16")
            (out / f"{stem}.lab").write_text(u.text + "\n", encoding="utf-8")
            mapping[stem] = u.id
        (ROOT / args.out / f"{name}.ids.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        print(f"{name}: {len(utts)} Hangul-only utterances -> {out}", flush=True)


if __name__ == "__main__":
    main()
