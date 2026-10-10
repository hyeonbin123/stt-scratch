"""Global log-mel mean/std over the training manifest (no speed perturbation) -> manifests/mel_stats.json.

Usage: uv run python scripts/compute_mel_stats.py [--manifest train]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.features import LogMel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="train")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utts = read_manifest(ROOT / "manifests" / f"{args.manifest}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", args.manifest)
    logmel = LogMel().to(device)
    total = torch.zeros(80, dtype=torch.float64, device=device)
    total_sq = torch.zeros(80, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for i, u in enumerate(utts):
            wav = store.waveform(u.id).to(device)
            mel, frames = logmel(wav[None], torch.tensor([wav.numel()], device=device))
            m = mel[0, : int(frames[0])].double()
            total += m.sum(0)
            total_sq += (m * m).sum(0)
            count += m.shape[0]
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(utts)} utterances, {count} frames", flush=True)
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(1e-8).sqrt()
    out = {
        "manifest": args.manifest,
        "n_utts": len(utts),
        "n_frames": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (ROOT / "manifests" / "mel_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("mean range", float(mean.min()), float(mean.max()), "std range", float(std.min()), float(std.max()))


if __name__ == "__main__":
    main()
