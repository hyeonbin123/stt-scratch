"""Peak-First lambda for Stage 4 (docs/experiments.md rule change 18): with the continuation's starting model,
lambda = 0.1 * mean CTC loss / mean Peak-First term over 20 seeded training batches under the training
conditions (train mode: dropout, SpecAugment, speed perturbation, sampled chunk masks), so that the
regularizer is about 10 % of the CTC loss the continuation sees. Writes reports/stage4/peak_first_lambda.json.

Usage: uv run python scripts/calibrate_peak_first.py --checkpoint outputs/stage2/runA/avg10.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.evaluate import autocast, load_model
from stt_scratch.model.ctc import ctc_loss, peak_first_regularizer
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import sample_chunk_spec
from stt_scratch.text import Vocab
from stt_scratch.train import prepare_batch

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--target-ratio", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="reports/stage4/peak_first_lambda.json")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model, _ = load_model(ROOT / args.checkpoint, device)
    model.train()  # dropout on, as in the continuation
    cfg = model.cfg
    torch.manual_seed(0)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    utts = read_manifest(ROOT / "manifests" / "train.jsonl")
    dataset = SpeechDataset(utts, PcmStore(ROOT / "data" / "pcm", "train"), vocab)
    sampler = DurationBucketSampler(utts, args.max_duration, 0)
    sampler.set_epoch(0)
    logmel = LogMel().to(device)
    rng = random.Random(0)
    ctc_sum, pfr_sum = 0.0, 0.0
    with torch.no_grad():
        for n, idx in enumerate(sampler):
            if n >= args.batches:
                break
            batch = collate([dataset[i] for i in idx])
            mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, perturb=True)
            spec = sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
            with autocast(device, device.type == "cuda"):
                logits, _, lengths = model(mel, frames, spec, augment=True)
            b = logits.shape[0]
            ctc_sum += float(ctc_loss(logits, lengths, targets, target_lengths)) / b
            pfr_sum += float(peak_first_regularizer(logits, lengths))
    ctc_mean, pfr_mean = ctc_sum / args.batches, pfr_sum / args.batches
    lam = args.target_ratio * ctc_mean / pfr_mean
    result = {
        "checkpoint": args.checkpoint,
        "batches": args.batches,
        "max_duration": args.max_duration,
        "sampler_seed": 0,
        "conditions": "train mode, SpecAugment, speed perturbation, sampled chunk masks",
        "mean_ctc_per_utt": ctc_mean,
        "mean_peak_first_per_utt": pfr_mean,
        "target_ratio": args.target_ratio,
        "lambda": lam,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
