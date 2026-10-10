"""Training loop: fp16 autocast, fp32 CTC, dynamic chunk training, warmup+cosine, per-epoch dev CER.

Usage (see docs/experiments.md for the pre-registered settings):
  uv run python -m stt_scratch.train --run outputs/stage2/runA --epochs 100 --lr 1e-3 --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.evaluate import evaluate_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import total_loss
from stt_scratch.model.features import LogMel, speed_perturb
from stt_scratch.model.mask import ChunkSpec, sample_chunk_spec
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[2]
SPEEDS = (0.9, 1.0, 1.1)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_model(vocab: Vocab, args: argparse.Namespace) -> tuple[ChunkConformerCTC, ModelConfig]:
    stats = json.loads((ROOT / "manifests" / "mel_stats.json").read_text(encoding="utf-8"))
    cfg = ModelConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        sc_layers=tuple(int(x) for x in args.sc_layers.split(",")) if args.sc_layers else (),
        sc_weight=args.sc_weight,
        bacon=not args.no_bacon,
        full_context_prob=0.0 if args.offline_only else 0.5,
        mel_mean=stats["mean"],
        mel_std=stats["std"],
    )
    return ChunkConformerCTC(cfg), cfg


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def prepare_batch(batch: dict, logmel: LogMel, device: torch.device, rng: random.Random, perturb: bool):
    wavs = []
    for wav in batch["wavs"]:
        wav = wav.to(device, non_blocking=True)
        if perturb:
            wav = speed_perturb(wav, rng.choice(SPEEDS))
        wavs.append(wav)
    lengths = torch.tensor([w.numel() for w in wavs], device=device)
    padded = torch.zeros(len(wavs), int(lengths.max()), device=device)
    for i, w in enumerate(wavs):
        padded[i, : w.numel()] = w
    mel, frames = logmel(padded, lengths)
    tokens = batch["tokens"]
    targets = torch.tensor([t for seq in tokens for t in seq], device=device, dtype=torch.int32)
    target_lengths = torch.tensor([len(seq) for seq in tokens], device=device, dtype=torch.int32)
    return mel, frames, targets, target_lengths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--train", default="train")
    p.add_argument("--devs", default="dev_mix,dev_h")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--sc-layers", default="3,6,9")
    p.add_argument("--sc-weight", type=float, default=0.5)
    p.add_argument("--no-bacon", action="store_true")
    p.add_argument(
        "--offline-only", action="store_true", help="control run: full context only, no chunk masks"
    )
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--eval-chunk", type=int, default=16, help="simulated chunk (frames) evaluated every epoch"
    )
    p.add_argument("--max-steps", type=int, default=0, help="stop early (benchmarks)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default="", help="checkpoint to start from (Stage 4 continuation)")
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args(argv)

    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = (
        False  # shapes change every batch; autotuning per shape dominated step time
    )

    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    train_utts = read_manifest(ROOT / "manifests" / f"{args.train}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", args.train)
    dataset = SpeechDataset(train_utts, store, vocab)
    sampler = DurationBucketSampler(train_utts, args.max_duration, args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs

    model, cfg = build_model(vocab, args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logmel = LogMel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-6)
    use_amp = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bf16_native = torch.cuda.is_bf16_supported(including_emulation=False) if device.type == "cuda" else None

    start_epoch, step = 0, 0
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    if args.resume and (run / "last.pt").exists():
        state = torch.load(run / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step = state["epoch"] + 1, state["step"]

    meta = {
        "args": vars(args),
        "config": asdict(cfg),
        "n_params": n_params,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "git": git_commit(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bf16_native": bf16_native,
        "amp_fp16": use_amp,
    }
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log = open(run / "train.jsonl", "a", encoding="utf-8")
    rng = random.Random(args.seed)
    nan_count, low_scale_streak, skipped = 0, 0, 0
    divergence = None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        t_epoch = time.perf_counter()
        wait, compute = 0.0, 0.0
        t_wait = time.perf_counter()
        for batch in loader:
            wait += time.perf_counter() - t_wait
            t_step = time.perf_counter()
            mel, frames, targets, target_lengths = prepare_batch(
                batch, logmel, device, rng, not args.no_perturb
            )
            spec = (
                ChunkSpec(0, -1)
                if args.offline_only
                else sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.warmup, args.lr)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, inter, out_lengths = model(mel, frames, spec, augment=not args.no_augment)
                loss, parts = total_loss(logits, inter, out_lengths, targets, target_lengths, cfg.sc_weight)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nan_count += 1
                skipped += 1
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped += 1
                parts["grad_norm"] = grad_norm
            scale = scaler.get_scale() if use_amp else 1.0
            low_scale_streak = low_scale_streak + 1 if (use_amp and scale < 1.0) else 0
            if nan_count >= 3 or low_scale_streak >= 100:
                divergence = {
                    "epoch": epoch,
                    "step": step,
                    "nan_count": nan_count,
                    "low_scale_streak": low_scale_streak,
                }
            compute += time.perf_counter() - t_step
            if step % args.log_every == 0:
                rec = {
                    "epoch": epoch,
                    "step": step,
                    "lr": opt.param_groups[0]["lr"],
                    "scale": scale,
                    "skipped": skipped,
                    "nan": nan_count,
                    "chunk": spec.chunk_size,
                    "left": spec.left_context,
                    "batch": len(batch["ids"]),
                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                    **parts,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            step += 1
            if divergence or (args.max_steps and step >= args.max_steps):
                break
            t_wait = time.perf_counter()
        epoch_time = time.perf_counter() - t_epoch
        if divergence or (args.max_steps and step >= args.max_steps):
            summary = {
                "epoch": epoch,
                "step": step,
                "epoch_time_s": epoch_time,
                "wait_s": wait,
                "compute_s": compute,
                "divergence": divergence,
                "stopped_early": True,
            }
            (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "config": asdict(cfg),
                },
                run / "last.pt",
            )
            break
        model.eval()
        dev_results = {}
        for dev in args.devs.split(","):
            dev_results[dev] = {
                "full": evaluate_manifest(model, vocab, dev, ChunkSpec(0, -1), device, logmel=logmel)[
                    "cer_ns"
                ],
                f"chunk{args.eval_chunk}": evaluate_manifest(
                    model, vocab, dev, ChunkSpec(args.eval_chunk, 128), device, logmel=logmel
                )["cer_ns"],
            }
        summary = {
            "epoch": epoch,
            "step": step,
            "epoch_time_s": epoch_time,
            "wait_s": wait,
            "compute_s": compute,
            "skipped": skipped,
            "nan": nan_count,
            "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            "dev": dev_results,
        }
        (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": asdict(cfg),
        }
        torch.save(state, run / f"epoch-{epoch:03d}.pt")
        torch.save(state, run / "last.pt")
    log.close()


if __name__ == "__main__":
    main(sys.argv[1:])
