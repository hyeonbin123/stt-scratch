"""Stage 0 micro-benchmark (docs/experiments.md H0.2): peak memory, step time and data-wait share per
max-duration, plus an epoch-time estimate and the SDPA kernel used on this GPU.

Usage: uv run python scripts/bench_stage0.py --durations 200,300,400,600 --steps 40
Writes reports/stage0/bench.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import total_loss
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import sample_chunk_spec
from stt_scratch.text import Vocab
from stt_scratch.train import prepare_batch

ROOT = Path(__file__).resolve().parents[1]


def sdpa_backend_probe(device: torch.device) -> dict:
    """Which SDPA kernel runs for our shapes (fp16, bool mask) on this GPU."""
    q = torch.randn(8, 4, 256, 64, device=device, dtype=torch.float16)
    mask = torch.ones(8, 1, 256, 256, dtype=torch.bool, device=device)
    out = {}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask)
            out[name] = True
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--durations", default="200,300,400,600")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda")
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    stats = json.loads((ROOT / "manifests" / "mel_stats.json").read_text(encoding="utf-8"))
    utts = read_manifest(ROOT / "manifests" / "train.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", "train")
    dataset = SpeechDataset(utts, store, vocab)
    logmel = LogMel().to(device)
    total_hours = sum(u.duration_s for u in utts) / 3600
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "n_train_utts": len(utts),
        "train_hours": total_hours,
        "sdpa": sdpa_backend_probe(device),
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        "runs": [],
    }
    torch.backends.cudnn.benchmark = True
    for md in (float(x) for x in args.durations.split(",")):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=len(vocab), mel_mean=stats["mean"], mel_std=stats["std"])
        model = ChunkConformerCTC(cfg).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
        scaler = torch.amp.GradScaler("cuda")
        sampler = DurationBucketSampler(utts, md, seed=0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0
        )
        rng = random.Random(0)
        torch.cuda.reset_peak_memory_stats()
        times, waits, batch_sizes = [], [], []
        t_wait = time.perf_counter()
        ok = True
        try:
            for i, batch in enumerate(loader):
                wait = time.perf_counter() - t_wait
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, True)
                spec = sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, inter, out_lengths = model(mel, frames, spec, augment=True)
                    loss, _ = total_loss(logits, inter, out_lengths, targets, target_lengths, cfg.sc_weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize()
                if i >= args.warmup:
                    times.append(time.perf_counter() - t0)
                    waits.append(wait)
                    batch_sizes.append(len(batch["ids"]))
                if i + 1 >= args.steps + args.warmup:
                    break
                t_wait = time.perf_counter()
        except torch.cuda.OutOfMemoryError:
            ok = False
        peak = torch.cuda.max_memory_allocated() / 1e9
        steps_per_epoch = len(sampler)
        step_mean = sum(times) / len(times) if times else None
        wait_mean = sum(waits) / len(waits) if waits else None
        run = {
            "max_duration_s": md,
            "oom": not ok,
            "peak_mem_gb": peak,
            "steps_per_epoch": steps_per_epoch,
            "step_time_s": step_mean,
            "wait_time_s": wait_mean,
            "wait_share": (wait_mean / (wait_mean + step_mean)) if times else None,
            "batch_size_mean": sum(batch_sizes) / len(batch_sizes) if batch_sizes else None,
            "epoch_time_min_est": (steps_per_epoch * (step_mean + wait_mean) / 60) if times else None,
        }
        results["runs"].append(run)
        print(json.dumps(run), flush=True)
        del model, opt, scaler, loader
        torch.cuda.empty_cache()
    out = ROOT / "reports" / "stage0" / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(
        "| max-duration (s) | OOM | peak GB | steps/epoch | step s | wait share | batch | epoch min (est) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results["runs"]:
        print(
            f"| {r['max_duration_s']:.0f} | {r['oom']} | {r['peak_mem_gb']:.2f} | {r['steps_per_epoch']} | "
            f"{r['step_time_s'] or 0:.3f} | {(r['wait_share'] or 0) * 100:.1f} % | "
            f"{r['batch_size_mean'] or 0:.1f} | "
            f"{r['epoch_time_min_est'] or 0:.1f} |"
        )
    print("sdpa:", results["sdpa"], "bf16 native:", results["bf16_native"])


if __name__ == "__main__":
    main()
