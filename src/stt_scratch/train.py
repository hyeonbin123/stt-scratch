"""Training loop: fp16 autocast, fp32 CTC, dynamic chunk training, warmup+cosine, per-epoch dev CER and loss.

Usage (see docs/experiments.md for the pre-registered settings):
  uv run python -m stt_scratch.train --run outputs/stage2/runA --epochs 100 --lr 1e-3 --seed 1

Counters in the logs: `step` = optimizer step attempts (the LR schedule, budgets and "same step count" rule
run on it); `updates` = attempts the GradScaler did not skip; `skipped` = skipped attempts plus non-finite
losses. Divergence (docs/experiments.md "fp16 규칙"): >= 3 non-finite losses, scale < 1 for 100 steps, or the
dev CTC loss after epoch 3 not below the value after epoch 1. A diverged run stops and keeps last.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from stt_scratch.data import DurationBucketSampler, PcmStore, SpeechDataset, collate, read_manifest
from stt_scratch.evaluate import ctc_loss_manifest, evaluate_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import cr_consistency, total_loss
from stt_scratch.model.features import LogMel, speed_perturb
from stt_scratch.model.mask import ChunkSpec, sample_chunk_spec
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[2]
SPEEDS = (0.9, 1.0, 1.1)
COUNTERS = ("nan_count", "low_scale_streak", "skipped", "updates")
# Arguments that define the training recipe. They are stored in every checkpoint and a --resume must match
# them (docs/experiments.md rule change 15); pass --allow-recipe-change to override deliberately.
RECIPE_KEYS = (
    "train",
    "lr",
    "warmup",
    "schedule_steps",
    "epochs",
    "max_duration",
    "seed",
    "d_model",
    "n_layers",
    "ffn_dim",
    "sc_layers",
    "sc_weight",
    "no_sc_feedback",
    "no_bacon",
    "no_qk_norm",
    "offline_only",
    "no_perturb",
    "no_augment",
    "augment_start_epoch",
    "augment_start_step",
    "perturb_start_epoch",
    "dev_loss_rule",
    "dev_loss_set",
    "fp32",
    "init",
    "block",
    "cgmlp_units",
    "cgmlp_kernel",
    "merge_kernel",
    "time_masks",
    "cr_ctc_alpha",
)


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
        sc_feedback=not args.no_sc_feedback,
        bacon=not args.no_bacon,
        qk_norm=not args.no_qk_norm,
        block=args.block,
        cgmlp_units=args.cgmlp_units,
        cgmlp_kernel=args.cgmlp_kernel,
        merge_kernel=args.merge_kernel,
        time_masks=args.time_masks,
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


def rng_states(rng: random.Random, device: torch.device) -> dict:
    states = {"python": rng.getstate(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["torch_cuda"] = torch.cuda.get_rng_state(device)
    return states


def restore_rng(states: dict, rng: random.Random, device: torch.device) -> None:
    version, internal, gauss = states["python"]
    rng.setstate((version, tuple(internal), gauss))
    torch.set_rng_state(states["torch_cpu"].cpu())  # map_location may have moved the state tensors
    if device.type == "cuda" and "torch_cuda" in states:
        torch.cuda.set_rng_state(states["torch_cuda"].cpu(), device)


def unique_path(path: Path) -> Path:
    """path, or path with -2, -3, ... before the suffix when it already exists (never overwrite records)."""
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def save_checkpoint(
    path: Path,
    model,
    opt,
    scaler,
    cfg,
    epoch: int,
    step: int,
    counters: dict,
    dev_loss,
    rng,
    device,
    complete,
    divergence=None,
    recipe=None,
    dev_rule_active=True,
):
    torch.save(
        {
            "recipe": recipe,
            "dev_loss_rule_active": dev_rule_active,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": asdict(cfg),
            "epoch_complete": complete,
            "divergence": divergence,
            "counters": dict(counters),
            "dev_loss": list(dev_loss),
            "rng": rng_states(rng, device),
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--train", default="train")
    p.add_argument("--devs", default="dev_mix,dev_h")
    p.add_argument("--dev-loss-set", default="dev_mix", help="manifest for the dev CTC loss divergence check")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-duration", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help="total steps of the warmup+cosine schedule; 0 = epochs x steps/epoch. Stage 1 runs pass the "
        "Stage 2 total so that the short run is a truncated copy of the long schedule",
    )
    p.add_argument(
        "--dev-loss-rule",
        default="1,3",
        help="A,B (1-based epochs): divergence when the dev CTC loss after epoch B is not below the value "
        "after epoch A (docs/experiments.md fp16 rule, rule change 3/13); skipped with --init",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--sc-layers", default="3,6,9")
    p.add_argument("--sc-weight", type=float, default=0.5)
    p.add_argument(
        "--no-sc-feedback", action="store_true", help="InterCTC only: intermediate losses, no feedback"
    )
    p.add_argument("--no-bacon", action="store_true")
    p.add_argument("--no-qk-norm", action="store_true", help="drop the per-head LayerNorm on q and k")
    p.add_argument("--block", default="conformer", choices=["conformer", "ebranchformer"])
    p.add_argument(
        "--cgmlp-units", type=int, default=1024, help="E-Branchformer cgMLP width (before the split)"
    )
    p.add_argument("--cgmlp-kernel", type=int, default=31)
    p.add_argument("--merge-kernel", type=int, default=31)
    p.add_argument("--time-masks", type=int, default=5, help="SpecAugment time masks per utterance")
    p.add_argument(
        "--cr-ctc-alpha",
        type=float,
        default=0.0,
        help="CR-CTC consistency weight (0 = off); each batch is duplicated into two augmented views",
    )
    p.add_argument(
        "--offline-only", action="store_true", help="control run: full context only, no chunk masks"
    )
    p.add_argument("--no-perturb", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--augment-start-epoch",
        type=int,
        default=3,
        help="SpecAugment is off before this epoch (0-based); the CTC alignment forms first (rule change 13)",
    )
    p.add_argument(
        "--perturb-start-epoch", type=int, default=0, help="speed perturbation is off before this epoch"
    )
    p.add_argument(
        "--augment-start-step",
        type=int,
        default=0,
        help="when > 0, SpecAugment starts at this optimizer step attempt instead of at "
        "--augment-start-epoch (for arms whose batch rule differs: same step count)",
    )
    p.add_argument(
        "--allow-recipe-change",
        action="store_true",
        help="resume even if the saved recipe differs from the current arguments (recorded in meta)",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--eval-chunk", type=int, default=16, help="simulated chunk (frames) evaluated every epoch"
    )
    p.add_argument("--max-steps", type=int, default=0, help="stop early (benchmarks)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default="", help="checkpoint to start from (Stage 4 continuation)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.0,
        help="cap PyTorch's GPU memory pool at this fraction of the card (0 = no cap). The caching allocator "
        "otherwise keeps fragmented blocks (9.4 GB reserved for 4.7 GB used) and Windows pages the excess",
    )
    args = p.parse_args(argv)

    run = ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((run / f).exists() for f in ("train.jsonl", "epochs.jsonl", "last.pt")):
        raise SystemExit(
            f"{run} already holds a run; pass --resume or use a new --run (logs are never mixed)"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.cuda_memory_fraction > 0:
        # not part of the recipe: it only makes the allocator release cached blocks earlier
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.cuda.current_device())
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
    schedule_steps = args.schedule_steps or total_steps
    rule_a, rule_b = (int(x) for x in args.dev_loss_rule.split(","))

    model, cfg = build_model(vocab, args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logmel = LogMel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-6)
    use_amp = device.type == "cuda" and not args.fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bf16_native = torch.cuda.is_bf16_supported(including_emulation=False) if device.type == "cuda" else None
    rng = random.Random(args.seed)
    counters = dict.fromkeys(COUNTERS, 0)
    dev_loss_hist: list[float] = []
    divergence = None

    start_epoch, step = 0, 0
    recipe = {k: getattr(args, k) for k in RECIPE_KEYS}
    dev_rule_active = not args.init  # the from-scratch divergence rule does not apply to continuations
    recipe_diff: dict = {}
    if args.resume and args.init:
        raise SystemExit("--resume restores the run's own weights; do not combine it with --init")
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    if args.resume:
        if not (run / "last.pt").exists():
            raise SystemExit(
                f"--resume needs {run / 'last.pt'}; a run without a completed epoch cannot resume"
            )
        state = torch.load(run / "last.pt", map_location=device, weights_only=True)
        if not state.get("epoch_complete", False):
            raise SystemExit(
                "last.pt was written mid-epoch (divergence or --max-steps); resume needs a completed epoch"
            )
        if state.get("divergence"):
            raise SystemExit(
                f"this run diverged ({state['divergence']}); restart with LR x 0.5 under a new --run"
            )
        saved = state.get("recipe")
        recipe_source = "checkpoint"
        if saved is None and (run / "meta.json").exists():
            # checkpoints written before the recipe guard: compare with the run's original meta.json; options
            # that did not exist yet count as their defaults
            meta_args = json.loads((run / "meta.json").read_text(encoding="utf-8"))["args"]
            saved = {k: meta_args.get(k, p.get_default(k)) for k in RECIPE_KEYS}
            recipe_source = "meta.json"
        if saved is None:
            recipe_diff = {"recipe": ("missing in checkpoint", "cannot verify")}
        else:
            saved = {k: saved.get(k, p.get_default(k)) for k in RECIPE_KEYS}  # keys added later = defaults
            recipe_diff = {k: (saved[k], recipe[k]) for k in RECIPE_KEYS if saved[k] != recipe[k]}
        if recipe_diff and not args.allow_recipe_change:
            raise SystemExit(
                f"recipe differs from the checkpoint (pass --allow-recipe-change to override): {recipe_diff}"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        counters.update(state.get("counters", {}))
        dev_loss_hist = list(state.get("dev_loss", []))
        dev_rule_active = bool(state.get("dev_loss_rule_active", dev_rule_active))
        if "rng" in state:
            restore_rng(state["rng"], rng, device)
        # step records of the interrupted epoch would otherwise appear twice in train.jsonl
        log_path = run / "train.jsonl"
        if log_path.exists():
            kept, dropped = [], []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line and json.loads(line).get("step", -1) >= step:
                    dropped.append(line)
                elif line:
                    kept.append(line)
            if dropped:
                unique_path(run / f"train-discarded-before-resume-{start_epoch:03d}.jsonl").write_text(
                    "\n".join(dropped) + "\n", encoding="utf-8"
                )
                log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    meta = {
        "args": vars(args),
        "config": asdict(cfg),
        "n_params": n_params,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "schedule_steps": schedule_steps,
        "dev_loss_rule": [rule_a, rule_b],
        "step_definition": "optimizer step attempts; `updates` counts the non-skipped ones",
        "dev_loss_rule_text": f"divergence when dev CTC loss after epoch {rule_b} is not below the value "
        f"after epoch {rule_a} (skipped with --init)",
        "git": git_commit(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bf16_native": bf16_native,
        "amp_fp16": use_amp,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    meta["augment_start_step_effective"] = (
        args.augment_start_step or args.augment_start_epoch * steps_per_epoch
    )
    meta["dev_loss_rule_active"] = dev_rule_active
    if args.resume:
        meta["resumed_from_epoch"] = start_epoch - 1
        meta["recipe_diff"] = {k: list(v) for k, v in recipe_diff.items()}
        meta["recipe_source"] = recipe_source
        meta_path = unique_path(run / f"meta-resume-{start_epoch:03d}.json")  # meta.json is kept
    else:
        meta_path = run / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log = open(run / "train.jsonl", "a", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        perturb = not args.no_perturb and epoch >= args.perturb_start_epoch
        augment = False
        t_epoch = time.perf_counter()
        wait, compute = 0.0, 0.0
        t_wait = time.perf_counter()
        for batch in loader:
            wait += time.perf_counter() - t_wait
            t_step = time.perf_counter()
            mel, frames, targets, target_lengths = prepare_batch(batch, logmel, device, rng, perturb)
            if args.augment_start_step > 0:
                augment = not args.no_augment and step >= args.augment_start_step
            else:
                augment = not args.no_augment and epoch >= args.augment_start_epoch
            spec = (
                ChunkSpec(0, -1)
                if args.offline_only
                else sample_chunk_spec(cfg.full_context_prob, cfg.chunk_sizes, cfg.left_contexts, rng)
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(step, schedule_steps, args.warmup, args.lr)
            if args.cr_ctc_alpha > 0:  # two views of the same batch; SpecAugment masks are drawn per row
                mel, frames = torch.cat([mel, mel]), torch.cat([frames, frames])
                targets, target_lengths = (
                    torch.cat([targets, targets]),
                    torch.cat([target_lengths, target_lengths]),
                )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, inter, out_lengths = model(mel, frames, spec, augment=augment)
                loss, parts = total_loss(logits, inter, out_lengths, targets, target_lengths, cfg.sc_weight)
                if args.cr_ctc_alpha > 0:
                    consistency = cr_consistency(logits, out_lengths)
                    loss = loss + args.cr_ctc_alpha * consistency
                    parts["cr"] = float(consistency.detach())
                    parts["loss"] = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                counters["nan_count"] += 1
                counters["skipped"] += 1
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:  # the scaler found inf/nan grads and skipped the step
                    counters["skipped"] += 1
                else:
                    counters["updates"] += 1
                parts["grad_norm"] = grad_norm
            scale = scaler.get_scale() if use_amp else 1.0
            counters["low_scale_streak"] = (
                counters["low_scale_streak"] + 1 if (use_amp and scale < 1.0) else 0
            )
            if counters["nan_count"] >= 3 or counters["low_scale_streak"] >= 100:
                divergence = {"reason": "nan_or_scale", "epoch": epoch, "step": step, **counters}
            compute += time.perf_counter() - t_step
            if step % args.log_every == 0:
                rec = {
                    "epoch": epoch,
                    "step": step,
                    "lr": opt.param_groups[0]["lr"],
                    "scale": scale,
                    "skipped": counters["skipped"],
                    "updates": counters["updates"],
                    "nan": counters["nan_count"],
                    "chunk": spec.chunk_size,
                    "left": spec.left_context,
                    "batch": len(batch["ids"]),
                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
                    # allocator health: reserved pool and how often it had to free cached blocks and retry
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else None,
                    "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0)
                    if device.type == "cuda"
                    else None,
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
        base = {
            "epoch": epoch,
            "step": step,
            "augment": augment,
            "perturb": perturb,
            "epoch_time_s": epoch_time,
            "wait_s": wait,
            "compute_s": compute,
            "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            **counters,
        }
        if divergence or (args.max_steps and step >= args.max_steps):
            summary = {**base, "divergence": divergence, "stopped_early": True}
            (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            save_checkpoint(
                run / "last.pt",
                model,
                opt,
                scaler,
                cfg,
                epoch,
                step,
                counters,
                dev_loss_hist,
                rng,
                device,
                False,
                divergence,
                recipe,
                dev_rule_active,
            )
            break
        model.eval()
        dev_results = {}
        for dev in args.devs.split(","):
            dev_results[dev] = {
                "full": evaluate_manifest(
                    model, vocab, dev, ChunkSpec(0, -1), device, logmel=logmel, amp=use_amp
                )["cer_ns"],
                f"chunk{args.eval_chunk}": evaluate_manifest(
                    model, vocab, dev, ChunkSpec(args.eval_chunk, 128), device, logmel=logmel, amp=use_amp
                )["cer_ns"],
            }
        dev_loss = ctc_loss_manifest(model, vocab, args.dev_loss_set, device, logmel=logmel, amp=use_amp)
        if device.type == "cuda":
            torch.cuda.empty_cache()  # evaluation batches have other shapes; drop their cached blocks
        dev_loss_hist.append(dev_loss)
        not_decreasing = len(dev_loss_hist) >= rule_b and not (
            math.isfinite(dev_loss_hist[rule_a - 1])
            and math.isfinite(dev_loss_hist[rule_b - 1])
            and dev_loss_hist[rule_b - 1] < dev_loss_hist[rule_a - 1]  # non-finite = not decreasing
        )
        if dev_rule_active and epoch == rule_b - 1 and not_decreasing:
            divergence = {
                "reason": "dev_loss_not_decreasing",
                "epoch": epoch,
                "step": step,
                "rule": [rule_a, rule_b],
                "dev_loss": dev_loss_hist[:rule_b],
            }
        summary = {**base, "dev": dev_results, "dev_loss": dev_loss, "dev_loss_set": args.dev_loss_set}
        if divergence:
            summary["divergence"] = divergence
        (run / "epochs.jsonl").open("a", encoding="utf-8").write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        for path in (run / f"epoch-{epoch:03d}.pt", run / "last.pt"):
            save_checkpoint(
                path,
                model,
                opt,
                scaler,
                cfg,
                epoch,
                step,
                counters,
                dev_loss_hist,
                rng,
                device,
                True,
                divergence,
                recipe,
                dev_rule_active,
            )
        if divergence:
            break
    log.close()


if __name__ == "__main__":
    main(sys.argv[1:])
