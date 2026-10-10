"""Decode a manifest with a model (full context, simulated chunk mask, or real cache streaming) and score it.

Reports CER_ns (whisper-ko-ft definition) and CER_ws on all utterances and on the Hangul-only subset,
with utterance-level bootstrap intervals, plus the checkpoint and decoding mode (docs/experiments.md "지표").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.metrics import score_ns, score_ws, summarize
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import ctc_loss
from stt_scratch.model.decode import GreedyState, greedy_batch
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel, StreamingFbank
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[2]


def load_model(path: Path, device: torch.device) -> tuple[ChunkConformerCTC, dict]:
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig(**state["config"])
    model = ChunkConformerCTC(cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def autocast(device, amp: bool):
    """The precision used for every decoding/loss path: fp16 autocast on CUDA when amp (training default)."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")


def duration_batches(utts, batch_seconds: float):
    """Indices sorted by duration, grouped so that padded seconds (longest * count) stay under the budget."""
    order = sorted(range(len(utts)), key=lambda i: utts[i].duration_s)
    i = 0
    while i < len(order):
        batch, longest = [], 0.0
        while i < len(order):
            d = utts[order[i]].duration_s
            if batch and max(longest, d) * (len(batch) + 1) > batch_seconds:
                break
            batch.append(order[i])
            longest = max(longest, d)
            i += 1
        yield batch


def batch_features(utts, batch, store, device, logmel):
    wavs = [store.waveform(utts[j].id).to(device) for j in batch]
    lengths = torch.tensor([w.numel() for w in wavs], device=device)
    padded = torch.zeros(len(wavs), int(lengths.max()), device=device)
    for k, w in enumerate(wavs):
        padded[k, : w.numel()] = w
    return logmel(padded, lengths)


@torch.no_grad()
def decode_offline(
    model, vocab, utts, store, spec: ChunkSpec, device, logmel, batch_seconds: float = 400.0, amp: bool = True
):
    hyps: dict[str, str] = {}
    for batch in duration_batches(utts, batch_seconds):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, amp):
            logits, _, out_lengths = model(mel, frames, spec)
        for j, ids in zip(batch, greedy_batch(logits.float(), out_lengths), strict=True):
            hyps[utts[j].id] = vocab.decode(ids)
    return hyps


@torch.no_grad()
def decode_streaming(model, vocab, utts, store, spec: ChunkSpec, device, logmel, amp: bool = True):
    """Real cache streaming: audio fed in packets of 4 * chunk_size * 160 samples (320 ms for chunk 8); the
    first packet carries FIRST_PACKET_EXTRA more samples so that every packet completes one encoder chunk.
    Runs under the same autocast as the offline paths so that differences come from the cache, not dtype."""
    enc = StreamingEncoder(model, spec)
    samples_per_chunk = spec.chunk_size * 4 * 160
    hyps: dict[str, str] = {}
    with autocast(device, amp):
        for u in utts:
            wav = store.waveform(u.id).to(device)
            fb = StreamingFbank(logmel)
            state = enc.new_state()
            greedy = GreedyState()
            pending = torch.zeros(0, 80, device=device)
            n_frames = 0
            pos, packet = 0, samples_per_chunk + FIRST_PACKET_EXTRA
            while pos < wav.numel():
                piece = wav[pos : pos + packet]
                pos += packet
                packet = samples_per_chunk
                pending = torch.cat([pending, fb.push(piece)], dim=0)
                while pending.shape[0] >= 4 * spec.chunk_size:
                    mel_chunk, pending = pending[: 4 * spec.chunk_size], pending[4 * spec.chunk_size :]
                    lp = enc.step(mel_chunk, state)
                    greedy.push(lp, n_frames)
                    n_frames += lp.shape[0]
            if pending.shape[0] > 0:  # partial last chunk: the causal front-end handles any length exactly
                lp = enc.step(pending, state)
                greedy.push(lp, n_frames)
            hyps[u.id] = vocab.decode(greedy.tokens)
    return hyps


@torch.no_grad()
def ctc_loss_manifest(
    model, vocab, name: str, device, logmel=None, amp: bool = True, batch_seconds: float = 400.0
) -> float:
    """Mean per-utterance CTC loss of the final head with full context (the dev-loss divergence check)."""
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", name)
    logmel = logmel or LogMel().to(device)
    total = 0.0
    for batch in duration_batches(utts, batch_seconds):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        tokens = [vocab.encode(utts[j].text) for j in batch]
        targets = torch.tensor([t for seq in tokens for t in seq], device=device, dtype=torch.int32)
        target_lengths = torch.tensor([len(seq) for seq in tokens], device=device, dtype=torch.int32)
        with autocast(device, amp):
            logits, _, out_lengths = model(mel, frames, ChunkSpec(0, -1))
            total += float(ctc_loss(logits, out_lengths, targets, target_lengths))
    return total / len(utts)


def score(utts, hyps: dict[str, str]) -> dict:
    out = {}
    for subset in ("all", "hangul_only"):
        rows = [u for u in utts if subset == "all" or u.hangul_only]
        ns = [s for s in (score_ns(u.text, hyps[u.id]) for u in rows) if s is not None]
        ws = [s for s in (score_ws(u.text, hyps[u.id]) for u in rows) if s is not None]
        out[subset] = {"cer_ns": summarize(ns), "cer_ws": summarize(ws)}
    return out


def evaluate_manifest(
    model,
    vocab,
    name: str,
    spec: ChunkSpec,
    device,
    logmel=None,
    streaming: bool = False,
    out_path: Path | None = None,
    checkpoint: str = "",
    amp: bool = True,
) -> dict:
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", name)
    logmel = logmel or LogMel().to(device)
    if streaming:
        hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = f"streaming chunk{spec.chunk_size} left{spec.left_context}"
    elif spec.full:
        hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = "full"
    else:
        hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
        mode = f"simulated chunk{spec.chunk_size} left{spec.left_context}"
    scored = score(utts, hyps)
    result = {
        "manifest": name,
        "mode": mode,
        "precision": "fp16-autocast" if amp and device.type == "cuda" else "fp32",
        "checkpoint": checkpoint,
        "n": len(utts),
        **scored,
        "cer_ns": scored["hangul_only"]["cer_ns"]["rate"],
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(out_path.with_suffix(".hyps.jsonl"), "w", encoding="utf-8") as f:
            for u in utts:
                f.write(json.dumps({"id": u.id, "ref": u.text, "hyp": hyps[u.id]}, ensure_ascii=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--chunk", type=int, default=0, help="frames; 0 = full context")
    p.add_argument("--left", type=int, default=128)
    p.add_argument(
        "--streaming", action="store_true", help="real cache streaming instead of a simulated mask"
    )
    p.add_argument("--out", default="")
    p.add_argument("--fp32", action="store_true", help="no fp16 autocast (for models trained with --fp32)")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    spec = ChunkSpec(args.chunk, args.left if args.chunk else -1)
    result = evaluate_manifest(
        model,
        vocab,
        args.manifest,
        spec,
        device,
        streaming=args.streaming,
        out_path=ROOT / args.out if args.out else None,
        checkpoint=args.checkpoint,
        amp=not args.fp32,
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "all" and k != "hangul_only"}, ensure_ascii=False)
    )
    for subset in ("all", "hangul_only"):
        print(subset, json.dumps(result[subset], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
