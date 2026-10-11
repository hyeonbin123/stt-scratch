"""Stage 2 measurements for one checkpoint (docs/experiments.md 2단계, 규칙 변경 11).

For each dev set: full-context greedy CER_ns/CER_ws (all + Hangul-only, bootstrap intervals), simulated
chunk masks {4, 8, 16, 32} frames x left 128, the intermediate self-conditioned heads (full context), and an
fp16 cache-streaming check against the offline chunk mask (max |dlog p|, max |dp|, frame argmax agreement).

Usage:
  uv run python scripts/stage2_report.py --checkpoint outputs/stage2/runA/avg10.pt --name runA-avg10
Writes reports/stage2/<name>.json and <name>.md. --limit N evaluates only the first N utterances (smoke).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, batch_features, decode_offline, duration_batches, load_model, score
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
FRAME_MS = 40


def rate(scored: dict) -> float:
    return scored["hangul_only"]["cer_ns"]["rate"]


@torch.no_grad()
def decode_heads(model, vocab, utts, store, device, logmel, amp: bool) -> dict[int, dict[str, str]]:
    """Greedy hypotheses of every intermediate head (full context); keys are the 1-based layer indices."""
    layers = sorted(model.sc_layers)
    hyps: dict[int, dict[str, str]] = {layer: {} for layer in layers}
    for batch in duration_batches(utts, 400.0):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, amp):
            _, inter, out_lengths = model(mel, frames, ChunkSpec(0, -1))
        for layer, logits in zip(layers, inter, strict=True):
            for j, ids in zip(batch, greedy_batch(logits.float(), out_lengths), strict=True):
                hyps[layer][utts[j].id] = vocab.decode(ids)
    return hyps


@torch.no_grad()
def streaming_check(model, utts, store, device, logmel, spec: ChunkSpec, amp: bool) -> dict:
    """fp16 cache streaming vs offline chunk mask on whole utterances: log-prob/prob gaps, argmax match."""
    enc = StreamingEncoder(model, spec)
    max_dlp, max_dp, agree, frames = 0.0, 0.0, 0, 0
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, n = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        with autocast(device, amp):
            offline = log_probs_fp32(model(mel, n, spec)[0])[0]
            state = enc.new_state()
            parts = []
            step = 4 * spec.chunk_size
            for i in range(0, mel.shape[1], step):
                parts.append(enc.step(mel[0, i : i + step], state))
            streamed = torch.cat(parts, dim=0)
        t = min(offline.shape[0], streamed.shape[0])
        offline, streamed = offline[:t], streamed[:t]
        max_dlp = max(max_dlp, float((offline - streamed).abs().max()))
        max_dp = max(max_dp, float((offline.exp() - streamed.exp()).abs().max()))
        agree += int((offline.argmax(-1) == streamed.argmax(-1)).sum())
        frames += t
    return {
        "n_utterances": len(utts),
        "frames": frames,
        "max_abs_dlogprob": max_dlp,
        "max_abs_dprob": max_dp,
        "argmax_agreement": agree / max(1, frames),
    }


@torch.no_grad()
def stability_stats(model, utts, store, device, logmel, amp: bool) -> dict:
    """QK-norm affine magnitudes and per-block attention-logit std on one dev batch (rule change 13 note:
    the LayerNorm bounds the logits only while its affine parameters stay near 1/0)."""
    blocks = model.blocks
    gains = [float(b.attn.q_norm.weight.abs().max()) for b in blocks if b.attn.q_norm is not None]
    kgains = [float(b.attn.k_norm.weight.abs().max()) for b in blocks if b.attn.k_norm is not None]
    batch = list(range(min(8, len(utts))))
    mel, frames = batch_features(utts, batch, store, device, logmel)
    stds = []
    with autocast(device, amp):
        x = model.normalize_mel(mel)
        x, lengths = model.subsampling(x, frames)
        t = x.shape[1]
        mask = torch.ones(x.shape[0], 1, t, t, dtype=torch.bool, device=device)
        pos = torch.arange(t, device=device)
        valid = pos[None, :] < lengths[:, None]
        for blk in blocks:
            h = blk.attn_norm(x + 0.5 * blk.ffn1(x))
            d = h.shape[-1]
            qkv = blk.attn.qkv(h)
            hd = blk.attn.head_dim
            q = qkv[..., :d].view(h.shape[0], t, -1, hd).transpose(1, 2)
            k = qkv[..., d : 2 * d].view(h.shape[0], t, -1, hd).transpose(1, 2)
            if blk.attn.q_norm is not None:
                q, k = blk.attn.q_norm(q), blk.attn.k_norm(k)
            stds.append(float(((q.float() @ k.float().transpose(-1, -2)) / hd**0.5).std()))
            x = blk(x, mask, pos, 0, valid)
    return {
        "qk_norm_max_gain_q": gains,
        "qk_norm_max_gain_k": kgains,
        "attention_logit_std_per_block": stds,
    }


def save_hyps(path: Path, utts, hyps: dict[str, str]) -> None:
    """One line per utterance: id, reference, hypothesis, hangul_only flag (input of compare_runs.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for u in utts:
            row = {"id": u.id, "ref": u.text, "hyp": hyps[u.id], "hangul_only": u.hangul_only}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True, help="report file stem under reports/stage2/")
    p.add_argument("--devs", default="dev_mix,dev_h")
    p.add_argument("--chunks", default="4,8,16,32")
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--streaming-check", type=int, default=200, help="dev_h utterances for the fp16 check")
    p.add_argument("--limit", type=int, default=0, help="first N utterances per set (smoke test)")
    p.add_argument(
        "--train-sample",
        type=int,
        default=1000,
        help="train utterances (seeded sample, full context, no augmentation) for the overfitting check",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    amp = not args.fp32 and device.type == "cuda"
    model, state = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    t0 = time.perf_counter()
    report: dict = {
        "checkpoint": args.checkpoint,
        "averaged_epochs": state.get("averaged"),
        "precision": "fp16-autocast" if amp else "fp32",
        "left_context": args.left,
        "sets": {},
    }
    for name in args.devs.split(","):
        utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
        if args.limit:
            utts = utts[: args.limit]
        store = PcmStore(ROOT / "data" / "pcm", name)
        entry: dict = {"n": len(utts), "modes": {}}
        hyp_dir = ROOT / "reports" / "stage2" / "hyps"
        hyps = decode_offline(model, vocab, utts, store, ChunkSpec(0, -1), device, logmel, amp=amp)
        save_hyps(hyp_dir / f"{args.name}-{name}-full.jsonl", utts, hyps)
        full = score(utts, hyps)
        entry["modes"]["full"] = full
        for chunk in (int(c) for c in args.chunks.split(",")):
            spec = ChunkSpec(chunk, args.left)
            hyps = decode_offline(model, vocab, utts, store, spec, device, logmel, amp=amp)
            mode = f"chunk{chunk}_{chunk * FRAME_MS}ms"
            save_hyps(hyp_dir / f"{args.name}-{name}-{mode}.jsonl", utts, hyps)
            scored = score(utts, hyps)
            scored["ratio_to_full"] = rate(scored) / rate(full) if rate(full) else None
            entry["modes"][mode] = scored
        heads = decode_heads(model, vocab, utts, store, device, logmel, amp)
        entry["heads"] = {f"layer{layer}": score(utts, hyps) for layer, hyps in heads.items()}
        report["sets"][name] = entry
        print(f"{name}: full {rate(full) * 100:.2f} %", flush=True)
    if args.train_sample:
        train = read_manifest(ROOT / "manifests" / "train.jsonl")
        picked = sorted(random.Random(0).sample(range(len(train)), min(args.train_sample, len(train))))
        utts = [train[i] for i in picked]
        store = PcmStore(ROOT / "data" / "pcm", "train")
        scored = score(
            utts, decode_offline(model, vocab, utts, store, ChunkSpec(0, -1), device, logmel, amp=amp)
        )
        report["train_sample"] = {"n": len(utts), "seed": 0, "full": scored}
        dev_h = report["sets"].get("dev_h", {}).get("modes", {}).get("full")
        if dev_h:
            report["train_sample"]["ratio_to_dev_h"] = rate(scored) / rate(dev_h) if rate(dev_h) else None
    if args.streaming_check:
        utts = read_manifest(ROOT / "manifests" / "dev_h.jsonl")[: args.streaming_check]
        store = PcmStore(ROOT / "data" / "pcm", "dev_h")
        report["streaming_check"] = {
            "spec": "chunk8 left128",
            **streaming_check(model, utts, store, device, logmel, ChunkSpec(8, args.left), amp),
        }
        report["stability"] = stability_stats(model, utts, store, device, logmel, amp)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out_dir = ROOT / "reports" / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lines = [f"# {args.name} ({report['precision']}, left {args.left}, checkpoint {args.checkpoint})", ""]
    lines.append(
        "| set | mode | CER_ns hangul-only % | 95 % interval | CER_ns all % | CER_ws hangul-only % | ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, entry in report["sets"].items():
        for mode, s in entry["modes"].items():
            h = s["hangul_only"]["cer_ns"]
            ratio = s.get("ratio_to_full")
            lines.append(
                f"| {name} | {mode} | {h['rate'] * 100:.2f} | "
                f"{h['ci'][0] * 100:.2f}-{h['ci'][1] * 100:.2f} | {s['all']['cer_ns']['rate'] * 100:.2f} | "
                f"{s['hangul_only']['cer_ws']['rate'] * 100:.2f} | "
                f"{'' if ratio is None else f'{ratio:.3f}'} |"
            )
        for head, s in entry["heads"].items():
            lines.append(
                f"| {name} | head {head} (full) | {rate(s) * 100:.2f} | | "
                f"{s['all']['cer_ns']['rate'] * 100:.2f} | | |"
            )
    if "train_sample" in report:
        ts = report["train_sample"]
        ratio = ts.get("ratio_to_dev_h")
        lines += [
            "",
            f"train sample ({ts['n']} utts, seed 0, full context, no augmentation): CER_ns "
            f"{rate(ts['full']) * 100:.2f} %"
            + ("" if ratio is None else f", ratio to dev_h full {ratio:.3f} (overfitting flag if < 0.5)"),
        ]
    if "streaming_check" in report:
        c = report["streaming_check"]
        lines += [
            "",
            f"fp16 streaming check ({c['spec']}, {c['n_utterances']} dev_h utts, {c['frames']} frames): "
            f"max |dlog p| {c['max_abs_dlogprob']:.2e}, max |dp| {c['max_abs_dprob']:.2e}, "
            f"argmax agreement {c['argmax_agreement'] * 100:.3f} %",
        ]
    if "stability" in report:
        s = report["stability"]
        lines += [
            "",
            "stability: attention logit std per block "
            f"{[round(v, 1) for v in s['attention_logit_std_per_block']]}, "
            f"QK-norm max |gain| q {[round(v, 2) for v in s['qk_norm_max_gain_q']]}",
        ]
    (out_dir / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
