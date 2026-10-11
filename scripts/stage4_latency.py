"""Stage 4 emission latency (docs/experiments.md 4단계, rule change 18) against MFA Korean 3.0 alignments.

For each checkpoint and simulated chunk size: greedy decoding under the chunk mask (identical to cache
streaming), the emission frame of every token (first frame of its CTC run) and its availability time
    t_emit = min(((4 C (k + 1) - 1) * 160 + 400) / 16000, duration),  k = frame // C
(the audio of the whole chunk must have arrived; compute time excluded). Reference: MFA word intervals
regrouped into the transcript's space-separated words (MFA splits Korean words into morphemes).
Hypothesis words are aligned to reference words by edit distance; for exactly matched words the delay is
t_emit(last syllable of the hypothesis word) - end of the reference word.
  FTD: delay of the first reference word, LTD: of the last one (utterances where that word is matched)
  MAD: mean delay over all matched words of an utterance
Reports P50/P90 over utterances, coverage counts and the CER_ns of the same decoding.

Usage: uv run python scripts/stage4_latency.py --checkpoint outputs/stage2/runA/avg10.pt --name runA-avg10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, batch_features, duration_batches, load_model, score
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
SILENCE = {"", "<eps>", "sil", "sp", "spn", "<unk>"}


def load_alignment(path: Path) -> list[tuple[float, float, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Type", "words") == "words" and r["Label"].strip() not in SILENCE:
                rows.append((float(r["Begin"]), float(r["End"]), r["Label"].strip()))
    return rows


def regroup(words: list[str], pieces: list[tuple[float, float, str]]) -> list[tuple[float, float]] | None:
    """Spans of the transcript words from MFA pieces whose labels concatenate to them; None if they don't."""
    spans, i = [], 0
    for w in words:
        if i >= len(pieces):
            return None
        start, end, acc = pieces[i][0], pieces[i][1], ""
        while i < len(pieces) and len(acc) < len(w):
            acc += pieces[i][2]
            end = pieces[i][1]
            i += 1
        if acc != w:
            return None
        spans.append((start, end))
    return spans if i == len(pieces) else None


def match_words(hyp: list[str], ref: list[str]) -> dict[int, int]:
    """Edit-distance alignment; returns {ref index: hyp index} for exact matches."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = {}, n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def emissions(best: list[int], blank: int = 0) -> list[tuple[int, int]]:
    """(token, frame) for every emitted token: the first frame of each run of a non-blank label."""
    out, prev = [], blank
    for f, tok in enumerate(best):
        if tok != blank and tok != prev:
            out.append((tok, f))
        prev = tok
    return out


def emit_time(frame: int, chunk: int, duration: float) -> float:
    k = frame // chunk
    return min(((4 * chunk * (k + 1) - 1) * 160 + 400) / 16000, duration)


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "mean": None}
    a = np.asarray(xs) * 1000
    return {
        "n": len(xs),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
    }


@torch.no_grad()
def measure(model, vocab, name, utts, store, refs, chunk, left, device, logmel, amp) -> dict:
    spec = ChunkSpec(chunk, left)
    hyps, ftd, ltd, mad = {}, [], [], []
    counts = {"utterances": len(utts), "with_alignment": 0, "first_matched": 0, "last_matched": 0}
    for batch in duration_batches(utts, 400.0):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, amp):
            logits, _, lengths = model(mel, frames, spec)
        best = logits.float().argmax(-1).cpu()
        for row, j in enumerate(batch):
            u = utts[j]
            em = emissions(best[row, : int(lengths[row])].tolist())
            hyps[u.id] = vocab.decode([t for t, _ in em])
            ref_spans = refs.get(u.id)
            if ref_spans is None:
                continue
            counts["with_alignment"] += 1
            # hypothesis words and the emission frame of each word's last syllable
            words, frames_of_words, cur, last = [], [], "", -1
            for tok, f in em:
                if tok == vocab.space_id:
                    if cur:
                        words.append(cur)
                        frames_of_words.append(last)
                    cur = ""
                else:
                    cur += vocab.decode([tok])
                    last = f
            if cur:
                words.append(cur)
                frames_of_words.append(last)
            ref_words = u.text.split()
            matched = match_words(words, ref_words)
            delays = {
                r: emit_time(frames_of_words[h], chunk, u.duration_s) - ref_spans[r][1]
                for r, h in matched.items()
            }
            if 0 in delays:
                counts["first_matched"] += 1
                ftd.append(delays[0])
            if len(ref_words) - 1 in delays:
                counts["last_matched"] += 1
                ltd.append(delays[len(ref_words) - 1])
            if delays:
                mad.append(float(np.mean(list(delays.values()))))
    scored = score(utts, hyps)
    return {
        "set": name,
        "chunk_frames": chunk,
        "chunk_ms": chunk * 40,
        "left": left,
        "cer_ns": scored["hangul_only"]["cer_ns"]["rate"],
        "cer_ci": scored["hangul_only"]["cer_ns"]["ci"],
        "ftd_ms": percentiles(ftd),
        "ltd_ms": percentiles(ltd),
        "mad_ms": percentiles(mad),
        "counts": counts,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--sets", default="dev_mix,dev_h")
    p.add_argument("--chunks", default="8,16")
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--alignments", default="work/mfa")
    p.add_argument("--report-dir", default="reports/stage4")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    amp = not args.fp32 and device.type == "cuda"
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    results = []
    for name in args.sets.split(","):
        utts = [u for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl") if u.hangul_only]
        store = PcmStore(ROOT / "data" / "pcm", name)
        ids = json.loads((ROOT / args.alignments / "corpus" / f"{name}.ids.json").read_text(encoding="utf-8"))
        refs, unregrouped = {}, 0
        text_by_id = {u.id: u.text for u in utts}
        for stem, uid in ids.items():
            path = ROOT / args.alignments / "aligned" / name / f"{stem}.csv"
            if not path.exists():
                continue
            spans = regroup(text_by_id[uid].split(), load_alignment(path))
            if spans is None:
                unregrouped += 1
                continue
            refs[uid] = spans
        for chunk in (int(c) for c in args.chunks.split(",")):
            r = measure(model, vocab, name, utts, store, refs, chunk, args.left, device, logmel, amp)
            r["alignment_files"] = len(ids)
            r["unregrouped"] = unregrouped
            results.append(r)
            print(
                f"{name} chunk {chunk * 40} ms: CER {r['cer_ns'] * 100:.2f} %, FTD P50/P90 "
                f"{r['ftd_ms']['p50']:.0f}/{r['ftd_ms']['p90']:.0f} ms (n {r['ftd_ms']['n']}), LTD P50/P90 "
                f"{r['ltd_ms']['p50']:.0f}/{r['ltd_ms']['p90']:.0f} ms (n {r['ltd_ms']['n']})",
                flush=True,
            )
    out = ROOT / args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "precision": "fp16-autocast" if amp else "fp32",
        "results": results,
    }
    (out / f"latency-{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
