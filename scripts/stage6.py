"""Stage 6 streaming-service measurements (docs/experiments.md 6단계, rule change 21).

Subcommands (each writes reports/stage6/<name>.json):
  grid     simulated CER for chunk {4, 8, 16} x left {64, 128}, algorithmic latency (mean/max), per-chunk
           compute p50/p90 and RTF on the GPU (fp16) and on the CPU with one thread (fp32); H6.2
  h61      real cache streaming vs simulated CER on all of dev_H and dev_mix for the service
           configuration; H6.1
  session  StreamSession runs on dev_mix (each utterance followed by 3 s of -60 dBFS noise): no-endpoint
           baseline, rule2 {0.8, 1.2} s with cache reset, then keep-cache for the chosen rule2;
           early-endpoint CER increase, EP50/EP90 (endpoint time - MFA end of the last word),
           UPWR/UPSR (H6.3), session CER vs measured CER
  drift    10 minutes of concatenated dev_H speech through one session: per-minute compute, GPU memory

Usage: uv run python scripts/stage6.py grid --checkpoint outputs/stage4/control/avg10.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, decode_offline, decode_streaming, load_model, score
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
PACKET = 5120
NOISE_DBFS = -60.0


def rate(scored: dict) -> float:
    return scored["hangul_only"]["cer_ns"]["rate"]


def algorithmic_latency_ms(chunk: int) -> dict:
    """A frame waits for the end of its chunk; the mel window adds 15 ms (25 ms window, 10 ms hop)."""
    chunk_ms = chunk * 40
    return {"mean_ms": chunk_ms / 2 + 15, "max_ms": chunk_ms + 15}


def noise(seconds: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g) * (10 ** (NOISE_DBFS / 20))


def packets(wav: torch.Tensor):
    pos, size = 0, PACKET + FIRST_PACKET_EXTRA
    while pos < wav.numel():
        yield pos, wav[pos : pos + size]
        pos += size
        size = PACKET


@torch.no_grad()
def time_steps(model, utts, store, spec, device, amp: bool) -> dict:
    """Per-chunk encoder step time (ms, synchronized) over whole utterances, plus the real-time factor."""
    logmel = LogMel().to(device)
    enc = StreamingEncoder(model, spec)
    times, audio_s = [], 0.0
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        state = enc.new_state()
        step = 4 * spec.chunk_size
        for i in range(
            0, mel.shape[1], step
        ):  # includes the partial last chunk (RTF over the whole utterance)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device, amp):
                enc.step(mel[0, i : i + step], state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        audio_s += u.duration_s
    a = np.asarray(times)
    return {
        "chunks": len(times),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "max_ms": float(a.max()),
        "rtf": float(a.sum() / 1000 / audio_s),
    }


def cmd_grid(args, model, vocab, device) -> dict:
    logmel = LogMel().to(device)
    out = {"configs": []}
    sets = {
        n: (
            read_manifest(ROOT / f"manifests/{n}.jsonl")[: args.limit or None],
            PcmStore(ROOT / "data/pcm", n),
        )
        for n in ("dev_mix", "dev_h")
    }
    timing_utts = sets["dev_h"][0][: args.timing_utts]
    cpu_model, _ = load_model(ROOT / args.checkpoint, torch.device("cpu"))
    for chunk in (4, 8, 16):
        for left in (64, 128):
            spec = ChunkSpec(chunk, left)
            row = {
                "chunk_frames": chunk,
                "chunk_ms": chunk * 40,
                "left": left,
                **algorithmic_latency_ms(chunk),
            }
            for name, (utts, store) in sets.items():
                row[f"{name}_cer_ns"] = rate(
                    score(utts, decode_offline(model, vocab, utts, store, spec, device, logmel))
                )
            row["gpu"] = time_steps(model, timing_utts, sets["dev_h"][1], spec, device, amp=True)
            torch.set_num_threads(1)
            row["cpu1"] = time_steps(
                cpu_model,
                timing_utts[: args.cpu_utts],
                sets["dev_h"][1],
                spec,
                torch.device("cpu"),
                amp=False,
            )
            print(json.dumps(row), flush=True)
            out["configs"].append(row)
    service = next(r for r in out["configs"] if r["chunk_frames"] == 8 and r["left"] == 128)
    out["h62"] = {
        "gpu_p90_ms": service["gpu"]["p90_ms"],
        "cpu1_p90_ms": service["cpu1"]["p90_ms"],
        "met": service["gpu"]["p90_ms"] <= 50 and service["cpu1"]["p90_ms"] <= 250,
    }
    return out


def cmd_h61(args, model, vocab, device) -> dict:
    logmel = LogMel().to(device)
    spec = ChunkSpec(args.chunk, args.left)
    out = {"spec": f"chunk{args.chunk} left{args.left}"}
    for name in ("dev_h", "dev_mix"):
        utts = read_manifest(ROOT / f"manifests/{name}.jsonl")[: args.limit or None]
        store = PcmStore(ROOT / "data/pcm", name)
        sim = rate(score(utts, decode_offline(model, vocab, utts, store, spec, device, logmel)))
        real_hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel)
        real = rate(score(utts, real_hyps))
        out[name] = {"simulated": sim, "real_streaming": real, "diff_pp": (real - sim) * 100}
        print(name, out[name], flush=True)
    out["h61_met"] = out["dev_h"]["diff_pp"] <= 0.1
    return out


def last_word_end(name: str) -> dict[str, float]:
    """MFA end time of the last word per utterance (Hangul-only utterances with an alignment)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from stage4_latency import load_alignment

    ids = json.loads((ROOT / f"work/mfa/corpus/{name}.ids.json").read_text(encoding="utf-8"))
    ends = {}
    for stem, uid in ids.items():
        path = ROOT / f"work/mfa/aligned/{name}/{stem}.csv"
        if path.exists():
            pieces = load_alignment(path)
            if pieces:
                ends[uid] = pieces[-1][1]
    return ends


def run_sessions(
    model, vocab, spec, utts, store, endpoint: EndpointConfig, reset: bool, device, tail_s: float
):
    """One StreamSession per utterance (+ tail noise). Returns hypotheses, endpoint times and partials."""
    logmel = LogMel().to(device)
    hyps, first_endpoint, partial_runs = {}, {}, []
    for k, u in enumerate(utts):
        wav = torch.cat([store.waveform(u.id), noise(tail_s, seed=k)]).to(device)
        session = StreamSession(model, vocab, spec, logmel, endpoint, reset)
        finals, partials_by_segment = [], {}
        for pos, piece in packets(wav):
            for ev in session.push_samples(piece):
                if ev.kind == "partial":
                    partials_by_segment.setdefault(ev.segment, []).append(ev.text)
                elif ev.kind in ("final", "endpoint"):
                    if ev.kind == "final":
                        finals.append(ev)
                    first_endpoint.setdefault(u.id, (pos + piece.numel()) / SR)
        for ev in session.flush():
            finals.append(ev)
        hyps[u.id] = " ".join(f.text for f in finals)
        for f in finals:
            partial_runs.append((partials_by_segment.get(f.segment, []), f.text))
    return hyps, first_endpoint, partial_runs


def stability(partial_runs) -> dict:
    """UPWR: complete words of partials (all but a trailing unfinished word) that differ from the final
    word at the same position, over final words; UPSR: segments with any such word. 'strict' also counts
    the trailing word whenever it is not yet the final word (an append-only decoder only extends it)."""
    unstable = strict = final_words = bad_segments = bad_strict = 0
    for partials, final in partial_runs:
        fw = final.split()
        final_words += len(fw)
        seg_bad = seg_strict = False
        for p in partials:
            words = p.split()
            complete = words if p.endswith(" ") else words[:-1]
            for i, w in enumerate(complete):
                if i >= len(fw) or fw[i] != w:
                    unstable += 1
                    seg_bad = True
            if words and not p.endswith(" "):
                i = len(words) - 1
                if i >= len(fw) or fw[i] != words[-1]:
                    strict += 1
                    seg_strict = True
        bad_segments += seg_bad
        bad_strict += seg_bad or seg_strict
    return {
        "upwr": unstable / max(1, final_words),
        "upsr": bad_segments / max(1, len(partial_runs)),
        "upwr_strict_trailing": (unstable + strict) / max(1, final_words),
        "upsr_strict_trailing": bad_strict / max(1, len(partial_runs)),
        "segments": len(partial_runs),
    }


def cmd_session(args, model, vocab, device) -> dict:
    spec = ChunkSpec(args.chunk, args.left)
    utts = read_manifest(ROOT / "manifests/dev_mix.jsonl")[: args.limit or None]
    store = PcmStore(ROOT / "data/pcm", "dev_mix")
    ends = last_word_end("dev_mix")
    off = EndpointConfig(rule1_s=1e9, rule2_s=1e9, rule3_s=1e9)
    out = {
        "spec": f"chunk{args.chunk} left{args.left}",
        "tail_noise_s": args.tail_s,
        "noise_dbfs": NOISE_DBFS,
    }
    base_hyps, _, base_partials = run_sessions(
        model, vocab, spec, utts, store, off, True, device, args.tail_s
    )
    base = rate(score(utts, base_hyps))
    out["baseline_no_endpoint"] = {"cer_ns": base, **stability(base_partials)}
    print("baseline", out["baseline_no_endpoint"], flush=True)
    runs = {}
    for rule2 in (0.8, 1.2):
        hyps, ep, partials = run_sessions(
            model, vocab, spec, utts, store, EndpointConfig(rule2_s=rule2), True, device, args.tail_s
        )
        runs[rule2] = (hyps, ep, partials)
        cer = rate(score(utts, hyps))
        delays = [ep[u] - ends[u] for u in ep if u in ends]
        out[f"rule2_{rule2}_reset"] = {
            "cer_ns": cer,
            "cer_increase_pp": (cer - base) * 100,
            "ep50_ms": float(np.percentile(delays, 50) * 1000) if delays else None,
            "ep90_ms": float(np.percentile(delays, 90) * 1000) if delays else None,
            "ep_n": len(delays),
            **stability(partials),
        }
        print(rule2, out[f"rule2_{rule2}_reset"], flush=True)
    ok = [r for r in (0.8, 1.2) if out[f"rule2_{r}_reset"]["cer_increase_pp"] <= 0.2]
    chosen = min(ok) if ok else 1.2
    out["rule2_chosen"] = chosen
    hyps, keep_ep, keep_partials = run_sessions(
        model, vocab, spec, utts, store, EndpointConfig(rule2_s=chosen), False, device, args.tail_s
    )
    keep = rate(score(utts, hyps))
    reset = out[f"rule2_{chosen}_reset"]["cer_ns"]
    keep_delays = [keep_ep[u] - ends[u] for u in keep_ep if u in ends]
    out["keep_cache"] = {
        "cer_ns": keep,
        "reset_cer_ns": reset,
        "ep50_ms": float(np.percentile(keep_delays, 50) * 1000) if keep_delays else None,
        "ep90_ms": float(np.percentile(keep_delays, 90) * 1000) if keep_delays else None,
        "ep_n": len(keep_delays),
        **stability(keep_partials),
    }
    out["cache_policy"] = "keep" if keep < reset else "reset"
    chosen_metrics = out["keep_cache"] if out["cache_policy"] == "keep" else out[f"rule2_{chosen}_reset"]
    out["chosen"] = {"rule2_s": chosen, "cache_policy": out["cache_policy"], **chosen_metrics}
    out["h63_met"] = chosen_metrics["upwr"] == 0 and out["baseline_no_endpoint"]["upwr"] == 0
    # service path CER (no-endpoint session) vs the measured real-streaming CER on dev_mix (rule: within 0.1)
    h61 = ROOT / "reports" / "stage6" / "h61.json"
    if h61.exists() and not args.limit:
        real = json.loads(h61.read_text(encoding="utf-8"))["dev_mix"]["real_streaming"]
        out["service_vs_measured"] = {
            "session_cer_ns": base,
            "real_streaming_cer_ns": real,
            "diff_pp": (base - real) * 100,
            "met": abs(base - real) * 100 <= 0.1,
        }
    return out


def cmd_drift(args, model, vocab, device) -> dict:
    spec = ChunkSpec(args.chunk, args.left)
    utts = read_manifest(ROOT / "manifests/dev_h.jsonl")
    store = PcmStore(ROOT / "data/pcm", "dev_h")
    pieces, total = [], 0.0
    for k, u in enumerate(utts):
        pieces += [store.waveform(u.id), noise(0.5, seed=10_000 + k)]
        total += u.duration_s + 0.5
        if total >= args.minutes * 60:
            break
    wav = torch.cat(pieces).to(device)
    endpoint = EndpointConfig(rule2_s=args.rule2)
    session = StreamSession(model, vocab, spec, LogMel().to(device), endpoint, args.policy == "reset")
    per_minute, mem = {}, []
    finals = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for pos, piece in packets(wav):
        n_before = len(session.stats.compute_ms)
        finals += sum(ev.kind == "final" for ev in session.push_samples(piece))
        minute = int(pos / SR // 60)
        per_minute.setdefault(minute, []).extend(session.stats.compute_ms[n_before:])
        if device.type == "cuda" and pos % (30 * SR) < PACKET:
            mem.append({"t_s": pos / SR, "allocated_mb": torch.cuda.memory_allocated() / 2**20})
    rows = {
        m: {"p50_ms": float(np.percentile(v, 50)), "p90_ms": float(np.percentile(v, 90)), "chunks": len(v)}
        for m, v in sorted(per_minute.items())
        if v
    }
    return {
        "audio_s": wav.numel() / SR,
        "segments_closed": finals,
        "per_minute": rows,
        "memory": mem,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "policy": args.policy,
        "rule2_s": args.rule2,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["grid", "h61", "session", "drift"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--timing-utts", type=int, default=30)
    p.add_argument("--cpu-utts", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tail-s", type=float, default=3.0)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--rule2", type=float, default=1.2)
    p.add_argument("--policy", default="reset", choices=["reset", "keep"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    t0 = time.perf_counter()
    result = {"grid": cmd_grid, "h61": cmd_h61, "session": cmd_session, "drift": cmd_drift}[args.command](
        args, model, vocab, device
    )
    import hashlib

    result.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(),
            "device": str(device),
            "spec": f"chunk{args.chunk} left{args.left}",
            "limit": args.limit,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    )
    out = ROOT / "reports" / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or args.command
    (out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote reports/stage6/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
