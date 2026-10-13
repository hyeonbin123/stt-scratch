"""Exploratory (not a Stage 6 verdict): where does the GPU streaming step time go?

Runs the service configuration (chunk 8, left 128) over a few dev_H utterances and reports, per step:
the number of ATen ops and CUDA kernels, CPU-side wall time, summed GPU kernel time, and the step time with
fp16 autocast vs plain fp32 on the GPU. Run only when no other GPU job is running.

    uv run python scripts/diag_step_profile.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs/stage4/trimtail/avg10.pt"


def chunks_of(model, store, utts, device, spec):
    logmel = LogMel().to(device)
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        step = 4 * spec.chunk_size
        yield [mel[0, i : i + step] for i in range(0, mel.shape[1], step)]


def timed(model, store, utts, device, spec, amp):
    enc = StreamingEncoder(model, spec)
    times = []
    for chunk_list in chunks_of(model, store, utts, device, spec):
        state = enc.new_state()
        for c in chunk_list:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device, amp):
                enc.step(c, state)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times[5:])  # drop warm-up steps
    return {"p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)), "n": len(a)}


def main() -> None:
    device = torch.device("cuda")
    model, _ = load_model(CKPT, device)
    spec = ChunkSpec(8, 128)
    utts = read_manifest(ROOT / "manifests/dev_h.jsonl")[:10]
    store = PcmStore(ROOT / "data/pcm", "dev_h")
    out = {"fp16_autocast": timed(model, store, utts, device, spec, True)}
    out["fp32"] = timed(model, store, utts, device, spec, False)
    # profile 20 steady-state steps (the left-context cache is full after 16 chunks)
    enc = StreamingEncoder(model, spec)
    chunk_list = next(
        chunks_of(model, store, read_manifest(ROOT / "manifests/dev_h.jsonl")[:1], device, spec)
    )
    state = enc.new_state()
    with autocast(device, True):
        for c in chunk_list[:16]:
            enc.step(c, state)
    steady = (chunk_list * 3)[16:36]
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for c in steady:
            with autocast(device, True):
                enc.step(c, state)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000 / len(steady)
    events = prof.key_averages()
    n_aten = sum(e.count for e in events if e.key.startswith("aten::") and e.cpu_parent is None)
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_ms = sum(e.device_time for e in kernels) / 1000 / len(steady)
    out["profile"] = {
        "steps": len(steady),
        "wall_ms_per_step_profiled": wall,
        "cuda_kernels_per_step": len(kernels) / len(steady),
        "gpu_kernel_ms_per_step": gpu_ms,
        "top_aten_ops": [
            {
                "op": e.key,
                "calls_per_step": e.count / len(steady),
                "cpu_ms_per_step": e.cpu_time_total / 1000 / len(steady),
            }
            for e in sorted(events, key=lambda e: -e.cpu_time_total)[:15]
        ],
    }
    out["aten_top_level_calls_hint"] = n_aten
    path = ROOT / "reports/stage6/diag-step_profile.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1))
    print(json.dumps({k: v for k, v in out["profile"].items() if k != "top_aten_ops"}, indent=1))
    for r in out["profile"]["top_aten_ops"]:
        print(r)


if __name__ == "__main__":
    main()
