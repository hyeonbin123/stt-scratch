"""Feed utterances to the WebSocket server in real time and record transcript/latency per utterance.

Used in Stage 6 to measure end-to-end latency (last audio byte sent -> final received) and to check that the
service path gives the same text as the offline streaming evaluation.

Usage: uv run python scripts/stream_client.py --manifest dev_h --limit 50 --url ws://127.0.0.1:8765/ws
Writes reports/stage6/client-<manifest>.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import websockets

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.features import FIRST_PACKET_EXTRA

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SAMPLES = 5120  # 320 ms at 16 kHz
FIRST_PACKET_SAMPLES = CHUNK_SAMPLES + FIRST_PACKET_EXTRA  # 5,360: the first packet completes 32 mel frames


async def stream_one(url: str, pcm: np.ndarray, realtime: bool) -> dict:
    events = []
    t_start = time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:

        async def reader() -> None:
            async for msg in ws:
                events.append({"t": time.perf_counter() - t_start, **json.loads(msg)})

        task = asyncio.create_task(reader())
        sent_t = []
        i, packet = 0, FIRST_PACKET_SAMPLES
        while i < len(pcm):
            piece = pcm[i : i + packet]
            i += packet
            packet = CHUNK_SAMPLES
            await ws.send(piece.astype("<i2").tobytes())
            sent_t.append(time.perf_counter() - t_start)
            if realtime:
                await asyncio.sleep(len(piece) / 16000)
        t_last_audio = time.perf_counter() - t_start
        await ws.send("flush")
        deadline = time.perf_counter() + 5.0  # the server answers flush with the events, then "stats"
        while time.perf_counter() < deadline and not any(e["type"] == "stats" for e in events):
            await asyncio.sleep(0.02)
        task.cancel()
    finals = [e for e in events if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    stats = next((e for e in events if e["type"] == "stats"), {})
    text = " ".join(e["text"] for e in finals)
    first_partial = partials[0]["t"] if partials else None
    # caption lag: arrival of each partial minus the send time of the latest packet sent before it
    # (network + queueing + compute; the chunk's own duration is the algorithmic latency, reported apart)
    lags = []
    for e in partials:
        before = [s for s in sent_t if s <= e["t"]]
        if before:
            lags.append((e["t"] - before[-1]) * 1000)
    last_final = finals[-1]["t"] if finals else None
    return {
        "text": text,
        "n_partials": len(partials),
        "n_finals": len(finals),
        "rules": [e["rule"] for e in finals],
        "first_partial_s": first_partial,
        "partial_lag_ms": lags,
        "final_after_last_audio_s": (last_final - t_last_audio) if last_final else None,
        "chunks": stats.get("chunks"),
        "compute_ms_p50": stats.get("compute_ms_p50"),  # per encoder chunk, from the server (H6.2)
        "compute_ms_p90": stats.get("compute_ms_p90"),
        "compute_ms_max": stats.get("compute_ms_max"),
    }


async def main_async(args: argparse.Namespace) -> None:
    utts = read_manifest(ROOT / "manifests" / f"{args.manifest}.jsonl")[: args.limit or None]
    store = PcmStore(ROOT / "data" / "pcm", args.manifest)
    out = ROOT / "reports" / "stage6" / f"client-{args.manifest}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            offset, length = store.index[u.id]
            pcm = np.asarray(store.bin[offset : offset + length])
            result = await stream_one(args.url, pcm, not args.fast)
            result.update({"id": u.id, "ref": u.text, "duration_s": u.duration_s})
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(utts)}", flush=True)
    print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="dev_h")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    p.add_argument("--fast", action="store_true", help="send without real-time pacing")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
