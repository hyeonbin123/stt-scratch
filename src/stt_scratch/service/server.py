"""WebSocket streaming STT server.

  uv run python -m stt_scratch.service.server --checkpoint outputs/stage2/runA/avg10.pt --chunk 8 --left 128

Protocol: the client opens ws://host:port/ws and sends binary frames of little-endian int16 mono 16 kHz
PCM (any size; 320 ms = 10,240 bytes recommended). The server answers with JSON text frames:
  {"type": "partial", "segment": n, "text": "..."}      append-only partial transcript of the open segment
  {"type": "final",   "segment": n, "text": "...", "rule": "rule2"}   the segment closed by an endpoint rule
A text frame "flush" closes the stream (partial last chunk + final) and is answered, after the events, by
  {"type": "stats", "chunks": n, "compute_ms_p50": ..., "compute_ms_p90": ..., "compute_ms_max": ...}
(per-chunk compute time of the whole session); "reset" starts a fresh session.
GET / serves web/index.html (microphone client). No authentication: local demo only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from stt_scratch.evaluate import load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web" / "index.html"


def build_app(
    checkpoint: Path,
    chunk: int,
    left: int,
    device: str,
    endpoint: EndpointConfig,
    reset_cache: bool,
    amp: bool | None = None,
) -> FastAPI:
    dev = torch.device(device)
    model, _ = load_model(checkpoint, dev)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(dev)
    spec = ChunkSpec(chunk, left)
    use_amp = (dev.type == "cuda") if amp is None else amp
    app = FastAPI(title="stt-scratch streaming")
    info = {
        "checkpoint": str(checkpoint),
        "chunk_frames": chunk,
        "chunk_ms": chunk * 40,
        "left_frames": left,
        "device": device,
        "precision": "fp16-autocast" if use_amp else "fp32",
        "endpoint": asdict(endpoint),
        "reset_cache_on_endpoint": reset_cache,
        "vocab": len(vocab),
    }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB)

    @app.get("/info")
    def get_info() -> JSONResponse:
        return JSONResponse(info)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, amp=use_amp)
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                stats = None
                if message.get("bytes") is not None:
                    events = session.push_pcm16(message["bytes"])
                elif message.get("text") == "flush":
                    events = session.flush()
                    stats = session.stats.summary()
                elif message.get("text") == "reset":
                    session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, amp=use_amp)
                    events = []
                else:
                    events = []
                for ev in events:
                    await socket.send_text(
                        json.dumps(
                            {
                                "type": ev.kind,
                                "segment": ev.segment,
                                "text": ev.text,
                                "rule": ev.rule,
                                "frames": ev.frames,
                                "compute_ms": round(ev.compute_ms, 1),
                            },
                            ensure_ascii=False,
                        )
                    )
                if stats is not None:
                    await socket.send_text(json.dumps({"type": "stats", **stats}))
        except WebSocketDisconnect:
            pass

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("STT_CHECKPOINT", ""))
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rule2", type=float, default=1.2)
    p.add_argument("--keep-cache", action="store_true", help="keep encoder caches across endpoints")
    p.add_argument("--fp32", action="store_true", help="no fp16 autocast (models trained with --fp32)")
    args = p.parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint (or STT_CHECKPOINT) is required")
    app = build_app(
        ROOT / args.checkpoint,
        args.chunk,
        args.left,
        args.device,
        EndpointConfig(rule2_s=args.rule2),
        not args.keep_cache,
        amp=False if args.fp32 else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])
