"""WebSocket streaming STT server.

  uv run python -m stt_scratch.service.server  # final model, 320 ms chunks, left 128, rule2 1.2 s, keep cache

Defaults are the Stage 6 choices (docs/experiments.md). At start-up the server flushes every possible partial
last-chunk length once (warm-up), so the first clients do not pay the one-time setup of new tensor shapes.

Protocol: the client opens ws://host:port/ws and sends binary frames of little-endian int16 mono 16 kHz
PCM (any size; 320 ms = 10,240 bytes recommended). The server answers with JSON text frames:
  {"type": "partial", "segment": n, "text": "..."}      append-only partial transcript of the open segment
  {"type": "final",   "segment": n, "text": "...", "rule": "rule2"}   the segment closed by an endpoint rule
  {"type": "endpoint", "segment": n, "text": "", "rule": "rule1"}     an endpoint that closed an empty segment
A text frame "flush" closes the stream (partial last chunk + final) and is answered, after the events, by
  {"type": "stats", "chunks": n, "compute_ms_p50": ..., "compute_ms_p90": ..., "compute_ms_max": ...}
(per-chunk compute time of the whole session); "reset" starts a fresh session.
GET / serves web/index.html (microphone client). No authentication: local demo only, bound to 127.0.0.1 by
default. Every HTTP response carries security headers (a hash-based CSP for the page), the interactive API
docs are off, and a WebSocket whose Origin is another site is refused (OWASP ZAP findings, docs/security.md).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from stt_scratch.evaluate import load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web" / "index.html"
DEFAULT_CHECKPOINT = "outputs/stage4/trimtail/avg10.pt"  # the final model (Stage 4 verdict)
COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
API_CSP = "default-src 'none'; frame-ancestors 'none'"


def _sha256_source(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


def page_csp(html: str) -> str:
    """CSP for the page: only its own inline <script> and <style> (by hash), the AudioWorklet blob,
    same-origin fetch/WebSocket; nothing else may load, frame it or submit forms."""
    scripts = [_sha256_source(s) for s in re.findall(r"<script>(.*?)</script>", html, re.S)]
    styles = [_sha256_source(s) for s in re.findall(r"<style>(.*?)</style>", html, re.S)]
    return (
        f"default-src 'none'; script-src {' '.join(scripts)} blob:; style-src {' '.join(styles)}; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def same_origin(origin: str | None, host: str | None) -> bool:
    """A browser WebSocket from another site carries its Origin; clients without one (scripts) pass."""
    if not origin:
        return True
    return host is not None and re.sub(r"^https?://", "", origin) == host


def shown_path(path: Path) -> str:
    """Checkpoint path for /info without the local directory layout."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def build_app(
    checkpoint: Path,
    chunk: int,
    left: int,
    device: str,
    endpoint: EndpointConfig,
    reset_cache: bool,
    amp: bool | None = None,
    warm: bool = True,
) -> FastAPI:
    dev = torch.device(device)
    model, _ = load_model(checkpoint, dev)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(dev)
    spec = ChunkSpec(chunk, left)
    use_amp = (dev.type == "cuda") if amp is None else amp
    warm_s = (
        warm_up(StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, amp=use_amp))
        if warm
        else 0.0
    )
    app = FastAPI(title="stt-scratch streaming", docs_url=None, redoc_url=None)
    page = WEB.read_text(encoding="utf-8")
    csp = page_csp(page)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in COMMON_HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers.setdefault("Content-Security-Policy", API_CSP)
        return response

    info = {
        "checkpoint": shown_path(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "chunk_frames": chunk,
        "chunk_ms": chunk * 40,
        "left_frames": left,
        "device": device,
        "precision": "fp16-autocast" if use_amp else "fp32",
        "endpoint": asdict(endpoint),
        "reset_cache_on_endpoint": reset_cache,
        "vocab": len(vocab),
        "warm_up_s": round(warm_s, 2),
    }

    @app.get("/", include_in_schema=False)  # the page is not part of the API
    def index() -> HTMLResponse:
        return HTMLResponse(page, headers={"Content-Security-Policy": csp})

    @app.get("/info")
    def get_info() -> JSONResponse:
        return JSONResponse(info)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not same_origin(socket.headers.get("origin"), socket.headers.get("host")):
            await socket.close(code=1008)  # cross-site WebSocket hijacking guard
            return
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
    p.add_argument("--checkpoint", default=os.environ.get("STT_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rule2", type=float, default=1.2)
    p.add_argument(
        "--reset-cache",
        action="store_true",
        help="reset encoder caches at endpoints (Stage 6 chose keeping them)",
    )
    p.add_argument("--keep-cache", action="store_true", help=argparse.SUPPRESS)  # the default since Stage 6
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
        args.reset_cache,
        amp=False if args.fp32 else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])
