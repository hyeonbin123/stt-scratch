"""Service gates: endpoint rules, session == offline streaming decode, flush, WebSocket round trip."""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel, n_mel_frames
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.subsampling import subsampled_length
from stt_scratch.service.endpoint import EndpointConfig, Endpointer
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))


def tiny_model(seed: int = 0) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(1,),
    )
    return ChunkConformerCTC(cfg).eval()


def silence(n: int, v: int = 8) -> torch.Tensor:
    lp = torch.full((n, v), -20.0)
    lp[:, 0] = 0.0
    return lp


def speech(n: int, v: int = 8) -> torch.Tensor:
    lp = torch.full((n, v), -20.0)
    lp[:, 3] = 0.0
    return lp


def test_endpoint_rules() -> None:
    ep = Endpointer(EndpointConfig(rule1_s=2.4, rule2_s=1.2, rule3_s=20.0))
    assert ep.update(silence(25), 0) is None  # 1.0 s of silence, nothing decoded yet
    assert ep.update(speech(5), 1) is None
    assert ep.update(silence(29), 0) is None  # 1.16 s after the token
    assert ep.update(silence(1), 0) == "rule2"  # 1.2 s
    ep.reset()
    assert ep.update(silence(59), 0) is None
    assert ep.update(silence(1), 0) == "rule1"  # 2.4 s without any token
    ep.reset()
    assert ep.update(speech(499), 1) is None
    assert ep.update(speech(1), 0) == "rule3"  # 20 s segment


def test_session_matches_offline_streaming_decode() -> None:
    """With endpointing disabled, the session's final text equals the batched chunk-masked greedy decode."""
    model = tiny_model()
    logmel = LogMel()
    spec = ChunkSpec(8, 64)
    g = torch.Generator().manual_seed(5)
    wav = torch.randn(16000 * 3 + 777, generator=g) * 0.05
    no_endpoint = EndpointConfig(rule1_s=1e9, rule2_s=1e9, rule3_s=1e9)
    session = StreamSession(model, VOCAB, spec, logmel, no_endpoint)
    events = []
    for i in range(0, wav.numel(), 5120):
        events.extend(session.push_samples(wav[i : i + 5120]))
    events.extend(session.flush())
    finals = [e for e in events if e.kind == "final"]
    partials = [e for e in events if e.kind == "partial"]
    # append-only: every partial is a prefix of the next
    for a, b in zip(partials, partials[1:], strict=False):
        assert b.text.startswith(a.text)
    with torch.no_grad():
        mel, frames = logmel(wav[None], torch.tensor([wav.numel()]))
        logits, _, lengths = model(mel, frames, spec)
    ref = VOCAB.decode(greedy_batch(log_probs_fp32(logits), lengths)[0])
    got = " ".join(f.text for f in finals)
    assert got.replace(" ", "") == ref.replace(" ", "")
    assert (
        session.stats.chunks == n_mel_frames(wav.numel()) // 32
    )  # whole chunks; the rest went through flush


def _run_aligned_packets(session: StreamSession, wav: torch.Tensor) -> tuple[list[int], list[int]]:
    """Feed 5,360 + 5,120-sample packets; returns (mel frames per encoder step, chunks per packet)."""
    fed: list[int] = []
    original = session.encoder.step
    session.encoder.step = lambda mel, state: fed.append(mel.shape[0]) or original(mel, state)
    per_packet: list[int] = []
    pos, packet = 0, 5120 + FIRST_PACKET_EXTRA
    while pos < wav.numel():
        before = len(fed)
        session.push_samples(wav[pos : pos + packet])
        per_packet.append(len(fed) - before)
        pos += packet
        packet = 5120
    session.flush()
    return fed, per_packet


@pytest.mark.parametrize("reset_cache", [True, False])
def test_endpoint_keeps_buffered_audio_and_packet_grid(reset_cache: bool) -> None:
    """Forced endpoints at chunks 5 and 12 must not drop pending mel frames or the fbank carry: every mel
    frame of the stream reaches the encoder and every packet after the first still yields one chunk."""
    model = tiny_model()
    session = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel(), EndpointConfig(), reset_cache)
    session.endpointer.update = lambda lp, n: "rule2" if session.stats.chunks in (5, 12) else None
    g = torch.Generator().manual_seed(8)
    wav = torch.randn(16000 * 5 + 1234, generator=g) * 0.05
    fed, per_packet = _run_aligned_packets(session, wav)
    n_mel = n_mel_frames(wav.numel())
    assert sum(fed) == n_mel
    assert all(
        n == 1 for n in per_packet[:-1]
    )  # incl. the packets right after the endpoints; last may be short
    assert session.stats.chunks == n_mel // 32
    assert sum(subsampled_length(n) for n in fed) == subsampled_length(n_mel)
    assert session.segment == 3  # two endpoints + the flush


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_session_runs_under_fp16_autocast_on_cuda() -> None:
    """The service decodes with the same precision as evaluate.decode_streaming (rule change 8/12)."""
    model = tiny_model().cuda()
    seen: list[torch.dtype] = []
    model.out_proj.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    session = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel().cuda(), EndpointConfig())
    g = torch.Generator().manual_seed(9)
    session.push_samples((torch.randn(5360, generator=g) * 0.05).cuda())
    assert seen and all(d == torch.float16 for d in seen)
    seen.clear()
    fp32 = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel().cuda(), EndpointConfig(), amp=False)
    fp32.push_samples((torch.randn(5360, generator=g) * 0.05).cuda())
    assert seen and all(d == torch.float32 for d in seen)


def test_websocket_round_trip(tmp_path) -> None:
    from stt_scratch.service import server

    model = tiny_model()
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    vocab_path = tmp_path / "vocab.json"
    VOCAB.save(vocab_path)
    original = server.ROOT
    server.ROOT = tmp_path  # vocab lookup
    (tmp_path / "manifests").mkdir()
    VOCAB.save(tmp_path / "manifests" / "vocab.json")
    try:
        app = server.build_app(ckpt, 8, 64, "cpu", EndpointConfig(), True)
    finally:
        server.ROOT = original
    client = TestClient(app)
    assert client.get("/info").json()["chunk_ms"] == 320
    g = torch.Generator().manual_seed(1)
    pcm = (torch.randn(16000, generator=g) * 2000).to(torch.int16).numpy().tobytes()
    assert client.get("/info").json()["precision"] == "fp32"
    with client.websocket_connect("/ws") as ws:
        for i in range(0, len(pcm), 10240):
            ws.send_bytes(pcm[i : i + 10240])
        ws.send_text("flush")
        got = []
        for _ in range(200):  # flush is always answered by the events and then one "stats" message
            got.append(json.loads(ws.receive_text()))
            if got[-1]["type"] == "stats":
                break
    assert got[-1]["type"] == "stats" and got[-1]["chunks"] == n_mel_frames(16000) // 32
    assert any(m["type"] == "final" for m in got)


@pytest.mark.parametrize("reset_cache", [True, False])
def test_endpoint_closes_segment(reset_cache: bool) -> None:
    model = tiny_model()
    session = StreamSession(
        model, VOCAB, ChunkSpec(8, 64), LogMel(), EndpointConfig(rule3_s=0.32), reset_cache
    )
    wav = torch.zeros(16000)
    events = []
    for i in range(0, wav.numel(), 5120):
        events.extend(session.push_samples(wav[i : i + 5120]))
    # rule3 at 0.32 s fires once per chunk: segments advance even if nothing was decoded
    assert session.segment >= 2
    assert all(e.kind in ("partial", "final", "endpoint") for e in events)
