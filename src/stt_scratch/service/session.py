"""One streaming recognition session: PCM in, partial/final transcript events out.

Pipeline per audio piece: StreamingFbank (frame-exact) -> whole encoder chunks (4 * chunk_size mel frames)
-> StreamingEncoder.step -> CTC greedy (append-only) -> rule endpointing. Partial results are never revised;
a final event closes the segment. An endpoint resets only the recogniser state (greedy state, endpoint
counters and, with `reset_cache_on_endpoint`, the encoder caches): buffered mel frames and the fbank carry
belong to the next segment and are kept, so no audio is lost and packets stay on the 32-frames grid. The
encoder runs under the same fp16 autocast as the evaluation paths (docs/experiments.md rule change 8/12).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.decode import GreedyState
from stt_scratch.model.features import FIRST_PACKET_EXTRA, HOP, N_MELS, LogMel, StreamingFbank
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.service.endpoint import EndpointConfig, Endpointer
from stt_scratch.text import Vocab


@dataclass
class Event:
    kind: str  # "partial" | "final"
    text: str
    segment: int
    frames: int  # encoder frames processed in this segment so far
    rule: str | None = None
    compute_ms: float = 0.0


@dataclass
class SessionStats:
    chunks: int = 0  # whole encoder chunks (the flush of a partial last chunk is not counted)
    compute_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        """Per-chunk compute time percentiles (H6.2 is defined on these, not on emitted events)."""
        if not self.compute_ms:
            return {
                "chunks": self.chunks,
                "compute_ms_p50": None,
                "compute_ms_p90": None,
                "compute_ms_max": None,
            }
        arr = np.asarray(self.compute_ms)
        return {
            "chunks": self.chunks,
            "compute_ms_p50": float(np.percentile(arr, 50)),
            "compute_ms_p90": float(np.percentile(arr, 90)),
            "compute_ms_max": float(arr.max()),
        }


class StreamSession:
    def __init__(
        self,
        model: ChunkConformerCTC,
        vocab: Vocab,
        spec: ChunkSpec,
        logmel: LogMel,
        endpoint: EndpointConfig | None = None,
        reset_cache_on_endpoint: bool = True,
        partial_interval_frames: int = 0,
        amp: bool | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.spec = spec
        self.device = next(model.parameters()).device
        self.amp = (self.device.type == "cuda") if amp is None else amp
        self.encoder = StreamingEncoder(model, spec)
        self.fbank = StreamingFbank(logmel)
        self.endpointer = Endpointer(endpoint or EndpointConfig())
        self.reset_cache_on_endpoint = reset_cache_on_endpoint
        self.partial_interval_frames = partial_interval_frames
        self.mel_per_chunk = 4 * spec.chunk_size
        self.stats = SessionStats()
        self.pending = torch.zeros(0, N_MELS, device=self.device)
        self.segment = 0
        self._new_segment(reset_cache=True)

    def _new_segment(self, reset_cache: bool) -> None:
        """Start a transcript segment; never touches `pending` or the fbank carry (see module docstring)."""
        self.greedy = GreedyState()
        self.endpointer.reset()
        self.segment_frames = 0
        self.frames_since_partial = 0
        if reset_cache:
            self.state = self.encoder.new_state()

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp)

    def push_pcm16(self, data: bytes) -> list[Event]:
        """data: little-endian int16 mono 16 kHz samples. Returns the events produced by this piece."""
        samples = torch.from_numpy(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
        return self.push_samples(samples.to(self.device))

    @torch.no_grad()
    def push_samples(self, samples: torch.Tensor) -> list[Event]:
        events: list[Event] = []
        self.pending = torch.cat([self.pending, self.fbank.push(samples)], dim=0)
        while self.pending.shape[0] >= self.mel_per_chunk:
            mel_chunk = self.pending[: self.mel_per_chunk]
            self.pending = self.pending[self.mel_per_chunk :]
            events.extend(self._process_chunk(mel_chunk))
        return events

    def _step(self, mel: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One encoder step under the session precision; returns (log_probs, encoder ms). The GPU is
        synchronized before and after, so queued feature work is not counted as encoder time."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self._autocast():
            log_probs = self.encoder.step(mel, self.state)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return log_probs, (time.perf_counter() - t0) * 1000

    def _process_chunk(self, mel_chunk: torch.Tensor) -> list[Event]:
        log_probs, ms = self._step(mel_chunk)
        new = self.greedy.push(log_probs, self.state.frames_seen - log_probs.shape[0])
        self.stats.chunks += 1
        self.stats.compute_ms.append(ms)
        self.segment_frames += log_probs.shape[0]
        self.frames_since_partial += log_probs.shape[0]
        events: list[Event] = []
        rule = self.endpointer.update(log_probs, len(new))
        text = self.vocab.decode(self.greedy.tokens)
        if rule is not None:
            if text.strip():
                events.append(Event("final", text, self.segment, self.segment_frames, rule, ms))
            else:  # an endpoint that closes an empty segment (e.g. leading silence); clients may ignore it
                events.append(Event("endpoint", "", self.segment, self.segment_frames, rule, ms))
            self.segment += 1
            self._new_segment(reset_cache=self.reset_cache_on_endpoint)
        elif new or self.frames_since_partial >= self.partial_interval_frames:
            if new:
                events.append(Event("partial", text, self.segment, self.segment_frames, None, ms))
            self.frames_since_partial = 0
        return events

    @torch.no_grad()
    def flush(self) -> list[Event]:
        """End of stream: process the partial last chunk, close the open segment, start a fresh stream."""
        events: list[Event] = []
        ms = 0.0
        if self.pending.shape[0] > 0:
            log_probs, ms = self._step(self.pending)
            self.pending = torch.zeros(0, N_MELS, device=self.device)
            self.greedy.push(log_probs, self.state.frames_seen - log_probs.shape[0])
            self.segment_frames += log_probs.shape[0]
        text = self.vocab.decode(self.greedy.tokens)
        if text.strip():
            events.append(Event("final", text, self.segment, self.segment_frames, "eos", ms))
        self.segment += 1
        self._new_segment(reset_cache=True)
        self.fbank.reset()
        return events


def warm_up(session: StreamSession) -> float:
    """Flush once for every possible length of a partial last chunk (1 .. 4*chunk-1 mel frames) so that
    no client pays the one-time setup of a new tensor shape: on the RTX 2080 Ti the first flush with a
    new encoder-frame count took about 0.47 s instead of 0.06 s. Leaves the session fresh; returns the
    seconds spent."""
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(0)
    first = FIRST_PACKET_EXTRA + session.mel_per_chunk * HOP  # one whole chunk
    for m in range(1, session.mel_per_chunk):
        n = first + m * HOP
        session.push_samples((torch.randn(n, generator=g) * 1e-3).to(session.device))
        session.flush()
    session.stats = SessionStats()
    session.segment = 0
    return time.perf_counter() - t0
