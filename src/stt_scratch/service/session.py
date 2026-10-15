"""스트리밍 인식 세션 하나: PCM을 받아 부분·확정 자막 이벤트를 낸다.

클라이언트 연결 하나의 인식 세션이다. 부분 자막(partial)·확정 자막(final)·빈 끝점(endpoint) 이벤트를
만든다. WebSocket과 무관한 순수 로직이라 서버·6단계 측정(scripts/stage6.py)·테스트가 같은 클래스를 쓴다.
음성 조각마다의 흐름: StreamingFbank(프레임 단위로 정확) → 인코더 청크 단위(4 * chunk_size mel 프레임)
→ StreamingEncoder.step → CTC greedy(덧붙이기만) → 규칙형 끝점 판정. 부분 자막은 절대 고치지 않으며,
final 이벤트가 세그먼트를 닫는다. 끝점은 인식 상태(greedy 상태, 끝점 카운터, `reset_cache_on_endpoint`면
인코더 캐시)만 초기화한다. 버퍼에 쌓인 mel 프레임과 fbank 이월 표본은 다음 세그먼트의 것이므로 그대로
두어, 음성을 잃지 않고 패킷이 32프레임 격자에 계속 맞는다. 인코더는 평가 경로와 같은 fp16 autocast
아래에서 돈다(docs/experiments.md 규칙 변경 8·14).
"""

# [핵심 결정] (1) 버퍼를 다음 세그먼트로 넘기는 이유: 처음 구현은 버퍼까지 지워 끝점마다 다음 음성 약 300
# ms가 사라지고 이후 패킷이 32프레임 격자에서 벗어났다(적대적 검토에서 발견, 규칙 변경 14). (2) 평가 경로와
# 같은 fp16 autocast(규칙 변경 8·14). (3) 끝점 뒤 인코더 캐시는 서버 기본값에서 유지한다: 6단계 비교에서
# 유지 26.862 % < 초기화 27.277 %(dev_mix)였다(docs/experiments.md 6단계 결과 '세션').
# [연결] service.server가 연결마다 StreamSession을 만들고 시작할 때 warm_up을 한 번 돌린다.
# scripts/stage6.py는 이 세션으로 끝점 규칙·캐시 정책·서비스 경로 CER을 쟀다.

from __future__ import annotations

import time
from dataclasses import dataclass, field

# numpy: 바이트 PCM 해석(np.frombuffer)과 계산 시간 백분위. torch: 텐서·autocast·GPU 동기화. 프로젝트 모듈:
# 특징(StreamingFbank), 캐시 스트리밍 인코더, greedy 상태, 끝점, 어휘.
import numpy as np
import torch

from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.decode import GreedyState
from stt_scratch.model.features import FIRST_PACKET_EXTRA, HOP, N_MELS, LogMel, StreamingFbank
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.service.endpoint import EndpointConfig, Endpointer
from stt_scratch.text import Vocab


# 세션이 내보내는 이벤트. kind: partial(열린 세그먼트의 지금까지 텍스트, 덧붙기만 함) / final(끝점 규칙이나
# 스트림 끝 "eos"로 닫힌 세그먼트) / endpoint(텍스트 없이 닫힌 빈 세그먼트, 6단계 EP 측정에서 '첫 끝점'으로
# 세기 위해 내보냄, 규칙 변경 24). compute_ms는 이 이벤트를 만든 청크의 인코더 계산 시간.
@dataclass
class Event:
    kind: str  # "partial" | "final" | "endpoint"
    text: str
    segment: int
    frames: int  # 이 세그먼트에서 지금까지 처리한 인코더 프레임 수
    rule: str | None = None
    compute_ms: float = 0.0


# 세션 전체의 청크당 인코더 계산 시간 기록. 서버는 flush 뒤 summary를 "stats" 메시지로 보내고, 6단계 측정은
# 이벤트(토큰이 나온 청크만 있음)가 아니라 이 전체 청크 통계로 계산 시간을 쟀다(규칙 변경 14,
# docs/ai-verification.md).
@dataclass
class SessionStats:
    chunks: int = 0  # 온전한 인코더 청크 수(마지막 부분 청크의 flush는 세지 않는다)
    compute_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        """청크당 계산 시간의 백분위(H6.2는 내보낸 이벤트가 아니라 이 값으로 정의한다)."""
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


# 스트리밍 인식 세션. 생성 인자: 모델·어휘·청크 설정(서비스는 ChunkSpec(8, 128)), LogMel, 끝점 설정, 끝점 뒤
# 캐시 초기화 여부(이 클래스의 기본값은 True지만 서버는 6단계 선택대로 False를 넘긴다), amp(None이면 GPU에서만
# fp16).
# partial_interval_frames는 부분 자막 간격 옵션이지만, 지금 구현은 새 토큰이 있을 때만 partial을 내므로 실제
# 동작에 영향이 없다(기본 0).
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
        # pending: 아직 인코더에 넣지 않은 mel 프레임(청크 하나가 차기 전까지 모아 둔다). segment: 지금 열린
        # 세그먼트 번호.
        self.pending = torch.zeros(0, N_MELS, device=self.device)
        self.segment = 0
        self._new_segment(reset_cache=True)

    # greedy 상태·끝점 카운터·세그먼트 프레임 수를 새로 하고, reset_cache면 인코더 캐시(StreamState)도 비운다.
    def _new_segment(self, reset_cache: bool) -> None:
        """새 자막 세그먼트를 시작한다. `pending`과 fbank 이월은 절대 건드리지 않는다(모듈 docstring 참고)."""
        self.greedy = GreedyState()
        self.endpointer.reset()
        self.segment_frames = 0
        self.frames_since_partial = 0
        if reset_cache:
            self.state = self.encoder.new_state()

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp)

    # WebSocket 바이너리 프레임을 float32 −1~1로 바꿔 push_samples로 넘긴다. 크기 제한이 없어 아무 길이로
    # 보내도 되지만, 320 ms(첫 패킷은 5,360표본) 단위로 보내야 패킷마다 청크 하나가 바로 나온다.
    def push_pcm16(self, data: bytes) -> list[Event]:
        """data: little-endian int16 mono 16 kHz 표본. 이 조각으로 생긴 이벤트들을 돌려준다."""
        samples = torch.from_numpy(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
        return self.push_samples(samples.to(self.device))

    # 파형 조각을 받아 완성된 mel 프레임을 pending에 쌓고, 청크 하나(4 × chunk_size = 32프레임)가 찰 때마다
    # 인코더를 돌려 이벤트를 모아 돌려준다. 남은 프레임은 다음 호출이나 flush로 넘어간다.
    @torch.no_grad()
    def push_samples(self, samples: torch.Tensor) -> list[Event]:
        events: list[Event] = []
        self.pending = torch.cat([self.pending, self.fbank.push(samples)], dim=0)
        while self.pending.shape[0] >= self.mel_per_chunk:
            mel_chunk = self.pending[: self.mel_per_chunk]
            self.pending = self.pending[self.mel_per_chunk :]
            events.extend(self._process_chunk(mel_chunk))
        return events

    # GPU 동기화는 Codex 4단계 검토 반영이다(규칙 변경 24). 이 값이 서버 stats의 청크당 계산 시간이다. 6단계
    # 측정에서 GPU p50은 약 62 ms였고, 연산 수가 많아 커널 실행 호출이 병목인 것으로
    # 보인다(docs/experiments.md 6단계 결과 H6.2).
    def _step(self, mel: torch.Tensor) -> tuple[torch.Tensor, float]:
        """세션 정밀도로 인코더 step 한 번을 돌려 (log_probs, 인코더 ms)를 돌려준다.

        앞뒤로 GPU를 동기화하므로 앞서 큐에 쌓인 특징 계산은 인코더 시간에 들어가지 않는다.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self._autocast():
            log_probs = self.encoder.step(mel, self.state)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return log_probs, (time.perf_counter() - t0) * 1000

    # 청크 하나 처리: 인코더 → greedy에 새 프레임 추가(절대 프레임 번호 기준) → 통계 기록 → 끝점 판정 →
    # 이벤트. 끝점이 나오면 지금까지의 텍스트로 final(텍스트가 비면 endpoint)을 내고 세그먼트를 넘긴다. 끝점이
    # 없고 새 토큰이 있으면 partial을 낸다.
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
        # 캐시를 유지하면 끝점이 인코더 출력은 바꾸지 않지만 greedy 상태는 세그먼트마다 새로 시작한다. 그래서
        # 같은 토큰이 이어지는 중에 끝점이 걸리면(묵음 없이 닫히는 rule3) 그 토큰이 한 번 더 나올 수
        # 있다(docs/experiments.md 6단계 결과 '세션').
        if rule is not None:
            if text.strip():
                events.append(Event("final", text, self.segment, self.segment_frames, rule, ms))
            else:  # 빈 세그먼트를 닫는 끝점(예: 앞쪽 묵음). 클라이언트는 무시해도 된다
                events.append(Event("endpoint", "", self.segment, self.segment_frames, rule, ms))
            self.segment += 1
            self._new_segment(reset_cache=self.reset_cache_on_endpoint)
        elif new or self.frames_since_partial >= self.partial_interval_frames:
            if new:
                events.append(Event("partial", text, self.segment, self.segment_frames, None, ms))
            self.frames_since_partial = 0
        return events

    # 클라이언트의 "flush"에 대응한다. 열린 세그먼트는 "eos" final로 닫고, 캐시·fbank를 모두 초기화한다.
    # 통계(stats)는 지우지 않아 서버가 세션 전체 요약을 보낼 수 있다. 부분 청크는 stats.chunks에 세지 않는다.
    @torch.no_grad()
    def flush(self) -> list[Event]:
        """스트림 끝: 청크보다 짧게 남은 부분 청크를 처리하고 열린 세그먼트를 닫은 뒤 새 스트림을 준비한다."""
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


# 서버 시작 때 한 번 한다. 6단계 측정에서 서버를 띄운 직후 몇몇 발화의 flush가 약 0.5 s 걸렸고, 이것이
# 새 인코더 프레임 수의 1회성 비용임을 재현으로 확인했다(scripts/diag_flush_warmup.py). 서버 시작이 약 7.7 s
# 늘어나는 대신, 수정 뒤 재측정에서 flush 왕복은 p90 64 ms였다(docs/experiments.md 6단계 결과 '측정 뒤 서비스
# 수정').
# 입력은 표준편차 1e-3의 작은 난수 잡음(시드 고정)이며, 끝나면 통계·세그먼트 번호를 되돌려 새 세션과 똑같이
# 동작한다(tests/test_service.py).
def warm_up(session: StreamSession) -> float:
    """예열: 가능한 마지막 부분 청크 길이(mel 1 .. 4*chunk-1프레임)마다 한 번씩 flush한다.

    그래서 어떤 클라이언트도 새 텐서 모양의 1회성 준비 비용을 치르지 않는다. RTX 2080 Ti에서 새 인코더
    프레임 수로 처음 flush하면 0.06 s가 아니라 약 0.47 s가 걸렸다. 세션을 새것 상태로 되돌리고 걸린 시간(초)을
    돌려준다.
    """
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(0)
    # first = 정확히 mel 32프레임(청크 하나)이 나오는 표본 수(5,360). 여기에 m × 160표본을 더하면 청크 하나 +
    # m프레임이 남고, flush가 그 m프레임짜리 부분 청크를 처리한다.
    first = FIRST_PACKET_EXTRA + session.mel_per_chunk * HOP
    for m in range(1, session.mel_per_chunk):
        n = first + m * HOP
        session.push_samples((torch.randn(n, generator=g) * 1e-3).to(session.device))
        session.flush()
    session.stats = SessionStats()
    session.segment = 0
    return time.perf_counter() - t0
