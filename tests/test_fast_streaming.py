"""빠른 스트리밍 경로(fast·graph)와 조기 종료(exit_layer) 테스트(docs/experiments.md 규칙 변경 26).

fast·graph 경로는 6·7단계에 쓴 eager 경로와 출력이 비트 단위로 같아야 한다(출력 동일성 관문의 CPU 부분).
CPU에서는 graph 경로가 CUDA Graph 없이 같은 고정 버퍼 코드를 그대로 돌리므로, 버퍼 묶기·풀기 논리와
정상 상태 step 본체를 여기서 확인한다. CUDA Graph 재생 자체와 GPU fp16 동일성은 측정 스크립트
(scripts/followup.py gate)가 GPU에서 확인한다.
"""

# [왜 비트 단위인가] 빠른 경로의 변경(마스크·RoPE 표 공유, 0을 0으로 바꾸던 탭 가리기 생략, autocast 가중치
# 미리 변환)은 모두 같은 값을 같은 순서로 계산하도록 고른 것이라, 허용 오차를 두지 않고 torch.equal로 본다.
# autocast 가중치 복사본은 CUDA fp16 대신 CPU bf16 autocast로 같은 원리(이미 낮은 정밀도인 가중치는 autocast가
# 다시 바꾸지 않음)를 확인한다.
# [exit_layer] 조기 종료 스트리밍 출력은 오프라인 forward의 같은 층 중간 헤드(청크 마스크)와 같아야 한다.

from __future__ import annotations

import pytest
import torch

from stt_scratch.model import streaming
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder, StreamState, autocast_weight_copy

# 좌측 문맥 16프레임 = 청크(4프레임) 4개라 다섯 번째 청크부터 정상 상태(고정 버퍼 step)가 된다.
SPEC = ChunkSpec(4, 16)
MEL_PER_CHUNK = 4 * SPEC.chunk_size


# 작은 모델: 폭 32, 3층, QK-norm 켬(최종 모델과 같음), 2층 뒤 중간 헤드(피드백 여부는 인자).
def make_model(seed: int = 0, **kw) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    base = dict(
        vocab_size=12,
        d_model=32,
        n_layers=3,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(2,),
        sc_feedback=False,
        qk_norm=True,
    )
    base.update(kw)
    return ChunkConformerCTC(ModelConfig(**base)).eval()


def random_mel(t: int, seed: int) -> torch.Tensor:
    return torch.randn(t, 80, generator=torch.Generator().manual_seed(seed))


# 청크 단위로 step을 돌려 청크별 log-prob 목록을 돌려준다. 마지막에 청크보다 짧은 부분 청크가 남으면 그것도
# 넣는다(서비스 flush와 같은 처리).
def run(enc: StreamingEncoder, mel: torch.Tensor, state=None) -> list[torch.Tensor]:
    state = state or enc.new_state()
    return [enc.step(mel[i : i + MEL_PER_CHUNK], state) for i in range(0, mel.shape[0], MEL_PER_CHUNK)]


def autocast_ctx(dtype: torch.dtype | None):
    return torch.autocast("cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None)


# 확인: 15개 청크 + 부분 청크에서 fast·graph의 청크별 출력과 마지막 캐시가 eager와 비트 단위로 같다(fp32와
# CPU bf16 autocast, 자기조건 피드백 있음·없음). 왜: 출력 동일성 관문을 통과해야 서버 기본값이 될 수 있다.
@pytest.mark.parametrize("path", ["fast", "graph"])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_fast_paths_are_bit_identical_to_eager(path: str, feedback: bool, amp_dtype) -> None:
    model = make_model(sc_feedback=feedback)
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3)
    eager, fast = StreamingEncoder(model, SPEC), StreamingEncoder(model, SPEC, path=path)
    s_eager, s_fast = eager.new_state(), fast.new_state()
    with torch.no_grad(), autocast_ctx(amp_dtype):
        out_eager = run(eager, mel, s_eager)
        out_fast = run(fast, mel, s_fast)
    assert len(out_eager) == len(out_fast) == 16
    for a, b in zip(out_eager, out_fast, strict=True):
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
    assert s_eager.frames_seen == s_fast.frames_seen
    eager_caches = s_eager.attn_caches + s_eager.conv_caches
    for a, b in zip(eager_caches, s_fast.attn_caches + s_fast.conv_caches, strict=True):
        assert torch.equal(a, b)


# 확인: graph 경로는 캐시가 꽉 찬 청크부터 고정 버퍼 step을 쓰고(16 mel 프레임 청크 15개 중 앞 4개는 fast,
# 나머지 11개), 그때 상태의 캐시 목록이 버퍼 자체를 가리킨다. 왜: 고정 버퍼 경로가 실제로 쓰였는지 확인해야
# 위 동일성 테스트가 의미가 있다.
def test_graph_path_uses_static_buffers_in_steady_state(monkeypatch) -> None:
    model = make_model()
    calls = []
    original = streaming._StaticStep._body
    monkeypatch.setattr(streaming._StaticStep, "_body", lambda self: calls.append(1) or original(self))
    enc = StreamingEncoder(model, SPEC, path="graph")
    state = enc.new_state()
    with torch.no_grad():
        run(enc, random_mel(MEL_PER_CHUNK * 15, seed=4), state)
    assert len(calls) == 15 - SPEC.left_context // SPEC.chunk_size
    stepper = streaming._static_stepper(enc)
    assert all(a is b for a, b in zip(state.attn_caches, stepper.attn, strict=True))


# 확인: 두 스트림을 번갈아 graph 경로로 돌려도(고정 버퍼를 서로 넘겨받음) 각자 eager로 따로 돌린 결과와 비트
# 단위로 같고, 버퍼를 넘긴 쪽 상태의 캐시는 복제본으로 남아 값이 바뀌지 않는다. 왜: 서버는 연결마다 세션을
# 만들지만 모델·설정이 같으면 고정 버퍼와 그래프 하나를 함께 쓴다.
def test_interleaved_streams_share_static_buffers_safely() -> None:
    model = make_model()
    mels = [random_mel(MEL_PER_CHUNK * 9, seed=s) for s in (5, 6)]
    with torch.no_grad():
        expected = [run(StreamingEncoder(model, SPEC), m) for m in mels]
        enc = StreamingEncoder(model, SPEC, path="graph")
        states = [enc.new_state(), enc.new_state()]
        got: list[list[torch.Tensor]] = [[], []]
        for i in range(0, mels[0].shape[0], MEL_PER_CHUNK):
            for k in (0, 1):
                got[k].append(enc.step(mels[k][i : i + MEL_PER_CHUNK], states[k]))
    for k in (0, 1):
        assert all(torch.equal(a, b) for a, b in zip(expected[k], got[k], strict=True))
    stepper = streaming._static_stepper(enc)
    # 마지막에 버퍼를 쓴 것은 두 번째 스트림이다. 첫 번째 스트림의 캐시는 버퍼가 아니다.
    assert all(a is not b for a, b in zip(states[0].attn_caches, stepper.attn, strict=True))


# 확인: 조기 종료(exit_layer = 2)의 스트리밍 출력이 오프라인 forward(같은 청크 마스크)의 2층 중간 헤드
# log-prob과 fp32 1e-5 안에서 같고(피드백 있음·없음; 중간 헤드는 피드백을 더하기 전의 로짓), fast 경로는
# eager와 비트 단위로 같다. 왜: 6층 조기 종료 측정(P4)이 '중간 헤드의 스트리밍 성능'을 잰다는 해석이 맞는지
# 고정한다.
@pytest.mark.parametrize("feedback", [False, True])
def test_exit_layer_matches_intermediate_head(feedback: bool) -> None:
    model = make_model(sc_feedback=feedback)
    t = MEL_PER_CHUNK * 9
    mel = random_mel(t, seed=8)
    with torch.no_grad():
        _, inter, _ = model(mel[None], torch.tensor([t]), SPEC)
        offline = log_probs_fp32(inter[0])[0]
        streamed = torch.cat(run(StreamingEncoder(model, SPEC, exit_layer=2), mel))
        fast = torch.cat(run(StreamingEncoder(model, SPEC, exit_layer=2, path="graph"), mel))
    assert float((streamed - offline).abs().max()) <= 1e-5
    assert torch.equal(streamed, fast)


# 확인: autocast 가중치 복사본은 Linear·conv 가중치만 낮은 정밀도로 바꾸고 LayerNorm·정규화 버퍼는 fp32로
# 두며, 원래 모델은 바꾸지 않고, 같은 모델·dtype이면 같은 복사본을 돌려준다.
def test_autocast_weight_copy_leaves_the_original_model_alone() -> None:
    model = make_model()
    twin = autocast_weight_copy(model, torch.bfloat16)
    assert twin is autocast_weight_copy(model, torch.bfloat16)
    assert twin.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert twin.blocks[0].conv.depthwise.weight.dtype == torch.bfloat16
    assert twin.subsampling.conv1.weight.dtype == torch.bfloat16
    assert twin.blocks[0].attn_norm.weight.dtype == torch.float32
    assert twin.mel_mean.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in model.parameters())


# 확인: 잘못된 설정을 거부한다(조기 종료 층 범위, 경로 이름, fast·graph의 E-Branchformer).
def test_streaming_encoder_rejects_bad_options() -> None:
    model = make_model()
    for kw in ({"exit_layer": 0}, {"exit_layer": 4}, {"path": "jit"}):
        with pytest.raises(ValueError):
            StreamingEncoder(model, SPEC, **kw)
    branch = make_model(block="ebranchformer", cgmlp_units=32, cgmlp_kernel=7, merge_kernel=7)
    StreamingEncoder(branch, SPEC)  # eager는 그대로 지원
    with pytest.raises(ValueError):
        StreamingEncoder(branch, SPEC, path="fast")


# 확인(GPU 전용, CPU·CI에서는 건너뜀): CUDA fp16 autocast에서 fast·graph 경로가 eager와 비트 단위로 같고,
# graph는 정상 상태 step을 실제로 CUDA Graph로 캡처해 재생한다. 왜: 그래프 캡처·재생과 가중치 복사본(fp16)은
# CUDA에서만 동작하므로, GPU 측정 전에 이 테스트로 먼저 확인한다(측정 스크립트의 gate가 실제 모델로 다시
# 확인한다).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("path", ["fast", "graph"])
def test_fast_paths_on_cuda_fp16(path: str) -> None:
    model = make_model().cuda()
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        eager = run(StreamingEncoder(model, SPEC), mel)
        fast = run(StreamingEncoder(model, SPEC, path=path), mel)
    assert all(torch.equal(a, b) for a, b in zip(eager, fast, strict=True))
    if path == "graph":
        stepper = streaming._static_stepper(StreamingEncoder(model, SPEC, path="graph"))
        assert stepper.graph is not None, stepper.graph_failed


# ---- 긴 세션의 위치 재기준(streaming.REBASE_AT, streaming.rebase_positions) ----
# 서버 기본값(끝점 뒤 캐시 유지)에서는 frames_seen이 연결 내내 커져 RoPE fp32 각도(위치 × 주파수)의 반올림
# 오차가 함께 커진다. 그래서 step 앞에서 frames_seen이 임계(2^16프레임, 약 44분) 이상이면 청크 배수만큼
# 당긴다. 수정 전에 이 작은 모델(CPU fp32)로 시작값만 바꿔 재 보면 0 시작 대비 최대 절대차가 2^16에서 5.5e-5,
# 2^20에서 1.2e-3, 2^24에서 0.12였고, 재기준이 만드는 시작값 16(좌측 문맥)에서는 4.8e-7이었다. 그러므로
# 아래 1e-4 비교는 2^24 경우에서만 재기준의 유무를 가르고, 2^16 경우는 당긴 양과 frames_seen 값의 단언이
# 가른다. CPU bf16 autocast(CUDA fp16보다 거친 정밀도)에서는 시작값이 16이든 65,536이든 0 시작과의 차이가
# 0.012~0.028이라 저정밀도 비교에는 5e-2를 쓴다.


def run_from(enc: StreamingEncoder, mel: torch.Tensor, start: int) -> tuple[torch.Tensor, StreamState]:
    """frames_seen을 start로 둔 새 상태로 mel 전체를 돌려 (이어 붙인 log-prob, 마지막 상태)를 돌려준다."""
    state = enc.new_state()
    state.frames_seen = start
    return torch.cat(run(enc, mel, state)), state


# rebase_positions가 돌려준 '당긴 양'을 step마다 기록하는 spy. StreamingEncoder.step이 모듈 전역 이름으로
# 부르므로 monkeypatch로 바꿔 끼울 수 있고, 임계(REBASE_AT)는 원래 함수가 호출 시점의 전역값을 읽는다.
def spy_rebase(monkeypatch) -> list[int]:
    shifts: list[int] = []
    original = streaming.rebase_positions

    def spy(state, spec, threshold=None):
        shifts.append(original(state, spec, threshold))
        return shifts[-1]

    monkeypatch.setattr(streaming, "rebase_positions", spy)
    return shifts


# 확인: 임계 미만 스트림(15청크 + 부분 청크)은 임계를 무한대로 둔 실행과 출력·frames_seen이 비트 단위로 같고
# 당긴 양은 모두 0이다. 왜: 봉인 test·6단계·후속 측정은 발화마다 새 세션이고 10분 드리프트도 15,000프레임이라
# 모두 임계 미만이므로, 이 변경이 기록 수치와 무관함을 코드로 고정한다.
@pytest.mark.parametrize("path", ["eager", "fast", "graph"])
def test_streams_below_the_rebase_threshold_are_bit_identical(path: str, monkeypatch) -> None:
    model = make_model()
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3)
    shifts = spy_rebase(monkeypatch)
    with torch.no_grad():
        out, state = run_from(StreamingEncoder(model, SPEC, path=path), mel, 0)
        monkeypatch.setattr(streaming, "REBASE_AT", 1 << 62)
        plain, s_plain = run_from(StreamingEncoder(model, SPEC, path=path), mel, 0)
    assert torch.equal(out, plain)
    assert state.frames_seen == s_plain.frames_seen
    assert shifts == [0] * 32


# 확인: 시작값이 임계 이상이면 재기준이 한 번 일어나(2^24는 첫 청크에서, 2^16 − 8청크는 캐시가 꽉 찬
# 9번째 청크에서) 당긴 양은 '시작값 + k청크 − 좌측 문맥'(청크 배수)이고, 재기준 뒤 frames_seen은 좌측
# 문맥(16)에서 다시 커지며, 세 경로가 비트 단위로 같은 값을 내고 0에서 시작한 eager 실행과 log-prob이
# 1e-4 안이다. 왜: RoPE는 상대 위치이고 캐시는 회전 전 입력을 저장하므로 모든 위치를 같은 청크 배수만큼
# 옮기는 것은 같은 계산이어야 하고, 세 경로가 같은 frames_seen 한 곳만 읽어야 경로 간 동일성이 유지된다.
@pytest.mark.parametrize("start", [1 << 24, (1 << 16) - 8 * SPEC.chunk_size])
def test_rebased_stream_matches_a_stream_started_at_zero(start: int, monkeypatch) -> None:
    model = make_model()
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3)
    shifts = spy_rebase(monkeypatch)
    with torch.no_grad():
        ref, s_ref = run_from(StreamingEncoder(model, SPEC), mel, 0)
        outs, states, seen = {}, {}, {}
        for path in streaming.PATHS:
            shifts.clear()
            outs[path], states[path] = run_from(StreamingEncoder(model, SPEC, path=path), mel, start)
            seen[path] = list(shifts)
    # 재기준이 일어나는 step k = frames_seen이 처음 임계에 이르는 청크 번호(시작값이 임계 이상이면 0).
    k = max(0, -(-(streaming.REBASE_AT - start) // SPEC.chunk_size))
    expected = [0] * 16
    expected[k] = start + k * SPEC.chunk_size - SPEC.left_context
    for path in streaming.PATHS:
        assert seen[path] == expected, path
        assert torch.equal(outs[path], outs["eager"]), path
        assert states[path].frames_seen == s_ref.frames_seen + SPEC.left_context - k * SPEC.chunk_size
    assert float((outs["eager"] - ref).abs().max()) <= 1e-4


# 확인: 임계를 좌측 문맥의 2배(32프레임)로 낮추면 15청크 스트림에서 재기준이 두 번 일어나는데, 당긴 양은
# 매번 청크 배수이고 당긴 뒤 frames_seen은 좌측 문맥 이상(키 위치가 음수가 되지 않음)이며, 모든 step 뒤
# frames_seen이 청크 배수로 남아(graph 경로의 정상 상태 조건) 고정 버퍼 step이 재기준 사이에도 계속 쓰이고,
# 출력은 재기준 없는 같은 경로 실행과 1e-4 안이다. 왜: 재기준이 반복돼도 frames_seen이 임계 근처에 머물며
# 세 경로의 조건이 깨지지 않아야 한다.
@pytest.mark.parametrize("path", ["eager", "fast", "graph"])
def test_repeated_rebases_keep_the_chunk_grid_and_the_left_context(path: str, monkeypatch) -> None:
    model = make_model()
    mel = random_mel(MEL_PER_CHUNK * 15, seed=4)
    calls: list[int] = []
    original_body = streaming._StaticStep._body
    monkeypatch.setattr(streaming._StaticStep, "_body", lambda self: calls.append(1) or original_body(self))
    with torch.no_grad():
        ref, _ = run_from(StreamingEncoder(model, SPEC, path=path), mel, 0)
        calls.clear()
        monkeypatch.setattr(streaming, "REBASE_AT", 2 * SPEC.left_context)
        shifts = spy_rebase(monkeypatch)
        enc = StreamingEncoder(model, SPEC, path=path)
        state = enc.new_state()
        outs, after = [], []
        for i in range(0, mel.shape[0], MEL_PER_CHUNK):
            outs.append(enc.step(mel[i : i + MEL_PER_CHUNK], state))
            after.append(state.frames_seen)
    assert sum(s > 0 for s in shifts) == 2
    for before, shift in zip([0, *after[:-1]], shifts, strict=True):
        assert shift % SPEC.chunk_size == 0
        if shift:
            assert before >= streaming.REBASE_AT and before - shift >= SPEC.left_context
    assert all(f % SPEC.chunk_size == 0 for f in after) and max(after) <= 2 * SPEC.left_context
    assert float((torch.cat(outs) - ref).abs().max()) <= 1e-4
    if path == "graph":
        assert len(calls) == 15 - SPEC.left_context // SPEC.chunk_size


# 캐시가 꽉 찬 뒤(9번째 청크) 재기준이 일어나는 시작값으로 eager와 graph를 돌려, 둘이 비트 단위로 같고
# 0 시작 eager와 tol 안이며 frames_seen이 '전체 프레임 + 좌측 문맥 − 8청크'인지 확인하고, graph의 고정 버퍼
# stepper를 돌려준다(호출자가 캡처 여부를 본다). CPU 저정밀도 테스트와 CUDA fp16 테스트가 같은 본체를 쓰므로
# CUDA 테스트의 논리는 CPU에서 먼저 검증된다.
def steady_state_rebase_check(
    model: ChunkConformerCTC, mel: torch.Tensor, tol: float
) -> streaming._StaticStep:
    start = (1 << 16) - 8 * SPEC.chunk_size
    ref, s_ref = run_from(StreamingEncoder(model, SPEC), mel, 0)
    eager, s_eager = run_from(StreamingEncoder(model, SPEC), mel, start)
    graph, s_graph = run_from(StreamingEncoder(model, SPEC, path="graph"), mel, start)
    assert torch.equal(eager, graph)
    expected_frames = s_ref.frames_seen + SPEC.left_context - 8 * SPEC.chunk_size
    assert s_eager.frames_seen == s_graph.frames_seen == expected_frames
    assert float((graph - ref).abs().max()) <= tol
    return streaming._static_stepper(StreamingEncoder(model, SPEC, path="graph"))


# 확인: CPU bf16 autocast(가중치 복사본 경로)에서도 정상 상태 재기준이 eager·graph에서 같은 값을 내고
# 0 시작과 5e-2 안이다(위 메모의 bf16 측정 0.012~0.028). 왜: 아래 CUDA fp16 테스트와 같은 본체를 CPU에서
# 먼저 돌려 둔다.
def test_rebase_in_steady_state_under_cpu_bf16_autocast() -> None:
    model = make_model()
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3)
    with torch.no_grad(), autocast_ctx(torch.bfloat16):
        stepper = steady_state_rebase_check(model, mel, tol=5e-2)
    assert stepper.graph is None  # CPU에서는 그래프를 캡처하지 않는다


# 확인(GPU 전용, CPU·CI에서는 건너뜀): CUDA fp16에서 그래프가 실제로 캡처·재생되는 정상 상태 step 사이에
# 재기준이 일어나도(offset 버퍼에 당긴 값이 들어감) graph가 eager와 비트 단위로 같고 0 시작 실행과 5e-2
# 안이다. 왜: 캡처된 그래프는 offset 버퍼를 읽으므로 버퍼 값만 바뀌면 되는데, 그 사실은 CUDA에서만 확인할 수
# 있다.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rebase_on_the_cuda_graph_path_fp16() -> None:
    model = make_model().cuda()
    mel = random_mel(MEL_PER_CHUNK * 15 + 7, seed=3).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        stepper = steady_state_rebase_check(model, mel, tol=5e-2)
    assert stepper.graph is not None, stepper.graph_failed
