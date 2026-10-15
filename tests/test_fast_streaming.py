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
from stt_scratch.model.streaming import StreamingEncoder, autocast_weight_copy

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
