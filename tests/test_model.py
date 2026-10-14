"""0단계 게이트(docs/experiments.md 0단계): 미래 누출 없음, 스트리밍 == 오프라인, CTC dtype, 과적합."""

# [테스트 묶음] 0단계 "할 것"의 게이트: 이 테스트가 모두 통과하기 전에는 학습을 시작하지 않는다는 규칙의
# 대상이다. 확인하는 것: (1) 미래 누출 없음, (2) 캐시 스트리밍 == 전체 발화 + 청크
# 마스크(fp32 1e-5, fp16 1e-2), (3) CTC 입력이 fp32, (4) 작은 데이터 과적합으로 학습 경로 전체가 동작함. 이후
# 단계에서 더한 부품(QK-norm, InterCTC, E-Branchformer, CR-CTC, Peak-First, TrimTail)과 버그 회귀 테스트도
# 여기 있다.
# [방식] 대부분 작은 모델(d=32, 3층)을 CPU에서 돌려 빠르게 확인하고, 실제 크기(12층 d=256, 1,288 클래스)의
# fp16 게이트는 CUDA가 있을 때만 돈다.

from __future__ import annotations

import math

import pytest
import torch

# 테스트 대상: 모델·설정, CTC 손실, greedy 디코딩, 특징(LogMel·StreamingFbank·SpecAugment), 청크 설정, 캐시
# 스트리밍 추론기, 서브샘플링 길이 공식.
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32, total_loss
from stt_scratch.model.decode import GreedyState, greedy_batch
from stt_scratch.model.features import (
    FIRST_PACKET_EXTRA,
    HOP,
    N_FFT,
    LogMel,
    StreamingFbank,
    n_mel_frames,
    spec_augment,
)
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.model.subsampling import subsampled_length

# 작은 가짜 어휘 크기. id 0~2는 blank·<unk>·공백 자리로 두고, 정답 토큰은 3 이상에서 뽑는다.
VOCAB = 12


# 테스트용 작은 설정: 폭 32, 3층, dropout 0(결정적 출력), 중간 헤드는 2층 하나(자기조건 피드백 경로도 함께
# 시험). 커널 15는 실제 모델과 같게 둬 conv 캐시 길이가 실제와 같다. 키워드로 필드를 바꿔 여러 변형을 만든다.
def small_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=VOCAB,
        n_mels=80,
        d_model=32,
        n_layers=3,
        n_heads=2,
        ffn_dim=64,
        conv_kernel=15,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(2,),
    )
    base.update(kw)
    return ModelConfig(**base)


# 시드를 고정한 작은 모델을 eval 모드로 만든다(같은 시드 = 같은 가중치).
def make_model(seed: int = 0, **kw) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    return ChunkConformerCTC(small_cfg(**kw)).eval()


def random_mel(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, t, 80, generator=g)


# 독립 검토 반영(docs/ai-verification.md).
def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """등록된 게이트는 '최대 절대 차이'다(allclose는 상대 오차 항까지 허용하므로 쓰지 않는다)."""
    return float((a.float() - b.float()).abs().max())


# fp16 autocast 게이트는 CUDA에서만 의미가 있어, GPU가 없으면 건너뛴다(README: CUDA가 없으면 fp16 테스트는
# 건너뜀).
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# 확인: subsampled_length 공식이 실제 서브샘플링 모듈의 출력 길이, ceil(ceil(T/2)/2)와 홀수·경계 길이에서 모두
# 같다. 왜: 이 공식이 CTC 길이 조건, 유효 프레임 마스크, 전처리 제외 규칙에 쓰이므로 하나라도 어긋나면
# 손실·마스크가 틀어진다.
def test_subsampled_length_matches_module() -> None:
    model = make_model()
    for t in (4, 5, 31, 32, 33, 128, 129):
        mel = random_mel(t)
        x, lengths = model.subsampling(mel, torch.tensor([t]))
        assert x.shape[1] == int(lengths[0]) == subsampled_length(t) == math.ceil(math.ceil(t / 2) / 2)


# 확인(게이트 1, 미래 누출): 청크 c의 시작부터 뒤쪽 mel을 큰 난수로 바꿔도 그 앞 청크까지의 출력이 1e-5 안에서
# 그대로이고, 뒤쪽 출력은 실제로 바뀐다(검사가 헛돌지 않음을 함께 확인). 청크 4/8/16/32 × 좌측 64/128 전부.
# 왜: 서브샘플링·어텐션·conv 어디서든 청크 경계 뒤를 보면 스트리밍에서 쓸 수 없는 정보로 학습하게 된다.
@pytest.mark.parametrize("chunk", [4, 8, 16, 32])
@pytest.mark.parametrize("left", [64, 128])
def test_no_future_leakage(chunk: int, left: int) -> None:
    """청크 경계 뒤의 음성을 바꿔도 그 경계까지의 출력은 바뀌면 안 된다."""
    model = make_model()
    spec = ChunkSpec(chunk, left)
    t_mel = 4 * chunk * 6  # 청크 6개
    mel = random_mel(t_mel)
    frames = torch.tensor([t_mel])
    with torch.no_grad():
        ref, _, _ = model(mel, frames, spec)
        for c in range(1, 6):
            cut_mel = 4 * chunk * c  # 청크 c에 속하는 첫 mel 프레임
            noisy = mel.clone()
            noisy[:, cut_mel:] = torch.randn_like(noisy[:, cut_mel:]) * 5
            out, _, _ = model(noisy, frames, spec)
            assert torch.allclose(out[:, : chunk * c], ref[:, : chunk * c], atol=1e-5), (chunk, left, c)
            assert not torch.allclose(out[:, chunk * c :], ref[:, chunk * c :], atol=1e-3)


# 확인(게이트 2): 청크를 하나씩 StreamingEncoder.step에 넣은 결과가 전체 발화 + 같은 청크 마스크의 오프라인
# 결과와 fp32 최대 차이 1e-5 안. 왜: 학습·simulated 평가는 오프라인 경로, 서비스는 캐시 경로라 둘이 같아야 dev
# 수치가 서비스 성능을 대표한다.
@pytest.mark.parametrize("chunk,left", [(4, 64), (8, 64), (8, 128), (16, 128), (32, 128)])
def test_streaming_equals_offline(chunk: int, left: int) -> None:
    model = make_model()
    spec = ChunkSpec(chunk, left)
    n_chunks = 7
    t_mel = 4 * chunk * n_chunks
    mel = random_mel(t_mel, seed=3)
    with torch.no_grad():
        logits, _, _ = model(mel, torch.tensor([t_mel]), spec)
        offline = log_probs_fp32(logits)[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        parts = [enc.step(mel[0, i * 4 * chunk : (i + 1) * 4 * chunk], state) for i in range(n_chunks)]
        streamed = torch.cat(parts, dim=0)
    assert streamed.shape == offline.shape
    assert max_abs(streamed, offline) <= 1e-5


# QK-norm은 규칙 변경 13으로 넣었다. 왜: QK-norm은 프레임 단위 연산이라 정확성이 그대로여야 한다는 주장을
# 테스트로 고정한다.
def test_streaming_equals_offline_qk_norm() -> None:
    """QK-LayerNorm은 프레임 단위라, 켜도 캐시 스트리밍이 정확하고 미래를 보지 않아야 한다."""
    model = make_model(qk_norm=True)
    assert model.blocks[0].attn.q_norm is not None
    spec = ChunkSpec(8, 64)
    t_mel = 32 * 6
    mel = random_mel(t_mel, seed=41)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(6)], dim=0)
        noisy = mel.clone()
        noisy[:, 96:] = torch.randn_like(noisy[:, 96:]) * 5
        out = log_probs_fp32(model(noisy, torch.tensor([t_mel]), spec)[0])[0]
    assert max_abs(streamed, offline) <= 1e-5
    assert max_abs(out[:24], offline[:24]) <= 1e-5


# 여기서는 13 mel 프레임짜리 부분 청크를 쓴다. 왜: 서비스 flush와 실제 스트리밍 평가가 발화 끝을 이렇게
# 처리한다.
def test_streaming_partial_last_chunk() -> None:
    """4 * chunk_size 프레임보다 짧은 마지막 청크도 오프라인 결과와 같아야 한다."""
    model = make_model()
    spec = ChunkSpec(8, 64)
    t_mel = 4 * 8 * 4 + 13
    mel = random_mel(t_mel, seed=9)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        parts = [enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(4)]
        parts.append(enc.step(mel[0, 4 * 32 :], state))
        streamed = torch.cat(parts, dim=0)
    assert streamed.shape == offline.shape
    assert max_abs(streamed, offline) <= 1e-5


# 확인: BACON을 끈 폴백(전 채널 인과 conv)에서도 캐시 스트리밍이 정확하다. 왜: 폴백 플래그가 실제로 쓸 수 있는
# 상태임을 보장한다.
def test_streaming_equals_offline_all_causal_conv() -> None:
    model = make_model(bacon=False)
    spec = ChunkSpec(8, 64)
    t_mel = 4 * 8 * 5
    mel = random_mel(t_mel, seed=5)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(5)], dim=0)
    assert max_abs(streamed, offline) <= 1e-5


# NaN이 없는지도 본다. 왜: 패딩 쿼리 행의 NaN이 BACON 오른쪽 탭으로 유효 프레임에 번질 수 있는 경로를 설계
# 검토에서 찾아 막았고(mask.py의 대각선, conv 입력 0), 그 회귀를 막는다.
def test_padding_does_not_change_valid_frames() -> None:
    """긴 발화와 함께 패딩된 배치에서도 짧은 발화의 출력은 혼자 돌린 결과와 같다."""
    model = make_model()
    spec = ChunkSpec(8, 64)
    a, b = random_mel(4 * 8 * 5, seed=7)[0], random_mel(4 * 8 * 3, seed=8)[0]
    batch = torch.zeros(2, a.shape[0], 80)
    batch[0] = a
    batch[1, : b.shape[0]] = b
    frames = torch.tensor([a.shape[0], b.shape[0]])
    with torch.no_grad():
        out, _, lengths = model(batch, frames, spec)
        alone, _, _ = model(b[None], torch.tensor([b.shape[0]]), spec)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1, : int(lengths[1])], alone[0], atol=1e-5)


# 확인: 5,120표본 패킷으로 나눠 넣은 StreamingFbank의 프레임이 한 번에 계산한 log-mel과 같고, 이월 표본이 자리
# 잡은 뒤에는 패킷마다 32프레임이 나온다. 마지막 단언은 첫 5,120표본 패킷이 30프레임뿐이라는
# 사실(FIRST_PACKET_EXTRA가 필요한 이유)을 고정한다.
def test_streaming_fbank_matches_offline() -> None:
    logmel = LogMel()
    g = torch.Generator().manual_seed(11)
    wav = torch.randn(5120 * 4 + 1000, generator=g) * 0.1
    ref, frames = logmel(wav[None], torch.tensor([wav.numel()]))
    sf = StreamingFbank(logmel)
    parts = [sf.push(wav[i * 5120 : (i + 1) * 5120]) for i in range(4)]
    parts.append(sf.push(wav[4 * 5120 :]))
    streamed = torch.cat(parts, dim=0)
    assert streamed.shape[0] == int(frames[0]) == n_mel_frames(wav.numel())
    assert parts[1].shape[0] == 32  # 이월 표본이 자리 잡으면 320 ms → 32프레임
    assert torch.allclose(streamed, ref[0], atol=1e-4)
    assert (5120 - N_FFT) // HOP + 1 == 30  # 맨 첫 청크에는 240표본 이월분이 없다


# 확인(게이트 3): autocast 아래에서도 CTC에 넘기는 log-prob이 float32이고 손실(중간 헤드 포함)이 유한하다.
# CUDA면 fp16, CPU면 CPU autocast가 지원하는 bf16으로 같은 성질을 본다. 왜: fp16 CTC는 오류나 수치 문제를 낼
# 수 있어 'CTC 입력 fp32'를 사전 등록 규칙으로 정했다.
def test_ctc_input_is_fp32_under_autocast() -> None:
    model = make_model()
    mel = random_mel(64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        logits, inter, lengths = model(mel.to(device), torch.tensor([64], device=device), ChunkSpec(0, -1))
        assert log_probs_fp32(logits).dtype == torch.float32
        targets = torch.tensor([3, 4, 5], device=device)
        loss, parts = total_loss(logits, inter, lengths, targets, torch.tensor([3], device=device), 0.5)
    assert torch.isfinite(loss)
    assert "ctc_inter" in parts


# 확인: 같은 토큰이 청크 경계(8번째 프레임)를 가로질러 이어지게 만든 로짓에서, 청크마다 GreedyState.push한
# 결과가 배치 greedy와 같다. 왜: 청크 사이에 직전 토큰을 넘기지 않으면 경계에서 같은 음절이 두 번 나온다.
def test_greedy_state_matches_batch_decoding() -> None:
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 40, VOCAB, generator=g)
    logits[0, 6:10, 0] -= 10  # 8번 프레임의 청크 경계를 가로질러 같은 비blank 토큰이 이어지게 만든다
    logits[0, 6:10, 3] += 10
    ref = greedy_batch(logits, torch.tensor([40]))[0]
    state = GreedyState()
    for i in range(0, 40, 8):
        state.push(logits[0, i : i + 8], i)
    assert state.tokens == ref


# 확인(게이트 4): 무작위 mel 20개와 무작위 정답으로 250 step 학습한다. 왜: 모델·손실·역전파·디코딩이 처음부터
# 끝까지 연결돼 실제로 학습되는지 보는 가장 싼 확인이다.
def test_overfit_tiny_dataset() -> None:
    """합성 발화 20개에서 오류 0(greedy 결과 == 정답)에 도달해야 한다: 학습 경로 전체가 동작한다."""
    torch.manual_seed(0)
    model = ChunkConformerCTC(small_cfg(dropout=0.0, sc_layers=(2,))).train()
    g = torch.Generator().manual_seed(4)
    n, t_mel = 20, 96
    mels = torch.randn(n, t_mel, 80, generator=g)
    labels = [[int(x) for x in torch.randint(3, VOCAB, (5,), generator=g)] for _ in range(n)]
    targets = torch.tensor([tok for seq in labels for tok in seq])
    target_lengths = torch.tensor([len(seq) for seq in labels])
    frames = torch.full((n,), t_mel)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(250):
        logits, inter, lengths = model(mels, frames, ChunkSpec(0, -1))
        loss, _ = total_loss(logits, inter, lengths, targets, target_lengths, 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, _, lengths = model(mels, frames, ChunkSpec(0, -1))
    hyps = greedy_batch(logits, lengths)
    assert hyps == labels


# unfold로 프레임마다 k개 탭을 펼치고, 양방향 절반은 '청크 끝과 신호 끝을 넘지 않는 탭'만 허용 마스크로 남겨
# 곱한다. 실제 구현(grouped conv1d + 오른쪽 탭 보정)과 계산 방식이 달라 대조에 쓸 수 있다.
def _reference_split_conv(conv, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """분할 depthwise conv의 독립 unfold 기준 구현(오프라인, 왼쪽 과거는 0)."""
    b, c, t = x.shape
    k, half, cc = conv.kernel, conv.half, conv.causal_channels
    w, bias = conv.weight, conv.bias
    xc = torch.nn.functional.pad(x, (k - 1, 0)).unfold(2, k, 1)  # (B, C, T, k)
    out_c = (xc[:, :cc] * w[:cc, None, :]).sum(-1)
    xb = torch.nn.functional.pad(x, (half, half)).unfold(2, k, 1)[:, cc:]
    idx = torch.arange(t)
    end = (idx // chunk_size + 1) * chunk_size - 1 if chunk_size > 0 else torch.full_like(idx, t - 1)
    taps = torch.arange(k) - half
    allowed = ((idx[:, None] + taps[None, :]) <= end[:, None]) & ((idx[:, None] + taps[None, :]) <= t - 1)
    out_b = (xb * allowed.to(xb.dtype)[None, None] * w[cc:, None, :]).sum(-1)
    return torch.cat([out_c, out_b], dim=1) + bias[None, :, None]


# 확인: 빠른 구현으로 바꾼 분할 conv가 독립 unfold 구현과 1e-5 안에서 같다(전체 문맥 0과 청크 4/8/16, BACON
# 켬/끔). 왜: 0단계에서 속도 때문에 구현을 바꿨으므로 출력이 바뀌지 않았음을 보장한다(docs/experiments.md
# 0단계 결과).
@pytest.mark.parametrize("chunk", [0, 4, 8, 16])
@pytest.mark.parametrize("bacon", [True, False])
def test_split_conv_matches_reference(chunk: int, bacon: bool) -> None:
    from stt_scratch.model.conv import SplitDepthwiseConv1d

    torch.manual_seed(3)
    conv = SplitDepthwiseConv1d(16, 15, bacon)
    x = torch.randn(2, 16, 45)
    with torch.no_grad():
        got = conv(x, chunk)
        ref = _reference_split_conv(conv, x, chunk)
    assert torch.allclose(got, ref, atol=1e-5)


# 어텐션 캐시가 정확히 64프레임으로 잘리는지도 본다. 왜: 잘라 내는 위치가 하나만 어긋나도 긴 발화에서만
# 나타나는 캐시 버그가 된다.
def test_streaming_cache_truncation_long_utterance() -> None:
    """좌측 문맥(64프레임)보다 청크가 많아도, 잘라 낸 캐시로 오프라인 마스크 결과가 재현돼야 한다."""
    model = make_model()
    spec = ChunkSpec(8, 64)
    n_chunks = 14  # 112프레임 > 좌측 문맥 64
    t_mel = 32 * n_chunks
    mel = random_mel(t_mel, seed=21)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(n_chunks)], dim=0)
    assert state.attn_caches[0].shape[1] == 64
    assert max_abs(streamed, offline) <= 1e-5


# 허용 차이는 1e-5다. 왜: 서비스가 실제로 타는 경로 전체(특징 경계 프레임 포함)를 등록 게이트로 확인한다.
def test_streaming_raw_audio_path_matches_offline() -> None:
    """파형 → StreamingFbank(320 ms 패킷, 첫 패킷 5,360표본) → 캐시 스트리밍이 오프라인 결과와 같다.

    오프라인은 fbank + 청크 마스크 forward다. 첫 패킷 뒤의(마지막을 뺀) 모든 패킷이 정확히 32프레임을 채운다.
    """
    model = make_model()
    spec = ChunkSpec(8, 64)
    logmel = LogMel()
    g = torch.Generator().manual_seed(13)
    wav = torch.randn(16000 * 3 + 700, generator=g) * 0.1
    with torch.no_grad():
        mel, frames = logmel(wav[None], torch.tensor([wav.numel()]))
        offline = log_probs_fp32(model(mel, frames, spec)[0])[0]
        fb = StreamingFbank(logmel)
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        pending = torch.zeros(0, 80)
        parts, frames_per_packet = [], []
        pos, packet = 0, 5120 + FIRST_PACKET_EXTRA
        while pos < wav.numel():
            new = fb.push(wav[pos : pos + packet])
            frames_per_packet.append(new.shape[0])
            pos += packet
            packet = 5120
            pending = torch.cat([pending, new], dim=0)
            while pending.shape[0] >= 32:
                parts.append(enc.step(pending[:32], state))
                pending = pending[32:]
        if pending.shape[0]:
            parts.append(enc.step(pending, state))
        streamed = torch.cat(parts, dim=0)
    assert all(n == 32 for n in frames_per_packet[:-1])
    assert streamed.shape == offline.shape
    assert max_abs(streamed, offline) <= 1e-5  # 등록된 fp32 게이트, fbank 경계 프레임 포함


# 확인: 좌측 문맥 0인 스트리밍 설정은 거부된다(독립 검토 반영).
def test_streaming_rejects_zero_left_context() -> None:
    with pytest.raises(ValueError):
        StreamingEncoder(make_model(), ChunkSpec(8, 0))


# sc_proj가 없는지도 본다. 왜: 3단계 (a)와 최종 모델이 이 설정이다.
def test_sc_feedback_off_keeps_intermediate_heads() -> None:
    """InterCTC 단독: softmax 피드백 없이 중간 손실(중간 로짓)은 남고, 스트리밍도 여전히 정확하다."""
    model = make_model(sc_feedback=False)
    assert model.sc_proj is None
    spec = ChunkSpec(8, 64)
    mel = random_mel(64, seed=23)
    with torch.no_grad():
        logits, inter, _ = model(mel, torch.tensor([64]), spec)
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(2)], dim=0)
    assert len(inter) == 1
    assert max_abs(streamed, log_probs_fp32(logits)[0]) <= 1e-5


# 확인: SpecAugment가 마지막 주파수 bin과 마지막 프레임도 가릴 수 있고(시작 위치 범위가 양끝 포함), 한 번에
# 전부를 지우지는 않는다. 왜: 범위를 하나 작게 잡는 실수를 독립 검토에서 지적받아
# 고쳤다(docs/ai-verification.md).
def test_spec_augment_can_mask_last_bin_and_last_frame() -> None:
    g = torch.Generator().manual_seed(0)
    mel = torch.ones(1, 100, 80)
    frames = torch.tensor([100])
    last_bin = any(
        bool((spec_augment(mel, frames, 2, 27, 0, 0.05, g)[0, :, 79] == 0).any()) for _ in range(300)
    )
    last_frame = any(
        bool((spec_augment(mel, frames, 0, 27, 5, 0.05, g)[0, 99, :] == 0).any()) for _ in range(300)
    )
    assert last_bin and last_frame
    out = spec_augment(mel, frames, 2, 27, 5, 0.05, g)
    assert out.shape == mel.shape and (out[0, 50:] == 1).any()  # 전부 지워지지는 않는다


# 확인: <unk>가 U+318D 한 글자로 디코딩되고 "unk" 글자가 나오지 않으며, CER_ns에서 <unk> 두 개가 오류 2개로
# 센다(규칙 변경 7).
def test_unk_decodes_to_a_scored_placeholder() -> None:
    from stt_scratch.metrics import score_ns
    from stt_scratch.text import UNK_CHAR, Vocab

    vocab = Vocab(("<blank>", "<unk>", " ", "\uac00"))
    hyp = vocab.decode([3, 0, 1, 1, 2, 3])
    assert hyp == "\uac00" + UNK_CHAR + UNK_CHAR + " " + "\uac00"
    assert "unk" not in hyp
    scored = score_ns("\uac00 \uac00", hyp)
    assert scored is not None and scored.edits == 2  # <unk> 두 개는 삽입 오류 두 글자로 센다


# 청크 뒤 값을 2048(fp16 덧셈에서 작은 값을 삼킬 만큼 큰 값)로 바꾼 입력과 원래 입력의 첫 청크 출력을 함께
# 돌려준다.
def _reference_split_conv_future_invariance(conv, x: torch.Tensor, chunk: int):
    y = x.clone()
    y[:, :, chunk:] = 2048.0  # 미래 탭을 더했다 빼면 fp16 덧셈에서 작은 값을 삼켜 버릴 만큼 큰 값
    return conv(x, chunk)[:, :, :chunk], conv(y, chunk)[:, :, :chunk]


# 확인: fp16에서 분할 conv의 첫 청크 출력이 청크 뒤 값과 무관하게 비트 단위로 같다. 왜: 오른쪽 탭을 '더한 뒤
# 빼는' 구현은 fp16 반올림에 미래 값이 섞여 이 성질이 깨졌다. '가린 뒤 더하는' 구현으로 바꾼 것을 고정한다.
@needs_cuda
def test_split_conv_future_invariant_fp16() -> None:
    from stt_scratch.model.conv import SplitDepthwiseConv1d

    torch.manual_seed(5)
    conv = SplitDepthwiseConv1d(16, 15, True).to(
        "cuda", torch.float16
    )  # self.half 속성이 .half()를 가리므로 .to로 바꾼다
    x = (torch.randn(1, 16, 32) * 4).to("cuda", torch.float16)
    with torch.no_grad():
        a, b = _reference_split_conv_future_invariance(conv, x, 8)
    assert torch.equal(a, b)


# 실제 크기는 12층, d=256, 1,288 클래스이고 허용 차이는 1e-3이다.
@needs_cuda
def test_no_future_leakage_fp16_full_size() -> None:
    """실제 크기 모델을 fp16 autocast로 돌려도 청크 경계 앞 프레임은 뒤쪽 음성에 영향받지 않는다."""
    torch.manual_seed(0)
    model = ChunkConformerCTC(ModelConfig(vocab_size=1288)).cuda().eval()
    spec = ChunkSpec(8, 128)
    t_mel = 32 * 6
    mel = random_mel(t_mel, seed=31).cuda()
    frames = torch.tensor([t_mel], device="cuda")
    noisy = mel.clone()
    noisy[:, 32 * 3 :] = torch.randn_like(noisy[:, 32 * 3 :]) * 50
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        ref = log_probs_fp32(model(mel, frames, spec)[0])[0]
        out = log_probs_fp32(model(noisy, frames, spec)[0])[0]
    assert max_abs(out[:24], ref[:24]) <= 1e-3
    assert max_abs(out[24:], ref[24:]) > 1e-3


# 비교는 log-prob 최대 차이로 한다. 실측은 3.4e-3~4.0e-3였다(docs/experiments.md 0단계 결과 보강).
@needs_cuda
def test_streaming_equals_offline_fp16_full_size() -> None:
    """등록 게이트: fp16 캐시 스트리밍과 오프라인 청크 마스크가 1e-2 안에서 같다.

    12층, d=256, 1,288 클래스. 25청크라 128프레임 캐시 절단도 포함된다.
    """
    torch.manual_seed(0)
    model = ChunkConformerCTC(ModelConfig(vocab_size=1288)).cuda().eval()
    spec = ChunkSpec(8, 128)
    n_chunks = 25
    t_mel = 32 * n_chunks
    mel = random_mel(t_mel, seed=17).cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel], device="cuda"), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(n_chunks)], dim=0)
    assert state.attn_caches[0].shape[1] == 128
    assert max_abs(streamed, offline) <= 1e-2


# 배치·패딩·메모리 배치 문제가 끼어들 여지가 없는 기준 계산이다.
def _per_utterance_ctc_reference(
    log_probs: torch.Tensor, out_lengths, targets, target_lengths
) -> torch.Tensor:
    """기준 CTC: 발화마다 패딩 없는 자기 길이의 (T_b, 1, V) 조각으로 따로 계산한 CTC 손실의 합."""
    total = log_probs.new_zeros(())
    off = 0
    for b in range(log_probs.shape[0]):
        t_b, u_b = int(out_lengths[b]), int(target_lengths[b])
        total = total + torch.nn.functional.ctc_loss(
            log_probs[b, :t_b].unsqueeze(1),
            targets[off : off + u_b].unsqueeze(0),
            torch.tensor([t_b], dtype=torch.int32),
            torch.tensor([u_b], dtype=torch.int32),
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        off += u_b
    return total


# 확인: 패딩된 배치에서 ctc_loss의 손실(1e-3 안)과 로짓 기울기(1e-4 안)가 발화별 기준과 같고, 기울기가
# 유한하며, 패딩 프레임의 기울기는 정확히 0이다. 왜: 1단계 첫 시도를 무효로 만든 CUDA CTC backward 문제(비연속
# 뷰, 초기화되지 않은 패딩 기울기)의 회귀 테스트다.
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_ctc_loss_matches_per_utterance_reference(device: str) -> None:
    """1단계 NaN의 회귀 테스트: 패딩된 배치, 연속 메모리 (T,B,V) 입력, 패딩한 (B,S) 정답.

    손실과 로짓 기울기가 발화별 기준과 같아야 한다(버그 경로는 최대 55까지 달랐다).
    """
    from stt_scratch.model.ctc import ctc_loss

    g = torch.Generator().manual_seed(6)
    out_lengths = torch.tensor([50, 37, 12], device=device)
    target_lengths = torch.tensor([9, 14, 3], device=device)
    logits = (torch.randn(3, 50, VOCAB, generator=g) * 3).to(device)
    targets = torch.randint(3, VOCAB, (int(target_lengths.sum()),), generator=g).to(device)
    x = logits.clone().requires_grad_(True)
    loss = ctc_loss(x, out_lengths, targets, target_lengths)
    loss.backward()
    y = logits.clone().requires_grad_(True)
    ref = _per_utterance_ctc_reference(log_probs_fp32(y), out_lengths, targets, target_lengths)
    ref.backward()
    assert torch.isfinite(loss) and abs(float(loss.detach()) - float(ref.detach())) <= 1e-3
    assert torch.isfinite(x.grad).all()
    assert max_abs(x.grad, y.grad) <= 1e-4
    for b, t_b in enumerate(out_lengths.tolist()):  # 패딩 프레임의 기울기는 정확히 0
        assert float(x.grad[b, t_b:].abs().max()) == 0.0 if t_b < 50 else True


# 확인: <unk>가 CER_ws에서도 오류 1자로 센다(가설 쪽 정규화가 U+318D를 남김). 이전에는 CER_ws에서 <unk>가
# 공백으로 사라졌다(적대적 검토에서 발견).
def test_unk_counts_as_an_error_in_cer_ws_too() -> None:
    from stt_scratch.metrics import score_ns, score_ws
    from stt_scratch.text import UNK_CHAR

    ref = "\uac00 \ub098"
    hyp = "\uac00" + UNK_CHAR + " \ub098"
    assert score_ns(ref, hyp).edits == 1
    assert score_ws(ref, hyp).edits == 1
    assert score_ws(ref, ref).edits == 0


# E-Branchformer 작은 설정(3단계 (b) 테스트용). cgMLP 폭 64, 커널은 실제와 같은 31이라 캐시 길이가 실제와
# 같다.
EB = dict(block="ebranchformer", cgmlp_units=64, cgmlp_kernel=31, merge_kernel=31)


# BACON 켬/끔 둘 다 본다. 왜: 3단계 (b)도 Run A와 같은 스트리밍 조건에서 비교해야 공정하다.
@pytest.mark.parametrize("chunk", [4, 8, 16])
@pytest.mark.parametrize("bacon", [True, False])
def test_ebranchformer_no_future_leakage(chunk: int, bacon: bool) -> None:
    """3단계 후보 (b): E-Branchformer의 cgMLP·병합 conv도 청크 경계를 넘지 않아야 한다."""
    model = make_model(bacon=bacon, **EB)
    spec = ChunkSpec(chunk, 64)
    t_mel = 4 * chunk * 6
    mel = random_mel(t_mel, seed=51)
    frames = torch.tensor([t_mel])
    with torch.no_grad():
        ref, _, _ = model(mel, frames, spec)
        for c in range(1, 6):
            noisy = mel.clone()
            noisy[:, 4 * chunk * c :] = torch.randn_like(noisy[:, 4 * chunk * c :]) * 5
            out, _, _ = model(noisy, frames, spec)
            assert max_abs(out[:, : chunk * c], ref[:, : chunk * c]) <= 1e-5, (chunk, c)
            assert max_abs(out[:, chunk * c :], ref[:, chunk * c :]) > 1e-3


# E-Branchformer에서 확인하며, conv 캐시가 블록마다 튜플(두 개)인지도 본다.
@pytest.mark.parametrize("bacon", [True, False])
def test_ebranchformer_streaming_equals_offline(bacon: bool) -> None:
    """캐시 스트리밍 == 오프라인 청크 마스크(64프레임 캐시 절단과 마지막 부분 청크 포함)."""
    model = make_model(bacon=bacon, **EB)
    spec = ChunkSpec(8, 64)
    n_full = 12
    t_mel = 32 * n_full + 19
    mel = random_mel(t_mel, seed=52)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        parts = [enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(n_full)]
        parts.append(enc.step(mel[0, n_full * 32 :], state))
        streamed = torch.cat(parts, dim=0)
    assert isinstance(state.conv_caches[0], tuple) and state.attn_caches[0].shape[1] == 64
    assert streamed.shape == offline.shape
    assert max_abs(streamed, offline) <= 1e-5


# 확인: E-Branchformer에서도 배치 패딩이 짧은 발화의 출력을 바꾸지 않고 NaN이 없다.
def test_ebranchformer_padding_does_not_change_valid_frames() -> None:
    model = make_model(**EB)
    spec = ChunkSpec(8, 64)
    a, b = random_mel(4 * 8 * 5, seed=53)[0], random_mel(4 * 8 * 3, seed=54)[0]
    batch = torch.zeros(2, a.shape[0], 80)
    batch[0] = a
    batch[1, : b.shape[0]] = b
    with torch.no_grad():
        out, _, lengths = model(batch, torch.tensor([a.shape[0], b.shape[0]]), spec)
        alone, _, _ = model(b[None], torch.tensor([b.shape[0]]), spec)
    assert torch.isfinite(out).all()
    assert max_abs(out[1, : int(lengths[1])], alone[0]) <= 1e-5


# 3단계 (d)에서 쓴 항이다. 패딩 값을 바꿔도 손실이 그대로인지도 본다.
def test_cr_consistency_properties() -> None:
    """CR-CTC 항: 두 뷰가 같으면 0, 다르면 양수, 패딩 프레임은 무시한다.

    기울기는 각 방향의 detach되지 않은 쪽을 통해 두 뷰 모두에 간다.
    """
    from stt_scratch.model.ctc import cr_consistency

    g = torch.Generator().manual_seed(7)
    view = torch.randn(3, 20, VOCAB, generator=g)
    lengths = torch.tensor([20, 15, 9, 20, 15, 9])
    same = cr_consistency(torch.cat([view, view]), lengths)
    assert abs(float(same)) < 1e-6
    other = torch.randn(3, 20, VOCAB, generator=g)
    x = torch.cat([view, other]).requires_grad_(True)
    loss = cr_consistency(x, lengths)
    assert float(loss.detach()) > 0
    loss.backward()
    assert float(x.grad[:3].abs().sum()) > 0 and float(x.grad[3:].abs().sum()) > 0
    assert float(x.grad[1, 15:].abs().max()) == 0.0 and float(x.grad[4, 15:].abs().max()) == 0.0
    changed = other.clone()
    changed[2, 9:] += 100.0  # 발화 2의 유효 길이 밖
    assert abs(float(cr_consistency(torch.cat([view, changed]), lengths)) - float(loss)) < 1e-3


# 확인(4단계 Peak-First 항): 마지막 유효 프레임은 목표로만 쓰여 기울기가 없고, 앞 프레임들에는 기울기가 간다.
def test_peak_first_regularizer_properties() -> None:
    """4단계: 이웃 프레임 분포가 같으면 0, 패딩은 무시, 기울기는 짝 가운데 앞 프레임에만 간다."""
    from stt_scratch.model.ctc import peak_first_regularizer

    g = torch.Generator().manual_seed(8)
    flat = torch.randn(2, 1, VOCAB, generator=g).expand(2, 12, VOCAB).contiguous()
    assert abs(float(peak_first_regularizer(flat, torch.tensor([12, 7])))) < 1e-6
    x = torch.randn(2, 12, VOCAB, generator=g).requires_grad_(True)
    lengths = torch.tensor([12, 7])
    loss = peak_first_regularizer(x, lengths)
    assert float(loss.detach()) > 0
    loss.backward()
    assert float(x.grad[0, -1].abs().max()) == 0.0  # 마지막 유효 프레임은 (detach된) 목표일 뿐이다
    assert float(x.grad[1, 6:].abs().max()) == 0.0  # 프레임 6이 마지막 유효 프레임, 7부터는 패딩
    assert float(x.grad[0, :-1].abs().sum()) > 0


# 확인(4단계 TrimTail): 여러 번 뽑아도 새 길이가 원래 길이 이하·CTC 최소 프레임 이상이고 절단 폭이 20프레임
# 이하이며, 남은 구간은 원래 값 그대로, 잘린 구간은 패딩 값 log(LOG_EPS)로 덮인다. 최소 프레임과 길이가 같은
# 발화(세 번째)는 전혀 잘리지 않는다.
def test_trim_tail_keeps_ctc_condition_and_pads() -> None:
    import random

    from stt_scratch.model.features import LOG_EPS, trim_tail

    mel = torch.randn(3, 100, 80)
    frames = torch.tensor([100, 90, 40])
    need = torch.tensor([30, 85, 40])  # 발화 1은 많아야 5프레임, 발화 2는 한 프레임도 잘릴 수 없다
    rng = random.Random(3)
    for _ in range(20):
        out, new = trim_tail(mel, frames, 20, need, rng)
        assert bool((new <= frames).all()) and bool((new >= need).all())
        assert bool((frames - new <= 20).all())
        for i in range(3):
            n = int(new[i])
            assert torch.equal(out[i, :n], mel[i, :n])
            assert bool((out[i, n : int(frames[i])] == float(torch.log(torch.tensor(LOG_EPS)))).all())
    assert int(new[2]) == 40
