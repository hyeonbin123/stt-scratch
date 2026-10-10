"""Stage 0 gates (docs/experiments.md 0단계): no future leakage, streaming == offline, CTC dtype, overfit."""

from __future__ import annotations

import math

import pytest
import torch

from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32, total_loss
from stt_scratch.model.decode import GreedyState, greedy_batch
from stt_scratch.model.features import HOP, N_FFT, LogMel, StreamingFbank, n_mel_frames
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.model.subsampling import subsampled_length

VOCAB = 12


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


def make_model(seed: int = 0, **kw) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    return ChunkConformerCTC(small_cfg(**kw)).eval()


def random_mel(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, t, 80, generator=g)


def test_subsampled_length_matches_module() -> None:
    model = make_model()
    for t in (4, 5, 31, 32, 33, 128, 129):
        mel = random_mel(t)
        x, lengths = model.subsampling(mel, torch.tensor([t]))
        assert x.shape[1] == int(lengths[0]) == subsampled_length(t) == math.ceil(math.ceil(t / 2) / 2)


@pytest.mark.parametrize("chunk", [4, 8, 16, 32])
@pytest.mark.parametrize("left", [64, 128])
def test_no_future_leakage(chunk: int, left: int) -> None:
    """Changing audio after a chunk boundary must not change outputs up to that boundary."""
    model = make_model()
    spec = ChunkSpec(chunk, left)
    t_mel = 4 * chunk * 6  # six chunks
    mel = random_mel(t_mel)
    frames = torch.tensor([t_mel])
    with torch.no_grad():
        ref, _, _ = model(mel, frames, spec)
        for c in range(1, 6):
            cut_mel = 4 * chunk * c  # first mel frame that belongs to chunk c
            noisy = mel.clone()
            noisy[:, cut_mel:] = torch.randn_like(noisy[:, cut_mel:]) * 5
            out, _, _ = model(noisy, frames, spec)
            assert torch.allclose(out[:, : chunk * c], ref[:, : chunk * c], atol=1e-5), (chunk, left, c)
            assert not torch.allclose(out[:, chunk * c :], ref[:, chunk * c :], atol=1e-3)


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
    assert torch.allclose(streamed, offline, atol=1e-5), float((streamed - offline).abs().max())


def test_streaming_partial_last_chunk() -> None:
    """A final chunk shorter than 4 * chunk_size frames must still match the offline result."""
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
    assert torch.allclose(streamed, offline, atol=1e-5)


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
    assert torch.allclose(streamed, offline, atol=1e-5)


def test_padding_does_not_change_valid_frames() -> None:
    """A batch with padding gives the same result for the shorter utterance as running it alone."""
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
    assert parts[1].shape[0] == 32  # 320 ms -> 32 frames once the carry is in place
    assert torch.allclose(streamed, ref[0], atol=1e-4)
    assert (5120 - N_FFT) // HOP + 1 == 30  # the very first chunk lacks the 240-sample carry


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


def test_greedy_state_matches_batch_decoding() -> None:
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 40, VOCAB, generator=g)
    logits[0, 10:14, 0] -= 10  # force a run of the same non-blank token across a chunk boundary
    logits[0, 10:14, 3] += 10
    ref = greedy_batch(logits, torch.tensor([40]))[0]
    state = GreedyState()
    for i in range(0, 40, 8):
        state.push(logits[0, i : i + 8], i)
    assert state.tokens == ref


def test_overfit_tiny_dataset() -> None:
    """20 synthetic utterances must reach zero errors: the training path works end to end."""
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


def _reference_split_conv(conv, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Independent unfold implementation of the split depthwise conv (offline, zero left history)."""
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
