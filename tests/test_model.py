"""Stage 0 gates (docs/experiments.md 0단계): no future leakage, streaming == offline, CTC dtype, overfit."""

from __future__ import annotations

import math

import pytest
import torch

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


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """The registered gate is a maximum absolute difference (allclose would also allow a relative term)."""
    return float((a.float() - b.float()).abs().max())


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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
    assert max_abs(streamed, offline) <= 1e-5


def test_streaming_equals_offline_qk_norm() -> None:
    """QK-LayerNorm is frame-local: cache streaming must stay exact and future-blind with it on."""
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
    assert max_abs(streamed, offline) <= 1e-5


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
    logits[0, 6:10, 0] -= 10  # force a run of the same non-blank token across the chunk boundary at 8
    logits[0, 6:10, 3] += 10
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


def test_streaming_cache_truncation_long_utterance() -> None:
    """More chunks than the left context holds: the trimmed caches must still reproduce the offline mask."""
    model = make_model()
    spec = ChunkSpec(8, 64)
    n_chunks = 14  # 112 frames > left context 64
    t_mel = 32 * n_chunks
    mel = random_mel(t_mel, seed=21)
    with torch.no_grad():
        offline = log_probs_fp32(model(mel, torch.tensor([t_mel]), spec)[0])[0]
        enc = StreamingEncoder(model, spec)
        state = enc.new_state()
        streamed = torch.cat([enc.step(mel[0, i * 32 : (i + 1) * 32], state) for i in range(n_chunks)], dim=0)
    assert state.attn_caches[0].shape[1] == 64
    assert max_abs(streamed, offline) <= 1e-5


def test_streaming_raw_audio_path_matches_offline() -> None:
    """Waveform -> StreamingFbank (320 ms packets, 5,360-sample first packet) -> cache streaming equals the
    offline fbank + chunk-mask forward; every packet after the first completes exactly 32 frames."""
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
    assert max_abs(streamed, offline) <= 1e-5  # registered fp32 gate, fbank boundary frames included


def test_streaming_rejects_zero_left_context() -> None:
    with pytest.raises(ValueError):
        StreamingEncoder(make_model(), ChunkSpec(8, 0))


def test_sc_feedback_off_keeps_intermediate_heads() -> None:
    """InterCTC-only ablation: intermediate losses without the softmax feedback, streaming still exact."""
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
    assert out.shape == mel.shape and (out[0, 50:] == 1).any()  # not everything is wiped


def test_unk_decodes_to_a_scored_placeholder() -> None:
    from stt_scratch.metrics import score_ns
    from stt_scratch.text import UNK_CHAR, Vocab

    vocab = Vocab(("<blank>", "<unk>", " ", "\uac00"))
    hyp = vocab.decode([3, 0, 1, 1, 2, 3])
    assert hyp == "\uac00" + UNK_CHAR + UNK_CHAR + " " + "\uac00"
    assert "unk" not in hyp
    scored = score_ns("\uac00 \uac00", hyp)
    assert scored is not None and scored.edits == 2  # the two <unk> count as two inserted characters


def _reference_split_conv_future_invariance(conv, x: torch.Tensor, chunk: int):
    y = x.clone()
    y[:, :, chunk:] = 2048.0  # values that swamp fp16 addition if a future tap were summed then removed
    return conv(x, chunk)[:, :, :chunk], conv(y, chunk)[:, :, :chunk]


@needs_cuda
def test_split_conv_future_invariant_fp16() -> None:
    from stt_scratch.model.conv import SplitDepthwiseConv1d

    torch.manual_seed(5)
    conv = SplitDepthwiseConv1d(16, 15, True).to(
        "cuda", torch.float16
    )  # .half() is shadowed by the attribute
    x = (torch.randn(1, 16, 32) * 4).to("cuda", torch.float16)
    with torch.no_grad():
        a, b = _reference_split_conv_future_invariance(conv, x, 8)
    assert torch.equal(a, b)


@needs_cuda
def test_no_future_leakage_fp16_full_size() -> None:
    """Project-size model under fp16 autocast: frames before a chunk boundary ignore later audio."""
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


@needs_cuda
def test_streaming_equals_offline_fp16_full_size() -> None:
    """Registered gate: fp16 cache streaming vs offline chunk mask within 1e-2 (12 layers, d=256, 1,288
    classes, 25 chunks so the 128-frame cache is truncated)."""
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


def _per_utterance_ctc_reference(
    log_probs: torch.Tensor, out_lengths, targets, target_lengths
) -> torch.Tensor:
    """Sum of per-utterance CTC losses, each computed on its own unpadded (T_b, 1, V) slice."""
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_ctc_loss_matches_per_utterance_reference(device: str) -> None:
    """Regression test for the Stage 1 NaN: padded batch, contiguous (T,B,V) input, padded (B,S) targets.
    Loss and logits gradient must match a per-utterance reference (the buggy path differed by up to 55)."""
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
    for b, t_b in enumerate(out_lengths.tolist()):  # padded frames get exactly zero gradient
        assert float(x.grad[b, t_b:].abs().max()) == 0.0 if t_b < 50 else True


def test_unk_counts_as_an_error_in_cer_ws_too() -> None:
    from stt_scratch.metrics import score_ns, score_ws
    from stt_scratch.text import UNK_CHAR

    ref = "\uac00 \ub098"
    hyp = "\uac00" + UNK_CHAR + " \ub098"
    assert score_ns(ref, hyp).edits == 1
    assert score_ws(ref, hyp).edits == 1
    assert score_ws(ref, ref).edits == 0


EB = dict(block="ebranchformer", cgmlp_units=64, cgmlp_kernel=31, merge_kernel=31)


@pytest.mark.parametrize("chunk", [4, 8, 16])
@pytest.mark.parametrize("bacon", [True, False])
def test_ebranchformer_no_future_leakage(chunk: int, bacon: bool) -> None:
    """Stage 3 arm (b): the cgMLP and merge convolutions must stay inside the chunk too."""
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


@pytest.mark.parametrize("bacon", [True, False])
def test_ebranchformer_streaming_equals_offline(bacon: bool) -> None:
    """Cache streaming == offline chunk mask, with the 64-frame cache truncated and a partial last chunk."""
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


def test_cr_consistency_properties() -> None:
    """CR-CTC term: zero for identical views, positive otherwise, padded frames ignored, gradients reach
    both views through the non-detached side."""
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
    changed[2, 9:] += 100.0  # beyond the valid length of utterance 2
    assert abs(float(cr_consistency(torch.cat([view, changed]), lengths)) - float(loss)) < 1e-3
