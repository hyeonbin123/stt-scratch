"""Cache-based streaming inference for ChunkConformerCTC.

The stream state holds, per block, the last L attention inputs (post-LayerNorm) and the last k-1 conv
inputs, plus the subsampling caches. Each chunk is forwarded once; keys/values of cached frames are
re-projected ("input cache + re-projection"), which makes the result identical to an offline forward
with the same chunk mask (checked by tests/test_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.mask import ChunkSpec


@dataclass
class StreamState:
    spec: ChunkSpec
    frames_seen: int = 0  # encoder frames (25 fps) processed so far
    sub_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    attn_caches: list[torch.Tensor] = field(default_factory=list)
    conv_caches: list = field(default_factory=list)  # per block: a tensor or a tuple of tensors


class StreamingEncoder:
    def __init__(self, model: ChunkConformerCTC, spec: ChunkSpec) -> None:
        if spec.full or spec.left_context <= 0:
            raise ValueError("streaming needs a finite chunk size and a positive left context")
        if spec.left_context % spec.chunk_size:
            raise ValueError("left context must be a multiple of the chunk size")
        self.model = model.eval()
        self.spec = spec
        self.device = next(model.parameters()).device

    def new_state(self) -> StreamState:
        return StreamState(spec=self.spec)

    @torch.no_grad()
    def step(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        """mel: (T_mel, 80) raw log-mel frames for one chunk, T_mel = 4 * chunk_size. Returns (chunk_size, V)
        fp32 log-probs for the new encoder frames and updates the state in place."""
        m = self.model
        cfg = m.cfg
        x = m.normalize_mel(mel.to(self.device).unsqueeze(0))
        x, state.sub_cache = m.subsampling.forward_streaming(x, state.sub_cache)
        t_c = x.shape[1]
        offset = state.frames_seen
        positions = torch.arange(offset, offset + t_c, device=self.device)
        if not state.attn_caches:
            d = cfg.d_model
            state.attn_caches = [x.new_zeros(1, 0, d) for _ in m.blocks]
            state.conv_caches = [block.init_conv_cache(x) for block in m.blocks]
        for i, block in enumerate(m.blocks, start=1):
            x, kv, new_conv = block.forward_streaming(
                x, state.attn_caches[i - 1], state.conv_caches[i - 1], positions, self.spec, offset
            )
            state.attn_caches[i - 1] = kv[:, -self.spec.left_context :]
            state.conv_caches[i - 1] = new_conv
            if i in m.sc_layers and i < len(m.blocks) and m.sc_proj is not None:
                probs = torch.softmax(m.out_proj(x).float(), dim=-1).to(x.dtype)
                x = x + m.sc_proj(probs)
        state.frames_seen += t_c
        return log_probs_fp32(m.out_proj(x))[0]
