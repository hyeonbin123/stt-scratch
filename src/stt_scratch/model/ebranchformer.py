"""Chunk-causal E-Branchformer block (Kim et al., SLT 2022, arXiv 2210.00077), Stage 3 arm (b).

Macaron FFN(1/2) -> [MHSA branch | cgMLP branch] -> merge (concat -> depthwise conv -> add -> linear) ->
FFN(1/2) -> LayerNorm. Both depthwise convolutions (the cgMLP gate and the merge) use the same split
depthwise convolution as the Conformer conv module (half causal, half bounded by the chunk end), so the
chunk mask, the future-leakage gates and exact cache streaming carry over unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from stt_scratch.model.attention import RelPosSelfAttention
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conv import SplitDepthwiseConv1d
from stt_scratch.model.mask import ChunkSpec


def _cached_depthwise(
    conv: SplitDepthwiseConv1d,
    y: torch.Tensor,
    chunk_size: int,
    left_cache: torch.Tensor | None,
    frame_offset: int,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y: (B, C, T). Zeroes padded frames, runs the split conv and returns (output, new k-1 frame cache)."""
    if valid is not None:
        y = y * valid[:, None, :].to(y.dtype)
    k1 = conv.kernel - 1
    new_cache = torch.cat([left_cache, y] if left_cache is not None else [y], dim=2)[:, :, -k1:]
    if new_cache.shape[2] < k1:
        new_cache = nn.functional.pad(new_cache, (k1 - new_cache.shape[2], 0))
    return conv(y, chunk_size, left_cache, frame_offset), new_cache


class ConvolutionalGatingMLP(nn.Module):
    """cgMLP: LN -> Linear(d, U) -> GELU -> split (r, g); g: LN -> depthwise conv; r * g -> Linear(U/2, d)."""

    def __init__(self, d_model: int, units: int, kernel: int, dropout: float, bacon: bool) -> None:
        super().__init__()
        assert units % 2 == 0
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, units)
        self.act = nn.GELU()
        self.gate_norm = nn.LayerNorm(units // 2)
        self.conv = SplitDepthwiseConv1d(units // 2, kernel, bacon)
        self.proj2 = nn.Linear(units // 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        left_cache: torch.Tensor | None = None,
        frame_offset: int = 0,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.act(self.proj1(self.norm(x)))
        r, g = y.chunk(2, dim=-1)
        g = self.gate_norm(g).transpose(1, 2)  # (B, U/2, T)
        g, new_cache = _cached_depthwise(self.conv, g, chunk_size, left_cache, frame_offset, valid)
        return self.dropout(self.proj2(r * g.transpose(1, 2))), new_cache


class EBranchformerBlock(nn.Module):
    """Same interface as ConformerBlock (forward, forward_streaming, init_conv_cache)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # attribute names shared with ConformerBlock (ffn1, attn_norm, attn) so reports can inspect both
        from stt_scratch.model.conformer import FeedForward

        d = cfg.d_model
        self.ffn1 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = RelPosSelfAttention(d, cfg.n_heads, cfg.dropout, cfg.rope_theta, cfg.qk_norm)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.cgmlp = ConvolutionalGatingMLP(d, cfg.cgmlp_units, cfg.cgmlp_kernel, cfg.dropout, cfg.bacon)
        self.merge_conv = SplitDepthwiseConv1d(2 * d, cfg.merge_kernel, cfg.bacon)
        self.merge_proj = nn.Linear(2 * d, d)
        self.merge_dropout = nn.Dropout(cfg.dropout)
        self.ffn2 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.final_norm = nn.LayerNorm(d)

    def init_conv_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            x.new_zeros(1, self.cgmlp.conv.weight.shape[0], self.cgmlp.conv.kernel - 1),
            x.new_zeros(1, self.merge_conv.weight.shape[0], self.merge_conv.kernel - 1),
        )

    def _merge(
        self,
        x: torch.Tensor,
        att: torch.Tensor,
        chunk_size: int,
        caches: tuple[torch.Tensor, torch.Tensor] | None,
        frame_offset: int,
        valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cg_cache, merge_cache = caches if caches is not None else (None, None)
        local, new_cg = self.cgmlp(x, chunk_size, cg_cache, frame_offset, valid)
        cat = torch.cat([self.attn_dropout(att), local], dim=-1)  # (B, T, 2d)
        fused, new_merge = _cached_depthwise(
            self.merge_conv, cat.transpose(1, 2), chunk_size, merge_cache, frame_offset, valid
        )
        x = x + self.merge_dropout(self.merge_proj(cat + fused.transpose(1, 2)))
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x), (new_cg, new_merge)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        att = self.attn(self.attn_norm(x), mask, positions)
        out, _ = self._merge(x, att, chunk_size, None, 0, valid)
        return out

    def forward_streaming(
        self,
        x: torch.Tensor,
        attn_cache: torch.Tensor,
        conv_cache: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        spec: ChunkSpec,
        frame_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = x + 0.5 * self.ffn1(x)
        q_in = self.attn_norm(x)
        kv = torch.cat([attn_cache, q_in], dim=1)
        t_c, n_cached = x.shape[1], attn_cache.shape[1]
        kv_positions = torch.arange(frame_offset - n_cached, frame_offset + t_c, device=x.device)
        cq = positions // spec.chunk_size
        ck = kv_positions // spec.chunk_size
        allowed = ck[None, :] <= cq[:, None]
        if spec.left_chunks >= 0:
            allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
        att = self.attn(q_in, allowed[None, None], positions, kv=kv, kv_positions=kv_positions)
        out, caches = self._merge(x, att, spec.chunk_size, conv_cache, frame_offset, None)
        return out, kv, caches
