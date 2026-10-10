"""Multi-head self-attention with rotary positions and a boolean chunk/padding mask (SDPA)."""

from __future__ import annotations

import torch
from torch import nn


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple:
    """positions: (T,) absolute frame indices. Returns cos, sin (T, head_dim), computed in fp32 then cast."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half))
    angles = positions.float()[:, None] * inv_freq[None, :]  # (T, half)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D). Rotates pairs (x[:half], x[half:])."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, theta: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.theta = theta
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, Tq, D) queries (and keys/values when kv is None). mask: bool (B or 1, 1, Tq, Tk),
        True = attend. kv: (B, Tk, D) key/value source for streaming (cache + chunk); absolute positions."""
        b, tq, d = x.shape
        q = self.qkv(x)[..., :d]
        src = x if kv is None else kv
        kvp = self.qkv(src)[..., d:]
        k, v = kvp.chunk(2, dim=-1)
        tk = src.shape[1]
        q = q.view(b, tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        cos_q, sin_q = rope_cos_sin(positions, self.head_dim, self.theta, q.dtype)
        if kv_positions is None:
            cos_k, sin_k = cos_q, sin_q
        else:
            cos_k, sin_k = rope_cos_sin(kv_positions, self.head_dim, self.theta, k.dtype)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, tq, d)
        return self.out(out)
