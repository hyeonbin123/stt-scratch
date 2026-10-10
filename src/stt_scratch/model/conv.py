"""Conformer convolution module with a BACON-style split depthwise convolution.

Half of the channels are strictly causal (taps i-k+1..i). The other half may also look right, but only
up to the end of the frame's own chunk (taps i-(k-1)/2 .. min(i+(k-1)/2, chunk_end)). With an unlimited
chunk this half is an ordinary symmetric depthwise convolution. Both halves see the past across chunk
boundaries, so streaming with a left cache of k-1 frames reproduces the offline result exactly.
"""

from __future__ import annotations

import torch
from torch import nn


def chunk_end_index(t: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    """For every frame i, the last frame index of its chunk (t-1 when chunk_size <= 0)."""
    idx = torch.arange(t, device=device)
    if chunk_size <= 0:
        return torch.full_like(idx, t - 1)
    return torch.clamp((idx // chunk_size + 1) * chunk_size - 1, max=t - 1)


class SplitDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel: int, bacon: bool) -> None:
        super().__init__()
        assert kernel % 2 == 1
        self.kernel = kernel
        self.half = kernel // 2
        self.bacon = bacon
        self.causal_channels = channels // 2 if bacon else channels
        self.weight = nn.Parameter(torch.empty(channels, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(
        self, x: torch.Tensor, chunk_size: int, left_cache: torch.Tensor | None = None, frame_offset: int = 0
    ) -> torch.Tensor:
        """x: (B, C, T). left_cache: (B, C, k-1) past frames for streaming (zeros when None).
        frame_offset: absolute index of x[..., 0] (so chunk boundaries stay aligned in streaming)."""
        b, c, t = x.shape
        k = self.kernel
        cache = x.new_zeros(b, c, k - 1) if left_cache is None else left_cache
        # causal half: windows ending at i
        xc = torch.cat([cache, x], dim=2)  # (B, C, T + k - 1)
        win_c = xc.unfold(2, k, 1)  # (B, C, T, k) window j covers frames i-k+1 .. i
        out_c = (win_c[:, : self.causal_channels] * self.weight[: self.causal_channels, None, :]).sum(-1)
        if self.causal_channels == c:
            return out_c + self.bias[None, :, None]
        # chunk-bounded bidirectional half: windows centred at i, right taps masked beyond the chunk end
        xb = torch.cat([cache[:, :, -self.half :], x, x.new_zeros(b, c, self.half)], dim=2)
        win_b = xb.unfold(2, k, 1)[:, self.causal_channels :]  # (B, C/2, T, k): tap j -> frame i - half + j
        idx = torch.arange(t, device=x.device) + frame_offset
        if chunk_size > 0:
            end = (idx // chunk_size + 1) * chunk_size - 1  # absolute chunk end per frame
        else:
            end = torch.full_like(idx, idx[-1] + 10**9)
        taps = torch.arange(k, device=x.device) - self.half  # relative tap offsets
        allowed = (idx[:, None] + taps[None, :]) <= end[:, None]  # (T, k)
        allowed &= (idx[:, None] + taps[None, :]) <= (frame_offset + t - 1)  # nothing beyond the given input
        win_b = win_b * allowed.to(win_b.dtype)[None, None]
        out_b = (win_b * self.weight[self.causal_channels :, None, :]).sum(-1)
        return torch.cat([out_c, out_b], dim=1) + self.bias[None, :, None]


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float, bacon: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = SplitDepthwiseConv1d(d_model, kernel, bacon)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.cache_frames = kernel - 1

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        left_cache: torch.Tensor | None = None,
        frame_offset: int = 0,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D); valid: (B, T) bool, padded frames are zeroed before the depthwise conv.
        Returns (output, new left cache (B, D, k-1) taken from the post-GLU signal)."""
        y = self.norm(x)
        y = nn.functional.glu(self.pointwise1(y), dim=-1).transpose(1, 2)  # (B, D, T)
        if valid is not None:
            y = y * valid[:, None, :].to(y.dtype)
        new_cache = torch.cat([left_cache, y] if left_cache is not None else [y], dim=2)[
            :, :, -self.cache_frames :
        ]
        if new_cache.shape[2] < self.cache_frames:
            new_cache = nn.functional.pad(new_cache, (self.cache_frames - new_cache.shape[2], 0))
        y = self.depthwise(y, chunk_size, left_cache, frame_offset).transpose(1, 2)
        y = self.pointwise2(self.act(self.norm2(y)))
        return self.dropout(y), new_cache
