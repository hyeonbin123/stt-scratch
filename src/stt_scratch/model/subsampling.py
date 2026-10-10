"""Causal Conv2d subsampling (4x in time): output frame i sees input frames <= 4i + 3, so the
front-end adds no look-ahead beyond the current chunk. Output length = ceil(ceil(T/2)/2)."""

from __future__ import annotations

import torch
from torch import nn


def subsampled_length(t: torch.Tensor | int) -> torch.Tensor | int:
    if isinstance(t, int):
        return ((t + 1) // 2 + 1) // 2  # ceil(ceil(t/2)/2)
    return torch.div(torch.div(t + 1, 2, rounding_mode="floor") + 1, 2, rounding_mode="floor")


class CausalConv2dSubsampling(nn.Module):
    def __init__(self, n_mels: int, channels: int, d_model: int, causal: bool = True) -> None:
        super().__init__()
        self.causal = causal
        # time padding is applied manually (left only when causal); frequency padding is symmetric
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=(0, 1))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=(0, 1))
        self.act = nn.SiLU()
        freq = (n_mels + 2 - 3) // 2 + 1
        freq = (freq + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * freq, d_model)
        self.cache_frames = 2  # input frames each conv needs from the past

    def _pad_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F). Causal: pad 2 on the left. Non-causal: 1 on each side (standard, 20 ms look-ahead).
        if self.causal:
            return nn.functional.pad(x, (0, 0, 2, 0))
        return nn.functional.pad(x, (0, 0, 1, 1))

    def forward(self, mel: torch.Tensor, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = mel.unsqueeze(1)  # (B, 1, T, F)
        x = self.act(self.conv1(self._pad_time(x)))
        x = self.act(self.conv2(self._pad_time(x)))
        b, c, t, f = x.shape
        x = self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return x, subsampled_length(frames)

    def forward_streaming(
        self, mel: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """mel: (1, T, F) with T a multiple of 4. cache = (last 2 mel frames, last 2 conv1 output frames)."""
        assert self.causal, "streaming needs the causal front-end"
        x = mel.unsqueeze(1)
        if cache is None:
            x1_in = nn.functional.pad(x, (0, 0, 2, 0))
        else:
            x1_in = torch.cat([cache[0], x], dim=2)
        new_mel_cache = x1_in[:, :, -2:]
        x1 = self.act(self.conv1(x1_in))
        if cache is None:
            x2_in = nn.functional.pad(x1, (0, 0, 2, 0))
        else:
            x2_in = torch.cat([cache[1], x1], dim=2)
        new_x1_cache = x2_in[:, :, -2:]
        x2 = self.act(self.conv2(x2_in))
        b, c, t, f = x2.shape
        out = self.out(x2.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return out, (new_mel_cache, new_x1_cache)
