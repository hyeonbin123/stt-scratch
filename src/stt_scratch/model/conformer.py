"""Chunk-causal Conformer encoder with self-conditioned CTC heads."""

from __future__ import annotations

import torch
from torch import nn

from stt_scratch.model.attention import RelPosSelfAttention
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conv import ConvModule
from stt_scratch.model.features import spec_augment
from stt_scratch.model.mask import ChunkSpec, attention_mask
from stt_scratch.model.subsampling import CausalConv2dSubsampling


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    """Macaron FFN(1/2) -> MHSA -> Conv -> FFN(1/2) -> LayerNorm, all pre-norm residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ffn1 = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = RelPosSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout, cfg.bacon)
        self.ffn2 = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), mask, positions))
        y, _ = self.conv(x, chunk_size, valid=valid)
        x = x + y
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)

    def forward_streaming(
        self,
        x: torch.Tensor,
        attn_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        positions: torch.Tensor,
        spec: ChunkSpec,
        frame_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (1, Tc, D) current chunk whose first frame is absolute frame `frame_offset`.
        attn_cache: (1, L, D) LayerNorm'ed attention inputs of the L frames before the chunk.
        conv_cache: (1, D, k-1) conv inputs of the k-1 frames before the chunk.
        Returns the block output, the full key/value source (cache + chunk; caller trims it) and the new
        conv cache."""
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
        mask = allowed[None, None]  # (1, 1, Tc, L + Tc)
        attn_out = self.attn(q_in, mask, positions, kv=kv, kv_positions=kv_positions)
        x = x + self.attn_dropout(attn_out)
        y, new_conv_cache = self.conv(x, spec.chunk_size, conv_cache, frame_offset)
        x = x + y
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x), kv, new_conv_cache


class ChunkConformerCTC(nn.Module):
    """Log-mel -> normalization (+SpecAugment in training) -> subsampling -> blocks -> CTC heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        mean = torch.tensor(cfg.mel_mean or [0.0] * cfg.n_mels)
        std = torch.tensor(cfg.mel_std or [1.0] * cfg.n_mels)
        self.register_buffer("mel_mean", mean.float())
        self.register_buffer("mel_std", std.float())
        self.subsampling = CausalConv2dSubsampling(
            cfg.n_mels, cfg.subsampling_channels, cfg.d_model, cfg.causal_subsampling
        )
        self.in_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)  # shared by the final and intermediate heads
        self.sc_layers = set(cfg.sc_layers)
        self.sc_proj = nn.Linear(cfg.vocab_size, cfg.d_model) if self.sc_layers else None

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mel_mean) / self.mel_std

    def forward(
        self, mel: torch.Tensor, frames: torch.Tensor, spec: ChunkSpec, augment: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """mel: (B, T, 80) raw log-mel; frames: valid frames. Returns final logits (B, T', V),
        intermediate logits list and output lengths (B,)."""
        x = self.normalize_mel(mel)
        if augment and self.training:
            x = spec_augment(
                x,
                frames,
                self.cfg.freq_masks,
                self.cfg.freq_mask_width,
                self.cfg.time_masks,
                self.cfg.time_mask_ratio,
            )
        x, out_lengths = self.subsampling(x, frames)
        x = self.in_dropout(x)
        t = x.shape[1]
        mask = attention_mask(out_lengths, t, spec, x.device)
        positions = torch.arange(t, device=x.device)
        valid = positions[None, :] < out_lengths[:, None]
        inter: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks, start=1):
            x = block(x, mask, positions, spec.chunk_size, valid)
            if i in self.sc_layers and i < len(self.blocks):
                logits = self.out_proj(x)
                inter.append(logits)
                probs = torch.softmax(logits.float(), dim=-1).to(x.dtype)
                x = x + self.sc_proj(probs)
        return self.out_proj(x), inter, out_lengths
