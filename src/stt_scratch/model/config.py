"""Model and streaming configuration (decided in docs/survey/README.md "결정")."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int
    n_mels: int = 80
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 4
    ffn_dim: int = 1024
    conv_kernel: int = 15
    dropout: float = 0.1
    subsampling_channels: int = 256
    causal_subsampling: bool = True
    rope_theta: float = 10000.0
    # LayerNorm on q and k per head before RoPE (bounds the attention logits; see docs/experiments.md
    # 1단계 진단). False = the architecture as first registered.
    qk_norm: bool = False
    # Self-conditioned CTC: 1-based block indices whose output gets an intermediate CTC head
    # and whose softmax is fed back into the next block. Empty tuple = plain CTC.
    sc_layers: tuple[int, ...] = (3, 6, 9)
    sc_weight: float = 0.5  # total = (1 - w) * final + w * mean(intermediate)
    # False = plain InterCTC: the intermediate losses stay, the softmax feedback into the next block is off
    sc_feedback: bool = True
    # Convolution module: True = BACON split (half causal, half chunk-bounded bidirectional);
    # False = all channels causal.
    bacon: bool = True
    # Dynamic chunk training (U2): with prob. full_context_prob the batch sees full context,
    # otherwise a chunk size (frames at 25 fps) and a left-context size (frames) are drawn uniformly.
    full_context_prob: float = 0.5
    chunk_sizes: tuple[int, ...] = (4, 8, 16, 32)
    left_contexts: tuple[int, ...] = (64, 128, -1)  # -1 = unlimited
    # SpecAugment (applied on the log-mel inside the model, training only)
    freq_masks: int = 2
    freq_mask_width: int = 27
    time_masks: int = 5
    time_mask_ratio: float = 0.05
    mel_mean: list[float] = field(default_factory=list)  # global normalization, set in Stage 0
    mel_std: list[float] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
