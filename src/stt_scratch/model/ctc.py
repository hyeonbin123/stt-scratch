"""CTC losses computed from fp32 log-probabilities (docs/experiments.md "fp16 규칙")."""

from __future__ import annotations

import torch
from torch import nn


def log_probs_fp32(logits: torch.Tensor) -> torch.Tensor:
    """log_softmax in float32 regardless of autocast; this is the only tensor handed to ctc_loss."""
    return nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)


def ctc_loss(
    logits: torch.Tensor, out_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    """Sum over the batch of per-utterance CTC losses (blank = 0, zero_infinity)."""
    lp = log_probs_fp32(logits).transpose(0, 1)  # (T, B, V)
    assert lp.dtype == torch.float32
    return nn.functional.ctc_loss(
        lp,
        targets,
        out_lengths.to(torch.int32),
        target_lengths.to(torch.int32),
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )


def total_loss(
    final_logits: torch.Tensor,
    inter_logits: list[torch.Tensor],
    out_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    sc_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """(1 - w) * final + w * mean(intermediate), normalized by the number of utterances."""
    b = final_logits.shape[0]
    final = ctc_loss(final_logits, out_lengths, targets, target_lengths) / b
    parts = {"ctc_final": float(final.detach())}
    if inter_logits and sc_weight > 0:
        inter = torch.stack(
            [ctc_loss(lg, out_lengths, targets, target_lengths) / b for lg in inter_logits]
        ).mean()
        parts["ctc_inter"] = float(inter.detach())
        loss = (1.0 - sc_weight) * final + sc_weight * inter
    else:
        loss = final
    parts["loss"] = float(loss.detach())
    return loss, parts
