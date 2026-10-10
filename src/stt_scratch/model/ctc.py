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
    """Sum over the batch of per-utterance CTC losses (blank = 0, zero_infinity).

    Two safeguards against the native CUDA CTC backward (torch 2.11+cu128, RTX 2080 Ti), both found by
    comparing against a per-utterance reference gradient:
    1. log_probs must be a contiguous (T, B, V) tensor. A transposed view produced gradients that
       differed from the reference by up to 55 in valid frames (non-deterministically) and sometimes NaN.
    2. Frames at or beyond each utterance's out_length are replaced through torch.where; the kernel leaves
       their gradient uninitialized and torch.where's backward drops whatever arrives there.
    Targets are passed as a padded (B, S) int32 CPU tensor and lengths as int32 CPU tensors: this layout
    matched the reference to 1e-7, while 1-D targets with CUDA int32 lengths differed by 1e-3.
    """
    lp = log_probs_fp32(logits)  # (B, T, V)
    valid = torch.arange(lp.shape[1], device=lp.device)[None, :] < out_lengths[:, None]
    lp = torch.where(valid[:, :, None], lp, torch.zeros((), dtype=lp.dtype, device=lp.device))
    lp = lp.transpose(0, 1).contiguous()  # (T, B, V)
    assert lp.dtype == torch.float32 and lp.is_contiguous()
    tl_cpu = target_lengths.to(torch.int32).cpu()
    flat = targets.to(torch.int32).cpu()
    padded_targets = torch.zeros(lp.shape[1], int(tl_cpu.max()), dtype=torch.int32)  # (B, S) layout
    off = 0
    for b, n in enumerate(tl_cpu.tolist()):
        padded_targets[b, :n] = flat[off : off + n]
        off += n
    return nn.functional.ctc_loss(
        lp,
        padded_targets,
        out_lengths.to(torch.int32).cpu(),
        tl_cpu,
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
