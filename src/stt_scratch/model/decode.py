"""CTC greedy decoding, batched (offline) and incremental (streaming, append-only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def greedy_batch(logits: torch.Tensor, out_lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Argmax per frame, collapse repeats, drop blanks. logits: (B, T, V)."""
    best = logits.argmax(dim=-1).cpu()
    results: list[list[int]] = []
    for b in range(best.shape[0]):
        seq = best[b, : int(out_lengths[b])].tolist()
        out: list[int] = []
        prev = blank
        for tok in seq:
            if tok != blank and tok != prev:
                out.append(tok)
            prev = tok
        results.append(out)
    return results


@dataclass
class GreedyState:
    """Carries the last frame's token across chunks so repeats spanning a boundary collapse correctly."""

    prev: int = 0
    tokens: list[int] = field(default_factory=list)
    frame_of_token: list[int] = field(
        default_factory=list
    )  # absolute output frame where each token was emitted

    def push(self, logits: torch.Tensor, frame_offset: int, blank: int = 0) -> list[int]:
        """logits: (T, V) for new frames. Returns the newly appended tokens (never modifies earlier ones)."""
        new: list[int] = []
        for i, tok in enumerate(logits.argmax(dim=-1).tolist()):
            if tok != blank and tok != self.prev:
                new.append(tok)
                self.frame_of_token.append(frame_offset + i)
            self.prev = tok
        self.tokens.extend(new)
        return new

    def reset(self) -> None:
        self.prev = 0
        self.tokens = []
        self.frame_of_token = []
