"""Chunk-boundary attention masks (U2 style) and dynamic chunk sampling.

Frame i may attend frame j iff chunk(j) <= chunk(i) and chunk(j) >= chunk(i) - left_chunks, and j is a
valid (non-padded) frame. All layers share the same boundaries, so look-ahead never accumulates with
depth: a frame sees at most the end of its own chunk.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkSpec:
    chunk_size: int  # frames at 25 fps; <= 0 means full context
    left_context: int  # frames; < 0 means unlimited

    @property
    def full(self) -> bool:
        return self.chunk_size <= 0

    @property
    def left_chunks(self) -> int:
        if self.full or self.left_context < 0:
            return -1
        return max(0, self.left_context // self.chunk_size)


def sample_chunk_spec(
    full_context_prob: float, chunk_sizes: tuple[int, ...], left_contexts: tuple[int, ...], rng: random.Random
) -> ChunkSpec:
    if rng.random() < full_context_prob:
        return ChunkSpec(0, -1)
    return ChunkSpec(rng.choice(chunk_sizes), rng.choice(left_contexts))


def attention_mask(
    lengths: torch.Tensor,
    t: int,
    spec: ChunkSpec,
    device: torch.device,
    query_offset: int = 0,
    t_q: int | None = None,
) -> torch.Tensor:
    """Boolean mask (B, 1, Tq, Tk) with True where attention is allowed.

    lengths: valid key frames per item (B,). Keys cover absolute frames [0, t); queries cover absolute
    frames [query_offset, query_offset + t_q) (defaults to all keys). Used both offline and in streaming,
    where keys are the cache plus the current chunk."""
    t_q = t if t_q is None else t_q
    key_idx = torch.arange(t, device=device)
    query_idx = torch.arange(t_q, device=device) + query_offset
    valid = key_idx[None, :] < lengths[:, None]  # (B, Tk)
    if spec.full:
        allowed = torch.ones(t_q, t, dtype=torch.bool, device=device)
    else:
        cq = query_idx // spec.chunk_size
        ck = key_idx // spec.chunk_size
        allowed = ck[None, :] <= cq[:, None]
        if spec.left_chunks >= 0:
            allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
    mask = allowed[None, :, :] & valid[:, None, :]
    # padded query rows attend themselves so softmax never sees an all-False row (their outputs are ignored)
    if query_offset == 0 and t_q == t:
        eye = torch.eye(t, dtype=torch.bool, device=device)
        mask = mask | eye[None]
    return mask[:, None, :, :]
