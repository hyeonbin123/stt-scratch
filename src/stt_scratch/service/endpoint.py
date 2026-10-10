"""Rule-based CTC endpointing (docs/experiments.md 6단계; sherpa/WeNet style).

A frame is "silence" when the blank posterior is at least `blank_threshold`. An endpoint fires when
  rule 1: trailing silence >= rule1_s (regardless of whether anything was decoded),
  rule 2: something was decoded in this segment and trailing silence >= rule2_s,
  rule 3: the segment is at least rule3_s long.
Frames are encoder frames at 25 fps (40 ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FRAME_S = 0.04


@dataclass
class EndpointConfig:
    blank_threshold: float = 0.8
    rule1_s: float = 2.4
    rule2_s: float = 1.2
    rule3_s: float = 20.0


class Endpointer:
    def __init__(self, cfg: EndpointConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.trailing_silence = 0
        self.segment_frames = 0
        self.decoded_something = False

    def update(self, log_probs: torch.Tensor, new_tokens: int) -> str | None:
        """log_probs: (T, V) fp32 for the new frames; new_tokens: tokens emitted in these frames.
        Returns the rule name that fired, or None."""
        blank = log_probs[:, 0].exp()
        is_sil = (blank >= self.cfg.blank_threshold).tolist()
        if new_tokens > 0:
            self.decoded_something = True
        for s in is_sil:
            self.trailing_silence = self.trailing_silence + 1 if s else 0
            self.segment_frames += 1
        trailing_s = self.trailing_silence * FRAME_S
        if trailing_s >= self.cfg.rule1_s:
            return "rule1"
        if self.decoded_something and trailing_s >= self.cfg.rule2_s:
            return "rule2"
        if self.segment_frames * FRAME_S >= self.cfg.rule3_s:
            return "rule3"
        return None
