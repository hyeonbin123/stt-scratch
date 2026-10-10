"""Manifests, the int16 PCM store written by scripts/prepare_data.py, and duration-bucketed batching."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stt_scratch.text import Vocab


@dataclass(frozen=True)
class Utterance:
    id: str
    corpus: str
    text: str
    duration_s: float
    hangul_only: bool
    speaker: str | None = None


def read_manifest(path: Path) -> list[Utterance]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                Utterance(
                    id=d["id"],
                    corpus=d["corpus"],
                    text=d["text"],
                    duration_s=float(d["duration_s"]),
                    hangul_only=bool(d.get("hangul_only", True)),
                    speaker=d.get("speaker"),
                )
            )
    return rows


class PcmStore:
    """Memory-mapped int16 16 kHz audio of one manifest: data/pcm/<name>.bin + <name>.index.jsonl."""

    def __init__(self, pcm_dir: Path, name: str) -> None:
        self.bin = np.memmap(pcm_dir / f"{name}.bin", dtype=np.int16, mode="r")
        self.index: dict[str, tuple[int, int]] = {}
        with open(pcm_dir / f"{name}.index.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.index[d["id"]] = (int(d["offset"]), int(d["length"]))

    def waveform(self, utt_id: str) -> torch.Tensor:
        offset, length = self.index[utt_id]
        arr = np.asarray(self.bin[offset : offset + length], dtype=np.float32) / 32768.0
        return torch.from_numpy(arr)


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, utts: list[Utterance], store: PcmStore, vocab: Vocab) -> None:
        self.utts = utts
        self.store = store
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.utts)

    def __getitem__(self, i: int) -> dict:
        u = self.utts[i]
        return {
            "id": u.id,
            "wav": self.store.waveform(u.id),
            "tokens": self.vocab.encode(u.text),
            "text": u.text,
        }


def collate(items: list[dict]) -> dict:
    """Keeps waveforms as a list (speed perturbation happens later, on the GPU)."""
    return {
        "ids": [it["id"] for it in items],
        "wavs": [it["wav"] for it in items],
        "tokens": [it["tokens"] for it in items],
        "texts": [it["text"] for it in items],
    }


class DurationBucketSampler(torch.utils.data.Sampler[list[int]]):
    """Batches of similar-length utterances whose summed duration stays under max_duration seconds.

    Each epoch: sort by duration, cut into batches, shuffle the batches with a seed derived from the epoch.
    Durations are padded with a 10 % margin so that speed perturbation (x1/0.9) still fits the budget.
    """

    def __init__(self, utts: list[Utterance], max_duration: float, seed: int, shuffle: bool = True) -> None:
        self.durations = [u.duration_s for u in utts]
        self.max_duration = max_duration
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self._batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0.0
        for i in order:
            d = self.durations[i] * 1.1
            new_max = max(cur_max, d)
            # padded batch cost = longest utterance x batch size
            if cur and new_max * (len(cur) + 1) > self.max_duration:
                batches.append(cur)
                cur, cur_max = [], 0.0
                new_max = d
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed * 1000 + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)
