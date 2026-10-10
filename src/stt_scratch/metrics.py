"""Character error rates with utterance-level bootstrap intervals (rules: docs/experiments.md "지표").

CER_ns ("no space") uses the whisper-ko-ft scoring normalization so numbers are comparable with that
project; CER_ws ("with space") keeps spaces as characters (KsponSpeech convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import numpy as np

from stt_scratch.text import normalize_hyp_ws, normalize_score, normalize_train

BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Scored:
    reference: str
    hypothesis: str
    edits: int
    length: int


def _edits(ref: str, hyp: str) -> Scored | None:
    if not ref:
        return None
    out = jiwer.process_characters(ref, hyp) if hyp else None
    if out is None:
        return Scored(ref, hyp, len(ref), len(ref))
    edits = out.substitutions + out.deletions + out.insertions
    return Scored(ref, hyp, edits, out.hits + out.substitutions + out.deletions)


def score_ns(reference: str, hypothesis: str) -> Scored | None:
    """CER without spaces (whisper-ko-ft definition). None when the normalized reference is empty."""
    return _edits(normalize_score(reference), normalize_score(hypothesis))


def score_ws(reference: str, hypothesis: str) -> Scored | None:
    """CER with spaces counted as characters (training normalization; the hypothesis keeps UNK_CHAR)."""
    return _edits(normalize_train(reference), normalize_hyp_ws(hypothesis))


def error_rate(edits: np.ndarray, lengths: np.ndarray) -> float:
    """Corpus-level rate: total edits over total reference length."""
    return float(edits.sum() / lengths.sum())


def _resample_indices(count: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))


def interval(edits: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95 % bootstrap interval of the corpus-level rate, resampling utterances."""
    picks = _resample_indices(len(edits))
    rates = edits[picks].sum(axis=1) / lengths[picks].sum(axis=1)
    low, high = np.percentile(rates, [2.5, 97.5])
    return float(low), float(high)


def difference_interval(edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """95 % interval of rate(a) - rate(b) over the same utterances (paired resampling)."""
    picks = _resample_indices(len(lengths))
    totals = lengths[picks].sum(axis=1)
    deltas = edits_a[picks].sum(axis=1) / totals - edits_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def summarize(scored: list[Scored]) -> dict:
    """Rate, interval and counts for a list of scored utterances (empty references already dropped)."""
    edits = np.array([s.edits for s in scored], dtype=np.int64)
    lengths = np.array([s.length for s in scored], dtype=np.int64)
    if len(scored) == 0:
        return {"n": 0, "rate": None, "ci": None, "edits": 0, "length": 0}
    low, high = interval(edits, lengths)
    return {
        "n": len(scored),
        "rate": error_rate(edits, lengths),
        "ci": [low, high],
        "edits": int(edits.sum()),
        "length": int(lengths.sum()),
    }
