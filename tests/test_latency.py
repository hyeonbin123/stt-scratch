"""Stage 4 latency helpers (scripts/stage4_latency.py): regrouping, word matching, emission frames, times."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage4_latency", ROOT / "scripts" / "stage4_latency.py")
lat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lat)


def test_regroup_morphemes_into_words() -> None:
    pieces = [(0.1, 0.4, "자동차세"), (0.4, 0.5, "는"), (0.6, 0.8, "덜")]
    words = ["자동차세는", "덜"]
    assert lat.regroup(words, pieces) == [(0.1, 0.5), (0.6, 0.8)]
    assert lat.regroup(["자동차"], pieces) is None  # labels do not concatenate to the words
    assert lat.regroup(words + ["더"], pieces) is None  # more words than pieces


def test_match_words_exact_only() -> None:
    ref = ["a", "b", "c", "d"]
    assert lat.match_words(["a", "x", "c", "d"], ref) == {0: 0, 2: 2, 3: 3}
    assert lat.match_words(["b", "c"], ref) == {1: 0, 2: 1}
    assert lat.match_words([], ref) == {}


def test_emissions_first_frame_of_each_run() -> None:
    assert lat.emissions([0, 3, 3, 0, 3, 4, 4, 0]) == [(3, 1), (3, 4), (4, 5)]


@pytest.mark.parametrize("chunk", [8, 16])
def test_emit_time_is_end_of_chunk_audio(chunk: int) -> None:
    # frame 0 is available once the first chunk's audio (4 * chunk mel frames) has arrived
    first = lat.emit_time(0, chunk, 100.0)
    assert first == pytest.approx(0.04 * chunk + 0.015)
    assert lat.emit_time(chunk - 1, chunk, 100.0) == first
    assert lat.emit_time(chunk, chunk, 100.0) == pytest.approx(0.08 * chunk + 0.015)
    assert lat.emit_time(10 * chunk, chunk, 1.0) == 1.0  # capped at the end of the utterance
