"""Unit tests for scripts/prepare_data.py: hold-out hash, CTC frame formula, Common Voice split logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from stt_scratch.text import Vocab, ctc_min_frames

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


pd = load_script()


# ---------------------------------------------------------------- hold-out hash
def test_holdout_bucket_is_deterministic_and_matches_definition():
    import hashlib

    sentence = "몬터규는 자녀들이 사랑을 제대로 못 받고 크면"
    expected = int(hashlib.sha256(("zeroth-holdout:" + sentence).encode("utf-8")).hexdigest()[:8], 16) % 100
    assert pd.holdout_bucket(sentence) == expected
    assert pd.holdout_bucket(sentence) == pd.holdout_bucket(sentence)
    # Pinned value: changing the salt or the slice would silently move the hold-out.
    assert pd.holdout_bucket("안녕하세요") == 33


def test_zeroth_target_partitions_by_bucket():
    cases = ((0, "dev_h"), (4, "dev_h"), (5, "test_h"), (9, "test_h"), (10, "train"), (99, "train"))
    for bucket, target in cases:
        sentence = next(s for s in (f"문장 {i}" for i in range(100_000)) if pd.holdout_bucket(s) == bucket)
        assert pd.zeroth_target(sentence) == target


def test_holdout_share_is_about_ten_percent_and_dev_test_disjoint():
    sentences = [f"합성 문장 번호 {i}" for i in range(20_000)]
    targets = {s: pd.zeroth_target(s) for s in sentences}
    dev_h = {s for s, t in targets.items() if t == "dev_h"}
    test_h = {s for s, t in targets.items() if t == "test_h"}
    train = {s for s, t in targets.items() if t == "train"}
    assert not dev_h & test_h
    assert not (dev_h | test_h) & train
    assert len(dev_h) + len(test_h) + len(train) == len(sentences)
    holdout_share = (len(dev_h) + len(test_h)) / len(sentences)
    assert 0.09 <= holdout_share <= 0.11
    assert 0.04 <= len(dev_h) / len(sentences) <= 0.06


# ---------------------------------------------------------------- CTC frames
@pytest.mark.parametrize(
    ("n_samples", "n_mel", "frames_out"),
    [
        (0, 0, 0),
        (399, 0, 0),
        (400, 1, 1),  # exactly one window
        (559, 1, 1),
        (560, 2, 1),  # ceil(ceil(2/2)/2) = 1
        (16_000, 98, 25),  # one second -> 25 fps
        (160_000, 998, 250),
    ],
)
def test_frame_formula(n_samples, n_mel, frames_out):
    assert pd.mel_frames(n_samples) == n_mel
    assert pd.output_frames(n_samples) == frames_out


def test_ctc_fits_requires_min_frames_plus_one():
    vocab = Vocab.build(["아 가"])
    ids = vocab.encode("아아 가")  # U = 4, one repeat -> ctc_min_frames = 5, need >= 6 frames
    assert ctc_min_frames(ids) == 5
    # 6 output frames <=> n_mel in 21..24 <=> samples in 400 + 20*160 .. 400 + 23*160
    assert pd.output_frames(400 + 20 * 160) == 6
    assert pd.ctc_fits(400 + 20 * 160, ids)
    assert pd.output_frames(400 + 19 * 160) == 5
    assert not pd.ctc_fits(400 + 19 * 160, ids)


def test_exclusion_reasons():
    assert pd.exclusion_reason("삼 년", 3.0) is None
    assert pd.exclusion_reason("3 년", 3.0) == "digit"
    assert pd.exclusion_reason("abc 년", 3.0) == "latin"
    assert pd.exclusion_reason("긴 발화", 20.001) == "too_long"
    assert pd.exclusion_reason("경계", 20.0) is None
    assert pd.exclusion_reason("3 abc", 30.0) == "digit"  # first matching rule wins


# ---------------------------------------------------------------- Common Voice split
def test_cv_path_in_dev_or_test_is_not_in_train():
    validated = ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3"]
    dev = ["b.mp3"]
    test = ["d.mp3", "zzz.mp3"]
    train = pd.cv_train_paths(validated, dev, test)
    assert train == ["a.mp3", "c.mp3", "e.mp3"]
    assert "b.mp3" not in train
    assert "d.mp3" not in train
    assert not set(train) & (set(dev) | set(test))
