"""Text normalization, scoring normalization and syllable tokenization (rules: docs/experiments.md "공통")."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]")
# Anything that is not Hangul syllable, Latin letter, digit or whitespace becomes a space.
_NON_TEXT = re.compile(r"[^가-힣A-Za-z0-9\s]")
# whisper-ko-ft scorer: anything that isn't a letter, digit, whitespace or apostrophe (\w covers Hangul).
_SCORE_PUNCT = re.compile(r"[^\w\s']|_")

BLANK, UNK, SPACE = "<blank>", "<unk>", " "
# Decoded form of <unk>: U+318D (Hangul letter araea). It is a word character, so the whisper-ko-ft scoring
# normalization keeps it and every emitted <unk> counts as one wrong character; it can never occur in a
# reference (normalize_train keeps only Hangul syllables, Latin letters, digits and spaces).
UNK_CHAR = "\u318d"
# normalize_train for hypotheses: identical, except that UNK_CHAR is kept (used by CER_ws scoring).
_NON_TEXT_HYP = re.compile("[^\uac00-\ud7a3A-Za-z0-9\\s" + UNK_CHAR + "]")


def normalize_train(text: str) -> str:
    """Training/evaluation reference normalization: NFC, punctuation -> space, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    text = _NON_TEXT.sub(" ", text)
    return " ".join(text.split())


def normalize_hyp_ws(text: str) -> str:
    """Training normalization for a hypothesis: like normalize_train but keeps UNK_CHAR, so an emitted
    <unk> stays one wrong character in CER_ws as well (references never contain it)."""
    text = unicodedata.normalize("NFC", text)
    text = _NON_TEXT_HYP.sub(" ", text)
    return " ".join(text.split())


def normalize_score(text: str) -> str:
    """Scoring normalization identical to whisper-ko-ft (Korean): lowercase, drop punctuation,
    drop apostrophes and every space, so CER is measured on characters without spaces."""
    text = _SCORE_PUNCT.sub(" ", text.lower())
    return "".join(text.replace("'", " ").split())


def has_digit(text: str) -> bool:
    return bool(DIGIT.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN.search(text))


def hangul_only(text: str) -> bool:
    """True when the normalized text contains only Hangul syllables and spaces (and is non-empty)."""
    return bool(text) and not has_digit(text) and not has_latin(text)


@dataclass(frozen=True)
class Vocab:
    """Syllable vocabulary: 0 = CTC blank, 1 = <unk>, 2 = space, then Hangul syllables."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens[:3] != (BLANK, UNK, SPACE):
            raise ValueError("vocab must start with <blank>, <unk>, space")

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def space_id(self) -> int:
        return 2

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, texts: list[str]) -> Vocab:
        """All Hangul syllables seen at least once in `texts` (already normalized), by frequency."""
        counts: dict[str, int] = {}
        for text in texts:
            for ch in HANGUL.findall(text):
                counts[ch] = counts.get(ch, 0) + 1
        ordered = sorted(counts, key=lambda ch: (-counts[ch], ch))
        return cls((BLANK, UNK, SPACE, *ordered))

    def encode(self, text: str) -> list[int]:
        """Characters of the normalized text; spaces -> space id; anything unknown -> <unk>."""
        index = self._index()
        return [index.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Drops blanks; <unk> becomes UNK_CHAR so it is scored as one wrong character."""
        return "".join(UNK_CHAR if i == self.unk_id else self.tokens[i] for i in ids if i != self.blank_id)

    def _index(self) -> dict[str, int]:
        cached = getattr(self, "_cached_index", None)
        if cached is None:
            cached = {tok: i for i, tok in enumerate(self.tokens)}
            object.__setattr__(self, "_cached_index", cached)
        return cached

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"tokens": list(self.tokens)}, ensure_ascii=False, indent=0), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Vocab:
        return cls(tuple(json.loads(path.read_text(encoding="utf-8"))["tokens"]))


def ctc_min_frames(ids: list[int]) -> int:
    """Smallest number of output frames CTC needs for this label sequence: U + number of repeats."""
    repeats = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a == b)
    return len(ids) + repeats
