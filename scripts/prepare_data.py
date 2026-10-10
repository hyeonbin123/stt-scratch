"""Stage 0 data preparation (rules: docs/experiments.md "공통", 2026-10-11).

Reads the raw corpora under data/raw (Zeroth-Korean and FLEURS ko parquet, Common Voice 22 ko tsv + tar)
and writes

  manifests/<name>.jsonl          train, dev_mix, dev_h, test_zeroth, test_h, test_fleurs, test_cv
                                  (sorted by id)
  manifests/train_excluded.jsonl  train candidates dropped by the exclusion rules, with a `reason`
  manifests/vocab.json            syllable vocabulary built on the final train texts
  data/pcm/<name>.bin             int16 little-endian mono 16 kHz audio of every kept utterance (not
                                  committed)
  data/pcm/<name>.index.jsonl     {"id", "offset", "length"} in samples, one line per utterance
  reports/stage0/data_summary.json / data_summary.md

The parquet corpora are streamed twice: pass A reads text and audio headers (durations, splits, exclusions,
vocabulary), pass B decodes audio and writes the PCM store. Common Voice mp3 is decoded once in pass A
(mp3 headers do not give exact sample counts) and the PCM is kept in memory until pass B.

Usage: uv run python scripts/prepare_data.py [--raw data/raw] [--manifests manifests] [--pcm data/pcm]
                                             [--reports reports/stage0]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

from stt_scratch.text import HANGUL, Vocab, ctc_min_frames, hangul_only, has_digit, has_latin, normalize_train

SAMPLE_RATE = 16_000
WINDOW, HOP = 400, 160  # 25 ms window, 10 ms hop at 16 kHz, center=False
SUBSAMPLING_STRIDES = (2, 2)  # two causal stride-2 convolutions -> 25 frames per second
MAX_TRAIN_SECONDS = 20.0
HOLDOUT_SALT = "zeroth-holdout:"
HOLDOUT_PERCENT, DEV_H_PERCENT = 10, 5
H0_1_THRESHOLD = 0.005
MANIFEST_NAMES = ("train", "dev_mix", "dev_h", "test_zeroth", "test_h", "test_fleurs", "test_cv")
EVAL_NAMES = MANIFEST_NAMES[1:]
EXCLUSION_REASONS = ("digit", "latin", "too_long", "ctc_length", "missing_audio")
PROGRESS_EVERY = 2_000
PARQUET_BATCH = 256


# ----------------------------------------------------------------------------------------------------
# Split rules
# ----------------------------------------------------------------------------------------------------
def holdout_bucket(sentence: str) -> int:
    """0..99 bucket of a normalized Zeroth sentence: int(sha256("zeroth-holdout:" + s)[:8], 16) % 100."""
    digest = hashlib.sha256((HOLDOUT_SALT + sentence).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def zeroth_target(sentence: str) -> str:
    """Manifest of a Zeroth *train* utterance by sentence bucket: dev_h (< 5), test_h (< 10), else train."""
    bucket = holdout_bucket(sentence)
    if bucket < DEV_H_PERCENT:
        return "dev_h"
    if bucket < HOLDOUT_PERCENT:
        return "test_h"
    return "train"


def cv_train_paths(validated: Iterable[str], dev: Iterable[str], test: Iterable[str]) -> list[str]:
    """Common Voice train = validated.tsv minus every `path` that appears in dev.tsv or test.tsv."""
    held = set(dev) | set(test)
    return [path for path in validated if path not in held]


# ----------------------------------------------------------------------------------------------------
# Exclusion rules
# ----------------------------------------------------------------------------------------------------
def mel_frames(n_samples: int) -> int:
    """Fbank frames of a 16 kHz signal with center=False: floor((n - 400) / 160) + 1, never negative."""
    return max(0, (n_samples - WINDOW) // HOP + 1)


def output_frames(n_samples: int) -> int:
    """Encoder output frames after the stride-2 convolutions: ceil(ceil(n_mel / 2) / 2)."""
    frames = mel_frames(n_samples)
    for stride in SUBSAMPLING_STRIDES:
        frames = math.ceil(frames / stride)
    return frames


def ctc_fits(n_samples: int, ids: Sequence) -> bool:
    """Training length constraint: output frames >= U + repeats + 1 (ids or characters: same repeats)."""
    return output_frames(n_samples) >= ctc_min_frames(list(ids)) + 1


def exclusion_reason(text: str, duration_s: float) -> str | None:
    """Training exclusions that do not need the vocabulary (text is normalized and non-empty)."""
    if has_digit(text):
        return "digit"
    if has_latin(text):
        return "latin"
    if duration_s > MAX_TRAIN_SECONDS:
        return "too_long"
    return None


# ----------------------------------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------------------------------
def decode_pcm16(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode an audio file to int16 mono 16 kHz; returns (pcm, source sample rate).

    Multi-channel audio is averaged to mono; a source rate other than 16 kHz is resampled on CPU with
    torchaudio.functional.resample; samples are rounded and clipped to the int16 range.
    """
    data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != SAMPLE_RATE:
        wave = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
        mono = torchaudio.functional.resample(wave, rate, SAMPLE_RATE).squeeze(0).numpy()
    pcm = np.clip(np.rint(mono * 32768.0), -32768, 32767).astype("<i2")
    return pcm, rate


def probe_samples(blob: bytes) -> tuple[int, int]:
    """Header-only (frames at 16 kHz, source rate). Exact for FLAC/WAV; the 16 kHz count follows the
    output length of torchaudio's resampler when the source rate differs."""
    with sf.SoundFile(io.BytesIO(blob)) as handle:
        frames, rate = handle.frames, handle.samplerate
    if rate != SAMPLE_RATE:
        frames = math.ceil(frames * SAMPLE_RATE / rate)
    return frames, rate


# ----------------------------------------------------------------------------------------------------
# Raw sources
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RawUtt:
    id: str
    corpus: str
    src_split: str
    speaker: str | None
    text_raw: str
    blob: bytes


def iter_parquet_rows(files: list[Path], columns: list[str]) -> Iterator[dict]:
    """Stream rows of several parquet files in order, one small batch at a time."""
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH, columns=columns):
            yield from batch.to_pylist()


def iter_zeroth(raw: Path, split: str) -> Iterator[RawUtt]:
    """Zeroth-Korean rows; id = zeroth/<split>/<row index in parquet order>, speaker = speaker_id."""
    files = sorted((raw / "zeroth_korean" / "data").glob(f"{split}-*.parquet"))
    for index, row in enumerate(iter_parquet_rows(files, ["speaker_id", "audio", "text"])):
        yield RawUtt(
            f"zeroth/{split}/{index}",
            "zeroth",
            split,
            str(row["speaker_id"]),
            (row["text"] or "").strip(),
            row["audio"]["bytes"],
        )


def iter_fleurs(raw: Path, split: str) -> Iterator[RawUtt]:
    """FLEURS ko rows. The `id` column is a *sentence* id shared by up to three speakers, so the row
    index is appended to keep utterance ids unique: fleurs/<split>/<id>/<row index>."""
    files = sorted((raw / "fleurs_ko" / "ko_kr" / split).glob("*.parquet"))
    for index, row in enumerate(iter_parquet_rows(files, ["id", "audio", "transcription"])):
        yield RawUtt(
            f"fleurs/{split}/{row['id']}/{index}",
            "fleurs",
            split,
            None,
            (row["transcription"] or "").strip(),
            row["audio"]["bytes"],
        )


def read_cv_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


# ----------------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------------
@dataclass
class Utt:
    id: str
    corpus: str
    src_split: str
    speaker: str | None
    text_raw: str
    text: str
    n_samples: int = 0  # 16 kHz samples (header in pass A, verified against the decoded length in pass B)
    manifest: str | None = None  # target manifest; None when excluded from train
    reason: str | None = None  # train exclusion reason
    pcm: np.ndarray | None = None  # decoded audio kept in memory (Common Voice only)

    @property
    def duration_s(self) -> float:
        return self.n_samples / SAMPLE_RATE


class Progress:
    """Prints a line every PROGRESS_EVERY utterances."""

    def __init__(self, label: str) -> None:
        self.label, self.count, self.start = label, 0, time.perf_counter()

    def tick(self, n: int = 1) -> None:
        before = self.count
        self.count += n
        if self.count // PROGRESS_EVERY != before // PROGRESS_EVERY:
            log(f"{self.label}: {self.count:,} utterances ({time.perf_counter() - self.start:.0f} s)")


def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------------------------------------------
# Pass A: text, durations, splits
# ----------------------------------------------------------------------------------------------------
def make_utt(src: RawUtt, n_samples: int) -> Utt:
    return Utt(
        src.id, src.corpus, src.src_split, src.speaker, src.text_raw, normalize_train(src.text_raw), n_samples
    )


def collect_parquet(raw: Path, stats: dict, progress: Progress) -> list[Utt]:
    """Zeroth (train hold-out by sentence hash, test -> test_zeroth) and FLEURS (train / dev_mix / test)."""
    utts: list[Utt] = []
    plan: list[tuple[str, Iterator[RawUtt], str | None]] = [
        ("zeroth/train", iter_zeroth(raw, "train"), None),
        ("zeroth/test", iter_zeroth(raw, "test"), "test_zeroth"),
        ("fleurs/train", iter_fleurs(raw, "train"), "train"),
        ("fleurs/validation", iter_fleurs(raw, "validation"), "dev_mix"),
        ("fleurs/test", iter_fleurs(raw, "test"), "test_fleurs"),
    ]
    for source, rows, manifest in plan:
        for src in rows:
            n_samples, rate = probe_samples(src.blob)
            if rate != SAMPLE_RATE:
                stats["resampled"][source] += 1
            utt = make_utt(src, n_samples)
            progress.tick()
            target = manifest if manifest is not None else zeroth_target(utt.text)
            if not utt.text:
                require_train(utt, target, "empty normalized transcript")
                stats["empty_text"][source] += 1
                continue
            utt.manifest = target
            utts.append(utt)
    return utts


def decode_cv_tars(cv_root: Path, splits: Iterable[str], wanted: set[str], progress: Progress) -> dict:
    """Sequentially decode the wanted mp3 members of audio/ko/<split>/*.tar: basename -> (pcm, rate)."""
    decoded: dict[str, tuple[np.ndarray, int]] = {}
    for split in splits:
        for tar_path in sorted((cv_root / "audio" / "ko" / split).glob("*.tar")):
            with tarfile.open(tar_path) as tar:
                for member in tar:
                    name = Path(member.name).name
                    if not member.isfile() or name not in wanted:
                        continue
                    blob = tar.extractfile(member).read()  # type: ignore[union-attr]
                    decoded[name] = decode_pcm16(blob)
                    progress.tick()
    return decoded


def collect_common_voice(raw: Path, stats: dict, progress: Progress) -> list[Utt]:
    """CV22 ko: validated - dev - test -> train, dev -> dev_mix, test -> test_cv (audio decoded here)."""
    cv_root = raw / "common_voice_22_ko"
    tsv = {
        name: read_cv_tsv(cv_root / "transcript" / "ko" / f"{name}.tsv")
        for name in ("validated", "dev", "test")
    }
    paths = {name: [row["path"] for row in rows] for name, rows in tsv.items()}
    train_paths = set(cv_train_paths(paths["validated"], paths["dev"], paths["test"]))
    plan = [
        ("validated", "train", [row for row in tsv["validated"] if row["path"] in train_paths]),
        ("dev", "dev_mix", tsv["dev"]),
        ("test", "test_cv", tsv["test"]),
    ]
    wanted = {row["path"] for _, _, rows in plan for row in rows}
    audio = decode_cv_tars(cv_root, ("train", "dev", "test"), wanted, progress)
    utts: list[Utt] = []
    for src_split, manifest, rows in plan:
        source = f"cv22/{src_split}"
        for row in rows:
            text_raw = row["sentence"].strip()
            utt = Utt(
                f"cv22/{src_split}/{row['path']}",
                "cv22",
                src_split,
                row["client_id"],
                text_raw,
                normalize_train(text_raw),
            )
            if not utt.text:
                require_train(utt, manifest, "empty normalized transcript")
                stats["empty_text"][source] += 1
                continue
            hit = audio.get(row["path"])
            if hit is None:
                require_train(utt, manifest, "no audio in the tar files")
                utt.reason = "missing_audio"
                utts.append(utt)
                continue
            utt.pcm, rate = hit
            if rate == SAMPLE_RATE:
                stats["cv_not_resampled"][source] += 1
            utt.n_samples = len(utt.pcm)
            utt.manifest = manifest
            utts.append(utt)
    return utts


def require_train(utt: Utt, target: str | None, problem: str) -> None:
    """Evaluation sets are fixed: an utterance that cannot be kept must stop the run, not shrink the set."""
    if target != "train":
        raise RuntimeError(f"evaluation utterance {utt.id} ({target}) cannot be kept: {problem}")


def apply_train_exclusions(utts: list[Utt], stats: dict) -> Vocab:
    """Exclusions 1-3 and the CTC length exclusion (on characters, which gives the same repeat count as
    token ids because every surviving training character is in the vocabulary), then the vocabulary on
    the final train texts."""
    checked = 0
    for utt in utts:
        if utt.manifest == "train":
            reason = exclusion_reason(utt.text, utt.duration_s)
            if reason is None:
                checked += 1
                if not ctc_fits(utt.n_samples, utt.text):
                    reason = "ctc_length"
            if reason is not None:
                utt.reason, utt.manifest = reason, None
    stats["ctc_checked"] = checked
    return Vocab.build([utt.text for utt in utts if utt.manifest == "train"])


# ----------------------------------------------------------------------------------------------------
# Pass B: PCM store
# ----------------------------------------------------------------------------------------------------
class PcmWriter:
    """Appends int16 little-endian samples to <name>.bin and one index line per utterance."""

    def __init__(self, pcm_dir: Path, name: str) -> None:
        self.bin = open(pcm_dir / f"{name}.bin", "wb")
        self.index = open(pcm_dir / f"{name}.index.jsonl", "w", encoding="utf-8", newline="\n")
        self.offset = 0
        self.count = 0

    def write(self, utt_id: str, pcm: np.ndarray) -> None:
        self.bin.write(pcm.astype("<i2", copy=False).tobytes())
        self.index.write(json.dumps({"id": utt_id, "offset": self.offset, "length": int(len(pcm))}) + "\n")
        self.offset += int(len(pcm))
        self.count += 1

    def close(self) -> None:
        self.bin.close()
        self.index.close()


def write_pcm_store(raw: Path, utts: list[Utt], pcm_dir: Path, stats: dict) -> None:
    pcm_dir.mkdir(parents=True, exist_ok=True)
    by_id = {utt.id: utt for utt in utts}
    writers = {name: PcmWriter(pcm_dir, name) for name in MANIFEST_NAMES}
    progress = Progress("pass B (decode)")
    sources: list[Iterator[RawUtt]] = [
        iter_zeroth(raw, "train"),
        iter_zeroth(raw, "test"),
        iter_fleurs(raw, "train"),
        iter_fleurs(raw, "validation"),
        iter_fleurs(raw, "test"),
    ]
    try:
        for rows in sources:
            for src in rows:
                utt = by_id.get(src.id)
                if utt is None or utt.manifest is None:
                    continue
                pcm, _rate = decode_pcm16(src.blob)
                if len(pcm) != utt.n_samples:
                    raise RuntimeError(f"{src.id}: header says {utt.n_samples} samples, decoded {len(pcm)}")
                writers[utt.manifest].write(utt.id, pcm)
                progress.tick()
        for utt in utts:
            if utt.corpus == "cv22" and utt.manifest is not None:
                writers[utt.manifest].write(utt.id, utt.pcm)
                utt.pcm = None
                progress.tick()
    finally:
        for writer in writers.values():
            writer.close()
    stats["audio_store"] = {
        name: {
            "bin": str(pcm_dir / f"{name}.bin"),
            "utterances": writer.count,
            "samples": writer.offset,
            "bytes": writer.offset * 2,
        }
        for name, writer in writers.items()
    }


# ----------------------------------------------------------------------------------------------------
# Manifests
# ----------------------------------------------------------------------------------------------------
def manifest_row(utt: Utt, vocab: Vocab) -> dict:
    ids = vocab.encode(utt.text)
    return {
        "id": utt.id,
        "corpus": utt.corpus,
        "src_split": utt.src_split,
        "speaker": utt.speaker,
        "duration_s": round(utt.duration_s, 3),
        "text": utt.text,
        "text_raw": utt.text_raw,
        "hangul_only": hangul_only(utt.text),
        "n_tokens": len(ids),
        "ctc_min_frames": ctc_min_frames(ids),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_vocab(vocab: Vocab, path: Path) -> None:
    """Same JSON as Vocab.save, written with LF line endings (Vocab.load reads it back)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"tokens": list(vocab.tokens)}, ensure_ascii=False, indent=0))


def write_manifests(utts: list[Utt], vocab: Vocab, manifest_dir: Path) -> dict[str, list[dict]]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {name: [] for name in MANIFEST_NAMES}
    excluded: list[dict] = []
    for utt in utts:
        if utt.manifest is not None:
            manifests[utt.manifest].append(manifest_row(utt, vocab))
        elif utt.reason is not None:
            excluded.append({**manifest_row(utt, vocab), "reason": utt.reason})
    for name, rows in manifests.items():
        rows.sort(key=lambda row: row["id"])
        write_jsonl(manifest_dir / f"{name}.jsonl", rows)
    excluded.sort(key=lambda row: row["id"])
    write_jsonl(manifest_dir / "train_excluded.jsonl", excluded)
    write_vocab(vocab, manifest_dir / "vocab.json")
    manifests["train_excluded"] = excluded
    return manifests


# ----------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------
def hours(rows: Iterable[dict]) -> float:
    return round(sum(row["duration_s"] for row in rows) / 3600, 3)


def summarize(manifests: dict[str, list[dict]], vocab: Vocab, stats: dict, wall_s: float) -> dict:
    train = manifests["train"]
    excluded = manifests["train_excluded"]
    per_manifest = {}
    for name in MANIFEST_NAMES:
        rows = manifests[name]
        corpora = sorted({row["corpus"] for row in rows})
        speakers_of = {
            c: {row["speaker"] for row in rows if row["corpus"] == c and row["speaker"]} for c in corpora
        }
        per_manifest[name] = {
            "utterances": len(rows),
            "hours": hours(rows),
            "speakers": len({row["speaker"] for row in rows if row["speaker"] is not None}),
            "utterances_without_speaker": sum(1 for row in rows if row["speaker"] is None),
            "by_corpus": {
                corpus: {
                    "utterances": sum(1 for row in rows if row["corpus"] == corpus),
                    "hours": hours(row for row in rows if row["corpus"] == corpus),
                    "speakers": len(speakers_of[corpus]),
                }
                for corpus in corpora
            },
        }
    exclusions = {
        reason: {
            "utterances": sum(1 for row in excluded if row["reason"] == reason),
            "hours": hours(row for row in excluded if row["reason"] == reason),
            "by_corpus": dict(Counter(row["corpus"] for row in excluded if row["reason"] == reason)),
        }
        for reason in EXCLUSION_REASONS
    }
    exclusions["total"] = {"utterances": len(excluded), "hours": hours(excluded)}

    zeroth_train_rows = [
        row for name in ("train", "dev_h", "test_h") for row in manifests[name] if row["corpus"] == "zeroth"
    ]
    dev_h_sentences = {row["text"] for row in manifests["dev_h"]}
    test_h_sentences = {row["text"] for row in manifests["test_h"]}
    holdout_sentences = dev_h_sentences | test_h_sentences
    train_sentences = {row["text"] for row in train}
    test_zeroth_sentences = {row["text"] for row in manifests["test_zeroth"]}
    holdout = {
        "zeroth_train_unique_sentences": len({row["text"] for row in zeroth_train_rows}),
        "zeroth_train_utterances_after_empty_drop": len(zeroth_train_rows),
        "H": {
            "sentences": len(holdout_sentences),
            "utterances": len(manifests["dev_h"]) + len(manifests["test_h"]),
        },
        "dev_h": {"sentences": len(dev_h_sentences), "utterances": len(manifests["dev_h"])},
        "test_h": {"sentences": len(test_h_sentences), "utterances": len(manifests["test_h"])},
        "dev_h_test_h_sentence_overlap": len(dev_h_sentences & test_h_sentences),
        "H_train_sentence_overlap": len(holdout_sentences & train_sentences),
        "test_zeroth_sentences_in_train": len(test_zeroth_sentences & train_sentences),
        "test_zeroth_utterances_with_sentence_in_train": sum(
            1 for row in manifests["test_zeroth"] if row["text"] in train_sentences
        ),
        "test_zeroth_sentences_in_H": len(test_zeroth_sentences & holdout_sentences),
    }

    vocab_tokens = set(vocab.tokens)
    eval_oov = {}
    for name in EVAL_NAMES:
        rows = manifests[name]
        syllables = [ch for row in rows for ch in HANGUL.findall(row["text"])]
        oov = [ch for ch in syllables if ch not in vocab_tokens]
        n_hangul_only = sum(1 for row in rows if row["hangul_only"])
        eval_oov[name] = {
            "syllable_tokens": len(syllables),
            "oov_tokens": len(oov),
            "oov_types": len(set(oov)),
            "oov_token_rate": round(len(oov) / len(syllables), 6) if syllables else 0.0,
            "hangul_only_utterances": n_hangul_only,
            "hangul_only_share": round(n_hangul_only / len(rows), 4) if rows else 0.0,
        }

    rates = np.array([row["n_tokens"] / row["duration_s"] for row in train]) if train else np.zeros(1)
    token_rate = {
        "p50": round(float(np.percentile(rates, 50)), 3),
        "p95": round(float(np.percentile(rates, 95)), 3),
        "max": round(float(rates.max()), 3),
    }
    ctc_excluded = exclusions["ctc_length"]["utterances"]
    h0_1 = {
        "checked_utterances": stats["ctc_checked"],
        "excluded_utterances": ctc_excluded,
        "ratio": round(ctc_excluded / stats["ctc_checked"], 6) if stats["ctc_checked"] else 0.0,
        "threshold": H0_1_THRESHOLD,
        "pass": (ctc_excluded / stats["ctc_checked"] if stats["ctc_checked"] else 0.0) <= H0_1_THRESHOLD,
    }
    return {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rules": 'docs/experiments.md "공통" (2026-10-11)',
        "script": "scripts/prepare_data.py",
        "wall_time_s": round(wall_s, 1),
        "manifests": per_manifest,
        "train_exclusions": exclusions,
        "holdout": holdout,
        "vocab": {"size": len(vocab), "syllables": len(vocab) - 3, "path": "manifests/vocab.json"},
        "eval_oov": eval_oov,
        "train_token_rate_per_s": token_rate,
        "h0_1": h0_1,
        "dropped_empty_text": dict(stats["empty_text"]),
        "dropped_missing_audio_eval": dict(stats["missing_audio_eval"]),
        "zeroth_fleurs_resampled": dict(stats["resampled"]),
        "cv_already_16k": dict(stats["cv_not_resampled"]),
        "audio_store": stats["audio_store"],
    }


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(summary: dict) -> str:
    m = summary["manifests"]
    ex = summary["train_exclusions"]
    ho = summary["holdout"]
    h01 = summary["h0_1"]
    rate = summary["train_token_rate_per_s"]
    parts = [
        "# 0단계 데이터 요약",
        "",
        f"생성: {summary['created']} · 스크립트 `{summary['script']}` · 규칙 {summary['rules']} · "
        f"벽시계 {summary['wall_time_s']:.0f} s",
        "",
        "## 매니페스트 (`manifests/<이름>.jsonl`)",
        "",
        table(
            ["이름", "발화", "시간(h)", "화자", "구성"],
            [
                [
                    name,
                    f"{m[name]['utterances']:,}",
                    f"{m[name]['hours']:.2f}",
                    f"{m[name]['speakers']:,}"
                    + (" (FLEURS 화자 정보 없음)" if m[name]["utterances_without_speaker"] else ""),
                    ", ".join(
                        f"{corpus} {v['utterances']:,}발화 {v['hours']:.2f} h"
                        + (f" {v['speakers']}명" if v["speakers"] else "")
                        for corpus, v in m[name]["by_corpus"].items()
                    ),
                ]
                for name in MANIFEST_NAMES
            ],
        ),
        "",
        "## 학습 제외 (`manifests/train_excluded.jsonl`, 오디오 미기록)",
        "",
        table(
            ["이유", "발화", "시간(h)", "코퍼스별"],
            [
                [
                    reason,
                    f"{ex[reason]['utterances']:,}",
                    f"{ex[reason]['hours']:.2f}",
                    ", ".join(f"{c} {n:,}" for c, n in sorted(ex[reason]["by_corpus"].items())) or "-",
                ]
                for reason in EXCLUSION_REASONS
            ]
            + [["합계", f"{ex['total']['utterances']:,}", f"{ex['total']['hours']:.2f}", ""]],
        ),
        "",
        f"- **H0.1** (25 fps CTC 길이 제약 위반 ≤ 0.5 %): 검사 {h01['checked_utterances']:,}발화 중 "
        f"{h01['excluded_utterances']:,}개 제외 = {100 * h01['ratio']:.3f} % → "
        f"{'충족' if h01['pass'] else '위반'}",
        "- 제외 발화의 길이는 헤더(Zeroth·FLEURS) 또는 디코딩(CV) 샘플 수에서 계산했다",
        "",
        '## Zeroth 문장 홀드아웃 (sha256("zeroth-holdout:" + 문장)[:8] % 100)',
        "",
        table(
            ["세트", "고유 문장", "발화"],
            [
                [
                    "Zeroth train 전체",
                    f"{ho['zeroth_train_unique_sentences']:,}",
                    f"{ho['zeroth_train_utterances_after_empty_drop']:,}",
                ],
                ["H (< 10)", f"{ho['H']['sentences']:,}", f"{ho['H']['utterances']:,}"],
                ["dev_h (< 5)", f"{ho['dev_h']['sentences']:,}", f"{ho['dev_h']['utterances']:,}"],
                ["test_h (5–9)", f"{ho['test_h']['sentences']:,}", f"{ho['test_h']['utterances']:,}"],
            ],
        ),
        "",
        f"- dev_h ∩ test_h 문장 = {ho['dev_h_test_h_sentence_overlap']}, "
        f"H ∩ train 문장 = {ho['H_train_sentence_overlap']}",
        f"- (참고) Zeroth 공식 test 문장 중 train에 있는 것 {ho['test_zeroth_sentences_in_train']:,}개"
        f"(발화 {ho['test_zeroth_utterances_with_sentence_in_train']:,}/{m['test_zeroth']['utterances']:,}), "
        f"H에 있는 것 {ho['test_zeroth_sentences_in_H']:,}개",
        "",
        "## 어휘 (`manifests/vocab.json`)",
        "",
        f"- 크기 {summary['vocab']['size']:,} = blank, `<unk>`, 공백 + "
        f"한글 음절 {summary['vocab']['syllables']:,}종 (최종 학습 전사 기준)",
        "",
        "## 평가 세트 OOV 음절과 한글 전용 비율",
        "",
        table(
            ["세트", "음절 토큰", "OOV 토큰", "OOV 종류", "OOV 비율", "한글 전용 발화", "비율"],
            [
                [
                    name,
                    f"{v['syllable_tokens']:,}",
                    f"{v['oov_tokens']:,}",
                    f"{v['oov_types']:,}",
                    f"{100 * v['oov_token_rate']:.3f} %",
                    f"{v['hangul_only_utterances']:,}/{m[name]['utterances']:,}",
                    f"{100 * v['hangul_only_share']:.1f} %",
                ]
                for name, v in summary["eval_oov"].items()
            ],
        ),
        "",
        "## 학습 토큰 속도 (공백 포함 토큰/초)",
        "",
        f"- p50 {rate['p50']:.2f}, p95 {rate['p95']:.2f}, 최대 {rate['max']:.2f}",
        "",
        "## 기타",
        "",
        f"- 정규화 뒤 빈 전사로 모든 세트에서 버린 발화: {summary['dropped_empty_text'] or '없음'}",
        f"- 오디오가 없어 버린 평가 발화: {summary['dropped_missing_audio_eval'] or '없음'}",
        "- Zeroth·FLEURS 가운데 16 kHz가 아니어서 리샘플링한 발화: "
        f"{summary['zeroth_fleurs_resampled'] or '없음'}",
        "- 오디오 저장소: "
        + ", ".join(
            f"{name} {v['utterances']:,}발화 {v['bytes'] / 1e9:.2f} GB"
            for name, v in summary["audio_store"].items()
        ),
        "- FLEURS `id`는 문장 id라 여러 화자가 공유하므로 발화 id는 `fleurs/<split>/<id>/<행 번호>`로 "
        "만들었다",
        "- CV22 validated 가운데 어느 tar에도 mp3가 없는 행은 `missing_audio`로 학습 제외 목록에 적었다",
        "",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--manifests", default="manifests")
    parser.add_argument("--pcm", default="data/pcm")
    parser.add_argument("--reports", default="reports/stage0")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    start = time.perf_counter()
    raw = Path(args.raw)
    stats: dict = {
        "empty_text": Counter(),
        "resampled": Counter(),
        "missing_audio_eval": Counter(),
        "cv_not_resampled": Counter(),
    }

    log("pass A: text, durations, splits")
    progress = Progress("pass A (probe)")
    utts = collect_parquet(raw, stats, progress)
    utts += collect_common_voice(raw, stats, progress)
    vocab = apply_train_exclusions(utts, stats)
    log(f"pass A done: {len(utts):,} utterances, vocab {len(vocab):,} ({time.perf_counter() - start:.0f} s)")

    log("pass B: decode audio and write the PCM store")
    write_pcm_store(raw, utts, Path(args.pcm), stats)
    log(f"pass B done ({time.perf_counter() - start:.0f} s)")

    manifests = write_manifests(utts, vocab, Path(args.manifests))
    summary = summarize(manifests, vocab, stats, time.perf_counter() - start)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "data_summary.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    with open(reports / "data_summary.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(summary))
    for name in MANIFEST_NAMES:
        info = summary["manifests"][name]
        log(f"{name:12s} {info['utterances']:7,} utts {info['hours']:7.2f} h")
    log(f"excluded: {json.dumps({k: v['utterances'] for k, v in summary['train_exclusions'].items()})}")
    log(f"H0.1 ratio {summary['h0_1']['ratio']:.5f} ({'pass' if summary['h0_1']['pass'] else 'FAIL'})")
    log(f"total wall time {summary['wall_time_s']:.0f} s")


if __name__ == "__main__":
    main()
