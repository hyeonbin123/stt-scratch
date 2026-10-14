"""0단계 데이터 준비(규칙: docs/experiments.md "공통", 2026-10-11).

data/raw 아래의 원본 코퍼스(Zeroth-Korean·FLEURS ko parquet, Common Voice 22 ko tsv + tar)를 읽어
사전 등록 규칙대로 정규화·분할·제외하고, 학습·평가가 읽는 다음 파일을 쓴다.

  manifests/<name>.jsonl          train, dev_mix, dev_h, test_zeroth, test_h, test_fleurs, test_cv
                                  (id 순 정렬)
  manifests/train_excluded.jsonl  제외 규칙으로 학습에서 빠진 후보와 그 사유(`reason`)
  manifests/vocab.json            최종 학습 전사로 만든 음절 어휘
  data/pcm/<name>.bin             남긴 모든 발화의 int16 little-endian mono 16 kHz 음성(커밋하지
                                  않음)
  data/pcm/<name>.index.jsonl     발화마다 한 줄의 {"id", "offset", "length"}(표본 단위)
  reports/stage0/data_summary.json / data_summary.md

parquet 코퍼스는 두 번 훑는다. 패스 A는 전사와 음성 헤더만 읽어 길이·분할·제외·어휘를 정하고, 패스 B는
음성을 디코딩해 PCM 저장소를 쓴다. Common Voice mp3는 헤더로 정확한 표본 수를 알 수 없어 패스 A에서
한 번 디코딩하고, 그 PCM을 패스 B까지 메모리에 둔다.

사용법: uv run python scripts/prepare_data.py [--raw data/raw] [--manifests manifests] [--pcm data/pcm]
                                              [--reports reports/stage0]
"""

# [왜 필요한가] 모든 단계가 같은 분할·같은 정규화·같은 음성을 써야 비교가 성립한다. 그래서 분할은 seed 없는
# SHA-256 해시로, 정규화는 평가와 같은 함수(stt_scratch.text.normalize_train) 하나로, 평가 세트는 '줄어들면
# 멈춤'으로 고정한다(규칙 변경 6). 매니페스트·어휘는 커밋하고 두 번 실행해 SHA-256이 같음을
# 확인했다(docs/ai-verification.md).
# [설계 이유] parquet 코퍼스를 두 번 훑는 것은 학습 세트 전체 PCM(약 52 h)을 메모리에 들고 있지 않으려는
# 것이다. CV mp3는 헤더의 표본 수가 실제 디코딩 길이와 달라서(표본 40/40에서 불일치) 패스 A에서 바로
# 디코딩하며, CV는 약 2 h로 작아 그 PCM을 패스 B까지 메모리에 둘 수 있다.
# [흐름] 패스 A(collect_parquet, collect_common_voice) → 학습 제외와 어휘(apply_train_exclusions) → 패스
# B(write_pcm_store) → 매니페스트·어휘 저장(write_manifests) → 요약(summarize, render_markdown).
# [연결] 입력은 scripts/download_data.py의 data/raw/. 산출물은 stt_scratch.data(PcmStore, read_manifest)와
# text.Vocab이 읽고, 그 뒤 scripts/compute_mel_stats.py를 돌린다. 규칙 함수(해시, 프레임 공식, CV 분할,
# 제외)는 tests/test_prepare_data.py가 이 파일을 직접 불러 검사한다.

from __future__ import annotations

# 표준 라이브러리: hashlib(문장 홀드아웃 해시), csv·tarfile(CV tsv 전사와 mp3 tar), io(메모리 바이트를
# soundfile에 파일처럼 넘김), math(프레임 수 올림), Counter(제외·리샘플링 집계).
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

# numpy: int16 PCM 배열과 통계. pyarrow.parquet: parquet를 배치 단위로 흘려 읽는다(파일 전체를 메모리에 올리지
# 않음). soundfile: FLAC/mp3 디코딩과 헤더 길이 읽기. torch·torchaudio: 16 kHz가 아닌 음성(CV mp3 32 kHz)을
# 리샘플링한다.
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

# 프로젝트 텍스트 모듈: 학습·평가 공통 정규화(normalize_train), 숫자·라틴·한글 전용 판별, 음절 어휘(Vocab),
# CTC 최소 프레임 수(ctc_min_frames). 정규화를 "스크립트 하나"로 두라는 규칙(docs/experiments.md "텍스트
# 정규화") 때문에 여기서 따로 구현하지 않고 가져온다.
from stt_scratch.text import HANGUL, Vocab, ctc_min_frames, hangul_only, has_digit, has_latin, normalize_train

# 모델 입력과 같은 프레임 규칙(model/features.py: 16 kHz에서 25 ms 창 = 400표본, 10 ms 간격 = 160표본,
# center=False)과 서브샘플링(인과 stride-2 합성곱 두 번 = 4배, 초당 25프레임). CTC 길이 제외를 실제 모델 출력
# 길이로 계산하려고 같은 값을 둔다.
SAMPLE_RATE = 16_000
WINDOW, HOP = 400, 160
SUBSAMPLING_STRIDES = (2, 2)
# 학습 발화 길이 상한(초). 20 s를 넘는 발화는 학습에서 뺀다(docs/experiments.md "데이터 분할"; 실제로 FLEURS
# 24발화).
MAX_TRAIN_SECONDS = 20.0
# Zeroth 문장 홀드아웃 규칙: sha256("zeroth-holdout:" + 문장)의 앞 8자리 16진수를 정수로 보아 % 100이 10
# 미만이면 홀드아웃 H, 그중 5 미만이면 dev_H, 5~9면 test_H. 접두 문자열(salt)은 이 해시를 다른 용도의 해시와
# 구분해 결과가 우연히 겹치지 않게 한다.
HOLDOUT_SALT = "zeroth-holdout:"
HOLDOUT_PERCENT, DEV_H_PERCENT = 10, 5
# 가설 H0.1의 기준: 25 fps에서 CTC 길이 제약에 걸리는 학습 발화 비율 ≤ 0.5 %(실측 0 %).
H0_1_THRESHOLD = 0.005
# 만드는 매니페스트 7개. train 외 6개는 평가 세트(dev_mix·dev_h는 선택용 dev, 나머지 4개는 7단계 봉인
# test)이고 평가 세트는 결손이 생기면 실행을 멈춘다.
MANIFEST_NAMES = ("train", "dev_mix", "dev_h", "test_zeroth", "test_h", "test_fleurs", "test_cv")
EVAL_NAMES = MANIFEST_NAMES[1:]
# 학습 제외 사유(보고서 표 순서). digit·latin은 정규화 규칙 4·5, too_long은 20 s 상한, ctc_length는 CTC 길이
# 제약, missing_audio는 tar에 음성이 없는 CV 행(규칙에 없던 사유라 추가해 기록).
EXCLUSION_REASONS = ("digit", "latin", "too_long", "ctc_length", "missing_audio")
# 진행 상황 출력 간격(발화 수)과 parquet를 한 번에 읽는 행 수.
PROGRESS_EVERY = 2_000
PARQUET_BATCH = 256


# ----------------------------------------------------------------------------------------------------
# 분할 규칙
# ----------------------------------------------------------------------------------------------------
# 난수 seed 대신 문장 자체의 SHA-256을 쓰므로 실행 순서·환경과 무관하게 같은 문장은 언제나 같은 구간에
# 들어간다(분할은 seed 있는 Generator나 SHA-256만 쓴다는 프로젝트 규칙). 앞 8자리(32비트)면 % 100의 치우침이
# 무시할 만큼 작다.
def holdout_bucket(sentence: str) -> int:
    """정규화된 Zeroth 문장의 구간 번호(0..99): int(sha256("zeroth-holdout:" + s)[:8], 16) % 100."""
    digest = hashlib.sha256((HOLDOUT_SALT + sentence).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


# 왜 문장 단위인가: Zeroth train은 3,003개 문장을 105명이 나눠 읽은 구조라, 발화나 화자 단위로 떼면 같은
# 문장이 학습에도 들어가 '본 문장' 평가가 된다. 문장을 통째로 빼야 '처음 보는 문장' 평가(dev_H·test_H)가
# 된다(docs/data.md). 결과: H 318문장 2,393발화, dev_h 151문장, test_h 167문장, 서로·학습과 겹침 0.
def zeroth_target(sentence: str) -> str:
    """Zeroth *train* 발화의 매니페스트를 문장 구간으로 정한다: dev_h(< 5), test_h(< 10), 나머지 train."""
    bucket = holdout_bucket(sentence)
    if bucket < DEV_H_PERCENT:
        return "dev_h"
    if bucket < HOLDOUT_PERCENT:
        return "test_h"
    return "train"


# 파일 경로 기준으로 빼서 같은 클립이 학습과 평가에 동시에 들어가지 않게 한다(tests/test_prepare_data.py가
# 검사).
def cv_train_paths(validated: Iterable[str], dev: Iterable[str], test: Iterable[str]) -> list[str]:
    """Common Voice 학습 = validated.tsv에서 dev.tsv나 test.tsv에 나오는 `path`를 모두 뺀 것."""
    held = set(dev) | set(test)
    return [path for path in validated if path not in held]


# ----------------------------------------------------------------------------------------------------
# 제외 규칙
# ----------------------------------------------------------------------------------------------------
# model/features.py의 n_mel_frames와 같은 공식이다.
def mel_frames(n_samples: int) -> int:
    """center=False일 때 16 kHz 신호의 fbank 프레임 수: floor((n - 400) / 160) + 1, 음수는 0으로 둔다."""
    return max(0, (n_samples - WINDOW) // HOP + 1)


# 표본 수에서 바로 구한다. 인과 Conv2d 서브샘플링(model/subsampling.py subsampled_length)과 같은 공식이다.
def output_frames(n_samples: int) -> int:
    """stride-2 합성곱 두 번을 거친 인코더 출력 프레임 수(25 fps): ceil(ceil(n_mel / 2) / 2)."""
    frames = mel_frames(n_samples)
    for stride in SUBSAMPLING_STRIDES:
        frames = math.ceil(frames / stride)
    return frames


# U는 토큰 수, 반복 수는 연속 같은 토큰 수다. CTC는 같은 토큰이 이어지면 사이에 blank가 하나 있어야 하므로
# 최소 U + 반복 수 프레임이 필요하고, 모자라면 손실이 무한대가 된다. +1은 등록 규칙(docs/experiments.md
# "데이터 분할")의 여유분이다.
# 주의: 원래 길이로만 검사한다. 학습 중 속도 교란 1.1배는 음성을 약 9 % 짧게 만들지만, 학습 토큰률이 최대
# 10.64 토큰/s라 25 fps에 여유가 크다(reports/stage0/data_summary.md).
def ctc_fits(n_samples: int, ids: Sequence) -> bool:
    """학습 길이 조건: 출력 프레임 >= U + 반복 수 + 1(토큰 id로 세든 글자로 세든 반복 수는 같다)."""
    return output_frames(n_samples) >= ctc_min_frames(list(ids)) + 1


# 숫자가 든 발화는 한국어 숫자 읽기가 두 가지라 자동 변환이 틀리기 쉬워서, 라틴 문자가 하나라도 든 발화는
# 어휘가 한글 음절뿐이라서 뺀다(규칙 4·5, 규칙 변경 4). 여러 사유에 걸리면 앞의 것 하나만 기록한다.
def exclusion_reason(text: str, duration_s: float) -> str | None:
    """어휘 없이 정할 수 있는 학습 제외 사유를 돌려준다(없으면 None). text는 정규화됐고 비어 있지 않다."""
    if has_digit(text):
        return "digit"
    if has_latin(text):
        return "latin"
    if duration_s > MAX_TRAIN_SECONDS:
        return "too_long"
    return None


# ----------------------------------------------------------------------------------------------------
# 음성
# ----------------------------------------------------------------------------------------------------
# 패스 B(Zeroth·FLEURS)와 패스 A(CV mp3)에서 쓴다.
def decode_pcm16(blob: bytes) -> tuple[np.ndarray, int]:
    """음성 파일 바이트를 int16 mono 16 kHz로 디코딩해 (pcm, 원래 표본화율)을 돌려준다.

    여러 채널이면 평균해 mono로 만들고, 원래 표본화율이 16 kHz가 아니면 CPU에서
    torchaudio.functional.resample로 리샘플링하며, 표본은 반올림한 뒤 int16 범위로 자른다.
    """
    # always_2d=True로 채널 수와 무관하게 (표본, 채널) 모양을 받고, 여러 채널이면 평균해 mono로 만든다.
    data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    # 16 kHz가 아니면(CV mp3는 32 kHz) torchaudio로 리샘플링한다. probe_samples의 길이 계산이 이 리샘플러의
    # 출력 길이 공식을 따른다.
    if rate != SAMPLE_RATE:
        wave = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
        mono = torchaudio.functional.resample(wave, rate, SAMPLE_RATE).squeeze(0).numpy()
    # -1~1 실수를 int16 범위로 반올림하고 넘치는 값은 자른다. '<i2'는 little-endian int16으로, PcmStore가 읽는
    # 형식과 같다.
    pcm = np.clip(np.rint(mono * 32768.0), -32768, 32767).astype("<i2")
    return pcm, rate


# 패스 A가 Zeroth·FLEURS 2만여 발화의 길이를 디코딩 없이 빨리 얻으려고 쓴다. 패스 B가 실제 디코딩 길이와
# 같은지 다시 확인한다. 16 kHz가 아니면 ceil(표본 수 × 16000 ÷ 원래 표본화율)로 환산한다.
def probe_samples(blob: bytes) -> tuple[int, int]:
    """헤더만 읽어 (16 kHz 기준 표본 수, 원래 표본화율)을 돌려준다.

    FLAC/WAV 헤더 값은 정확하다. 원래 표본화율이 16 kHz와 다르면 16 kHz 표본 수는 torchaudio 리샘플러의
    출력 길이와 같은 식을 따른다.
    """
    with sf.SoundFile(io.BytesIO(blob)) as handle:
        frames, rate = handle.frames, handle.samplerate
    if rate != SAMPLE_RATE:
        frames = math.ceil(frames * SAMPLE_RATE / rate)
    return frames, rate


# ----------------------------------------------------------------------------------------------------
# 원본 읽기
# ----------------------------------------------------------------------------------------------------
# 원본에서 읽은 발화 한 줄: id, 코퍼스, 원래 분할, 화자, 원문 전사, 인코딩된 음성 바이트(blob). 정규화·분할
# 전의 상태다.
@dataclass(frozen=True)
class RawUtt:
    id: str
    corpus: str
    src_split: str
    speaker: str | None
    text_raw: str
    blob: bytes


# 음성 바이트가 든 큰 파일도 메모리에 한꺼번에 올리지 않으려는 것이다.
def iter_parquet_rows(files: list[Path], columns: list[str]) -> Iterator[dict]:
    """여러 parquet 파일의 행을 순서대로, 작은 배치(PARQUET_BATCH 행)씩 끊어 행 사전으로 내준다."""
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH, columns=columns):
            yield from batch.to_pylist()


# id는 정렬된 parquet 순서의 행 번호라, 같은 파일(download manifest에 커밋 sha 기록)이면 언제나 같은 id가
# 된다. 패스 A·B가 같은 순서로 두 번 읽어 id로 맞춘다.
def iter_zeroth(raw: Path, split: str) -> Iterator[RawUtt]:
    """Zeroth-Korean 한 분할의 행을 RawUtt로 내준다.

    id = zeroth/<split>/<parquet 순서의 행 번호>, speaker = speaker_id.
    """
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


# 행 번호를 덧붙이는 것은 docs/ai-verification.md에 기록된 수정이다. FLEURS에는 화자 정보가 없어 speaker는
# None이다.
def iter_fleurs(raw: Path, split: str) -> Iterator[RawUtt]:
    """FLEURS 한국어 한 분할의 행을 RawUtt로 내준다.

    `id` 열은 최대 세 화자가 공유하는 *문장* id이므로, 발화 id가 유일하도록 행 번호를 덧붙인다:
    fleurs/<split>/<id>/<행 번호>.
    """
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


# CV tsv를 행 사전 목록으로 읽는다. QUOTE_NONE은 문장 안의 따옴표를 글자 그대로 두게 한다(CV tsv는 따옴표로
# 감싼 CSV가 아니다).
def read_cv_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


# ----------------------------------------------------------------------------------------------------
# 발화 기록
# ----------------------------------------------------------------------------------------------------
# 전처리 중 한 발화의 상태: 원문·정규화 전사, 16 kHz 표본 수, 들어갈 매니페스트(학습에서 빠지면 None), 제외
# 사유, CV의 경우 디코딩된 PCM. 패스 A에서 만들어 제외 판정·패스 B·매니페스트 작성까지 이어 쓴다.
@dataclass
class Utt:
    id: str
    corpus: str
    src_split: str
    speaker: str | None
    text_raw: str
    text: str
    n_samples: int = 0  # 16 kHz 표본 수(패스 A는 헤더 값, 패스 B에서 디코딩 길이와 대조)
    manifest: str | None = None  # 들어갈 매니페스트. 학습에서 제외되면 None
    reason: str | None = None  # 학습 제외 사유
    pcm: np.ndarray | None = None  # 메모리에 둔 디코딩 음성(Common Voice만)

    @property
    def duration_s(self) -> float:
        return self.n_samples / SAMPLE_RATE


class Progress:
    """긴 작업(디코딩에 수 분)의 진행 상황을 PROGRESS_EVERY 발화마다 경과 시간과 함께 한 줄씩 찍는다."""

    def __init__(self, label: str) -> None:
        self.label, self.count, self.start = label, 0, time.perf_counter()

    def tick(self, n: int = 1) -> None:
        before = self.count
        self.count += n
        if self.count // PROGRESS_EVERY != before // PROGRESS_EVERY:
            log(f"{self.label}: {self.count:,} utterances ({time.perf_counter() - self.start:.0f} s)")


# 즉시 출력(flush) 도우미. 로그를 파일로 돌려 받을 때도 줄이 늦게 쌓이지 않게 한다.
def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------------------------------------------
# 패스 A: 전사, 길이, 분할
# ----------------------------------------------------------------------------------------------------
# RawUtt를 Utt로 바꾸며 전사를 학습·평가 공통 규칙(NFC, 문장부호 → 공백, 공백 축약)으로 정규화한다.
def make_utt(src: RawUtt, n_samples: int) -> Utt:
    return Utt(
        src.id, src.corpus, src.src_split, src.speaker, src.text_raw, normalize_train(src.text_raw), n_samples
    )


# 헤더 길이와 정규화 전사로 Utt 목록을 만들고 각 발화의 매니페스트를 정한다. Zeroth train은 문장 해시로
# train/dev_h/test_h, FLEURS train/validation/test는 train/dev_mix/test_fleurs로 간다. 학습 제외(숫자·라틴·
# 길이)는 아직 하지 않고 apply_train_exclusions에서 한꺼번에 한다.
def collect_parquet(raw: Path, stats: dict, progress: Progress) -> list[Utt]:
    """패스 A의 parquet 부분: Zeroth(train은 문장 해시 홀드아웃, test → test_zeroth)와 FLEURS를 모은다.

    FLEURS는 train / dev_mix / test(test_fleurs)로 나눈다.
    """
    utts: list[Utt] = []
    # (출처 이름, 행 반복자, 매니페스트) 목록. 매니페스트가 None이면 문장 해시로 정한다(Zeroth train).
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
            # 16 kHz가 아닌 발화 수를 출처별로 센다(보고서의 '리샘플링한 발화'; 실제로는 없음).
            if rate != SAMPLE_RATE:
                stats["resampled"][source] += 1
            utt = make_utt(src, n_samples)
            progress.tick()
            target = manifest if manifest is not None else zeroth_target(utt.text)
            # 정규화 뒤 전사가 빈 발화: 학습 세트면 세기만 하고 버리고, 평가 세트면 require_train이 실행을
            # 멈춘다(평가 세트를 조용히 줄이지 않음).
            if not utt.text:
                require_train(utt, target, "empty normalized transcript")
                stats["empty_text"][source] += 1
                continue
            utt.manifest = target
            utts.append(utt)
    return utts


# tar마다 처음부터 끝까지 한 번씩 훑는다. tar는 임의 접근보다 순차 읽기가 빠르므로 tsv 순서가 아니라 tar 안의
# 순서대로 처리한다.
def decode_cv_tars(cv_root: Path, splits: Iterable[str], wanted: set[str], progress: Progress) -> dict:
    """audio/ko/<split>/*.tar 안의 필요한 mp3만 차례로 디코딩해 파일 이름 → (pcm, rate) 사전을 돌려준다."""
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


# 음성을 여기서 디코딩하는 것은 정확한 표본 수를 얻으려는 것이고, PCM은 Utt에 담아 둔다. id는
# cv22/<원래 분할>/<mp3 파일 이름>, 화자는 client_id.
def collect_common_voice(raw: Path, stats: dict, progress: Progress) -> list[Utt]:
    """패스 A의 CV22 ko 부분: validated − dev − test → train, dev → dev_mix, test → test_cv(음성도 디코딩)."""
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
    # 세 계획에 들어간 클립 이름만 디코딩한다. tar는 train·dev·test 폴더에서 찾는다.
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
            # 어느 tar에도 mp3가 없는 행: 학습이면 missing_audio 사유로 제외 목록에 남기고(manifest None),
            # 평가 세트면 실행을 멈춘다. 실제로 validated의 1행이 이 경우였다(docs/ai-verification.md).
            if hit is None:
                require_train(utt, manifest, "no audio in the tar files")
                utt.reason = "missing_audio"
                utts.append(utt)
                continue
            utt.pcm, rate = hit
            if rate == SAMPLE_RATE:
                stats["cv_not_resampled"][source] += 1
            # CV는 헤더가 아니라 디코딩한 실제 길이를 쓴다.
            utt.n_samples = len(utt.pcm)
            utt.manifest = manifest
            utts.append(utt)
    return utts


# 버려야 하는 발화가 평가 세트(dev_mix, dev_h, 봉인 test) 것이면 RuntimeError로 멈춘다. 평가 세트를 조용히
# 줄이지 않는다는 규칙 변경 6을 코드로 강제한 것이다. 학습 세트면 아무 일도 하지 않는다.
def require_train(utt: Utt, target: str | None, problem: str) -> None:
    """평가 세트는 고정이다: 남길 수 없는 평가 발화가 나오면 세트를 줄이지 않고 실행을 멈춘다."""
    if target != "train":
        raise RuntimeError(f"evaluation utterance {utt.id} ({target}) cannot be kept: {problem}")


# 제외 규칙은 숫자, 라틴, 20 s 초과, CTC 길이 순이다.
# 왜 이 순서인가: 어휘는 CTC 길이 제외까지 모든 학습 제외를 적용한 뒤의 전사로 만든다(규칙 변경 5).
def apply_train_exclusions(utts: list[Utt], stats: dict) -> Vocab:
    """학습 매니페스트 발화에 제외 규칙 1~3과 CTC 길이 제외를 적용한 뒤 최종 학습 전사로 어휘를 만든다.

    CTC 길이는 토큰 id 대신 글자로 검사한다. 끝까지 남는 학습 글자는 모두 어휘에 들어가므로 토큰 id로 셀
    때와 반복 수가 같다.
    """
    checked = 0
    for utt in utts:
        if utt.manifest == "train":
            reason = exclusion_reason(utt.text, utt.duration_s)
            # 텍스트·길이 제외를 통과한 발화만 CTC 길이를 검사하고 그 수를 센다. 이 수가 H0.1 비율의 분모다.
            if reason is None:
                checked += 1
                if not ctc_fits(utt.n_samples, utt.text):
                    reason = "ctc_length"
            if reason is not None:
                # 제외된 발화는 매니페스트를 None으로 바꾸고 사유를 남긴다. train_excluded.jsonl에 기록되고
                # PCM은 쓰지 않는다.
                utt.reason, utt.manifest = reason, None
    stats["ctc_checked"] = checked
    # 최종 학습 전사에 한 번 이상 나온 한글 음절을 빈도순으로 모아 blank·<unk>·공백 뒤에 붙인 어휘(실측 1,288
    # = 3 + 음절 1,285).
    return Vocab.build([utt.text for utt in utts if utt.manifest == "train"])


# ----------------------------------------------------------------------------------------------------
# 패스 B: PCM 저장소
# ----------------------------------------------------------------------------------------------------
# 모든 발화를 큰 파일 하나에 이어 쓰고, 색인 한 줄은 {id, offset, length}(표본 단위)다.
# 왜 이 형식인가: 학습 때 stt_scratch.data.PcmStore가 이 파일을 numpy 메모리 맵으로 열어 오프셋으로 바로 잘라
# 읽는다. 매 epoch FLAC/mp3를 다시 디코딩하지 않고, 수만 개의 작은 파일을 여닫지 않아도 된다.
class PcmWriter:
    """매니페스트 하나의 PCM 저장소 작성기.

    int16 little-endian 표본을 <name>.bin에 이어 쓰고, 발화마다 색인(<name>.index.jsonl)에 한 줄을 쓴다.
    """

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


# 패스 B: parquet를 패스 A와 같은 순서로 다시 훑어 매니페스트에 남은 발화만 디코딩해 해당 저장소에 쓰고, CV는
# 메모리에 둔 PCM을 쓴다. 매니페스트별 발화 수·표본 수·바이트를 stats에 남긴다.
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
                # 빈 전사로 버렸거나 학습에서 제외된 발화는 음성을 쓰지 않는다.
                if utt is None or utt.manifest is None:
                    continue
                pcm, _rate = decode_pcm16(src.blob)
                # 패스 A의 헤더 길이와 실제 디코딩 길이가 다르면 멈춘다. 매니페스트의 길이(duration_s)와
                # 저장된 음성이 반드시 일치하게 하는 검사다.
                if len(pcm) != utt.n_samples:
                    raise RuntimeError(f"{src.id}: header says {utt.n_samples} samples, decoded {len(pcm)}")
                writers[utt.manifest].write(utt.id, pcm)
                progress.tick()
        # CV 발화는 패스 A에서 디코딩해 둔 PCM을 쓰고, 쓴 뒤 바로 메모리를 놓는다.
        for utt in utts:
            if utt.corpus == "cv22" and utt.manifest is not None:
                writers[utt.manifest].write(utt.id, utt.pcm)
                utt.pcm = None
                progress.tick()
    finally:
        # 중간에 예외가 나도 파일 핸들을 닫는다(finally).
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
# 매니페스트
# ----------------------------------------------------------------------------------------------------
# 매니페스트 한 줄. 정규화 전사(text)와 원문(text_raw), 길이, hangul_only(숫자·라틴이 없는 발화; 평가의 '한글
# 전용 부분집합' 수치에 쓴다), 토큰 수와 CTC 최소 프레임 수(참고값)를 담는다.
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


# JSONL 저장. 한글을 그대로(ensure_ascii=False) 쓰고 줄 끝을 LF로 고정해, Windows에서 실행해도 같은 바이트가
# 나와 커밋 diff와 SHA-256 비교가 안정적이다.
def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_vocab(vocab: Vocab, path: Path) -> None:
    """어휘를 저장한다. Vocab.save와 같은 JSON을 줄 끝 LF로 고정해 쓴다(Vocab.load로 다시 읽는다)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"tokens": list(vocab.tokens)}, ensure_ascii=False, indent=0))


# Utt 목록을 매니페스트별로 나눠 id 순으로 정렬해 쓴다. 처리 순서와 무관하게 같은 파일이 나오도록 정렬한다.
# 학습에서 제외된 발화는 사유와 함께 train_excluded.jsonl에 쓰고, 어휘도 여기서 저장한다. 요약에 쓰려고 제외
# 목록도 함께 돌려준다.
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
# 요약
# ----------------------------------------------------------------------------------------------------
# 행들의 길이 합(시간, 소수 셋째 자리).
def hours(rows: Iterable[dict]) -> float:
    return round(sum(row["duration_s"] for row in rows) / 3600, 3)


# 0단계 요약 수치를 만든다: 매니페스트별 발화·시간·화자, 제외 사유별 수, 홀드아웃 검사, 평가 세트 OOV, 학습
# 토큰률, H0.1 판정. docs/experiments.md "0단계 결과"의 수치가 여기서 나왔다.
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
    # 홀드아웃 검사: dev_h와 test_h, 홀드아웃과 학습 사이의 문장 겹침이 0이어야 '처음 보는 문장' 평가가
    # 성립한다. Zeroth 공식 test 문장 가운데 학습·홀드아웃에 있는 수도 참고로 센다(실측: 학습 367문장, H
    # 55문장).
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

    # 평가 세트별 OOV 음절(어휘에 없어 모델이 낼 수 없는 음절) 수와 비율, 한글 전용 발화 비율. 어휘 규칙이
    # 평가 세트별 OOV 음절 수를 CER 하한으로 보고하라고 정했다(docs/experiments.md "어휘").
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

    # 학습 발화의 토큰률(공백 포함 토큰/초) p50·p95·최대. 25 fps 출력에 얼마나 여유가 있는지 보여 준다(실측
    # 최대 10.64).
    rates = np.array([row["n_tokens"] / row["duration_s"] for row in train]) if train else np.zeros(1)
    token_rate = {
        "p50": round(float(np.percentile(rates, 50)), 3),
        "p95": round(float(np.percentile(rates, 95)), 3),
        "max": round(float(rates.max()), 3),
    }
    # H0.1 판정: CTC 길이로 제외된 발화 ÷ 검사한 발화 ≤ 0.5 %.
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


# Markdown 표 문자열을 만드는 도우미.
def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# 요약 사전을 사람이 읽는 한국어 보고서(reports/stage0/data_summary.md)로 만든다. 매니페스트 표, 학습 제외
# 표와 H0.1, Zeroth 홀드아웃 표, 어휘 크기, 평가 세트 OOV, 토큰률, 기타 처리 기록 순서다.
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
# 전체 실행: 패스 A → 제외·어휘 → 패스 B → 매니페스트 → 요약 저장. 인자는 원본·매니페스트·PCM·보고서 폴더(모두
# 현재 작업 폴더 기준이므로 저장소 루트에서 실행한다).
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--manifests", default="manifests")
    parser.add_argument("--pcm", default="data/pcm")
    parser.add_argument("--reports", default="reports/stage0")
    args = parser.parse_args()
    # Windows 콘솔의 기본 인코딩에서 한글 로그가 깨지거나 예외가 나지 않게 표준 출력을 UTF-8로 바꾼다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    start = time.perf_counter()
    raw = Path(args.raw)
    # 출처별 집계: 빈 전사로 버린 수, 리샘플링한 수, 이미 16 kHz였던 CV 수. missing_audio_eval은 보고 칸만
    # 있고, 평가 세트 결손은 require_train이 실행을 멈추므로 실제로는 늘지 않는다.
    stats: dict = {
        "empty_text": Counter(),
        "resampled": Counter(),
        "missing_audio_eval": Counter(),
        "cv_not_resampled": Counter(),
    }

    # 패스 A: 전사·길이·분할을 정하고 학습 제외와 어휘까지 끝낸다.
    log("pass A: text, durations, splits")
    progress = Progress("pass A (probe)")
    utts = collect_parquet(raw, stats, progress)
    utts += collect_common_voice(raw, stats, progress)
    vocab = apply_train_exclusions(utts, stats)
    log(f"pass A done: {len(utts):,} utterances, vocab {len(vocab):,} ({time.perf_counter() - start:.0f} s)")

    # 패스 B: 남은 발화의 음성만 디코딩해 PCM 저장소를 쓴다.
    log("pass B: decode audio and write the PCM store")
    write_pcm_store(raw, utts, Path(args.pcm), stats)
    log(f"pass B done ({time.perf_counter() - start:.0f} s)")

    # 매니페스트·어휘를 쓰고 요약 JSON과 Markdown을 저장한 뒤 콘솔에 핵심 수치를 찍는다.
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
