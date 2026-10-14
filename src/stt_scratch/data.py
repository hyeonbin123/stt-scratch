"""매니페스트, scripts/prepare_data.py가 쓴 int16 PCM 저장소, 길이별 묶음 배치."""

# [파일 역할] 학습·평가 데이터 입력부: 매니페스트(jsonl) 읽기, 전처리가 만든 int16 PCM 저장소에서 파형 꺼내기,
# 학습용 Dataset·collate, 길이별 묶음 배치 샘플러.
# [왜 PCM 저장소인가] scripts/prepare_data.py가 FLAC·mp3를 한 번만 16 kHz int16으로 풀어 매니페스트별 .bin
# 하나에 이어 붙여 둔다. 학습 중에는 메모리 맵으로 필요한 구간만 읽으므로 매 epoch 디코딩 비용이 없고, 0단계
# 측정에서 데이터 대기 비율은 1.1 %였다(docs/experiments.md 0단계 결과). 원본 parquet 음성과 비트 단위로
# 같음도 확인했다(Zeroth 37발화 표본, docs/experiments.md 1단계 진단).
# [연결] train.py가 SpeechDataset·collate·DurationBucketSampler로 DataLoader를 만들고, evaluate.py와
# scripts/의 평가 도구가 read_manifest·PcmStore를 쓴다.

from __future__ import annotations

# numpy: int16 .bin 파일을 np.memmap으로 열어 통째로 메모리에 올리지 않고 필요한 구간만 읽는다. torch:
# Dataset·Sampler 기반 클래스와 텐서 변환. random: epoch별 배치 순서 섞기(시드 고정).
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stt_scratch.text import Vocab


# 매니페스트 한 줄 = 발화 하나. text는 정규화된 전사, hangul_only는 숫자·라틴이 없는 발화 표시(평가의 '한글
# 전용' 부분집합), corpus는 zeroth·fleurs·cv22 같은 출처다. 매니페스트에 있는 다른 필드(원문 전사, 토큰 수
# 등)는 여기서 읽지 않는다.
@dataclass(frozen=True)
class Utterance:
    id: str
    corpus: str
    text: str
    duration_s: float
    hangul_only: bool
    speaker: str | None = None


# manifests/<이름>.jsonl을 읽어 Utterance 목록으로. 빈 줄은 건너뛰고, hangul_only가 없으면 True로 본다.
# 학습(train)·dev(dev_mix, dev_h)·봉인 test(test_zeroth 등) 매니페스트가 모두 같은 형식이다.
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


# index.jsonl에는 발화 id → (.bin 안의 시작 표본 위치, 표본 수)가 들어 있다. id로 바로 찾아 읽으므로
# Dataset·평가 모두 순서와 무관하게 접근할 수 있다.
class PcmStore:
    """매니페스트 하나의 메모리 맵 int16 16 kHz 오디오: data/pcm/<name>.bin + <name>.index.jsonl."""

    def __init__(self, pcm_dir: Path, name: str) -> None:
        self.bin = np.memmap(pcm_dir / f"{name}.bin", dtype=np.int16, mode="r")
        self.index: dict[str, tuple[int, int]] = {}
        with open(pcm_dir / f"{name}.index.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.index[d["id"]] = (int(d["offset"]), int(d["length"]))

    # 발화 하나의 파형을 float32 −1~1(int16 ÷ 32768) 1차원 텐서로 돌려준다. memmap 구간을 np.asarray로 복사해
    # 파일과 분리된 배열을 만든다.
    def waveform(self, utt_id: str) -> torch.Tensor:
        offset, length = self.index[utt_id]
        arr = np.asarray(self.bin[offset : offset + length], dtype=np.float32) / 32768.0
        return torch.from_numpy(arr)


# 학습용 Dataset: 발화마다 파형(특징 추출 전), 음절 id 목록, 원문 전사를 돌려준다. log-mel·속도
# 교란·SpecAugment는 GPU에서 배치 단위로 하므로 여기서는 하지 않는다.
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


# 길이가 다른 파형을 여기서 패딩하지 않는 이유: 속도 교란이 길이를 바꾸므로 train.prepare_batch가 GPU에서
# 교란한 뒤에 패딩한다.
def collate(items: list[dict]) -> dict:
    """배치를 리스트로만 묶는다. 파형도 리스트로 둔다(속도 교란은 나중에 GPU에서 한다)."""
    return {
        "ids": [it["id"] for it in items],
        "wavs": [it["wav"] for it in items],
        "tokens": [it["tokens"] for it in items],
        "texts": [it["text"] for it in items],
    }


# 길이순으로 묶으면 패딩 낭비가 작다. 같은 시드·epoch면 데이터 순서가 같아 --resume 재개가 원래 순서를 그대로
# 잇는다(규칙 변경 10). max_duration 300 s에서 epoch당 697 step, 평균 배치 32발화였다(docs/experiments.md
# 0단계 결과).
# 같은 비교 안의 런은 메모리가 남아도 배치를 키우지 않는다는 규칙(docs/experiments.md 선택 규칙)에 따라
# max_duration은 처방의 일부로 기록된다(train.RECIPE_KEYS).
class DurationBucketSampler(torch.utils.data.Sampler[list[int]]):
    """길이가 비슷한 발화끼리 묶되, 배치의 패딩 포함 길이가 max_duration 초를 넘지 않게 하는 배치 샘플러.

    배치 비용은 길이의 합이 아니라 '가장 긴 발화 × 발화 수'(패딩까지 포함한 계산량)로 잰다. 생성할 때 한 번
    길이순으로 정렬해 배치로 자르고, epoch마다 그 배치들의 순서만 epoch에서 만든 시드(seed × 1000 + epoch)로
    섞는다. 속도 교란(×1/0.9)을 걸어도 예산 안에 들도록 길이에 10 % 여유를 곱한다.
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
            # 속도 교란 0.9배는 발화를 약 1.11배 늘리므로, 길이에 10 % 여유를 곱해 예산을 잡는다.
            d = self.durations[i] * 1.1
            new_max = max(cur_max, d)
            # 패딩 기준 배치 비용 = 가장 긴 발화 × 배치 크기. 이 발화를 넣으면 비용이 예산을 넘을 때 현재
            # 배치를 닫고 새 배치를 시작한다.
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

    # epoch별 시드로 배치 순서를 섞는다(shuffle=False면 길이순 그대로). DataLoader의 batch_sampler로 쓰여 배치
    # 하나가 인덱스 목록 하나다.
    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed * 1000 + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)
