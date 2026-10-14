"""내려받은 코퍼스의 재고를 조사한다: 시간, 발화 수, 텍스트 통계, 문장 겹침, 음절 범위.

download_data.py가 받은 data/raw/*의 세 코퍼스(Zeroth-Korean, FLEURS ko, Common Voice ko)를 읽어 분할별
발화 수·시간·고유 문장 수·음절 종류·글자율(공백 뺀 글자/초)·숫자/라틴/문장부호 비율·표본화율, 분할 사이
문장 겹침, 평가 분할의 학습 밖 음절(OOV)을 센다. 결과는 data/inventory.json에 쓰고 Markdown 요약을 표준
출력에 찍는다. 음성은 길이와 표본화율을 재려고만 디코딩한다(특징은 계산하지 않는다).

사용법: uv run python scripts/inventory_data.py [--raw data/raw] [--out data/inventory.json]
"""

# [왜 필요한가] 데이터 설계를 숫자로 정하기 위한 조사다. 여기서 Zeroth 공식 test 문장의 99.8 %(423개 중
# 422개)가 train에 그대로 있다는 것을 찾았고, 이 발견으로 평가 설계를 바꿨다: Zeroth 문장 단위
# 홀드아웃(dev_H·test_H)과 다른 도메인 test(FLEURS·CV)로 판정하고 공식 test는 비교용으로만 본다(docs/data.md
# "문장 겹침", docs/ai-verification.md). 그 밖에 FLEURS 정규화 열에도 숫자가 26.8 % 남아 있다는 것, CV mp3가
# 32 kHz라는 것, 최대 글자율로 본 서브샘플링 배율(8배는 CTC 길이 제약에 걸림)도 이 표에서 나왔다.
# [흐름] 코퍼스별 읽기(parquet 또는 tsv+tar) → text_stats로 분할 통계 → 문장 겹침(overlap) → 학습 분할 음절
# 집합 대비 test 분할 OOV → 원문 목록은 빼고 JSON 저장 → 표 출력.
# [연결] 입력은 scripts/download_data.py가 받은 data/raw/. 결과는 docs/data.md "재고 조사 결과" 표와
# reports/data/inventory-2026-10-11.json에 옮겨 적었다. 이 단계는 정규화 규칙이 정해지기 전이라 원문 전사로
# 센다(실제 전처리는 scripts/prepare_data.py).

from __future__ import annotations

# 표준 라이브러리: csv(CV의 tsv 전사), tarfile(CV 오디오 tar 안의 mp3 표본), io(메모리의 바이트를 파일처럼
# soundfile에 넘김), Counter(표본화율별 개수).
import argparse
import csv
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

# numpy: 비율·백분위 계산. pyarrow.parquet: Zeroth·FLEURS parquet(전사 + 음성 바이트 열)를 datasets 라이브러리
# 없이 바로 읽는다. soundfile: 음성 바이트의 길이와 표본화율을 읽는다.
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# 글자 종류 판별용 정규식. HANGUL: 완성형 한글 음절(가~힣, 11,172자). DIGIT·LATIN: 아라비아 숫자와 로마자(학습
# 어휘에 넣을지 정하려고 센다). PUNCT: 글자·숫자·밑줄·공백이 아닌 것(문장부호·기호; \w가 이미 한글을
# 포함한다).
HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]")
PUNCT = re.compile(r"[^\w\s가-힣]")


# 텍스트 통계: 발화 수, 고유 문장 수, 발화당 음절 수 평균·최대, 숫자·라틴·문장부호가 든 발화 비율(%), 음절
# 종류 수.
# 길이 통계(길이가 주어졌을 때만): 총 시간, 길이 평균·p95·최대, 글자율 p50·p95·최대. 글자율 최대는 CTC가
# 프레임당 하나씩 토큰을 내야 하므로 서브샘플링 배율의 상한을 정하는 데 쓰였다(docs/data.md "음절").
def text_stats(texts: list[str], durations: list[float]) -> dict:
    """한 분할의 전사 목록으로 텍스트·길이 통계 사전을 만든다.

    durations는 texts와 같은 순서의 길이(초) 목록이며 비어 있을 수 있다(그때는 길이 통계를 뺀다).
    """
    n = len(texts)
    syl = np.array([len(HANGUL.findall(t)) for t in texts])
    chars_no_space = np.array([len(t.replace(" ", "")) for t in texts])
    out = {
        "n_utts": n,
        "n_unique_texts": len(set(texts)),
        "syllables_per_utt_mean": float(syl.mean()) if n else 0.0,
        "syllables_per_utt_max": int(syl.max()) if n else 0,
        "pct_with_digits": float(100 * np.mean([bool(DIGIT.search(t)) for t in texts])) if n else 0.0,
        "pct_with_latin": float(100 * np.mean([bool(LATIN.search(t)) for t in texts])) if n else 0.0,
        "pct_with_punct": float(100 * np.mean([bool(PUNCT.search(t)) for t in texts])) if n else 0.0,
        "syllable_inventory": len(set(ch for t in texts for ch in HANGUL.findall(t))),
    }
    if durations and len(durations) == n:
        d = np.array(durations)
        # 글자율 = 공백을 뺀 글자 수 ÷ 길이(초). 길이 0으로 나누지 않게 하한을 둔다.
        rate = chars_no_space / np.maximum(d, 1e-6)
        out.update(
            {
                "hours": float(d.sum() / 3600),
                "duration_mean_s": float(d.mean()),
                "duration_p95_s": float(np.percentile(d, 95)),
                "duration_max_s": float(d.max()),
                "chars_per_sec_p50": float(np.percentile(rate, 50)),
                "chars_per_sec_p95": float(np.percentile(rate, 95)),
                "chars_per_sec_max": float(rate.max()),
            }
        )
    return out


# 음성 파일 바이트(FLAC·mp3 등)를 메모리에서 열어 (길이 초, 표본화율)을 돌려준다. 디스크에 풀지 않고
# parquet·tar 안의 바이트를 그대로 쓴다.
def decode_duration(blob: bytes) -> tuple[float, int]:
    with sf.SoundFile(io.BytesIO(blob)) as f:
        return f.frames / f.samplerate, f.samplerate


# 한 분할의 parquet 파일들에서 전사·길이·표본화율 분포를 모은다.
# text_keys: 전사 열 이름 후보를 순서대로 찾는다(데이터셋마다 열 이름이 달라서). audio 열의 각 행은 {"bytes":
# ...} 형태라 그 바이트로 길이를 잰다.
def read_parquet_split(
    files: list[Path], text_keys: tuple[str, ...]
) -> tuple[list[str], list[float], Counter]:
    texts: list[str] = []
    durations: list[float] = []
    rates: Counter = Counter()
    for file in files:
        table = pq.read_table(file)
        names = table.column_names
        text_col = next(k for k in text_keys if k in names)
        audio = table.column("audio").to_pylist()
        for row_text, row_audio in zip(table.column(text_col).to_pylist(), audio, strict=True):
            texts.append(row_text.strip())
            dur, rate = decode_duration(row_audio["bytes"])
            durations.append(dur)
            rates[rate] += 1
    return texts, durations, rates


# Zeroth-Korean의 train·test 분할 통계. 파일은 data/<분할>-*.parquet. 겹침 계산에 쓰려고 전사 목록(texts)도
# 잠시 결과에 넣어 둔다(저장 전에 뺀다).
def inventory_zeroth(root: Path) -> dict:
    out = {}
    for split in ("train", "test"):
        files = sorted((root / "data").glob(f"{split}-*.parquet"))
        texts, durations, rates = read_parquet_split(files, ("text", "sentence", "transcription"))
        out[split] = {
            "files": len(files),
            "sample_rates": dict(rates),
            "texts": texts,
            **text_stats(texts, durations),
        }
    return out


# FLEURS 한국어의 train·validation·test 분할 통계. 통계는 정규화된 transcription 열로 내고, 원문
# raw_transcription의 숫자 비율도 따로 센다. 정규화 열에도 숫자가 남는지 확인하려는 것이다(결과: 남아 있음,
# docs/data.md "표기 차이").
def inventory_fleurs(root: Path) -> dict:
    out = {}
    for split in ("train", "validation", "test"):
        files = sorted((root / "ko_kr" / split).glob("*.parquet"))
        texts, durations, rates = read_parquet_split(files, ("transcription",))
        raw_texts: list[str] = []
        for file in files:
            raw_texts.extend(
                t.strip() for t in pq.read_table(file, columns=["raw_transcription"]).column(0).to_pylist()
            )
        stats = text_stats(texts, durations)
        stats["raw_transcription_pct_with_digits"] = float(
            100 * np.mean([bool(DIGIT.search(t)) for t in raw_texts])
        )
        out[split] = {"files": len(files), "sample_rates": dict(rates), "texts": texts, **stats}
    return out


# Common Voice 한국어 미러의 분할별 통계(train·dev·test·validated·other·invalidated 중 있는 것).
# 길이는 음성을 디코딩하지 않고 배포본의 clip_durations.tsv(ms)에서 가져온다. 화자 수는 client_id로 센다.
# 표본화율은 첫 tar의 앞 20개 파일만 표본으로 확인한다(전부 열면 오래 걸려서).
def inventory_common_voice(root: Path) -> dict:
    out = {}
    durations_ms = {}
    with open(root / "transcript" / "ko" / "clip_durations.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            durations_ms[row["clip"]] = float(row["duration[ms]"])
    for split in ("train", "dev", "test", "validated", "other", "invalidated"):
        if not (root / "transcript" / "ko" / f"{split}.tsv").exists():
            continue
        with open(root / "transcript" / "ko" / f"{split}.tsv", encoding="utf-8") as f:
            # QUOTE_NONE: CV tsv는 따옴표로 감싼 CSV가 아니어서, 문장 안의 따옴표를 문자 그대로 읽게
            # 한다(따옴표 처리를 켜면 문장이 잘리거나 합쳐질 수 있다).
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        texts = [r["sentence"].strip() for r in rows]
        # 길이 표에 없는 클립은 0초로 둔다.
        durations = [durations_ms.get(r["path"], 0.0) / 1000 for r in rows]
        stats = text_stats(texts, durations)
        stats["n_speakers"] = len({r["client_id"] for r in rows})
        tars = sorted((root / "audio" / "ko" / split).glob("*.tar"))
        rates: Counter = Counter()
        if tars:
            with tarfile.open(tars[0]) as tf:
                for member in tf.getmembers()[:20]:
                    if member.isfile():
                        blob = tf.extractfile(member).read()
                        try:
                            # 디코더가 못 여는 파일이 있어도 조사 전체를 멈추지 않고 오류 종류를 표본화율 칸에
                            # 세어 보고한다.
                            _, rate = decode_duration(blob)
                            rates[rate] += 1
                        except Exception as exc:  # noqa: BLE001  # 디코더 문제는 세어 보고하고 멈추지 않는다
                            rates[f"decode_error:{type(exc).__name__}"] += 1
        out[split] = {"files": len(tars), "sample_rates": dict(rates), "texts": texts, **stats}
    return out


# 두 분할의 문장 겹침(정규화 없이 정확 일치). exact_match_unique: 겹치는 고유 문장 수. pct_of_b_utts_in_a: b의
# 발화 가운데 문장이 a에 있는 비율(%). 'test 문장이 train에 있는가'를 보려고 a = train, b = test로 부른다.
def overlap(a: list[str], b: list[str]) -> dict:
    sa, sb = set(a), set(b)
    inter = sa & sb
    return {
        "exact_match_unique": len(inter),
        "pct_of_b_utts_in_a": float(100 * np.mean([t in sa for t in b])) if b else 0.0,
    }


# 모든 코퍼스를 조사해 JSON을 쓰고 표를 출력한다. 인자: --raw(원본 폴더), --out(결과 JSON, 기본
# data/inventory.json; 커밋하지 않는 data/ 아래).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/inventory.json")
    args = parser.parse_args()
    raw = Path(args.raw)
    inv = {
        "zeroth_korean": inventory_zeroth(raw / "zeroth_korean"),
        "fleurs_ko": inventory_fleurs(raw / "fleurs_ko"),
    }
    # 받아 둔 Common Voice 판을 모두 조사하되, 겹침·OOV 계산에는 목록의 첫 번째(22가 있으면 22)를 쓴다. 22를
    # 쓰고 17은 폴백이라는 데이터 결정과 같은 순서다(docs/data.md).
    cv_names = [n for n in ("common_voice_22_ko", "common_voice_17_ko") if (raw / n).exists()]
    for name in cv_names:
        inv[name] = inventory_common_voice(raw / name)
    z, fl, cv = inv["zeroth_korean"], inv["fleurs_ko"], inv[cv_names[0]]
    # 평가 분할의 문장이 같은 코퍼스 학습 분할에 있는지 본다. 이 표에서 Zeroth test ↔ train 겹침 99.8 %가
    # 나왔고 FLEURS·CV는 0개였다.
    inv["overlap"] = {
        "zeroth_test_in_zeroth_train": overlap(z["train"]["texts"], z["test"]["texts"]),
        "fleurs_test_in_fleurs_train": overlap(fl["train"]["texts"], fl["test"]["texts"]),
        "fleurs_validation_in_fleurs_train": overlap(fl["train"]["texts"], fl["validation"]["texts"]),
        "cv_corpus": cv_names[0],
        "cv_test_in_cv_train": overlap(cv["train"]["texts"], cv["test"]["texts"]),
        "cv_dev_in_cv_train": overlap(cv["train"]["texts"], cv["dev"]["texts"]),
    }
    # 학습 후보 분할(Zeroth train + FLEURS train + CV train)에 나오는 음절 집합. 음절 어휘로 test 음절을
    # 얼마나 덮는지(OOV 토큰·종류 수) 세어 '어휘를 학습 음절로만 만들어도 되는가'를 판단했다(docs/data.md:
    # 미등장 음절은 <unk>로 두어도 CER 손실이 작음).
    train_syl = set(
        ch
        for t in z["train"]["texts"] + fl["train"]["texts"] + cv["train"]["texts"]
        for ch in HANGUL.findall(t)
    )
    inv["syllable_coverage"] = {"train_inventory": len(train_syl)}
    for name, split in (("zeroth_test", z["test"]), ("fleurs_test", fl["test"]), ("cv_test", cv["test"])):
        toks = [ch for t in split["texts"] for ch in HANGUL.findall(t)]
        inv["syllable_coverage"][name] = {
            "n_syllable_tokens": len(toks),
            "oov_tokens": sum(1 for ch in toks if ch not in train_syl),
            "oov_types": len(set(ch for ch in toks if ch not in train_syl)),
        }
    # 원문 전사 목록은 JSON에서 뺀다(파일을 작게 유지한다). 원본에서 언제든 다시 계산할 수 있다.
    for corpus in ["zeroth_korean", "fleurs_ko", *cv_names]:
        for split in inv[corpus].values():
            split.pop("texts", None)
    Path(args.out).write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    # docs/data.md에 옮기기 쉬운 Markdown 표와, 겹침·음절 범위 JSON을 출력한다. 길이가 없는 분할은 0으로
    # 찍힌다.
    print("| corpus | split | utts | hours | unique texts | syl inventory | chars/s p50/p95 | sample rates |")
    print("|---|---|---|---|---|---|---|---|")
    for corpus in ["zeroth_korean", "fleurs_ko", *cv_names]:
        for split, s in inv[corpus].items():
            row = [
                corpus,
                split,
                str(s["n_utts"]),
                f"{s.get('hours', 0):.2f}",
                str(s["n_unique_texts"]),
                str(s["syllable_inventory"]),
                f"{s.get('chars_per_sec_p50', 0):.1f}/{s.get('chars_per_sec_p95', 0):.1f}",
                str(s["sample_rates"]),
            ]
            print("| " + " | ".join(row) + " |")
    print(
        json.dumps(
            {"overlap": inv["overlap"], "syllable_coverage": inv["syllable_coverage"]},
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
