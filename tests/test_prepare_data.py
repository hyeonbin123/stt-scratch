"""scripts/prepare_data.py 단위 테스트: 홀드아웃 해시, CTC 프레임 공식, Common Voice 분할 규칙.

Zeroth 문장 홀드아웃 해시, dev_h·test_h·train 배정, mel·출력 프레임 수 공식과 CTC 길이 조건, 학습 제외 사유,
Common Voice 학습 분할을 확인한다.
"""

# [왜 중요한가] 이 규칙들은 docs/experiments.md "데이터 분할"에 측정 전에 등록한 것이다. 해시 소금이나 경계가
# 조용히 바뀌면 dev·봉인 test 문장이 학습으로 새거나 분할이 달라져, 이후 모든 수치의 전제가 깨진다. 실제
# 데이터 없이 순수 함수만 시험한다.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from stt_scratch.text import Vocab, ctc_min_frames

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


# scripts/prepare_data.py를 파일 경로로 불러와 모듈로 쓴다(scripts/는 패키지가 아님). 모듈 안의 dataclass가
# 정의될 때 sys.modules에서 자기 모듈을 찾으므로 실행 전에 등록해 둔다.
def load_script():
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pd = load_script()


# ---------------------------------------------------------------- 홀드아웃 해시
# 확인: 버킷 = int(sha256("zeroth-holdout:" + 문장)의 앞 8자리 16진수) % 100이 등록 정의 그대로이고
# 결정적이다. "안녕하세요"의 버킷 33을 고정값으로 박아 두어 소금 문자열이나 자르는 위치가 바뀌면 바로
# 실패한다.
def test_holdout_bucket_is_deterministic_and_matches_definition():
    import hashlib

    sentence = "몬터규는 자녀들이 사랑을 제대로 못 받고 크면"
    expected = int(hashlib.sha256(("zeroth-holdout:" + sentence).encode("utf-8")).hexdigest()[:8], 16) % 100
    assert pd.holdout_bucket(sentence) == expected
    assert pd.holdout_bucket(sentence) == pd.holdout_bucket(sentence)
    # 고정값: 소금 문자열이나 자르는 위치가 바뀌면 홀드아웃이 조용히 옮겨 가므로 여기서 잡는다.
    assert pd.holdout_bucket("안녕하세요") == 33


# 확인: 버킷 0~4는 dev_h, 5~9는 test_h(봉인), 10 이상은 train으로 간다(경계값 0, 4, 5, 9, 10, 99를 각각 시험).
def test_zeroth_target_partitions_by_bucket():
    cases = ((0, "dev_h"), (4, "dev_h"), (5, "test_h"), (9, "test_h"), (10, "train"), (99, "train"))
    for bucket, target in cases:
        sentence = next(s for s in (f"문장 {i}" for i in range(100_000)) if pd.holdout_bucket(s) == bucket)
        assert pd.zeroth_target(sentence) == target


# 확인: 합성 문장 2만 개에서 홀드아웃 비율이 약 10 %, dev_h가 약 5 %이고, dev_h·test_h·train이 서로 겹치지
# 않으며 빠짐없이 나뉜다. 왜: 같은 문장이 dev와 학습에 동시에 들어가면 처음 보는 문장 평가(dev_H, test_H)가
# 무의미해진다.
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


# ---------------------------------------------------------------- CTC 프레임
# 확인: 표본 수 → mel 프레임 수(창 400, 이동 160, center=False) → 인코더 출력 프레임 수(ceil 두 번) 공식. 창
# 하나가 안 되면 0, 1초는 mel 98 → 출력 25(25 fps), 10초는 출력 250.
@pytest.mark.parametrize(
    ("n_samples", "n_mel", "frames_out"),
    [
        (0, 0, 0),
        (399, 0, 0),
        (400, 1, 1),  # 창 정확히 하나
        (559, 1, 1),
        (560, 2, 1),  # ceil(ceil(2/2)/2) = 1
        (16_000, 98, 25),  # 1초 → 25 fps
        (160_000, 998, 250),
    ],
)
def test_frame_formula(n_samples, n_mel, frames_out):
    assert pd.mel_frames(n_samples) == n_mel
    assert pd.output_frames(n_samples) == frames_out


# 확인: CTC 길이 조건은 '출력 프레임 ≥ 토큰 수 + 반복 수 + 1'이다. "아아 가"는 토큰 4개에 반복 1번이라 최소
# 5프레임이고, 출력 6프레임이면 통과, 5프레임이면 제외된다(docs/experiments.md 데이터 분할의 제외 규칙). 경계
# 바로 위·아래 표본 수로 고정한다.
def test_ctc_fits_requires_min_frames_plus_one():
    vocab = Vocab.build(["아 가"])
    ids = vocab.encode("아아 가")  # U = 4, 반복 1번 → ctc_min_frames = 5, 6프레임 이상 필요
    assert ctc_min_frames(ids) == 5
    # 출력 6프레임 <=> n_mel 21..24 <=> 표본 수 400 + 20*160 .. 400 + 23*160
    assert pd.output_frames(400 + 20 * 160) == 6
    assert pd.ctc_fits(400 + 20 * 160, ids)
    assert pd.output_frames(400 + 19 * 160) == 5
    assert not pd.ctc_fits(400 + 19 * 160, ids)


# 확인: 학습 제외 사유와 우선순위. 숫자가 있으면 digit, 라틴 문자가 하나라도 있으면 latin(규칙 변경 4), 20 s
# 초과면 too_long(정확히 20.0 s는 포함), 여러 사유가 겹치면 숫자가 먼저다.
def test_exclusion_reasons():
    assert pd.exclusion_reason("삼 년", 3.0) is None
    assert pd.exclusion_reason("3 년", 3.0) == "digit"
    assert pd.exclusion_reason("abc 년", 3.0) == "latin"
    assert pd.exclusion_reason("긴 발화", 20.001) == "too_long"
    assert pd.exclusion_reason("경계", 20.0) is None
    assert pd.exclusion_reason("3 abc", 30.0) == "digit"  # 먼저 맞는 규칙이 이긴다


# ---------------------------------------------------------------- Common Voice 분할
# 확인: Common Voice 학습 분할 = validated에서 dev·test에 있는 경로(path)를 뺀 것. validated에 없는 test
# 경로가 섞여 있어도 문제없다. 왜: CV dev는 dev_mix, test는 봉인 test라 같은 녹음이 학습에 들어가면 안 된다.
def test_cv_path_in_dev_or_test_is_not_in_train():
    validated = ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3"]
    dev = ["b.mp3"]
    test = ["d.mp3", "zzz.mp3"]
    train = pd.cv_train_paths(validated, dev, test)
    assert train == ["a.mp3", "c.mp3", "e.mp3"]
    assert "b.mp3" not in train
    assert "d.mp3" not in train
    assert not set(train) & (set(dev) | set(test))
