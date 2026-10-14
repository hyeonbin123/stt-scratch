"""4단계 방출 지연 도우미(scripts/stage4_latency.py) 테스트: 어절 재조합, 어절 맞추기, 방출 프레임, 방출 시각.

MFA 형태소 구간을 전사 어절로 다시 묶기, 가설·정답 어절 맞추기, 토큰 방출 프레임, 청크 기준 방출 시각을
확인한다.
"""

# [왜 필요한가] FTD/LTD(첫·마지막 어절 지연)는 이 도우미들의 조합으로 계산되고, 4단계 판정(TrimTail 채택, H4
# 미충족)이 그 값에 달려 있다. 정의(규칙 변경 18)대로 동작하는지 작은 예로 고정한다.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# scripts/는 패키지가 아니라서 import 문으로 가져올 수 없으므로, 파일 경로로 모듈을 불러온다(실행되는 것은
# 모듈 정의뿐이고 main은 돌지 않는다).
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage4_latency", ROOT / "scripts" / "stage4_latency.py")
lat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lat)


# 확인: MFA가 형태소로 나눈 구간("자동차세" + "는")을 전사 어절 "자동차세는"의 (시작, 끝)으로 다시 묶는다.
# 이어 붙인 글자가 어절과 맞지 않거나 조각이 모자라면 None을 돌려 그 발화를 지연 측정에서 뺀다. 왜: MFA 한국어
# 모델은 어절을 형태소로 나누므로 어절 단위 지연을 재려면 이 재조합이 필요하다.
def test_regroup_morphemes_into_words() -> None:
    pieces = [(0.1, 0.4, "자동차세"), (0.4, 0.5, "는"), (0.6, 0.8, "덜")]
    words = ["자동차세는", "덜"]
    assert lat.regroup(words, pieces) == [(0.1, 0.5), (0.6, 0.8)]
    assert lat.regroup(["자동차"], pieces) is None  # 레이블을 이어 붙여도 어절과 같아지지 않음
    assert lat.regroup(words + ["더"], pieces) is None  # 어절이 조각보다 많음


# 확인: 편집 거리 정렬로 가설·정답 어절을 맞추되 정확히 같은 어절만 {정답 위치: 가설 위치}로 돌려준다(치환된
# 어절은 빠진다). 지연은 정확히 맞힌 어절에서만 재기 때문이다.
def test_match_words_exact_only() -> None:
    ref = ["a", "b", "c", "d"]
    assert lat.match_words(["a", "x", "c", "d"], ref) == {0: 0, 2: 2, 3: 3}
    assert lat.match_words(["b", "c"], ref) == {1: 0, 2: 1}
    assert lat.match_words([], ref) == {}


# 확인: 프레임별 argmax 열에서 토큰이 '처음 나온 프레임'(같은 토큰이 이어지는 구간의 첫 프레임)을 방출
# 시점으로 잡는다. blank를 사이에 둔 같은 토큰은 두 번 방출된다.
def test_emissions_first_frame_of_each_run() -> None:
    assert lat.emissions([0, 3, 3, 0, 3, 4, 4, 0]) == [(3, 1), (3, 4), (4, 5)]


# 확인: 프레임의 방출 시각 = 그 프레임이 속한 청크의 음성이 모두 도착한 시각(0.04 × 청크 × (청크 번호 + 1) +
# 0.015 s, mel 창 25 ms가 15 ms를 더함)이고, 같은 청크 안의 프레임은 같은 시각, 발화 길이를 넘지 않는다(규칙
# 변경 18). 계산 시간은 넣지 않는다.
@pytest.mark.parametrize("chunk", [8, 16])
def test_emit_time_is_end_of_chunk_audio(chunk: int) -> None:
    # 프레임 0은 첫 청크의 음성(mel 4 * chunk프레임)이 모두 도착해야 나올 수 있다.
    first = lat.emit_time(0, chunk, 100.0)
    assert first == pytest.approx(0.04 * chunk + 0.015)
    assert lat.emit_time(chunk - 1, chunk, 100.0) == first
    assert lat.emit_time(chunk, chunk, 100.0) == pytest.approx(0.08 * chunk + 0.015)
    assert lat.emit_time(10 * chunk, chunk, 1.0) == 1.0  # 발화 끝 시각에서 자른다
