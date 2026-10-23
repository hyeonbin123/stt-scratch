"""문자 오류율(CER)과 발화 단위 부트스트랩 구간(규칙: docs/experiments.md "지표").

발화 하나의 채점(편집 수·정답 글자 수)과, 발화 묶음의 CER·부트스트랩 95 % 구간을 계산한다.
CER_ns("no space", 공백 제외)는 whisper-ko-ft의 채점 정규화를 써서 그 프로젝트 수치와 비교할 수 있게 하고,
CER_ws("with space", 공백 포함)는 공백도 글자로 센다(KsponSpeech 관례).
같은 문장·화자의 발화를 한 묶음(군집)으로 다시 뽑는 사후 보조 구간(cluster_interval·
cluster_difference_interval)도 여기 있다. 등록 규칙의 구간은 발화 단위 그대로다.
"""

# [규칙] CER은 발화별 비율의 평균이 아니라 '묶음 전체 편집 수 합 ÷ 정답 글자 수 합'이다. 구간은 발화 단위
# 부트스트랩 1,000회(seed 0), 두 모델 비교는 같은 발화 표본을 다시 뽑는 짝지은 구간이다(docs/experiments.md
# "지표"). 모든 보고 수치에 이 구간을 함께 적는다.
# [연결] evaluate.score·summarize가 평가 결과를, scripts/compare_runs.py가 짝지은 비교를,
# scripts/sealed_breakdown.py가 봉인 test 사후 분할과 whisper 앵커 짝 비교를, scripts/cluster_ci.py가 군집
# 보조 구간을 이 함수들로 계산한다.

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

# jiwer: 글자 단위 편집 거리 정렬(치환·삭제·삽입 수)을 검증된 구현으로 쓴다. numpy: 부트스트랩 재표본을 벡터
# 연산으로 계산하고, 시드를 받은 Generator로 재현 가능하게 만든다.
import jiwer
import numpy as np

from stt_scratch.text import normalize_hyp_ws, normalize_score, normalize_train

# 부트스트랩 1,000회, 시드 0(사전 등록 값). 시드가 고정이라 같은 발화 수면 모든 구간이 같은 재표본 인덱스를
# 쓴다.
BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED = 0


# 발화 하나의 채점 결과: 정규화된 정답·가설, 편집 수(치환 + 삭제 + 삽입), 정답 글자 수.
@dataclass(frozen=True)
class Scored:
    reference: str
    hypothesis: str
    edits: int
    length: int


# 정규화가 끝난 두 문자열의 글자 단위 편집 수. 정답이 비면 None(채점에서 뺀다). 가설이 비면 jiwer를 부르지
# 않고 정답 글자 전부를 삭제 오류로 센다. 정답 길이는 jiwer 정렬의 일치 + 치환 + 삭제 수(= 정답 글자 수)다.
def _edits(ref: str, hyp: str) -> Scored | None:
    if not ref:
        return None
    out = jiwer.process_characters(ref, hyp) if hyp else None
    if out is None:
        return Scored(ref, hyp, len(ref), len(ref))
    edits = out.substitutions + out.deletions + out.insertions
    return Scored(ref, hyp, edits, out.hits + out.substitutions + out.deletions)


# 주지표다. 채점 정규화(공백·작은따옴표 제거)를 정답·가설 양쪽에 건다. <unk>는 decode에서 U+318D 한 글자가
# 되어 오류 1자로 센다.
def score_ns(reference: str, hypothesis: str) -> Scored | None:
    """공백 제외 CER_ns 채점(whisper-ko-ft 정의). 정규화한 정답이 비면 None."""
    return _edits(normalize_score(reference), normalize_score(hypothesis))


# 보조 지표다(KsponSpeech 정의). 규칙 변경 7.
def score_ws(reference: str, hypothesis: str) -> Scored | None:
    """공백도 글자로 센 CER_ws 채점. 정답은 학습 정규화, 가설은 UNK_CHAR를 남기는 정규화를 쓴다."""
    return _edits(normalize_train(reference), normalize_hyp_ws(hypothesis))


def error_rate(edits: np.ndarray, lengths: np.ndarray) -> float:
    """묶음 전체 비율: 편집 수 합 ÷ 정답 길이 합."""
    return float(edits.sum() / lengths.sum())


# 발화 인덱스를 복원 추출한 (1,000, count) 표. 매번 같은 시드로 새 Generator를 만들므로 같은 count면 항상 같은
# 표가 나온다.
def _resample_indices(count: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))


# 재표본마다 '편집 합 ÷ 글자 합'을 다시 계산하고 그 분포의 2.5·97.5 백분위를 구간으로 쓴다(백분위 부트스트랩).
def interval(edits: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """발화를 재표본해 구한 묶음 전체 비율의 95 % 부트스트랩 구간."""
    picks = _resample_indices(len(edits))
    rates = edits[picks].sum(axis=1) / lengths[picks].sum(axis=1)
    low, high = np.percentile(rates, [2.5, 97.5])
    return float(low), float(high)


# 두 모델에 같은 재표본 인덱스를 쓰므로 두 CER이 같은 표본에서 함께 움직이고, 발화 난이도에서 오는 변동이
# 차이에서 상쇄된다. 이 구간이 0을 제외하는지가 개선·채택을 주장하는 판정에 쓰인다(docs/experiments.md 선택
# 규칙, 규칙 변경 23).
# 호출 위치: scripts/compare_runs.py.
def difference_interval(edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray) -> tuple[float, float]:
    """같은 발화 집합에서 rate(a) − rate(b)의 95 % 구간(짝지은 재표본)."""
    picks = _resample_indices(len(lengths))
    totals = lengths[picks].sum(axis=1)
    deltas = edits_a[picks].sum(axis=1) / totals - edits_b[picks].sum(axis=1) / totals
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


# 발화별 값 배열들을 군집별 합으로 접는다. 군집 번호는 키가 처음 나온 순서다(정렬 순서가 아니다).
# 그래서 키가 모두 다르면 군집 j = 발화 j라 합이 원래 배열과 같고, 아래 두 함수가 등록 함수
# (interval·difference_interval)와 비트 단위로 같은 값을 낸다. 정수 배열은 정수 그대로 더한다
# (np.bincount의 weights는 float64로 바꾸므로 쓰지 않는다).
# 입력: 같은 길이의 1차원 배열들과, 발화마다 하나씩인 군집 키(해시 가능한 값, 예: 화자 id·정규화 문장).
# 출력: 군집별 합 배열들(입력 순서대로).
# 주의: 키가 None이면 서로 다른 발화(예: 화자 정보가 없는 FLEURS 발화 전부)가 조용히 한 군집으로 묶이므로
# 거부한다. 발화가 없거나 길이가 다르면 ValueError.
def _cluster_totals(arrays: Sequence[np.ndarray], groups: Sequence[Hashable]) -> list[np.ndarray]:
    arrays = [np.asarray(a) for a in arrays]
    count = len(groups)
    if count == 0:
        raise ValueError("no utterances to resample")
    if any(a.ndim != 1 or len(a) != count for a in arrays):
        raise ValueError(f"every array needs one value per utterance ({count} cluster keys)")
    order: dict[Hashable, int] = {}
    index = np.empty(count, dtype=np.int64)
    for i, key in enumerate(groups):
        if key is None:
            raise ValueError(f"cluster key of utterance {i} is None")
        index[i] = order.setdefault(key, len(order))
    totals = []
    for a in arrays:
        out = np.zeros(len(order), dtype=a.dtype)
        np.add.at(out, index, a)
        totals.append(out)
    return totals


# 군집(문장·화자 등) 단위 부트스트랩: 군집마다 편집 합·글자 합을 만든 뒤 군집 인덱스를 복원 추출한다.
# CER은 '편집 합 ÷ 글자 합'이라 군집을 통째로 다시 뽑는 것은 군집 합 배열을 발화처럼 다시 뽑는 것과
# 같으므로, 군집 합에 등록 함수를 그대로 쓴다(같은 seed 0 Generator, 같은 (1,000, 군집 수) 표).
# 왜: 같은 문장을 여러 화자가 읽거나 소수 화자가 많은 발화를 내는 평가 세트에서 발화 단위 구간은 '같은
# 문장·화자에서 발화만 다시 뽑을 때'의 변동이라, 다른 문장·화자로 일반화할 때의 불확실성을 작게 보인다.
# 이 구간은 등록 규칙(발화 단위)을 바꾸지 않는 사후 보조 지표다(docs/experiments.md 7단계 결과의 사후
# 보조 구간).
# 주의: 군집이 적으면(예: 화자 10명) 백분위 끝값의 재표본 오차가 커서, 같은 seed라도 발화·군집 순서가
# 바뀌면 소수 둘째 자리(%)가 달라질 수 있다. 호출하는 쪽이 발화 순서를 고정해야 한다.
def cluster_interval(
    edits: np.ndarray, lengths: np.ndarray, groups: Sequence[Hashable]
) -> tuple[float, float]:
    """군집을 재표본해 구한 묶음 전체 비율의 95 % 부트스트랩 구간. 군집 = 발화이면 interval과 같다."""
    cluster_edits, cluster_lengths = _cluster_totals((edits, lengths), groups)
    return interval(cluster_edits, cluster_lengths)


# 두 시스템의 짝지은 차이를 군집 단위로 다시 뽑는다. 두 시스템에 같은 군집 표본을 쓰므로 군집 난이도는
# 차이에서 상쇄되고, 군집마다 다른 '차이'의 변동만 남는다.
def cluster_difference_interval(
    edits_a: np.ndarray, edits_b: np.ndarray, lengths: np.ndarray, groups: Sequence[Hashable]
) -> tuple[float, float]:
    """같은 발화 집합에서 rate(a) − rate(b)의 군집 단위 95 % 구간.

    군집 = 발화이면 difference_interval과 같다.
    """
    cluster_a, cluster_b, cluster_lengths = _cluster_totals((edits_a, edits_b, lengths), groups)
    return difference_interval(cluster_a, cluster_b, cluster_lengths)


# 반환: {발화 수, CER, 95 % 구간, 편집 합, 글자 합}. 빈 목록이면 비율·구간 없이 0을 돌려준다.
def summarize(scored: list[Scored]) -> dict:
    """채점된 발화 목록(정답이 빈 발화는 이미 뺀 것)의 비율, 구간, 개수를 요약한다."""
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
