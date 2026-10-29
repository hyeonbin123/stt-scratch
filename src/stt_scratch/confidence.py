"""확정 자막 어절 신뢰도: 2차 패스 프레임 사후확률 → 토큰·어절 점수, 정오 라벨, AUROC·ECE·Platt 보정.

끝점 뒤 2차 패스(service.session.StreamSession(final_pass="full"))가 세그먼트마다 계산한 전체 문맥
log-prob으로 확정 자막의 어절마다 '틀렸을 가능성'을 알려 주는 신뢰도를 만들고, 그 신뢰도가 실제 오류
어절을 얼마나 가려내는지(AUROC)와 보정한 확률이 실제 정답률과 맞는지(ECE)를 재는 numpy 함수들이다.
등록 규칙은 docs/experiments.md 규칙 변경 28("확정 자막 어절 신뢰도" 절), 측정 도구는
scripts/final_pass_dump.py(GPU, 프레임 통계 덤프)와 scripts/confidence_eval.py(CPU, 선택·판정)다.

  토큰 구간  greedy CTC가 토큰을 낸 프레임부터 같은 argmax가 이어지는 프레임까지
             (model/decode.greedy_batch와 같은 규칙)
  토큰 점수  entropy = 1 − (구간 최소 엔트로피) / log V,  maxprob = 구간 최대 사후확률
  어절 점수  공백 토큰으로 나눈 어절의 음절 토큰 점수 최솟값(min) 또는 평균(mean) → 후보 4개
  정오 라벨  편집 거리 최소 정렬 가운데 일치가 가장 많은 정렬에서 일치 = 정답, 대치·삽입 = 오류
  보정·평가  Platt 보정 σ(a·logit(s) + b), AUROC(같은 값 1/2), ECE(같은 폭 15구간), 발화 부트스트랩
"""

# [왜 엔트로피와 최대 확률 두 가지인가] 프레임 엔트로피를 정규화해 토큰·어절 단위로 모으는 학습 없는 신뢰도가
# 프레임 최대 확률 기준보다 오류 어절을 더 잘 가른다는 보고가 있다(Laptev·Ginsburg, SLT 2022,
# arXiv:2212.08703; CTC와 RNN-T). 이 모델에서도 그런지는 등록한 절반 A에서 고르고 절반 B에서 한 번만 판정한다.
# [왜 보정이 필요한가] CTC 사후확률은 한 프레임에 몰리는 뾰족한 분포라 맞든 틀리든 1에 가까운 값이 많아(과신)
# 그대로 확률로 읽을 수 없다. 순위(AUROC)는 보정과 무관하고, 화면 밑줄 임계처럼 '확률'로 쓰려면 Platt 보정
# 뒤의 ECE를 본다.
# [순수 numpy] 프레임 통계는 덤프 단계에서 이미 계산해 두므로 여기서는 torch가 필요 없다(CPU에서 수 초).

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# 부트스트랩 재표본 표는 등록 규칙의 다른 모든 구간과 같은 함수(seed 0, 1,000회)를 그대로 쓴다. 같은 발화 수면
# 같은 표가 나오므로 구간끼리 같은 재표본에서 계산된다(metrics.py "지표" 규칙).
from stt_scratch.metrics import _resample_indices
from stt_scratch.text import Vocab

# 후보 이름 → (토큰 점수 종류, 어절 집계). 이 순서가 절반 A 선택의 동점 순서다(등록 값, 규칙 변경 28).
CANDIDATES: dict[str, tuple[str, str]] = {
    "entropy_min": ("entropy", "min"),
    "entropy_mean": ("entropy", "mean"),
    "maxprob_min": ("maxprob", "min"),
    "maxprob_mean": ("maxprob", "mean"),
}
ECE_BINS = 15  # 같은 폭 15구간(등록 값)
# logit을 취하기 전에 점수를 [SCORE_CLIP, 1 − SCORE_CLIP]로 자른다. CTC 최대 확률은 fp32에서 정확히 1.0이 되는
# 프레임이 있어 logit이 무한대가 되기 때문이다. 1e-6이면 logit 범위가 약 ±13.8이다.
SCORE_CLIP = 1e-6


# --- 토큰 구간과 점수 ------------------------------------------------------------------------------------


# greedy CTC 축약을 프레임 구간까지 함께 돌려준다. 규칙은 model/decode.greedy_batch와 같다: blank가 아니고
# 직전 프레임 토큰과 다르면 새 토큰이고, 같은 토큰이 이어지는 프레임은 그 토큰의 구간에 붙는다. 그래서 돌려준
# id 열은 greedy_batch의 결과와 같다(tests/test_confidence.py). 입력: 프레임별 argmax 1차원 정수 배열. 출력:
# 토큰마다 (id, 시작 프레임, 끝 프레임(제외)). 주의: 같은 토큰이 blank를 사이에 두고 다시 나오면(가 _ 가) 토큰
# 두 개다(CTC 반복 규칙).
def token_spans(argmax: np.ndarray, blank: int = 0) -> list[tuple[int, int, int]]:
    """프레임 argmax 열에서 greedy CTC 토큰마다 (id, 시작, 끝(제외)) 프레임 구간을 돌려준다."""
    spans: list[list[int]] = []
    prev = blank
    for t, tok in enumerate(np.asarray(argmax).tolist()):
        if tok != blank:
            if tok != prev:
                spans.append([tok, t, t + 1])
            else:
                spans[-1][2] = t + 1
        prev = tok
    return [(tok, start, end) for tok, start, end in spans]


# 토큰마다 두 점수를 만든다. entropy: 프레임 엔트로피 H(전체 어휘 분포, nats)를 최대값 log V로 나눠 0~1로 만든
# 뒤 1에서 뺀 값을 구간에서 가장 큰 것(= 가장 확신한 프레임)으로 쓴다. maxprob: 구간의 최대 사후확률(구간
# 프레임에서는 argmax가 그 토큰이므로 max_logprob = 그 토큰의 log-prob). 둘 다 1에 가까울수록 확신한다. 입력:
# token_spans 결과, 프레임별 최대 log-prob·엔트로피(덤프 배열), 어휘 크기 V. 출력: {"entropy", "maxprob"} →
# 토큰 수 길이의 float64 배열(fp32 반올림으로 범위를 조금 벗어난 값은 0~1로 자름).
def token_scores(
    spans: Sequence[tuple[int, int, int]], max_logprob: np.ndarray, entropy: np.ndarray, vocab_size: int
) -> dict[str, np.ndarray]:
    """토큰 구간마다 정규화 엔트로피 신뢰도(entropy)와 최대 사후확률(maxprob)."""
    log_v = math.log(vocab_size)
    ent = [1.0 - float(np.min(entropy[s:e])) / log_v for _, s, e in spans]
    top = [math.exp(float(np.max(max_logprob[s:e]))) for _, s, e in spans]
    return {
        "entropy": np.clip(np.asarray(ent, dtype=np.float64), 0.0, 1.0),
        "maxprob": np.clip(np.asarray(top, dtype=np.float64), 0.0, 1.0),
    }


# 토큰 열을 공백 토큰에서 나눈다. 연속 공백·앞뒤 공백이 만드는 빈 어절은 버리므로 결과는 디코딩한 텍스트의
# str.split()과 같은 어절 열이다(평가는 이 성질을 덤프마다 확인한다).
# 입력: 토큰 id 열, 공백 id. 출력: 어절마다 그 어절 토큰들의 위치(토큰 열 안의 인덱스) 목록.
def split_words(token_ids: Sequence[int], space_id: int) -> list[list[int]]:
    """토큰 열을 공백에서 나눈 어절별 토큰 위치 목록(빈 어절 없음)."""
    words: list[list[int]] = []
    current: list[int] = []
    for i, tok in enumerate(token_ids):
        if tok == space_id:
            if current:
                words.append(current)
                current = []
        else:
            current.append(i)
    if current:
        words.append(current)
    return words


# 가설 어절 하나: 텍스트와, 토큰 점수 종류마다 그 어절 음절 토큰들의 점수 배열(공백 토큰은 들지 않는다).
@dataclass
class Word:
    text: str
    scores: dict[str, np.ndarray]


# 세그먼트 하나(2차 패스 한 번)의 프레임 통계에서 확정 텍스트의 어절 목록을 만든다. 어절 텍스트는
# Vocab.decode라 <unk>는 U+318D 한 글자로 어절 안에 남는다(채점과 같다). 출력 어절을 공백으로 이으면 그
# 세그먼트 확정 텍스트를 split()한 것과 같다.
def segment_words(
    argmax: np.ndarray, max_logprob: np.ndarray, entropy: np.ndarray, vocab: Vocab
) -> list[Word]:
    """세그먼트 프레임 통계 → 어절 목록(텍스트 + 토큰 점수)."""
    spans = token_spans(argmax, vocab.blank_id)
    ids = [tok for tok, _, _ in spans]
    scores = token_scores(spans, np.asarray(max_logprob), np.asarray(entropy), len(vocab))
    words = []
    for positions in split_words(ids, vocab.space_id):
        idx = np.asarray(positions, dtype=np.int64)
        words.append(Word(vocab.decode([ids[i] for i in positions]), {k: v[idx] for k, v in scores.items()}))
    return words


def word_score(word: Word, candidate: str) -> float:
    """후보 이름(CANDIDATES)의 어절 점수: 토큰 점수 종류의 최솟값 또는 평균."""
    kind, how = CANDIDATES[candidate]
    values = word.scores[kind]
    return float(values.min() if how == "min" else values.mean())


# --- 정오 라벨 -------------------------------------------------------------------------------------------


# 어절 열 두 개의 편집 거리 정렬. 비용은 대치·삽입·삭제 각 1(WER의 편집 수와 같다)이고, 편집 수가 같은 정렬이
# 여럿이면 일치가 가장 많은 것을 고른다(비용을 (편집 수, −일치 수)의 사전식 순서로 비교하는 동적 계획법; 두
# 성분이 경로를 따라 더해지므로 칸마다 사전식 최소를 고르면 전체 최적이 된다). 그래도 같으면 끝에서부터 되짚을
# 때 대치·일치 > 삭제 > 삽입 순으로 고른다(결정적). 왜 jiwer를 쓰지 않는가: jiwer(rapidfuzz)는 편집 수가 같은
# 정렬 사이에서 일치가 적은 쪽을 고를 수 있다. 예: 정답 '할 수 있다 가나', 가설 '할수 있다 다라 마'를 jiwer
# 4.0은 어절 4개 대치로 맞춰 맞게 나온 '있다'까지 오류로 센다(편집 4개, 다른 정렬은 같은 편집 4개로 '있다'를
# 일치로 둔다). 신뢰도 평가에서는 이 라벨이 맞혀야 할 목표이므로 일치를 최대로 하는 정의를 규칙으로 고정했다.
# 편집 수 자체는 jiwer와 같다(tests/test_confidence.py). 입력: 정답 어절 열, 가설 어절 열. 출력: (op, 정답
# 위치 또는 None, 가설 위치 또는 None) 목록(앞에서부터 순서), op ∈ equal·substitute·insert·delete(jiwer의
# 이름과 같다). 계산은 O(정답 어절 수 × 가설 어절 수).
def align_words(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int | None, int | None]]:
    """편집 수가 가장 적고 그 가운데 일치가 가장 많은 어절 정렬."""
    n, m = len(ref), len(hyp)
    cost = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = (i, 0)
    for j in range(1, m + 1):
        cost[0][j] = (j, 0)
    for i in range(1, n + 1):
        row, prev = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1]
            diag = (d[0], d[1] - 1) if ref[i - 1] == hyp[j - 1] else (d[0] + 1, d[1])
            row[j] = min(diag, (prev[j][0] + 1, prev[j][1]), (row[j - 1][0] + 1, row[j - 1][1]))
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            d = cost[i - 1][j - 1]
            if cost[i][j] == ((d[0], d[1] - 1) if same else (d[0] + 1, d[1])):
                ops.append(("equal" if same else "substitute", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == (cost[i - 1][j][0] + 1, cost[i - 1][j][1]):
            ops.append(("delete", i - 1, None))
            i -= 1
            continue
        ops.append(("insert", None, j - 1))
        j -= 1
    ops.reverse()
    return ops


# 가설 어절마다의 정오 라벨. correct: 정렬에서 일치면 True, 대치·삽입이면 False. spacing_only: 오류 어절
# 가운데 '띄어쓰기만 다른' 것 — 일치와 일치 사이(또는 앞뒤 끝)의 연속 비일치 구간에서 정답 어절들을 공백 없이
# 이은 것과 가설 어절들을 이은 것이 같으면 그 구간의 가설 어절(예: 정답 '할 수', 가설 '할수'). 판정 라벨은
# correct이고, spacing_only는 '띄어쓰기 오류를 정답으로 보면' 지표가 얼마나 달라지는지 보고만 하는
# 민감도용이다. counts: 정렬 op별 수.
@dataclass
class WordLabels:
    correct: list[bool]
    spacing_only: list[bool]
    counts: dict[str, int]


def hyp_word_labels(ref: Sequence[str], hyp: Sequence[str]) -> WordLabels:
    """가설 어절마다 정답 여부·띄어쓰기만 다른 오류 여부와 정렬 수."""
    ops = align_words(ref, hyp)
    correct = [False] * len(hyp)
    spacing = [False] * len(hyp)
    run_ref: list[int] = []
    run_hyp: list[int] = []

    def close_run() -> None:
        if run_ref and run_hyp and "".join(ref[i] for i in run_ref) == "".join(hyp[j] for j in run_hyp):
            for j in run_hyp:
                spacing[j] = True
        run_ref.clear()
        run_hyp.clear()

    for op, i, j in ops:
        if op == "equal":
            close_run()
            correct[j] = True
            continue
        if i is not None:
            run_ref.append(i)
        if j is not None:
            run_hyp.append(j)
    close_run()
    counts = Counter(op for op, _, _ in ops)
    return WordLabels(
        correct, spacing, {k: counts.get(k, 0) for k in ("equal", "substitute", "insert", "delete")}
    )


# --- 평가 지표 -------------------------------------------------------------------------------------------


# 오류 어절 검출 AUROC: 무작위로 고른 (정답 어절, 오류 어절) 쌍에서 정답 어절의 신뢰도가 더 높을 확률(같으면
# 1/2) = Mann–Whitney U ÷ 쌍 수. '신뢰도가 낮을수록 오류'로 오류를 가려내는 AUROC와 같은 값이다. 같은 값
# 묶음은 정렬 뒤 묶음별 합으로 한 번에 센다(O(n log n)). 입력: 신뢰도, 정답 여부(True = 정답 어절),
# weights(어절마다 0 이상의 배수; 부트스트랩 재표본에서 그 발화가 뽑힌 횟수. 정수 배수면 그만큼 복제한 집합의
# AUROC와 같다). 출력: 0~1, 한쪽 부류의 가중치 합이 0이면 None.
def auroc(conf: np.ndarray, correct: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """신뢰도로 정답 어절과 오류 어절을 가르는 AUROC(같은 값은 1/2로 센다)."""
    conf = np.asarray(conf, dtype=np.float64)
    good = np.asarray(correct, dtype=bool)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, dtype=np.float64)
    pos, neg = w * good, w * ~good
    n_pos, n_neg = pos.sum(), neg.sum()
    if conf.size == 0 or n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(conf, kind="mergesort")
    sorted_conf = conf[order]
    starts = np.flatnonzero(np.r_[True, sorted_conf[1:] != sorted_conf[:-1]])
    group_pos = np.add.reduceat(pos[order], starts)
    group_neg = np.add.reduceat(neg[order], starts)
    neg_below = np.cumsum(group_neg) - group_neg
    return float((group_pos * (neg_below + 0.5 * group_neg)).sum() / (n_pos * n_neg))


# 기대 보정 오차(ECE): [0, 1]을 같은 폭 bins개 구간으로 나누고(1.0은 마지막 구간) Σ_구간 (구간 어절 비율) ×
# |구간 정답률 − 구간 평균 확률| = Σ_구간 |정답 수 − 확률 합| ÷ 전체 어절 수. 빈 구간은 0으로 기여한다.
# 입력: 정답일 확률(0~1), 정답 여부, 구간 수, weights(auroc와 같은 배수). 출력: 0~1, 가중치 합이 0이면 None.
def ece(
    prob: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS, weights: np.ndarray | None = None
) -> float | None:
    """같은 폭 구간의 기대 보정 오차."""
    prob = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    good = np.asarray(correct, dtype=np.float64)
    w = np.ones_like(prob) if weights is None else np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if prob.size == 0 or total == 0:
        return None
    idx = np.minimum((prob * bins).astype(np.int64), bins - 1)
    hits = np.bincount(idx, weights=w * good, minlength=bins)
    mass = np.bincount(idx, weights=w * prob, minlength=bins)
    return float(np.abs(hits - mass).sum() / total)


def logit(s: np.ndarray) -> np.ndarray:
    """점수를 [SCORE_CLIP, 1 − SCORE_CLIP]로 자른 뒤 log(s / (1 − s))."""
    s = np.clip(np.asarray(s, dtype=np.float64), SCORE_CLIP, 1.0 - SCORE_CLIP)
    return np.log(s) - np.log1p(-s)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """수치적으로 안정한 로지스틱 함수(tanh 형태라 큰 |z|에서도 넘침이 없다)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


# Platt 보정: p(정답) = σ(a·logit(s) + b). 점수의 logit을 입력으로 쓰므로 a = 1, b = 0이면 원래 점수
# 그대로이고, a는 온도(뾰족함), b는 치우침을 고친다. 추정은 Platt(1999)의 목표값 평활(정답 어절 목표
# (N₊+1)/(N₊+2), 오류 어절 1/(N₋+2); 완전히 갈리는 자료에서도 a·b가 유한)과 Lin·Lin·Weng(2007, "A note on
# Platt's probabilistic outputs for support vector machines")의 뉴턴법(헤세 행렬 대각에 1e-12, 백트래킹 선
# 탐색, 기울기 1e-5 이하에서 멈춤, 최대 100회)을 따른다. 무작위성이 없어 같은 입력이면 같은 값이 나온다. 입력:
# 점수(0~1), 정답 여부. 출력: {"a", "b", "iterations", "converged"}. 한쪽 부류만 있어도 평활 목표값 덕에
# 유한한 값을 낸다(그런 자료는 AUROC가 없어 평가 도구가 note로 표시하고 판정은 not_met이 된다).
def platt_fit(scores: np.ndarray, correct: np.ndarray, max_iter: int = 100) -> dict:
    """σ(a·logit(s) + b)의 a, b를 평활 목표값 교차 엔트로피 최소화(뉴턴법)로 추정한다."""
    x = logit(scores)
    y = np.asarray(correct, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    target = np.where(y, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

    def objective(a: float, b: float) -> float:
        z = a * x + b
        return float((np.logaddexp(0.0, z) - target * z).sum())

    a, b = 0.0, math.log((n_pos + 1.0) / (n_neg + 1.0))
    value = objective(a, b)
    converged = False
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        p = sigmoid(a * x + b)
        grad_a = float(((p - target) * x).sum())
        grad_b = float((p - target).sum())
        if max(abs(grad_a), abs(grad_b)) < 1e-5:
            converged = True
            break
        h = p * (1.0 - p)
        h_aa = float((h * x * x).sum()) + 1e-12
        h_ab = float((h * x).sum())
        h_bb = float(h.sum()) + 1e-12
        det = h_aa * h_bb - h_ab * h_ab
        step_a = -(h_bb * grad_a - h_ab * grad_b) / det
        step_b = -(-h_ab * grad_a + h_aa * grad_b) / det
        slope = grad_a * step_a + grad_b * step_b
        step = 1.0
        while step >= 1e-10:
            new_a, new_b = a + step * step_a, b + step * step_b
            new_value = objective(new_a, new_b)
            if new_value < value + 1e-4 * step * slope:
                a, b, value = new_a, new_b, new_value
                break
            step /= 2.0
        else:
            break  # 선 탐색 실패: 더 줄일 수 없음(converged는 False로 남는다)
    return {"a": a, "b": b, "iterations": iterations, "converged": converged}


def platt_apply(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt 보정 확률 σ(a·logit(s) + b)."""
    return sigmoid(a * logit(scores) + b)


# --- 절반 나누기와 구간 ----------------------------------------------------------------------------------


# 등록한 절반: 발화 id(UTF-8)의 SHA-256 첫 바이트가 짝수면 A(선택), 홀수면 B(판정). seed가 없고 id만으로
# 정해져 결과를 보기 전에 크기를 셀 수 있다(docs/experiments.md "데이터 분할"과 같은 방식).
def half_of(utt_id: str) -> str:
    """발화 id의 SHA-256 첫 바이트 짝·홀로 'A' 또는 'B'."""
    return "A" if hashlib.sha256(utt_id.encode("utf-8")).digest()[0] % 2 == 0 else "B"


# 발화 단위 부트스트랩 95 % 구간. 어절은 같은 발화 안에서 서로 얽혀 있으므로(같은 화자·같은 2차 패스) 어절이
# 아니라 발화를 다시 뽑는다. 재표본 표는 등록 규칙의 다른 구간과 같은 metrics._resample_indices(seed 0,
# 1,000회)이고, 재표본마다 발화가 뽑힌 횟수를 그 발화 어절들의 가중치로 metric(weights)을 다시 계산해 2.5·97.5
# 백분위를 쓴다. 입력: 어절마다 발화 번호(0 .. n_utts−1; 어절이 없는 발화도 n_utts에 센다), 발화 수, 가중치를
# 받아 값(또는 None)을 내는 함수. 출력: {"low", "high", "resamples": 값이 나온 재표본 수}.
def bootstrap_interval(
    word_utt: np.ndarray, n_utts: int, metric: Callable[[np.ndarray], float | None]
) -> dict:
    """발화를 재표본한 metric의 95 % 백분위 구간."""
    word_utt = np.asarray(word_utt, dtype=np.int64)
    values = []
    for row in _resample_indices(n_utts):
        counts = np.bincount(row, minlength=n_utts).astype(np.float64)
        value = metric(counts[word_utt])
        if value is not None:
            values.append(value)
    if not values:
        return {"low": None, "high": None, "resamples": 0}
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "resamples": len(values)}
