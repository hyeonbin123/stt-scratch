"""확정 자막 핫워드 문맥 편향: 음절 trie 보너스를 얹은 top-k CTC prefix beam search(순수 Python).

끝점 뒤 2차 패스(service.session.StreamSession(final_pass="full"))의 전체 문맥 log-prob에서, 사용자가 준 편향
목록(고유명사·전문어 같은 어절)이 확정 자막에 나오도록 돕는 디코더다. 목록이 비면 이 모듈을 쓰지 않고 greedy
그대로다(model/decode.biased_batch). 등록 규칙은 docs/experiments.md 규칙 변경 29("확정 자막 핫워드 문맥 편향"
절), 평가 도구는 scripts/bias_eval.py다.

  목록   어절마다 학습 정규화(text.normalize_train) → 공백 제거 → 음절 id 열. 어휘에 없는 음절(<unk>)이 든
         어절은 모델이 낼 수 없어 뺀다(dropped)
  trie   그 id 열들의 접두사 나무. 공백은 trie 기호가 아니다(띄어 쓴 목록 어절도 붙여 쓴 어절과 같다)
  상태   가설 접두사마다 (trie 노드, 보류 보너스, 확정 보너스)
  보너스 어절 시작(세그먼트 처음·공백 뒤)부터 음절이 trie 경로를 따라가면 음절마다 b를 보류 보너스에 더한다.
         어절 경계(공백·세그먼트 끝)에서 trie 끝 노드(완결 어절)면 확정, 아니면 회수한다. 경로가 끊기면(자식
         없음 — 완결 어절 뒤에 음절이 더 이어지는 경우 포함) 회수하고 다음 공백까지 새 일치를 시작하지 않는다
  빔     프레임마다 상위 k 토큰만 확장하고, 접두사마다 blank로 끝나는 경로와 비blank로 끝나는 경로의 log
         확률을 따로 log-sum-exp로 모은다. (음향 점수 + 확정 + 보류 보너스)가 높은 beam개만 남기고, 끝에서
         보류 보너스를 정리(완결만 확정)한 점수가 가장 높은 접두사를 낸다
"""

# [왜 편향인가] 이 모델의 오류는 처음 보는 문장·어절에 몰린다(docs/experiments.md 7단계 결과: Zeroth test 본
# 문장 1.85 % 대 안 본 문장 16.01 %). 고유명사·전문어가 대표적인 처음 보는 어절이고, 학습이나 외부 LM 없이
# 사용자 사전으로 대응할 수 있다. 사용자별 구문 목록(연락처 등)으로 LM이 있을 때와 없을 때 모두 정확도를
# 올렸다고 보고한 WeNet 2.0의 문맥 편향(Zhang 외, arXiv:2203.15455)과 같은 계열이다.
# [왜 회수인가] 접두사에 주는 보너스는 빔이 목록 어절 경로를 놓치지 않게 하는 장치다. 끝까지 맞지 않은 경로의
# 보너스를 남기면 '목록 어절처럼 시작하는' 가설이 부당하게 이긴다. 그래서 끊기거나 어절 경계에서 미완결이면
# 그 경로에서 받은 보너스를 되돌리고, 완결된 어절의 보너스만 남긴다.
# [왜 어절 단위 일치인가] 목록 항목이 어절(조사 포함)이고 판정 지표가 어절 정확 일치율이라, 일치를 어절
# 시작에서만 시작하고 어절 경계에서만 완결로 친다. 그래야 다른 어절 안쪽에 우연히 든 음절열(예: '나가서' 속
# '가서')에 보너스가 붙지 않고, 띄어쓰기가 틀린 가설('아이 유가')보다 목록 어절을 그대로 쓴 가설('아이유가')이
# 이긴다.
# [근사] 프레임마다 상위 k 토큰만 보므로(WeNet ctc_prefix_beam_search와 같은 가지치기), 그 프레임에서 상위 k에
# 들지 않은 토큰으로 끝나는 접두사의 '같은 토큰 반복' 경로는 그 프레임에서 버린다. blank는 상위 k에 없어도
# 따로 받은 blank log 확률로 늘 확장한다.
# [결정성] 같은 입력이면 같은 출력이다. 프레임 안 후보를 토큰 id 순으로 정렬해 처리하고 동점은 접두사
# 번호(만든 순서)로 가르므로, 후보 배열의 순서(예: torch.topk의 같은 값 순서)가 결과를 바꾸지 않는다.

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from stt_scratch.text import Vocab, normalize_train

NEG_INF = float("-inf")
ROOT = 0  # 어절 시작에서 일치를 기다리는 상태(trie 뿌리 노드 번호)
DEAD = -1  # 어절 가운데서 일치가 끊긴 상태: 다음 공백까지 새 일치를 시작하지 않는다

# 접두사 하나의 문맥 상태: (trie 노드 또는 ROOT·DEAD, 보류 보너스, 확정 보너스). 보너스는 nats(log 확률과 같은
# 단위)다.
State = tuple[int, float, float]


# 두 log 확률의 log(exp(a) + exp(b)). 인자를 크기순으로 정렬해 계산하므로 a, b 순서를 바꿔도 비트 단위로 같은
# 값이 나온다(누적 순서와 무관한 결정성에 필요). 둘 다 -inf면 -inf.
def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


class ContextTrie:
    """편향 목록의 음절 trie와 보너스 상태 전이.

    sequences: 목록 어절마다의 토큰 id 열(공백 id는 무시, blank id 0은 거부). space_id: 어절 경계 토큰.
    bonus: 일치하는 음절마다 주는 보너스 b(nats, 0 이상의 유한한 값; 0이면 편향 없는 빔). 같은 열은 한
    항목이다.
    """

    def __init__(
        self, sequences: Iterable[Sequence[int]], space_id: int, bonus: float, blank: int = 0
    ) -> None:
        if not (math.isfinite(bonus) and bonus >= 0.0):
            raise ValueError(f"bonus must be a finite number >= 0, got {bonus!r}")
        self.space_id = int(space_id)
        self.bonus = float(bonus)
        # 노드 0 = 뿌리. children[노드] = {토큰 id: 자식 노드}, terminal[노드] = 목록 어절이 여기서 끝나는가.
        self.children: list[dict[int, int]] = [{}]
        self.terminal: list[bool] = [False]
        self.entries = 0  # 서로 다른 목록 어절 수(0이면 편향할 것이 없다 → 호출하는 쪽이 greedy를 쓴다)
        self.dropped: list[str] = []  # from_words가 뺀 어절(원래 문자열)
        self.initial: State = (ROOT, 0.0, 0.0)  # 세그먼트 처음 = 어절 시작
        for seq in sequences:
            ids = [int(t) for t in seq if int(t) != self.space_id]
            if blank in ids:
                raise ValueError("a context sequence cannot contain the CTC blank id")
            if not ids:
                continue
            node = ROOT
            for tok in ids:
                child = self.children[node].get(tok)
                if child is None:
                    child = len(self.children)
                    self.children[node][tok] = child
                    self.children.append({})
                    self.terminal.append(False)
                node = child
            if not self.terminal[node]:
                self.terminal[node] = True
                self.entries += 1

    # 사람이 준 어절 목록 → trie. 어절마다 학습 정규화(NFC, 문장부호 → 공백)한 뒤 공백을 빼고 음절 id로
    # 바꾼다. 비었거나 어휘에 없는 음절(id = <unk>, 숫자·라틴 포함)이 든 어절은 모델이 그대로 낼 수 없으므로
    # 빼고 dropped에 원래 문자열을 남긴다(평가 도구가 개수를 보고한다).
    @classmethod
    def from_words(cls, words: Iterable[str], vocab: Vocab, bonus: float) -> ContextTrie:
        """어절 목록으로 trie를 만든다(정규화·공백 제거, 어휘 밖 음절이 든 어절은 dropped로)."""
        kept: list[list[int]] = []
        dropped: list[str] = []
        for word in words:
            ids = vocab.encode(normalize_train(word).replace(" ", ""))
            if not ids or vocab.unk_id in ids:
                dropped.append(word)
            else:
                kept.append(ids)
        trie = cls(kept, vocab.space_id, bonus, vocab.blank_id)
        trie.dropped = dropped
        return trie

    # 접두사에 토큰 하나를 붙였을 때의 새 상태. 공백: 완결 어절(끝 노드)이면 보류 보너스를 확정에 더하고,
    # 아니면 버린 뒤(회수) 어절 시작으로. 음절(<unk> 포함): 어절 시작이나 일치 중인 노드에서 자식이 있으면
    # 내려가며 보류 보너스 + b, 없으면 회수하고 DEAD(다음 공백까지).
    def step(self, state: State, token: int) -> State:
        """state에 token을 붙인 뒤의 (노드, 보류 보너스, 확정 보너스)."""
        node, pending, locked = state
        if token == self.space_id:
            if node > ROOT and self.terminal[node]:
                return (ROOT, 0.0, locked + pending)
            return (ROOT, 0.0, locked)
        if node >= ROOT:
            child = self.children[node].get(token)
            if child is not None:
                return (child, pending + self.bonus, locked)
        return (DEAD, 0.0, locked)

    @staticmethod
    def score(state: State) -> float:
        """탐색 중 점수에 더할 보너스: 확정 + 보류(목록 어절을 따라가는 중인 접두사가 빔에 남게)."""
        return state[1] + state[2]

    def final_score(self, state: State) -> float:
        """디코딩 끝(= 어절 경계)의 보너스: 완결 어절이면 보류분까지 확정, 아니면 보류분을 회수한다."""
        node, pending, locked = state
        return locked + pending if node > ROOT and self.terminal[node] else locked


# top-k CTC prefix beam search. 입력은 한 세그먼트의 프레임별 상위 k (토큰 id, log 확률)와 blank log 확률로,
# scripts/final_pass_dump.py가 저장하는 배열(topk_ids, topk_logprobs, blank_logprob)이나
# model/decode.biased_batch가 로짓에서 만든 배열이다. 접두사는 (부모 번호, 마지막 토큰)으로 번호를 붙여(같은
# 토큰 열 = 같은 번호) 튜플 복사 없이 늘린다. 출력: 가장 높은 접두사의 토큰 id 열(blank 없음, CTC 축약 뒤;
# greedy_batch 결과와 같은 형식). 주의: 순수 Python이라 계산량은 프레임 수 × beam × k에 비례한다(5 s 세그먼트
# 125프레임, beam 8, k 10이면 약 1만 번 확장). 접두사 번호·상태 표는 세그먼트 하나 동안만 쓴다.
def prefix_beam_search(
    topk_ids: np.ndarray,
    topk_logprobs: np.ndarray,
    blank_logprob: np.ndarray,
    context: ContextTrie,
    beam: int = 8,
    blank: int = 0,
) -> list[int]:
    """프레임 상위 k 후보만 확장하는 CTC prefix beam에 context의 trie 보너스를 얹어 최고 접두사를 낸다."""
    ids = np.asarray(topk_ids)
    lps = np.asarray(topk_logprobs)
    blanks = np.asarray(blank_logprob)
    if ids.ndim != 2 or ids.shape != lps.shape or blanks.shape != (ids.shape[0],):
        raise ValueError(
            f"expected top-k arrays of one shape (T, k) and blank log-probs (T,), got "
            f"{ids.shape}, {lps.shape}, {blanks.shape}"
        )
    if beam < 1:
        raise ValueError(f"beam must be at least 1, got {beam}")
    # 프레임 안 후보를 토큰 id 순으로 정렬해 Python 목록으로(정렬은 결정성, 목록은 속도 때문).
    order = np.argsort(ids, axis=1, kind="stable")
    frame_ids = np.take_along_axis(ids, order, axis=1).astype(np.int64).tolist()
    frame_lps = np.take_along_axis(lps, order, axis=1).astype(np.float64).tolist()
    frame_blanks = blanks.astype(np.float64).tolist()

    parent = [-1]  # 접두사 번호 → 부모 번호(0 = 빈 접두사)
    last = [-1]  # 접두사 번호 → 마지막 토큰(빈 접두사는 -1)
    states: list[State] = [context.initial]
    child_of: dict[tuple[int, int], int] = {}
    step, score = context.step, context.score
    # 빔: (접두사 번호, blank로 끝나는 경로 log 확률, 비blank로 끝나는 경로 log 확률)
    hyps: list[tuple[int, float, float]] = [(0, 0.0, NEG_INF)]
    for cand_ids, cand_lps, lb in zip(frame_ids, frame_lps, frame_blanks, strict=True):
        cands = [(c, lp) for c, lp in zip(cand_ids, cand_lps, strict=True) if c != blank]
        nxt: dict[int, list[float]] = {}
        for pid, pb, pnb in hyps:
            total = _logaddexp(pb, pnb)
            # blank: 접두사는 그대로, blank로 끝나는 경로로.
            cell = nxt.get(pid)
            if cell is None:
                cell = nxt[pid] = [total + lb, NEG_INF]
            else:
                cell[0] = _logaddexp(cell[0], total + lb)
            tail = last[pid]
            for c, lp in cands:
                if c == tail:
                    # 같은 토큰 반복: 비blank로 끝난 경로는 축약돼 접두사 그대로, blank를 사이에 둔 경로만 새
                    # 토큰.
                    cell[1] = _logaddexp(cell[1], pnb + lp)
                    gain = pb + lp
                    if gain == NEG_INF:
                        continue
                else:
                    gain = total + lp
                key = (pid, c)
                cid = child_of.get(key)
                if cid is None:
                    cid = child_of[key] = len(parent)
                    parent.append(pid)
                    last.append(c)
                    states.append(step(states[pid], c))
                ccell = nxt.get(cid)
                if ccell is None:
                    nxt[cid] = [NEG_INF, gain]
                else:
                    ccell[1] = _logaddexp(ccell[1], gain)
        ranked = sorted(
            nxt.items(), key=lambda item: (-(_logaddexp(*item[1]) + score(states[item[0]])), item[0])
        )
        hyps = [(pid, cell[0], cell[1]) for pid, cell in ranked[:beam]]
    best_pid: int | None = None
    best = NEG_INF
    for pid, pb, pnb in hyps:
        value = _logaddexp(pb, pnb) + context.final_score(states[pid])
        if best_pid is None or value > best or (value == best and pid < best_pid):
            best_pid, best = pid, value
    out: list[int] = []
    while best_pid > 0:
        out.append(last[best_pid])
        best_pid = parent[best_pid]
    out.reverse()
    return out
