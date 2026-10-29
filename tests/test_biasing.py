"""확정 자막 핫워드 문맥 편향(docs/experiments.md 규칙 변경 29) 테스트: 음절 trie 보너스, top-k CTC prefix
beam, greedy 분기, 평가·지연 도구.

확인하는 것: (1) 편향 목록이 없거나 비면(모든 어절이 빠진 목록 포함) 확정 자막 디코딩이 greedy_batch와 비트
단위로 같다, (2) 합성 log-prob에서 목록 어절이 2위 후보일 때 보너스로 1위가 되고 목록 밖 출력은
그대로다(보너스가 작거나 목록과 무관하면 그대로), (3) 부분 일치 뒤 끊기면 그 경로에서 받은 보너스를 회수하고
완결 어절의 보너스는 남긴다(일치는 어절 시작에서만 시작하고 어절 경계·세그먼트 끝에서 완결), (4) 빔이
결정적이고 프레임 안 후보 순서와 무관하다, (5) 실시간 경로(biased_batch)와 덤프
경로(scripts/final_pass_dump.frame_stats 배열 → prefix_beam_search)가 같은 id를 낸다, (6) 평가 도구가 희귀
어절·편향 목록을 등록대로 만들고, 덤프 greedy가 기록된 확정 가설을 재현하는지 확인하며, 절반 A에서만 고르고
절반 B에서 판정하고, 지연 결과가 없으면 채택을 정하지 않는다, (7) 지연 측정이 작은 무작위 모델로 CPU에서
끝까지 돌고 있는 결과는 덮지 않는다.
"""

# [방식] 디코더는 손으로 짠 프레임 분포(토큰 프레임 + blank 프레임)로 결과를 미리 계산할 수 있게 하고, 평가
# 도구는 그런 프레임으로 만든 합성 덤프로 끝까지 돌린다. 지연 측정은 tests/test_confidence.py와 같은 작은
# 무작위 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 흐름만 본다. 실제 수치는 GPU 덤프·지연 측정 뒤에만 나온다.

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from stt_scratch import biasing
from stt_scratch import confidence as conf
from stt_scratch.data import Utterance
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.decode import biased_batch, greedy_batch
from stt_scratch.text import Vocab

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
followup = importlib.import_module("followup")
fpd = importlib.import_module("final_pass_dump")
ce = importlib.import_module("confidence_eval")
be = importlib.import_module("bias_eval")

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))
BLANK, UNK, SP, GA, NA, DA, RA, MA = range(8)
CHAR = {ch: i for i, ch in enumerate(VOCAB.tokens)} | {"_": BLANK}


# 프레임마다 정한 토큰 확률(글자로 적고 '_' = blank)과, 나머지 토큰에 남은 질량을 고르게 나눈 log 확률 (T, V).
def lattice(*frames: dict[str, float]) -> np.ndarray:
    rows = []
    for frame in frames:
        p = np.zeros(len(VOCAB))
        for ch, value in frame.items():
            p[CHAR[ch]] = value
        rest = p == 0
        p[rest] = (1.0 - p.sum()) / rest.sum()
        rows.append(np.log(p))
    return np.asarray(rows, dtype=np.float32)


# log 확률 (T, V) → (상위 k id, 그 log 확률, blank log 확률). 같은 값은 id가 작은 쪽이 앞(안정 정렬).
def topk(lp: np.ndarray, k: int):
    order = np.argsort(-lp, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(lp, order, axis=1), lp[:, BLANK]


def beam_text(lp: np.ndarray, words: list[str], bonus: float, beam: int = 8, k: int = 10) -> str:
    trie = biasing.ContextTrie.from_words(words, VOCAB, bonus)
    ids, vals, blank = topk(lp, k)
    return VOCAB.decode(biasing.prefix_beam_search(ids, vals, blank, trie, beam=beam))


def greedy_text(lp: np.ndarray) -> str:
    return VOCAB.decode(greedy_batch(torch.from_numpy(lp)[None], torch.tensor([lp.shape[0]]))[0])


# --- (1) 빈 목록 = greedy -----------------------------------------------------------------------------------


# 확인: 목록이 없을 때(None), 항목이 없는 trie, 모든 어절이 빠진 목록(라틴·빈 문자열·문장부호만)에서
# biased_batch가 greedy_batch와 같은 id 목록을 낸다(fp32·fp16 로짓 모두, 발화마다 길이가 다른 배치).
def test_empty_list_decodes_exactly_like_greedy() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 40, len(VOCAB), generator=g) * 3
    lengths = torch.tensor([40, 31, 7])
    empty = biasing.ContextTrie([], VOCAB.space_id, 2.0)
    dropped = biasing.ContextTrie.from_words(["abc", "", "!!"], VOCAB, 2.0)
    assert empty.entries == 0 and dropped.entries == 0 and dropped.dropped == ["abc", "", "!!"]
    for context in (None, empty, dropped):
        assert biased_batch(logits, lengths, context) == greedy_batch(logits, lengths)
        assert biased_batch(logits.half(), lengths, context) == greedy_batch(logits.half(), lengths)


# --- (2) 2위 후보의 목록 어절 -------------------------------------------------------------------------------

# '가나 다라'에서 '나' 프레임의 2위 후보가 '마'(0.6 대 0.3, log 차 약 0.69)인 격자.
SECOND = lattice(
    {"가": 0.97},
    {"_": 0.98},
    {"나": 0.6, "마": 0.3},
    {"_": 0.98},
    {" ": 0.97},
    {"_": 0.98},
    {"다": 0.97},
    {"_": 0.98},
    {"라": 0.97},
    {"_": 0.98},
)


# 확인: 목록 '가마'에 보너스 2를 주면 '마'(2위)가 1위가 되어 '가마 다라'가 나오고 목록 밖 '다라'는 그대로다.
# 보너스가 작으면(2 × 0.1 < 0.69) 그대로, 목록이 비거나 목록과 무관하면(빔만) greedy와 같은 텍스트다. 실시간
# 경로(biased_batch, log 확률을 로짓으로 넘김)도 같다.
def test_bonus_lifts_a_listed_word_from_second_place() -> None:
    assert greedy_text(SECOND) == "가나 다라"
    assert beam_text(SECOND, [], 2.0) == "가나 다라"
    assert beam_text(SECOND, ["가마"], 2.0) == "가마 다라"
    assert beam_text(SECOND, ["가마"], 2.0, beam=4) == "가마 다라"
    assert beam_text(SECOND, ["가마"], 0.1) == "가나 다라"
    assert beam_text(SECOND, ["라마"], 2.0) == "가나 다라"
    trie = biasing.ContextTrie.from_words(["가마"], VOCAB, 2.0)
    ids = biased_batch(torch.from_numpy(SECOND)[None], torch.tensor([len(SECOND)]), trie)[0]
    assert VOCAB.decode(ids) == "가마 다라"


# --- (3) 부분 일치 회수와 완결 어절 -------------------------------------------------------------------------

# '가마라'가 '가나라'보다 조금 낫다(0.5 대 0.45, log 차 약 0.11).
WITHDRAW = lattice({"가": 0.97}, {"_": 0.98}, {"나": 0.45, "마": 0.5}, {"_": 0.98}, {"라": 0.97}, {"_": 0.98})


# 확인 (가) 상태 전이: 일치하는 음절마다 보류 보너스 b가 쌓이고(탐색 점수에 보인다) 어절 경계(공백·세그먼트
# 끝)에서 완결이면 확정, 미완결이면 회수된다. 경로가 끊기면(자식 없음, 완결 어절 뒤에 음절이 이어짐) 받은
# 보너스를 회수하고 다음 공백까지 새 일치를 시작하지 않는다. 띄어 쓴 목록 어절은 공백을 뺀 어절과 같은 trie다.
# (나) 빔: '가나다'의 접두 '가나'가 받은 2 × 2.0은 '라'에서 끊겨 회수되므로 음향 점수가 나은 '가마라'가 남는다
# (회수하지 않았다면 4.0이 0.11 차이를 뒤집는다). 같은 자리에서 완결되는 목록 어절 '가나라'는 보너스를 유지해
# 1위가 된다. top-k 3이라 '라' 프레임에 '다'는 후보가 아니다.
def test_partial_match_bonus_is_withdrawn_and_complete_words_keep_it() -> None:
    trie = biasing.ContextTrie.from_words(["가나다", "마"], VOCAB, 1.5)

    def run(tokens: list[int]) -> tuple[list[float], float]:
        state, trace = trie.initial, []
        for tok in tokens:
            state = trie.step(state, tok)
            trace.append(trie.score(state))
        return trace, trie.final_score(state)

    assert run([GA, NA, DA]) == ([1.5, 3.0, 4.5], 4.5)
    assert run([GA, NA, DA, SP, RA]) == ([1.5, 3.0, 4.5, 4.5, 4.5], 4.5)
    assert run([GA, NA, RA, SP]) == ([1.5, 3.0, 0.0, 0.0], 0.0)
    assert run([GA, NA]) == ([1.5, 3.0], 0.0)
    assert run([GA, NA, DA, RA]) == ([1.5, 3.0, 4.5, 0.0], 0.0)
    assert run([RA, MA]) == ([0.0, 0.0], 0.0)
    assert run([RA, SP, MA]) == ([0.0, 0.0, 1.5], 1.5)
    assert run([MA, SP, MA, UNK]) == ([1.5, 1.5, 3.0, 1.5], 1.5)
    spaced = biasing.ContextTrie.from_words(["가 나다", "마"], VOCAB, 1.5)
    assert (spaced.children, spaced.terminal) == (trie.children, trie.terminal)

    assert greedy_text(WITHDRAW) == "가마라"
    assert beam_text(WITHDRAW, ["가나다"], 2.0, k=3) == "가마라"
    assert beam_text(WITHDRAW, ["가나라"], 2.0, k=3) == "가나라"


# --- (4) 결정성 ---------------------------------------------------------------------------------------------


# 확인: 같은 입력이면 같은 출력이고, 프레임마다 top-k 후보의 순서를 섞어도 출력이 같다(동점 처리가 입력 순서에
# 기대지 않는다). 빔 크기 0이나 모양이 어긋난 배열은 거부한다.
def test_beam_search_is_deterministic_and_ignores_candidate_order() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(60, len(VOCAB))) * 2.5
    lp = (logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)).astype(np.float32)
    ids, vals, blank = topk(lp, 5)
    trie = biasing.ContextTrie.from_words(["가나", "다라마", "마가"], VOCAB, 2.5)
    first = biasing.prefix_beam_search(ids, vals, blank, trie, beam=4)
    assert first and first == biasing.prefix_beam_search(ids, vals, blank, trie, beam=4)
    perm = rng.permuted(np.tile(np.arange(5), (60, 1)), axis=1)
    shuffled = biasing.prefix_beam_search(
        np.take_along_axis(ids, perm, axis=1), np.take_along_axis(vals, perm, axis=1), blank, trie, beam=4
    )
    assert shuffled == first
    assert biasing.prefix_beam_search(ids[:0], vals[:0], blank[:0], trie) == []
    with pytest.raises(ValueError):
        biasing.prefix_beam_search(ids, vals, blank, trie, beam=0)
    with pytest.raises(ValueError):
        biasing.prefix_beam_search(ids, vals[:, :3], blank, trie)


# 확인: 가지치기가 없으면(k = 어휘 전체, 빔이 모든 접두사를 담을 만큼 큼) 빔의 출력이 정의와 같다: 모든 정렬
# 경로(V^T개)를 직접 펼쳐 CTC 축약(반복 → blank 제거)으로 묶은 라벨열마다 정확한 log 확률을 더하고 그 라벨열의
# 끝 보너스(final_score)를 더한 값이 가장 큰 라벨열. 보너스 0(편향 없음)과 보너스 1.5 두 경우. blank·비blank
# 분리와 같은 토큰 반복 처리가 틀리면 여기서 어긋난다.
def test_beam_search_is_exact_without_pruning() -> None:
    rng = np.random.default_rng(7)
    steps = 5
    for bonus, scale in ((0.0, 1.5), (1.5, 1.0)):
        logits = rng.normal(size=(steps, len(VOCAB))) * scale
        lp = (logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)).astype(np.float32)
        trie = biasing.ContextTrie.from_words(["가나", "다", "마라"], VOCAB, bonus)
        totals: dict[tuple[int, ...], float] = {}
        for path in np.ndindex(*(len(VOCAB),) * steps):
            score = float(sum(float(lp[t, tok]) for t, tok in enumerate(path)))
            labels, prev = [], BLANK
            for tok in path:
                if tok != BLANK and tok != prev:
                    labels.append(tok)
                prev = tok
            key = tuple(labels)
            totals[key] = float(np.logaddexp(totals.get(key, -np.inf), score))

        def with_bonus(labels: tuple[int, ...], trie=trie, totals=totals) -> float:
            state = trie.initial
            for tok in labels:
                state = trie.step(state, tok)
            return totals[labels] + trie.final_score(state)

        expected = max(totals, key=with_bonus)
        ids, vals, blank = topk(lp, len(VOCAB))
        assert tuple(biasing.prefix_beam_search(ids, vals, blank, trie, beam=100_000)) == expected


# --- (5) 실시간 경로 = 덤프 경로 ----------------------------------------------------------------------------


# 확인: 로짓 배치를 biased_batch로 디코딩한 결과가, 같은 로짓에서 덤프 함수(final_pass_dump.frame_stats)가
# 남기는 top-k·blank 배열을 prefix_beam_search에 넣은 결과와 발화마다 같다(평가 도구가 덤프로 잰 출력 =
# 서비스가 낼 출력). 편향이 실제로 출력을 바꾸는 입력이다(보너스 0과 다름). top_k 0은 거부한다.
def test_live_decoding_matches_the_dump_arrays() -> None:
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 50, len(VOCAB), generator=g) * 2
    lengths = torch.tensor([50, 33])
    words = ["가나", "나다라", "마"]
    trie = biasing.ContextTrie.from_words(words, VOCAB, 3.0)
    live = biased_batch(logits, lengths, trie, beam=4, top_k=5)
    plain = biased_batch(logits, lengths, biasing.ContextTrie.from_words(words, VOCAB, 0.0), beam=4, top_k=5)
    assert live != plain
    for b, n in enumerate(lengths.tolist()):
        stats = fpd.frame_stats(logits[b, :n], 5)
        dumped = biasing.prefix_beam_search(
            stats["topk_ids"], stats["topk_logprobs"], stats["blank_logprob"], trie, beam=4
        )
        assert dumped == live[b]
    with pytest.raises(ValueError):
        biased_batch(logits, lengths, trie, top_k=0)


# 확인: 목록 어절을 학습 정규화한 뒤 공백을 빼서 넣고(같은 어절은 하나), 어휘에 없는 음절이 든 어절·빈 어절은
# 빼서 dropped에 원래 문자열로 남긴다. id 열로 직접 만든 trie와 같다. blank id가 든 열, 음수·NaN 보너스는
# 거부한다.
def test_context_trie_normalizes_words_and_rejects_bad_input() -> None:
    trie = biasing.ContextTrie.from_words(["가 나", "가나", "마라!", "abc", "", "가"], VOCAB, 1.0)
    assert trie.entries == 3 and trie.dropped == ["abc", ""]
    same = biasing.ContextTrie([[GA, NA], [MA, RA], [GA]], VOCAB.space_id, 1.0)
    assert (same.children, same.terminal, same.entries) == (trie.children, trie.terminal, 3)
    with pytest.raises(ValueError):
        biasing.ContextTrie([[GA, BLANK]], VOCAB.space_id, 1.0)
    with pytest.raises(ValueError):
        biasing.ContextTrie([[GA]], VOCAB.space_id, -1.0)
    with pytest.raises(ValueError):
        biasing.ContextTrie([[GA]], VOCAB.space_id, float("nan"))


# --- (6) 평가 도구 ------------------------------------------------------------------------------------------


def utterance(uid: str, text: str) -> Utterance:
    return Utterance(id=uid, corpus="cv22", text=text, duration_s=1.0, hangul_only=True)


# 확인: 희귀 어절 = 정답 어절 가운데 학습 어절 집합에 없는 것(등장 순서·중복 그대로). 목록 = 그 발화의 희귀
# 어절(중복 없이 첫 등장 순서) + 다른 발화 희귀 어절 풀에서 seed 0으로 뽑은 방해 어절 N개(자기 희귀 어절은
# 절대 들지 않고, 풀이 모자라면 전부). 같은 입력이면 같은 목록이다.
def test_rare_words_and_bias_lists_follow_the_registration() -> None:
    seen = {"가나", "다라"}
    assert be.rare_words("가나 마가 마가 다라 라마", seen) == ["마가", "마가", "라마"]
    texts = ["가나 마가 마가", "다라 라마 마가", "가나 다라", "나마 가가"]
    utts = [utterance(f"u{i}", t) for i, t in enumerate(texts)]
    lists = be.build_lists(utts, seen, 2)
    assert lists == be.build_lists(utts, seen, 2)
    pool = {"마가", "라마", "나마", "가가"}
    for u in utts:
        own = list(dict.fromkeys(be.rare_words(u.text, seen)))
        entry = lists[u.id]
        assert entry["own"] == own
        assert len(entry["distractors"]) == min(2, len(pool - set(own)))
        assert set(entry["distractors"]) <= pool - set(own)
        assert len(set(entry["distractors"])) == len(entry["distractors"])
    assert be.build_lists(utts, seen, 0)["u2"] == {"own": [], "distractors": []}
    assert sorted(be.build_lists(utts, seen, 10)["u2"]["distractors"]) == sorted(pool)


# 정한 후보(내림차순) 뒤에 나머지 토큰을 아주 작은 확률(1e-4, id 순)로 붙여 프레임마다 어휘 전체를
# 채운다(덤프의 top-10은 어휘가 8개면 8개 전부다).
def full(main: list[tuple[int, float]]) -> list[tuple[int, float]]:
    listed = {t for t, _ in main}
    return main + [(t, 1e-4) for t in range(len(VOCAB)) if t not in listed]


# 합성 덤프의 프레임: 토큰마다 토큰 프레임 + blank 프레임. alts = {토큰 위치: (대안 토큰, 확률)}이면 그 토큰
# 프레임의 2위가 대안이다. 항목 = (argmax, 내림차순 top-8 [(id, p)], blank 확률).
def token_frames(tokens: list[int], alts: dict[int, tuple[int, float]]) -> list[tuple]:
    rows = []
    for i, tok in enumerate(tokens):
        if i in alts:
            alt, p_alt = alts[i]
            rows.append((tok, full([(tok, 0.6), (alt, p_alt), (BLANK, 0.05)]), 0.05))
        else:
            rows.append((tok, full([(tok, 0.9), (BLANK, 0.05), (UNK, 0.01)]), 0.05))
        rows.append((BLANK, full([(BLANK, 0.97), (tok, 0.02), (UNK, 0.005)]), 0.97))
    return rows


TAIL = [(BLANK, full([(BLANK, 0.97), (GA, 0.02), (UNK, 0.005)]), 0.97)] * 4  # 텍스트 없는 끝점 세그먼트


# 발화 유형(i % 4): 0 = 희귀 어절 '마가'를 '나가'로 잘못 냈고 '마'가 2위(확률 0.3~0.07; 음향 log 차 약
# 0.7~2.2라 보너스 1이면 앞의 셋, 2 이상이면 넷 다 고쳐짐), 1 = 맞게 냈지만 방해 어절 '마라'의 '마'가 0.07로
# 2위(보너스 2 이상이면 '마라'가 잘못 들어감), 2 = 희귀 어절 없음, 3 = 희귀 어절 '마라'를 맞게 냄(풀에
# '마라'를 넣는다). 학습 어절은 '가나', '다라', '나가'.
def pattern(i: int) -> tuple[str, str, dict[int, tuple[int, float]]]:
    kind = i % 4
    if kind == 0:
        return "가나 마가", "가나 나가", {3: (MA, (0.3, 0.2, 0.12, 0.07)[(i // 4) % 4])}
    if kind == 1:
        return "다라 가나", "다라 가나", {0: (MA, 0.07)}
    if kind == 2:
        return "가나 다라", "가나 다라", {}
    return "마라 가나", "마라 가나", {}


def synthetic_dump(directory: Path, name: str, count: int, config: dict) -> dict[str, tuple[str, str]]:
    frames: dict[str, list] = {k: [] for k in fpd.FRAME_KEYS}
    segments, utterances, texts = [], [], {}
    offset = 0
    for i in range(count):
        uid = f"{name}/{i}"
        ref, hyp, alts = pattern(i)
        parts = [(token_frames(VOCAB.encode(hyp), alts), "final", hyp)]
        if i % 5 == 4:
            parts.append((TAIL, "endpoint", ""))
        seg_ids = []
        for rows, kind, text in parts:
            for argmax, top, p_blank in rows:
                frames["argmax"].append(argmax)
                frames["max_logprob"].append(math.log(top[0][1]))
                frames["entropy"].append(0.1)
                frames["blank_logprob"].append(math.log(p_blank))
                frames["topk_ids"].append([t for t, _ in top])
                frames["topk_logprobs"].append([math.log(p) for _, p in top])
            seg_ids.append(len(segments))
            segments.append(
                {
                    "utt": uid,
                    "segment": len(seg_ids) - 1,
                    "kind": kind,
                    "rule": "rule2",
                    "emitted": True,
                    "text": text,
                    "stream_text": hyp,
                    "offset": offset,
                    "frames": len(rows),
                    "mel_frames": 4 * len(rows),
                    "pass_ms": 1.0,
                }
            )
            offset += len(rows)
        utterances.append(
            {
                "id": uid,
                "corpus": "cv22",
                "hangul_only": True,
                "noise_seed": i,
                "hyp_final": hyp,
                "hyp_stream": hyp,
                "segments": seg_ids,
            }
        )
        texts[uid] = (ref, hyp)
    arrays = {
        k: np.asarray(v, dtype=np.int16 if k in ("argmax", "topk_ids") else np.float32)
        for k, v in frames.items()
    }
    meta = {
        "format": fpd.FORMAT_VERSION,
        "set": name,
        "provenance": {"checkpoint": "m.pt", "checkpoint_sha256": "c" * 64, "git_head": "x", "limit": 0},
        "config": config,
        "vocab_sha256": "v" * 64,
        "frames": offset,
        "segments": segments,
        "utterances": utterances,
    }
    directory.mkdir(parents=True, exist_ok=True)
    fpd.write_dump(directory, name, arrays, meta)
    return texts


P1_CONFIG = {
    "chunk": 8,
    "left": 128,
    "rule1_s": 2.4,
    "rule2_s": 0.8,
    "rule3_s": 20.0,
    "reset_cache_on_endpoint": False,
    "step_path": "graph",
    "final_pass": "full",
    "final_pass_bucket": 128,
    "precision": "fp16-autocast",
    "tail_s": 3.0,
    "top_k": 3,
}


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for uid, text in rows:
            line = {"id": uid, "corpus": "cv22", "text": text, "duration_s": 1.0, "hangul_only": True}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# 평가용 가짜 저장소: 합성 덤프 두 세트(dev_mix 40발화, dev_h 12발화), 매니페스트(정답), 학습 매니페스트(학습
# 어절 집합), P1 기록(session.json, 확정 가설 파일 = 덤프 greedy). dev_mix id는 절반 A·B에 고르게 든다.
def eval_root(tmp_path: Path) -> Path:
    session = {
        "provenance": {"checkpoint_sha256": "c" * 64},
        "step_path": "graph",
        "tail_noise_s": 3.0,
        "runs": {},
    }
    hyp_dir = tmp_path / "reports" / "followup" / "hyps"
    hyp_dir.mkdir(parents=True)
    write_manifest(tmp_path / "manifests" / "train.jsonl", [("t/0", "가나 다라"), ("t/1", "나가 가나")])
    for name, count in (("dev_mix", 40), ("dev_h", 12)):
        texts = synthetic_dump(tmp_path / "dump", name, count, P1_CONFIG)
        write_manifest(
            tmp_path / "manifests" / f"{name}.jsonl", [(uid, ref) for uid, (ref, _) in texts.items()]
        )
        utts = [utterance(uid, ref) for uid, (ref, _) in texts.items()]
        hyps = {uid: hyp for uid, (_, hyp) in texts.items()}
        session["runs"][f"{name}_rule2_0.8"] = {"cer_ns_final_pass": ce.cer_ns_hangul(utts, hyps)}
        with open(hyp_dir / f"{name}-session-rule2_0.8-final_pass.jsonl", "w", encoding="utf-8") as f:
            for u in utts:
                f.write(
                    json.dumps({"id": u.id, "ref": u.text, "hyp": hyps[u.id], "hangul_only": True}) + "\n"
                )
    (tmp_path / "reports" / "followup" / "session.json").write_text(json.dumps(session), encoding="utf-8")
    return tmp_path


def patch_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(be, "ROOT", root)
    monkeypatch.setattr(ce, "ROOT", root)
    monkeypatch.setattr(followup, "ROOT", root)
    monkeypatch.setattr(ce, "load_vocab", lambda: (VOCAB, "v" * 64))


def eval_args(**overrides) -> argparse.Namespace:
    values = {"dump_dir": "dump", "out": "out/biasing.json", "latency_dir": "lat", "overwrite": False}
    values.update(overrides)
    return argparse.Namespace(**values)


# 지연 결과 파일(등록 조건과 같은 측정: 체크포인트, 100세그먼트, N = 100·seed 0 목록, CPU 1스레드).
def write_latency(directory: Path, device: str, p90: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = {
        be.config_key(b, k): {"n": 100, "p50": p90 / 2, "p90": p90, "max": p90 * 1.5}
        for b in be.BONUSES
        for k in be.BEAMS
    }
    data = {
        "provenance": {"checkpoint_sha256": "c" * 64},
        "threads": 1,
        "segments": 100,
        "lists": {"distractors": 100, "seed": 0},
        "greedy_ms": {"n": 100, "p50": 10.0, "p90": 20.0, "max": 30.0},
        "biased_ms": rows,
        "load_before": {"idle": True},
    }
    (directory / f"biasing_latency-{device}.json").write_text(json.dumps(data), encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 확인: 재현 조건이 맞으면 평가가 끝까지 돌아 (가) 격자 8구성을 절반 A 발화로만 재고 등록 규칙(A CER 가드를
# 지키는 구성 가운데 희귀 어절 일치율 최대, 동점은 작은 보너스·작은 빔)으로 고른다(절반 B 정답을 바꿔도 선택이
# 그대로), (나) 절반 B 판정 수치가 모듈 함수로 B 발화에서 직접 계산한 값과 같고 덤프 greedy가 기록 확정 가설을
# 그대로 재현한다, (다) 목록이 빈 발화는 greedy 그대로다, (라) 지연 결과가 없으면 채택을 정하지 않고(정확도
# 조건이 깨졌으면 False), 등록 조건의 지연 결과가 있으면 판정에 넣는다(CPU p90이 500 ms를 넘으면 채택 아님).
# 결과 파일이 있으면 덮지 않는다.
def test_bias_eval_selects_on_half_a_and_judges_on_half_b(tmp_path, monkeypatch) -> None:
    root = eval_root(tmp_path)
    patch_root(monkeypatch, root)
    be.cmd_eval(eval_args())
    result = read(root / "out" / "biasing.json")
    assert result["reproduction"]["met"] is True
    sel = result["selection"]
    assert sel["distractors"] == be.JUDGED_DISTRACTORS == 100
    assert [(r["bonus"], r["beam"]) for r in sel["rows"]] == [(b, k) for b in be.BONUSES for k in be.BEAMS]
    eligible = [r for r in sel["rows"] if r["cer_ns_delta"] <= be.CER_GUARD]
    assert eligible
    expected = max(eligible, key=lambda r: (r["rare_match_biased"], -r["bonus"], -r["beam"]))
    assert sel["chosen"] == {"bonus": expected["bonus"], "beam": expected["beam"], "key": expected["key"]}

    utts = ce.read_dev_manifest("dev_mix")
    targets = [u for u in utts if u.hangul_only]
    seen = be.train_word_set()
    assert seen == {"가나", "다라", "나가"}
    meta, arrays = fpd.load_dump(root / "dump", "dev_mix")
    segs = be.dump_segments(meta, arrays, VOCAB, utts)
    all_ids = [u.id for u in targets]
    greedy = be.decode_set(segs, all_ids, None, VOCAB, 0.0, 1)
    assert greedy == {row["id"]: row["hyp_final"] for row in meta["utterances"]}
    lists = be.build_lists(targets, seen, be.JUDGED_DISTRACTORS)
    half_a = [u for u in targets if conf.half_of(u.id) == "A"]
    half_b = [u for u in targets if conf.half_of(u.id) == "B"]
    assert half_a and half_b and result["split"]["half_b_utterances"] == len(half_b)
    chosen = sel["chosen"]
    biased_b = be.decode_set(segs, [u.id for u in half_b], lists, VOCAB, chosen["bonus"], chosen["beam"])
    direct = be.compare(half_b, greedy, biased_b, seen, lists, VOCAB)
    judged = result["dev_mix_half_b"]["distractors_100"]
    assert judged == direct
    assert judged["cer_ns_greedy"] == pytest.approx(ce.cer_ns_hangul(half_b, greedy))
    assert judged["rare_words"] > 0 and judged["rare_match_gain"] > 0
    assert set(result["dev_mix_half_b"]) == {
        "distractors_0",
        "distractors_100",
        "distractors_500",
        "beam_only",
    }
    assert set(result["dev_h"]) == {"distractors_0", "distractors_100", "distractors_500"}

    lists0 = be.build_lists(targets, seen, 0)
    hyps0 = be.decode_set(segs, all_ids, lists0, VOCAB, 4.0, 8)
    empty = [uid for uid in all_ids if not lists0[uid]["own"]]
    assert empty and all(hyps0[uid] == greedy[uid] for uid in empty)

    v = result["verdict"]
    accuracy = v["rare_match_met"] and v["cer_guard_met"] and v["dev_h_guard_met"]
    assert v["rare_match_met"] == (judged["rare_match_gain"] >= 0.10 and judged["rare_match_gain_ci"][0] > 0)
    assert v["latency"]["cuda"]["measured"] is False and v["latency_met"] is None
    assert v["adopted"] == (None if accuracy else False)
    with pytest.raises(SystemExit):
        be.cmd_eval(eval_args())

    write_latency(root / "lat", "cuda", 100.0)
    write_latency(root / "lat", "cpu", 300.0)
    be.cmd_eval(eval_args(out="out/with_latency.json"))
    v2 = read(root / "out" / "with_latency.json")["verdict"]
    assert v2["latency"]["cuda"]["met"] is True and v2["latency"]["cpu"]["met"] is True
    assert v2["latency_met"] is True and v2["adopted"] == accuracy
    write_latency(root / "lat", "cpu", 600.0)
    be.cmd_eval(eval_args(out="out/slow_cpu.json"))
    v3 = read(root / "out" / "slow_cpu.json")["verdict"]
    assert v3["latency"]["cpu"]["met"] is False and v3["latency_met"] is False and v3["adopted"] is False

    # 절반 B 정답의 학습 어절만 바꾼다('가나' → '다라'). 희귀 어절이 그대로라 목록·방해 어절 풀이 같다.
    manifest = root / "manifests" / "dev_mix.jsonl"
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if conf.half_of(r["id"]) == "B":
            r["text"] = r["text"].replace("가나", "다라")
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(ce, "check_reproduction", lambda *a, **k: {"met": True, "applicable": True})
    be.cmd_eval(eval_args(out="out/changed.json"))
    changed = read(root / "out" / "changed.json")
    assert changed["selection"] == sel and changed["dev_mix_half_b"] != result["dev_mix_half_b"]


# 확인: 덤프 greedy가 기록된 P1 확정 가설과 다르면(재현 조건 미충족) 결과를 쓰지 않고 종료 코드 5로 멈추고,
# 체크포인트가 P1과 다르면(연기 시험) 재현 대조 없이 판정 없이 쓴다.
def test_bias_eval_stops_when_the_dump_does_not_reproduce_p1(tmp_path, monkeypatch) -> None:
    root = eval_root(tmp_path)
    patch_root(monkeypatch, root)
    path = root / "reports" / "followup" / "hyps" / "dev_mix-session-rule2_0.8-final_pass.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in rows[:10]:
        r["hyp"] = "마마"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    session_path = root / "reports" / "followup" / "session.json"
    session = read(session_path)
    session["runs"]["dev_mix_rule2_0.8"]["cer_ns_final_pass"] = ce.cer_ns_hangul(
        ce.read_dev_manifest("dev_mix"), {r["id"]: r["hyp"] for r in rows}
    )
    session_path.write_text(json.dumps(session), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        be.cmd_eval(eval_args())
    assert stopped.value.code == ce.REPRO_EXIT == 5
    assert not (root / "out" / "biasing.json").exists()
    session["provenance"]["checkpoint_sha256"] = "d" * 64
    session_path.write_text(json.dumps(session), encoding="utf-8")
    be.cmd_eval(eval_args())
    result = read(root / "out" / "biasing.json")
    assert result["reproduction"]["applicable"] is False and result["verdict"] is None


# --- (7) 지연 측정 ------------------------------------------------------------------------------------------


def tiny_model(seed: int = 0) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(1,),
    )
    return ChunkConformerCTC(cfg).eval()


# 도우미: root 아래 manifests/<name>.jsonl과 data/pcm/<name>.{bin,index.jsonl}(seed 고정 난수 PCM)을 만든다.
def fake_set(root: Path, name: str, utts: list[Utterance]) -> None:
    write_manifest(root / "manifests" / f"{name}.jsonl", [(u.id, u.text) for u in utts])
    pcm_dir = root / "data" / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name))
    pieces, offset = [], 0
    with open(pcm_dir / f"{name}.index.jsonl", "w", encoding="utf-8", newline="\n") as index:
        for u in utts:
            n = int(u.duration_s * 16000)
            index.write(json.dumps({"id": u.id, "offset": offset, "length": n}) + "\n")
            pieces.append(np.clip(rng.standard_normal(n) * 2000, -32768, 32767).astype("<i2"))
            offset += n
    np.concatenate(pieces).tofile(pcm_dir / f"{name}.bin")


def latency_args(**overrides) -> argparse.Namespace:
    values = {
        "latency": True,
        "device": "cpu",
        "threads": 1,
        "limit": 100,
        "checkpoint": "m.pt",
        "latency_dir": "lat",
        "wait_idle_min": 0.0,
        "tail_s": 3.0,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# 확인: 지연 모드가 작은 무작위 모델로 CPU에서 끝까지 돌아 greedy 2차 패스와 격자 8구성의 편향 2차 패스를 앞
# 세그먼트마다 재서(분위수 n = 세그먼트 수) 결과를 쓰고, greedy 텍스트가 덤프 세션이 낸 확정 텍스트와 같으며,
# 끝나면 torch 스레드 수를 되돌린다. 있는 결과는 덮지 않는다.
def test_latency_mode_runs_end_to_end_on_cpu(tmp_path, monkeypatch) -> None:
    fake_set(tmp_path, "dev_mix", [Utterance(f"cv/{i}", "cv22", "가나 마가", 1.5, True) for i in range(3)])
    write_manifest(tmp_path / "manifests" / "train.jsonl", [("t/0", "가나 다라")])
    VOCAB.save(tmp_path / "manifests" / "vocab.json")
    model = tiny_model()
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, tmp_path / "m.pt")
    monkeypatch.setattr(be, "ROOT", tmp_path)
    monkeypatch.setattr(followup, "ROOT", tmp_path)
    monkeypatch.setattr(
        followup, "wait_for_idle", lambda minutes: {"idle": True, "waited_s": 0.0, "snapshot": {}}
    )
    monkeypatch.setattr(followup, "load_snapshot", lambda: {"time": "t"})
    threads = torch.get_num_threads()
    be.cmd_latency(latency_args(limit=2))
    assert torch.get_num_threads() == threads
    result = read(tmp_path / "lat" / "biasing_latency-cpu.json")
    assert result["segments"] == 2 and result["threads"] == 1
    keys = {be.config_key(b, k) for b in be.BONUSES for k in be.BEAMS}
    assert set(result["biased_ms"]) == keys and set(result["beam_cpu_ms"]) == keys
    assert all(row["n"] == 2 for row in result["biased_ms"].values()) and result["greedy_ms"]["n"] == 2
    assert result["greedy_matches_session"] == 2
    assert result["lists"] == {
        "set": "dev_mix",
        "subset": "hangul-only utterances",
        "distractors": be.JUDGED_DISTRACTORS,
        "seed": be.LIST_SEED,
        "words_per_list": result["lists"]["words_per_list"],
    }
    with pytest.raises(SystemExit):
        be.cmd_latency(latency_args(limit=2))
