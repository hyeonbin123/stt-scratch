"""확정 자막 핫워드 문맥 편향 평가(docs/experiments.md 규칙 변경 29): 덤프 재디코딩(CPU)과 2차 패스 지연.

두 가지 일을 한다.
  평가(기본, CPU)  scripts/final_pass_dump.py의 덤프(dev_mix·dev_H)에서 2차 패스 프레임 top-10·blank
                   log-prob을 편향 디코딩(stt_scratch.biasing)으로 다시 푼다. 덤프 재현 조건(덤프 확정 가설
                   = P1 기록)은 규칙 변경 28과 같다(scripts/confidence_eval.check_reproduction).
    대상           한글 전용 발화(주지표와 같은 부분집합)
    희귀 어절      정답 어절(학습 정규화, 공백 분리) 가운데 manifests/train.jsonl 전사의 어절 집합에 없는 것
    목록           발화마다 자기 희귀 어절(중복 없이 첫 등장 순) + 같은 세트 희귀 어절 풀(종류)에서 seed 0
                   Generator로 뽑은 방해 어절 N개(자기 희귀 어절 제외, N ∈ {0, 100, 500}; 풀이 모자라면
                   전부)
    선택(절반 A)   dev_mix를 발화 id SHA-256 첫 바이트 짝·홀로 절반 A·B로 나누고(규칙 변경 28과 같음),
                   A에서 N = 100으로 보너스 b ∈ {1, 2, 3, 4} × beam ∈ {4, 8}을 잰다. A의 CER_ns 증가
                   ≤ 0.3포인트인 구성 가운데 희귀 어절 정확 일치율이 가장 높은 것(동점은 작은 b, 작은 beam;
                   없으면 CER 증가가 가장 작은 구성과 note)
    판정(절반 B)   N = 100에서 희귀 어절 정확 일치율 +10포인트 이상이고 발화 짝지은 부트스트랩 구간 하한
                   > 0, CER_ns 증가 ≤ 0.3포인트(절반 B, dev_mix 전체 둘 다), dev_H CER_ns 증가 ≤ 0.3포인트,
                   2차 패스 p90 GPU ≤ 250 ms·CPU 1스레드 ≤ 500 ms(지연 결과 파일)이면 채택(옵션 제공)
    보고           N = 0·500, 빔만(b = 0), 목록 밖 어절 일치율, 희귀 어절 없는 발화의 CER_ns, 방해 어절
                   삽입 수
  지연(--latency)  최종 모델로 dev_mix 한글 전용 발화를 P1 세션 조건(발화마다 새 세션, 발화 뒤 −60 dBFS
                   잡음 3 s·seed = 매니페스트 순번, 서버 명령행 기본 구성)으로 흘려 앞 100세그먼트의 mel을
                   모은다. 세그먼트마다 greedy 2차 패스(서비스 decode_full)와 격자 8구성의 편향 2차
                   패스(forward + top-k + 빔, 같은 길이 묶음·정밀도·앞뒤 GPU 동기화)를 재 p50·p90을
                   남긴다(GPU 또는 CPU 1스레드)
결과: reports/followup/biasing.json(평가), reports/followup/biasing_latency-<cuda|cpu>.json(지연). 이미 있으면
--overwrite 없이 거부한다. 평가는 지연 결과가 있으면 판정에 넣고, 없으면 지연 조건을 '측정 안 됨'으로 두어
채택을 정하지 않는다(정확도 조건이 깨졌으면 채택 아님).

사용법(측정 순서):
  uv run --no-sync python scripts/bias_eval.py --latency --device cuda --limit 100
  uv run --no-sync python scripts/bias_eval.py --latency --device cpu --threads 1 --limit 100
  uv run --no-sync python scripts/bias_eval.py      # CPU, 덤프(scripts/final_pass_dump.py) 뒤
"""

# [왜 목록을 정답에서 만드는가] 문맥 편향 평가의 관행(정답에 든 드문 단어 + 다른 발화의 드문 단어를 방해
# 단어로 섞은 목록으로, 목록이 정답 단어를 얼마나 끌어올리는지와 방해 단어가 잘못 들어오는지를 함께 봄)을
# 따른다. 실제 사용자 목록보다 낙관적인 조건이므로 결과에 그렇게 적는다(등록 문서의 주의).
# [왜 절반을 나누는가] 격자 8구성을 같은 자료에서 고르고 판정하면 판정 수치가 낙관적으로 치우친다. 절반 A에서
# 고른 것을 절반 B에서 한 번만 잰다. 방해 어절 풀은 세트 전체의 희귀 어절 종류라 두 절반이 같은 풀을 쓰지만,
# 선택은 절반 A 가설과 정답만 본다(풀은 결과를 보기 전에 정해지는 목록 정의다).
# [왜 덤프인가] 2차 패스 forward는 편향과 무관하므로 GPU에서 한 번 저장한 top-10·blank log-prob으로 CPU에서
# 모든 구성을 반복한다. 이 계산은 model/decode.biased_batch와 같다(tests/test_biasing.py).
# [지연] 판정 지연은 서비스 2차 패스(StreamSession.decode_full)와 같은 경로로 잰 forward + 빔 시간이다. 측정
# 전 쉬는 상태를 최대 10분 기다리고 아니면 부하 조건을 기록한다(scripts/followup.py와 같은 규칙). 편향 trie는
# 목록을 받을 때 한 번 만드는 것이라 시간에 넣지 않는다.

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# numpy: 목록 표본(seed 0 Generator)·발화별 집계. torch: 지연 측정의 모델 실행.
import numpy as np
import torch

# 프로젝트 모듈: 편향 디코더, 어절 정렬·절반 나누기(규칙 변경 28과 같은 정의), 매니페스트, 모델 로드·어휘
# 검사, 채점·짝지은 부트스트랩, 확정 자막 디코딩 분기, 특징, 청크 설정, 서비스 세션·예열, 어휘·정규화.
from stt_scratch import biasing
from stt_scratch import confidence as conf
from stt_scratch.data import read_manifest
from stt_scratch.evaluate import check_vocab, load_model
from stt_scratch.metrics import difference_interval, score_ns
from stt_scratch.model.decode import biased_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab, normalize_hyp_ws, normalize_train

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import confidence_eval as ce  # noqa: E402
import final_pass_dump as fpd  # noqa: E402
import followup  # noqa: E402
from stage6 import noise, packets  # noqa: E402

# 등록 값(docs/experiments.md 규칙 변경 29). 숫자는 여기 한곳에만 두고 선택·판정 코드가 읽는다.
JUDGED_SET = "dev_mix"  # 절반 A(선택)·B(판정)
GUARD_SET = "dev_h"  # CER 가드
BONUSES = (1.0, 2.0, 3.0, 4.0)  # 음절당 보너스 b(nats)
BEAMS = (4, 8)
TOP_K = 10  # 프레임당 후보 수(덤프가 남긴 top-10)
DISTRACTORS = (0, 100, 500)  # 방해 어절 수 N
JUDGED_DISTRACTORS = 100  # 선택·판정의 N
LIST_SEED = 0
MATCH_GAIN = 0.10  # 희귀 어절 정확 일치율 +10포인트
CER_GUARD = 0.003  # CER_ns 증가 0.3포인트
P90_LIMIT_MS = {"cuda": 250.0, "cpu": 500.0}  # 2차 패스(forward + 빔) p90 상한(CPU는 1스레드)
LATENCY_SEGMENTS = 100  # dev_mix 앞 100세그먼트


def config_key(bonus: float, beam: int) -> str:
    """격자 구성 이름(결과 파일의 키): 예 b2_beam8."""
    return f"b{bonus:g}_beam{beam}"


# --- 희귀 어절과 편향 목록 ---------------------------------------------------------------------------------


def train_word_set() -> set[str]:
    """manifests/train.jsonl 전사(학습 정규화)의 어절 집합(모델이 학습에서 본 어절)."""
    utts = read_manifest(ROOT / "manifests" / "train.jsonl")
    return {w for u in utts for w in normalize_train(u.text).split()}


def rare_words(text: str, seen: set[str]) -> list[str]:
    """정답 어절 가운데 학습 어절 집합에 없는 것(등장 순서, 중복 그대로)."""
    return [w for w in normalize_train(text).split() if w not in seen]


# 발화별 편향 목록. 풀 = 대상 발화들의 희귀 어절 종류(정렬). 발화마다 seed 0 Generator(하나를 매니페스트
# 순서로 이어 씀)의 풀 크기 순열을 하나 뽑아 그 순서대로 자기 희귀 어절이 아닌 어절을 N개 고른다. 순열 길이가
# 늘 풀 크기라 발화마다 쓰는 난수 양이 같으므로, 한 발화의 정답이 바뀌어도(풀이 같으면) 다른 발화의 목록은
# 바뀌지 않는다. 출력: id → {"own": 자기 희귀 어절, "distractors": 방해 어절}. 방해 어절은 정의상 그 발화
# 정답에 없다.
def build_lists(
    utts, seen: set[str], distractors: int, seed: int = LIST_SEED
) -> dict[str, dict[str, list[str]]]:
    own = {u.id: list(dict.fromkeys(rare_words(u.text, seen))) for u in utts}
    pool = sorted({w for words in own.values() for w in words})
    rng = np.random.default_rng(seed)
    lists = {}
    for u in utts:
        mine = set(own[u.id])
        picked: list[str] = []
        if distractors > 0 and pool:
            for j in rng.permutation(len(pool)).tolist():
                if pool[j] not in mine:
                    picked.append(pool[j])
                    if len(picked) == distractors:
                        break
        lists[u.id] = {"own": own[u.id], "distractors": picked}
    return lists


# 어휘 밖 음절(숫자·라틴 포함)이 든 어절: 모델이 그대로 낼 수 없어 목록에서 빠지지만(biasing.ContextTrie.
# from_words) 일치율 분모에는 남는다(고칠 수 없는 희귀 어절이 몇 개인지 보고한다).
def _unmatchable(word: str, vocab: Vocab) -> bool:
    return vocab.unk_id in vocab.encode(word)


def list_stats(targets, lists_by_n: dict[int, dict], seen: set[str], vocab: Vocab) -> dict:
    """목록 통계(결과를 보기 전에 정해지는 수): 발화·희귀 어절 수, 어휘 밖 음절이 든 희귀 어절, 목록 크기."""
    occurrences = [rare_words(u.text, seen) for u in targets]
    pool = {w for words in occurrences for w in words}
    return {
        "utterances": len(targets),
        "utterances_with_rare_words": sum(1 for words in occurrences if words),
        "rare_words": sum(len(words) for words in occurrences),
        "rare_words_unmatchable": sum(_unmatchable(w, vocab) for words in occurrences for w in words),
        "rare_word_types": len(pool),
        "list_words": {
            str(n): followup.percentiles(len(e["own"]) + len(e["distractors"]) for e in lists.values())
            for n, lists in lists_by_n.items()
        },
        "distractors_min": {
            str(n): min((len(e["distractors"]) for e in lists.values()), default=0)
            for n, lists in lists_by_n.items()
        },
    }


# --- 덤프 재디코딩 -----------------------------------------------------------------------------------------


# 발화 id → 2차 패스가 돈 세그먼트 목록(순서대로): {"greedy", "topk_ids", "topk_logprobs", "blank_logprob"}.
# 덤프가 매니페스트 발화를 순서대로 덮는지, 세그먼트 확정 텍스트가 argmax의 greedy 축약과 같은지(끝점
# 세그먼트는 공백뿐), 발화 확정 가설이 final 세그먼트 텍스트를 이은 것과 같은지 확인한다(어긋나면 덤프가 깨진
# 것이라 SystemExit). 2차 패스가 돌지 않은 세그먼트(넣은 mel 없음, 프레임 0)는 텍스트가 없어 뺀다.
def dump_segments(meta: dict, arrays: dict, vocab: Vocab, utts) -> dict[str, list[dict]]:
    if [row["id"] for row in meta["utterances"]] != [u.id for u in utts]:
        raise SystemExit(f"{meta['set']}: the dump does not cover the manifest utterances in order")
    out: dict[str, list[dict]] = {}
    for row in meta["utterances"]:
        segs, finals = [], []
        for si in row["segments"]:
            seg = meta["segments"][si]
            if seg["emitted"] and seg["kind"] == "final":
                finals.append(seg["text"])
            if seg["frames"] == 0:
                continue
            part = slice(seg["offset"], seg["offset"] + seg["frames"])
            greedy = vocab.decode(
                [tok for tok, _, _ in conf.token_spans(arrays["argmax"][part], vocab.blank_id)]
            )
            if (greedy != seg["text"]) if seg["kind"] == "final" else bool(greedy.strip()):
                raise SystemExit(
                    f"{row['id']} segment {seg['segment']}: argmax does not decode to the final text"
                )
            segs.append(
                {
                    "greedy": greedy,
                    "topk_ids": arrays["topk_ids"][part],
                    "topk_logprobs": arrays["topk_logprobs"][part],
                    "blank_logprob": arrays["blank_logprob"][part],
                }
            )
        if " ".join(finals) != row["hyp_final"]:
            raise SystemExit(f"{row['id']}: the final segments do not join to the final hypothesis")
        out[row["id"]] = segs
    return out


# 발화 하나의 확정 가설: 세그먼트마다 디코딩해 공백이 아닌 텍스트를 공백으로 잇는다(서비스가 텍스트 있는
# 세그먼트만 final로 내는 것과 같다). 편향 목록이 없거나 비면 덤프 greedy 텍스트 그대로(= biased_batch의
# greedy 분기).
def decode_utterance(segments: list[dict], context, beam: int, vocab: Vocab) -> str:
    texts = []
    for s in segments:
        if context is None or context.entries == 0:
            text = s["greedy"]
        else:
            ids = biasing.prefix_beam_search(
                s["topk_ids"],
                s["topk_logprobs"],
                s["blank_logprob"],
                context,
                beam=beam,
                blank=vocab.blank_id,
            )
            text = vocab.decode(ids)
        if text.strip():
            texts.append(text)
    return " ".join(texts)


def decode_set(segments: dict, utt_ids, lists: dict | None, vocab: Vocab, bonus: float, beam: int) -> dict:
    """발화 id → 확정 가설. lists가 None이면 greedy, 아니면 발화마다 자기 목록(own + distractors)의 trie로."""
    out = {}
    for uid in utt_ids:
        context = None
        if lists is not None:
            entry = lists[uid]
            context = biasing.ContextTrie.from_words(entry["own"] + entry["distractors"], vocab, bonus)
        out[uid] = decode_utterance(segments[uid], context, beam, vocab)
    return out


# --- 지표 --------------------------------------------------------------------------------------------------


# 발화별 정수 배열: CER_ns 편집·글자 수(metrics.score_ns), 희귀·나머지 정답 어절 수와 그 가운데 맞은 수(정렬은
# confidence.align_words: 편집 수 최소 가운데 일치 최대; 일치한 정답 어절 = 맞음), 가설에 든 방해 어절 수,
# 어휘 밖 음절이 든 희귀 어절 수.
def _utterance_arrays(utts, hyps: dict, seen: set[str], lists: dict | None, vocab: Vocab) -> dict:
    keys = ("edits", "length", "rare", "rare_hits", "other", "other_hits", "distractors", "unmatchable")
    rows: dict[str, list[int]] = {k: [] for k in keys}
    for u in utts:
        hyp = hyps[u.id]
        scored = score_ns(u.text, hyp)
        ref = normalize_train(u.text).split()
        words = normalize_hyp_ws(hyp).split()
        hit = [False] * len(ref)
        for op, i, _j in conf.align_words(ref, words):
            if op == "equal":
                hit[i] = True
        rare = [w not in seen for w in ref]
        distract = set(lists[u.id]["distractors"]) if lists is not None else set()
        rows["edits"].append(scored.edits if scored else 0)
        rows["length"].append(scored.length if scored else 0)
        rows["rare"].append(sum(rare))
        rows["rare_hits"].append(sum(h and r for h, r in zip(hit, rare, strict=True)))
        rows["other"].append(len(ref) - sum(rare))
        rows["other_hits"].append(sum(h and not r for h, r in zip(hit, rare, strict=True)))
        rows["distractors"].append(sum(w in distract for w in words))
        rows["unmatchable"].append(sum(r and _unmatchable(w, vocab) for w, r in zip(ref, rare, strict=True)))
    return {k: np.asarray(v, dtype=np.int64) for k, v in rows.items()}


# 묶음 비율 '합 ÷ 합'(분모 합이 0이면 None)과 두 값의 차(어느 쪽이 None이면 None). 연기 시험처럼 절반이 비는
# 작은 덤프에서도 나눗셈 오류 없이 None으로 남긴다.
def _rate(num: np.ndarray, den: np.ndarray) -> float | None:
    total = int(den.sum())
    return float(num.sum() / total) if total else None


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


# 짝지은 발화 부트스트랩 95 % 구간(metrics.difference_interval: seed 0, 1,000회; 두 디코딩에 같은 재표본)의
# rate(a) − rate(b). 발화가 없거나 분모 합이 0이면 None.
def _paired(a: np.ndarray, b: np.ndarray, lengths: np.ndarray) -> list[float] | None:
    if lengths.size == 0 or int(lengths.sum()) == 0:
        return None
    return list(difference_interval(a, b, lengths))


# 같은 발화 집합의 greedy 대 편향 확정 가설 비교. CER_ns는 '편집 합 ÷ 글자 합', 희귀 어절 정확 일치율은 '맞은
# 희귀 정답 어절 ÷ 희귀 정답 어절'이고 그 증가의 구간은 희귀 어절이 있는 발화만 다시 뽑는다. 방해 어절 수는
# lists의 그 발화 방해 어절이 가설에 나온 횟수(정답에는 없으므로 모두 잘못 들어온 어절이다).
def compare(utts, greedy: dict, biased: dict, seen: set[str], lists: dict | None, vocab: Vocab) -> dict:
    """greedy와 편향 확정 가설의 CER_ns·희귀 어절 일치율·방해 어절 비교(구간 포함)."""
    g = _utterance_arrays(utts, greedy, seen, lists, vocab)
    b = _utterance_arrays(utts, biased, seen, lists, vocab)
    has_rare = g["rare"] > 0
    no_rare = ~has_rare
    cer_g, cer_b = _rate(g["edits"], g["length"]), _rate(b["edits"], b["length"])
    match_g, match_b = _rate(g["rare_hits"], g["rare"]), _rate(b["rare_hits"], b["rare"])
    return {
        "utterances": len(utts),
        "cer_ns_greedy": cer_g,
        "cer_ns_biased": cer_b,
        "cer_ns_delta": _delta(cer_b, cer_g),
        "cer_ns_delta_ci": _paired(b["edits"], g["edits"], g["length"]),
        "rare_words": int(g["rare"].sum()),
        "rare_words_unmatchable": int(g["unmatchable"].sum()),
        "rare_match_greedy": match_g,
        "rare_match_biased": match_b,
        "rare_match_gain": _delta(match_b, match_g),
        "rare_match_gain_ci": _paired(
            b["rare_hits"][has_rare], g["rare_hits"][has_rare], g["rare"][has_rare]
        ),
        "other_words": int(g["other"].sum()),
        "other_match_greedy": _rate(g["other_hits"], g["other"]),
        "other_match_biased": _rate(b["other_hits"], b["other"]),
        "no_rare_utterances": int(no_rare.sum()),
        "cer_ns_no_rare_greedy": _rate(g["edits"][no_rare], g["length"][no_rare]),
        "cer_ns_no_rare_biased": _rate(b["edits"][no_rare], b["length"][no_rare]),
        "distractor_words_greedy": int(g["distractors"].sum()),
        "distractor_words_biased": int(b["distractors"].sum()),
        "changed_utterances": sum(greedy[u.id] != biased[u.id] for u in utts),
    }


# 절반 A 선택(등록 규칙): A의 CER_ns 증가 ≤ 0.3포인트인 구성 가운데 희귀 어절 정확 일치율이 가장 높은 것,
# 동점은 작은 보너스 → 작은 빔. 가드를 지키는 구성이 없으면 CER 증가가 가장 작은 구성(동점은 같은 순서)을 쓰고
# note를 남긴다(그 경우 판정의 CER 가드가 대개 깨진다).
def select(rows: list[dict]) -> dict:
    eligible = [r for r in rows if r["cer_ns_delta"] is not None and r["cer_ns_delta"] <= CER_GUARD]
    out: dict = {"half": "A", "distractors": JUDGED_DISTRACTORS, "rows": rows}
    if eligible:
        best = max(
            eligible,
            key=lambda r: (
                -1.0 if r["rare_match_biased"] is None else r["rare_match_biased"],
                -r["bonus"],
                -r["beam"],
            ),
        )
    else:
        best = min(
            rows,
            key=lambda r: (
                float("inf") if r["cer_ns_delta"] is None else r["cer_ns_delta"],
                r["bonus"],
                r["beam"],
            ),
        )
        out["note"] = (
            "no configuration met the half A CER guard; the one with the smallest CER increase is used"
        )
    out["chosen"] = {"bonus": best["bonus"], "beam": best["beam"], "key": best["key"]}
    return out


# 지연 결과 파일(--latency) 읽기. 등록 조건과 같은 측정(같은 체크포인트, 100세그먼트, N = 100·seed 0 목록,
# CPU는 1스레드)일 때만 상한과 비교하고, 아니면 comparable False와 사유(met = None)를 남긴다. 파일이 없으면
# measured False.
def read_latency(directory: Path, key: str, checkpoint_sha256: str) -> dict:
    out = {}
    for device, limit in P90_LIMIT_MS.items():
        path = directory / f"biasing_latency-{device}.json"
        if not path.exists():
            out[device] = {"measured": False, "limit_ms": limit, "met": None}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        reasons = []
        if data["provenance"].get("checkpoint_sha256") != checkpoint_sha256:
            reasons.append("checkpoint differs from the dump")
        if data.get("segments") != LATENCY_SEGMENTS:
            reasons.append(f"{data.get('segments')} segments instead of {LATENCY_SEGMENTS}")
        lists = data.get("lists", {})
        if lists.get("distractors") != JUDGED_DISTRACTORS or lists.get("seed") != LIST_SEED:
            reasons.append("bias lists differ from the registration")
        if device == "cpu" and data.get("threads") != 1:
            reasons.append("not a 1-thread CPU run")
        p90 = data["biased_ms"].get(key, {}).get("p90")
        if p90 is None:
            reasons.append(f"no timing for {key}")
        out[device] = {
            "measured": True,
            "file": path.name,
            "p90_ms": p90,
            "limit_ms": limit,
            "greedy_p90_ms": data.get("greedy_ms", {}).get("p90"),
            "idle_before": data.get("load_before", {}).get("idle"),
            "comparable": not reasons,
            "reasons": reasons,
            "met": (p90 <= limit) if not reasons else None,
        }
    return out


# 등록 판정. 정확도 조건(희귀 어절 +10포인트·구간 하한 > 0, CER 가드 셋)이 하나라도 깨지면 채택 아님(False).
# 다 맞으면 지연 조건을 본다: 하나라도 상한을 넘으면 False, 측정이 없거나 비교할 수 없는 측정이 있으면
# None(지연 측정 뒤 판정), 둘 다 상한 안이면 True.
def make_verdict(half_b: dict, mix_all: dict, dev_h: dict, latency: dict) -> dict:
    gain, ci = half_b["rare_match_gain"], half_b["rare_match_gain_ci"]
    rare_met = gain is not None and ci is not None and gain >= MATCH_GAIN and ci[0] > 0
    deltas = (half_b["cer_ns_delta"], mix_all["cer_ns_delta"])
    cer_met = all(d is not None and d <= CER_GUARD for d in deltas)
    h_met = dev_h["cer_ns_delta"] is not None and dev_h["cer_ns_delta"] <= CER_GUARD
    marks = [latency[d]["met"] for d in P90_LIMIT_MS]
    latency_met = (
        False if any(m is False for m in marks) else (None if any(m is None for m in marks) else True)
    )
    accuracy = rare_met and cer_met and h_met
    return {
        "match_gain_threshold": MATCH_GAIN,
        "cer_guard": CER_GUARD,
        "p90_limits_ms": P90_LIMIT_MS,
        "rare_match_met": rare_met,
        "cer_guard_met": cer_met,
        "dev_h_guard_met": h_met,
        "latency": latency,
        "latency_met": latency_met,
        "adopted": False if not accuracy or latency_met is False else latency_met,
        "adopted_means": "offer the biased final pass as an option in a separate commit; server defaults "
        "stay greedy and exposing a context message in the WebSocket protocol is a user decision",
    }


# --- 평가 명령 ---------------------------------------------------------------------------------------------


# 평가 명령. 순서: 결과 파일 확인(있으면 --overwrite 없이 거부) → 덤프 두 세트 읽기(npz 해시·어휘 해시 대조) →
# 재현 조건(맞지 않으면 결과를 쓰지 않고 종료 코드 5) → 목록·greedy 기준 → 절반 A 격자 8구성과 선택 → 절반 B의
# N = 0·100·500과 빔만 → dev_mix 전체(A는 선택 때의 같은 구성 가설) → dev_H → 지연 결과 읽기 → 판정. 재현
# 대조를 할 수 없는 덤프(체크포인트·구성이 P1과 다름, 일부 덤프)나 top-10이 아닌 덤프는 판정 없이(verdict
# None) 쓴다.
def cmd_eval(args) -> None:
    out_path = ROOT / args.out
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    vocab, vocab_sha = ce.load_vocab()
    dump_dir = ROOT / args.dump_dir
    dumps, manifests = {}, {}
    for name in (JUDGED_SET, GUARD_SET):
        meta, arrays = fpd.load_dump(dump_dir, name)
        if meta["vocab_sha256"] != vocab_sha:
            raise SystemExit(f"{name}: the dump was decoded with a different manifests/vocab.json")
        dumps[name] = (meta, arrays)
        # 연기 시험의 일부 덤프(--limit N)는 매니페스트 앞 N발화만 덮는다(재현 대조는 하지 않는다).
        limit = meta["provenance"].get("limit") or 0
        manifest_utts = ce.read_dev_manifest(name)
        manifests[name] = manifest_utts[:limit] if limit else manifest_utts
    repro = ce.check_reproduction({name: meta for name, (meta, _) in dumps.items()}, manifests)
    if repro["applicable"] and not repro["met"]:
        print(json.dumps(repro, ensure_ascii=False), flush=True)
        print(
            "the dump does not reproduce the recorded P1 final hypotheses; not judging (rule change 29)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(ce.REPRO_EXIT)
    # 등록 판정은 덤프가 프레임마다 top-10(어휘가 더 작으면 전부)을 남겼을 때만 한다.
    blocked = [
        f"{name}: the dump keeps top-{arrays['topk_ids'].shape[1]}, the registration uses top-{TOP_K}"
        for name, (_, arrays) in dumps.items()
        if arrays["topk_ids"].shape[1] != min(TOP_K, len(vocab))
    ]
    seen = train_word_set()
    sets = {}
    for name, (meta, arrays) in dumps.items():
        targets = [u for u in manifests[name] if u.hangul_only]
        segs = dump_segments(meta, arrays, vocab, manifests[name])
        lists = {n: build_lists(targets, seen, n) for n in DISTRACTORS}
        greedy = decode_set(segs, [u.id for u in targets], None, vocab, 0.0, 1)
        sets[name] = {"targets": targets, "segs": segs, "lists": lists, "greedy": greedy}
    mix, guard = sets[JUDGED_SET], sets[GUARD_SET]
    half_a = [u for u in mix["targets"] if conf.half_of(u.id) == "A"]
    half_b = [u for u in mix["targets"] if conf.half_of(u.id) == "B"]
    judged_lists = mix["lists"][JUDGED_DISTRACTORS]

    rows, hyps_a = [], {}
    for bonus in BONUSES:
        for beam in BEAMS:
            key = config_key(bonus, beam)
            hyps_a[key] = decode_set(mix["segs"], [u.id for u in half_a], judged_lists, vocab, bonus, beam)
            row = {"bonus": bonus, "beam": beam, "key": key}
            row.update(compare(half_a, mix["greedy"], hyps_a[key], seen, judged_lists, vocab))
            rows.append(row)
            print(
                f"half A {key}: rare match {row['rare_match_greedy']} -> {row['rare_match_biased']}, "
                f"CER_ns delta {row['cer_ns_delta']}",
                flush=True,
            )
    selection = select(rows)
    bonus, beam = selection["chosen"]["bonus"], selection["chosen"]["beam"]

    def run(s: dict, utts, n: int, b: float = bonus) -> tuple[dict, dict]:
        hyps = decode_set(s["segs"], [u.id for u in utts], s["lists"][n], vocab, b, beam)
        return hyps, compare(utts, s["greedy"], hyps, seen, s["lists"][n], vocab)

    half_b_report, hyps_b = {}, {}
    for n in DISTRACTORS:
        hyps_b[n], half_b_report[f"distractors_{n}"] = run(mix, half_b, n)
    half_b_report["beam_only"] = run(mix, half_b, JUDGED_DISTRACTORS, 0.0)[1]
    all_hyps = {**hyps_a[selection["chosen"]["key"]], **hyps_b[JUDGED_DISTRACTORS]}
    mix_all = compare(mix["targets"], mix["greedy"], all_hyps, seen, judged_lists, vocab)
    dev_h = {f"distractors_{n}": run(guard, guard["targets"], n)[1] for n in DISTRACTORS}
    checkpoint = dumps[JUDGED_SET][0]["provenance"]["checkpoint_sha256"]
    latency = read_latency(ROOT / args.latency_dir, selection["chosen"]["key"], checkpoint)
    judged = half_b_report[f"distractors_{JUDGED_DISTRACTORS}"]
    verdict = None
    if repro["applicable"] and not blocked:
        verdict = make_verdict(judged, mix_all, dev_h[f"distractors_{JUDGED_DISTRACTORS}"], latency)
    result = {
        "rule": "docs/experiments.md rule change 29",
        "evaluated": {**ce.git_state(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "dumps": {
            name: {
                "npz_sha256": meta["npz_sha256"],
                "provenance": meta["provenance"],
                "config": meta["config"],
                "graph": meta.get("graph"),
                "frames": meta["frames"],
                "segments": len(meta["segments"]),
                "utterances": len(meta["utterances"]),
                "top_k": int(arrays["topk_ids"].shape[1]),
            }
            for name, (meta, arrays) in dumps.items()
        },
        "reproduction": repro,
        "verdict_blocked": blocked,
        "subset": "hangul-only utterances",
        "rare_words": "reference words (training normalization, split on spaces) absent from the word set "
        "of manifests/train.jsonl",
        "lists": {
            "seed": LIST_SEED,
            "distractors": list(DISTRACTORS),
            "rule": "own rare words + N distractors drawn from the other utterances' rare words of the "
            "same set (seed 0 Generator, one permutation of the pool per utterance)",
            "sets": {name: list_stats(s["targets"], s["lists"], seen, vocab) for name, s in sets.items()},
        },
        "grid": {"bonus": list(BONUSES), "beam": list(BEAMS), "top_k": TOP_K},
        "split": {
            "rule": "SHA-256 of the utterance id (UTF-8): first byte even = A (selection), odd = B (verdict)",
            "half_a_utterances": len(half_a),
            "half_b_utterances": len(half_b),
        },
        "selection": selection,
        "dev_mix_half_b": half_b_report,
        "dev_mix_all": mix_all,
        "dev_h": dev_h,
        "note": "lists are built from the references (own rare words + distractors), as in contextual "
        "biasing evaluations; the numbers are optimistic compared with real user lists",
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print(
        f"chosen {selection['chosen']['key']}; half B rare match {judged['rare_match_greedy']} -> "
        f"{judged['rare_match_biased']} (gain CI {judged['rare_match_gain_ci']}), CER_ns delta "
        f"{judged['cer_ns_delta']}; verdict {None if verdict is None else verdict['adopted']}",
        flush=True,
    )


# --- 지연 명령 ---------------------------------------------------------------------------------------------


# P1 세션 조건으로 대상 발화를 차례로 흘려(덤프 세션이 세그먼트 mel을 남김) 2차 패스가 돈 세그먼트를
# 앞에서부터 limit개 모은다. 잡음 seed는 매니페스트 전체에서의 순번(scripts/final_pass_dump.dump_set과 같음).
def collect_segments(model, vocab, targets, store, session_args: dict, device, tail_s: float, limit: int):
    logmel = LogMel().to(device)
    out: list[dict] = []
    for k, u in targets:
        wav = torch.cat([store.waveform(u.id), noise(tail_s, seed=k)]).to(device)
        session = fpd.DumpSession(model, vocab, logmel=logmel, keep_mel=True, **session_args)
        for _pos, piece in packets(wav):
            session.push_samples(piece)
        session.flush()
        for rec in session.closed:
            cap = rec["capture"]
            if cap is None:
                continue
            out.append(
                {
                    "utt": u.id,
                    "text": rec["text"],
                    "mel": cap["mel"],
                    "mel_frames": cap["mel_frames"],
                    "frames": cap["frames"],
                }
            )
            if len(out) >= limit:
                return out
    return out


# StreamSession.decode_full과 같은 준비: 길이를 final_pass_bucket 배수로 0을 덧붙인다(실제 길이는 따로 넘김).
def _padded(session: StreamSession, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
    t = mel.shape[0]
    bucket = session.final_pass_bucket
    padded = -(-t // bucket) * bucket if bucket > 0 else t
    if padded > t:
        mel = torch.cat([mel, mel.new_zeros(padded - t, mel.shape[1])], dim=0)
    return mel, t


# GPU 비동기 실행을 기다린다(decode_full과 같게 시간 재기 앞뒤에서). CPU에서는 할 일이 없다.
def _sync(session: StreamSession) -> None:
    if session.device.type == "cuda":
        torch.cuda.synchronize()


def _forward(session: StreamSession, mel: torch.Tensor, t: int):
    # 세션 정밀도(GPU fp16 autocast)와 전체 문맥 모드는 decode_full과 같다.
    with session._autocast():
        frames = torch.tensor([t], device=session.device)
        return session.model(mel[None], frames, ChunkSpec(0, -1))


# 편향 2차 패스 한 번: decode_full과 같은 시간 범위(앞뒤 GPU 동기화 사이의 forward + 디코딩, 결과 id를 CPU로
# 가져오는 것까지)에서 greedy 대신 biased_batch(log_softmax → top-k → CPU 빔)를 쓴다. 출력: (텍스트, ms).
@torch.no_grad()
def biased_pass(session: StreamSession, mel: torch.Tensor, context, beam: int) -> tuple[str, float]:
    mel, t = _padded(session, mel)
    _sync(session)
    t0 = time.perf_counter()
    logits, _, lengths = _forward(session, mel, t)
    ids = biased_batch(logits, lengths, context, beam=beam, top_k=TOP_K, blank=session.vocab.blank_id)[0]
    _sync(session)
    return session.vocab.decode(ids), (time.perf_counter() - t0) * 1000


# 빔만의 CPU 시간을 따로 재려고(보고용) 같은 로짓에서 덤프와 같은 top-k·blank 배열을 만든다(시간에 넣지 않음).
@torch.no_grad()
def candidate_arrays(session: StreamSession, mel: torch.Tensor) -> dict[str, np.ndarray]:
    mel, t = _padded(session, mel)
    logits, _, lengths = _forward(session, mel, t)
    return fpd.frame_stats(logits[0, : int(lengths[0])], TOP_K)


# 지연 명령. 장치(cuda·cpu)·인자와 결과 파일(biasing_latency-<장치>.json, 있으면 --overwrite 없이 거부)을
# 무거운 일 전에 확인하고, 측정이 끝나거나 실패해도 torch 스레드 수를 원래대로 되돌린다(CPU 1스레드 측정이
# 같은 프로세스의 다른 계산을 느리게 하지 않게).
def cmd_latency(args) -> None:
    device = torch.device(args.device)
    if device.type not in P90_LIMIT_MS:
        raise SystemExit(f"--device must be cuda or cpu, got {args.device!r}")
    if args.limit < 1 or args.threads < 1:
        raise SystemExit("--limit and --threads must be at least 1")
    out_path = ROOT / args.latency_dir / f"biasing_latency-{device.type}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    threads = torch.get_num_threads()
    try:
        _measure_latency(args, device, out_path)
    finally:
        torch.set_num_threads(threads)  # 같은 프로세스의 다른 계산(테스트 등)에 1스레드가 남지 않게


# 지연 측정 본체. 순서: 모델·어휘 검사 → 서버 기본 구성(scripts/final_pass_dump.session_config) → 한글 전용
# 대상과 N = 100 목록(평가와 같은 정의) → 세그먼트 mel 수집(CPU는 이때 기본 스레드 수, 시간에 넣지 않음) →
# CPU면 스레드 수 고정 → 쉬는 상태 대기 → 같은 구성 세션 예열 → 세그먼트마다 greedy 1번, 8구성 편향 각 1번(+
# 빔만 CPU 시간) → 결과. greedy 텍스트가 수집 때 덤프 세션이 낸 확정 텍스트와 같은 세그먼트 수도 적는다(같은
# mel·같은 경로 확인).
def _measure_latency(args, device: torch.device, out_path: Path) -> None:
    model, _ = load_model(followup.ROOT / args.checkpoint, device)
    vocab = Vocab.load(followup.ROOT / "manifests" / "vocab.json")
    check_vocab(model, vocab)  # 어휘 토큰 수 ≠ 체크포인트 출력 크기면 측정 전에 멈춘다
    session_args = fpd.session_config(device)
    utts, store = followup.manifest(JUDGED_SET, 0)
    targets = [(k, u) for k, u in enumerate(utts) if u.hangul_only]
    lists = build_lists([u for _, u in targets], train_word_set(), JUDGED_DISTRACTORS)
    segments = collect_segments(model, vocab, targets, store, session_args, device, args.tail_s, args.limit)
    if not segments:
        raise SystemExit("no segment ran the final pass; nothing to time")
    tries = {
        (seg["utt"], b): biasing.ContextTrie.from_words(
            lists[seg["utt"]]["own"] + lists[seg["utt"]]["distractors"], vocab, b
        )
        for seg in segments
        for b in BONUSES
    }
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    load_before = followup.wait_for_idle(args.wait_idle_min)
    session = StreamSession(model, vocab, logmel=LogMel().to(device), **session_args)
    warm_s = warm_up(session)
    # 편향 경로의 첫 호출 준비(log_softmax·topk 커널 등)를 시간 밖에서 한 번 치른다.
    biased_pass(session, segments[0]["mel"].to(device), tries[(segments[0]["utt"], BONUSES[0])], BEAMS[-1])
    keys = [config_key(b, k) for b in BONUSES for k in BEAMS]
    greedy_ms: list[float] = []
    biased_ms: dict[str, list[float]] = {key: [] for key in keys}
    beam_ms: dict[str, list[float]] = {key: [] for key in keys}
    matches = 0
    for seg in segments:
        mel = seg["mel"].to(device)
        text, ms = session.decode_full(mel)
        greedy_ms.append(ms)
        matches += int(text == seg["text"] or (not text.strip() and not seg["text"]))
        stats = candidate_arrays(session, mel)
        for b in BONUSES:
            trie = tries[(seg["utt"], b)]
            for k in BEAMS:
                key = config_key(b, k)
                biased_ms[key].append(biased_pass(session, mel, trie, k)[1])
                t0 = time.perf_counter()
                biasing.prefix_beam_search(
                    stats["topk_ids"], stats["topk_logprobs"], stats["blank_logprob"], trie, beam=k
                )
                beam_ms[key].append((time.perf_counter() - t0) * 1000)
    spec, endpoint = session_args["spec"], session_args["endpoint"]
    result = {
        "rule": "docs/experiments.md rule change 29",
        "mode": "latency",
        "provenance": followup.provenance(args, device),
        "threads": torch.get_num_threads(),
        "config": {
            "chunk": spec.chunk_size,
            "left": spec.left_context,
            **asdict(endpoint),
            "reset_cache_on_endpoint": session_args["reset_cache_on_endpoint"],
            "step_path": session_args["step_path"],
            "final_pass": session_args["final_pass"],
            "final_pass_bucket": session.final_pass_bucket,
            "precision": "fp16-autocast" if session.amp else "fp32",
            "tail_s": args.tail_s,
        },
        "lists": {
            "set": JUDGED_SET,
            "subset": "hangul-only utterances",
            "distractors": JUDGED_DISTRACTORS,
            "seed": LIST_SEED,
            "words_per_list": followup.percentiles(
                len(lists[seg["utt"]]["own"]) + len(lists[seg["utt"]]["distractors"]) for seg in segments
            ),
        },
        "grid": {"bonus": list(BONUSES), "beam": list(BEAMS), "top_k": TOP_K},
        "load_before": load_before,
        "warm_up_s": warm_s,
        "segments": len(segments),
        "mel_frames": followup.percentiles(seg["mel_frames"] for seg in segments),
        "frames": followup.percentiles(seg["frames"] for seg in segments),
        "greedy_ms": followup.percentiles(greedy_ms),
        "biased_ms": {key: followup.percentiles(v) for key, v in biased_ms.items()},
        "beam_cpu_ms": {key: followup.percentiles(v) for key, v in beam_ms.items()},
        "greedy_matches_session": matches,
        "load_after": followup.load_snapshot(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    followup.write_json(out_path, result)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="확정 자막 핫워드 문맥 편향 평가(docs/experiments.md 규칙 변경 29). 기본은 2차 패스 "
        "덤프를 편향 디코딩으로 다시 풀어 dev_mix 한글 전용 절반 A에서 보너스·빔을 고르고 절반 B에서 "
        "판정한다(CPU). --latency는 최종 모델로 dev_mix 앞 세그먼트의 greedy·편향 2차 패스 시간을 잰다(GPU "
        "또는 CPU 1스레드). 봉인 test 세트는 열지 않는다.",
        epilog="측정 순서: --latency --device cuda → --latency --device cpu --threads 1 → 평가. 평가는 지연 "
        "결과가 없으면 지연 조건을 측정 안 됨으로 두고 채택을 정하지 않는다.",
    )
    p.add_argument(
        "--dump-dir",
        default="outputs/followup/final_pass_dump",
        help="scripts/final_pass_dump.py 결과 폴더(저장소 기준; dev_mix·dev_h 덤프가 모두 있어야 함)",
    )
    p.add_argument(
        "--out",
        default="reports/followup/biasing.json",
        help="평가 결과 JSON(저장소 기준). 연기 시험은 work/ 아래로",
    )
    p.add_argument(
        "--latency-dir",
        default="reports/followup",
        help="지연 결과 폴더(저장소 기준): --latency가 biasing_latency-<cuda|cpu>.json을 쓰고 평가가 읽는다",
    )
    p.add_argument("--latency", action="store_true", help="평가 대신 2차 패스 지연을 잰다(최종 모델 필요)")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="지연 측정 장치(cuda 또는 cpu; 평가는 늘 CPU)",
    )
    p.add_argument(
        "--threads", type=int, default=1, help="CPU 지연 측정의 torch 스레드 수(등록 1; GPU에서는 무시)"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=LATENCY_SEGMENTS,
        help="지연을 잴 앞 세그먼트 수(등록 100; 연기 시험은 작게, 그 결과는 판정에 쓰지 않음)",
    )
    p.add_argument(
        "--checkpoint", default="outputs/stage4/trimtail/avg10.pt", help="지연 측정 모델(최종 모델)"
    )
    p.add_argument(
        "--tail-s", type=float, default=3.0, help="발화 뒤에 붙이는 -60 dBFS 잡음 길이(초, P1과 같은 3)"
    )
    p.add_argument(
        "--wait-idle-min",
        type=float,
        default=10.0,
        help="지연 측정 전 쉬는 상태(GPU ≤ 10 %%, CPU ≤ 25 %%)를 기다리는 최대 분(넘으면 부하 조건으로 기록)",
    )
    p.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어쓴다")
    return p


# 진입점: 표준 출력을 UTF-8로(Windows 콘솔 기본 cp949에서 한글 경로·메시지가 깨지지 않게) 바꾸고 --latency면
# 지연 측정, 아니면 평가를 한다.
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.latency:
        cmd_latency(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main(sys.argv[1:])
