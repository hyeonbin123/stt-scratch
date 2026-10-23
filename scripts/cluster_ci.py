"""저장된 가설로 README 결과 표 행과 P1 비교의 군집 부트스트랩 보조 구간을 계산한다(사후 보조, 새 디코딩 없음,
판정 아님).

봉인 test 표 8행(4세트 × full·320 ms 스트리밍, 한글 전용 CER_ns)과 README가 인용하는 7단계 뒤 P1 비교(끝점 뒤
2차 패스 − 1-pass, dev_mix·dev_H, rule2 0.8 s)에 대해, 등록 구간(발화 단위)과 함께 같은 문장·화자의 발화를
한 묶음으로 다시 뽑은 구간을 낸다. 등록 구간·기록 수치·판정은 바꾸지 않는다.

    uv run python scripts/cluster_ci.py   # 결과: reports/final/cluster_ci.json
"""

# [왜 필요한가] 등록 규칙의 구간은 발화를 서로 독립으로 다시 뽑는다(docs/experiments.md "지표"). 그런데 평가
# 세트는 묶음 구조가 강하다: test_H는 1,280발화가 167문장, Zeroth 공식 test는 화자 10명, CV22 test는 화자
# 74명, dev_mix 한글 전용 글자의 61 %는 CV22 화자 7명의 것이다. 그래서 발화 단위 구간은 '같은
# 문장·화자에서 발화만 다시 뽑을 때'의 변동이고, 다른 문장·화자로 일반화할 때의 불확실성은 군집 구간이 더 잘
# 보여 준다. 짝지은 비교는 두 시스템에 같은 군집 표본을 쓰므로 군집 난이도가 상쇄된다(docs/experiments.md
# 7단계 결과의 사후 보조 구간).
# [군집 키(고정, 결과를 본 뒤 바꾸지 않는다)] Zeroth(test_zeroth·test_h·dev_h) = 채점 정규화한 정답
# 문장(normalize_score)과 화자(speaker) 두 가지, CV22 = 화자(client_id), FLEURS = id의 문장 부분
# (fleurs/<분할>/<문장>/<n>의 <문장>; FLEURS에는 화자 정보가 없다), dev_mix = CV22 발화는 화자·FLEURS 발화는
# 문장으로 섞은 한 가지.
# [순서] 발화는 등록 보고서와 같은 순서(봉인 표는 evaluate.score처럼 매니페스트 순서, P1 비교는
# scripts/compare_runs.py처럼 id 정렬 순서), 군집은 그 순서에서 처음 나온 순서다. 그래서 '군집 = 발화' 열은
# 기록된 구간과 같아야 하고, 다르면 입력(가설·매니페스트·채점)이 기록과 다르다는 뜻이라 멈춘다. 군집이 적으면
# 1,000회 백분위 끝값의 재표본 오차가 커서 순서가 바뀌면 소수 둘째 자리(%)가 달라질 수 있다.
# [연결] 입력: manifests/*.jsonl, reports/final/test_*-{full,streaming}.hyps.jsonl·.json(scripts/stage7.py),
# reports/followup/hyps/*-session-rule2_0.8-{final_pass,stream}.jsonl·reports/followup/session.json
# (scripts/followup.py). 출력: reports/final/cluster_ci.json. 테스트: tests/test_cluster_bootstrap.py.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# numpy: 발화별 편집·글자 수 배열. 기록된 구간과의 대조에 허용 오차 비교(np.allclose)를 쓴다.
import numpy as np

# 프로젝트 내부 모듈: 매니페스트 읽기, 평가와 같은 CER_ns 채점(score_ns)·묶음 비율(error_rate), 부트스트랩
# 상수와 군집 구간 함수(군집 = 발화이면 등록 함수와 비트 단위로 같다), 채점 정규화(문장 군집 키).
from stt_scratch.data import Utterance, read_manifest
from stt_scratch.metrics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    cluster_difference_interval,
    cluster_interval,
    error_rate,
    score_ns,
)
from stt_scratch.text import normalize_score

# 저장소 루트. 입력·출력 경로를 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]
# README 결과 표의 행: 봉인 test 4세트 × 두 디코딩 모드(7단계 결과 파일 이름과 같다).
SEALED_SETS = ("test_zeroth", "test_h", "test_fleurs", "test_cv")
MODES = ("full", "streaming")
# 세트별 군집 종류(위 [군집 키]). 모든 행에 등록 구간과 같은 'utterance'를 함께 낸다.
CLUSTER_KINDS = {
    "test_zeroth": ("sentence", "speaker"),
    "test_h": ("sentence", "speaker"),
    "test_fleurs": ("sentence",),
    "test_cv": ("speaker",),
    "dev_mix": ("speaker_or_sentence",),
    "dev_h": ("sentence", "speaker"),
}
# 결과 파일에 남기는 군집 키 설명(결과를 읽는 사람이 정의를 코드 없이 알 수 있게).
KIND_NOTES = {
    "utterance": "each utterance is its own cluster (the registered bootstrap)",
    "sentence": "Zeroth: normalize_score(reference); FLEURS: <sentence> of fleurs/<split>/<sentence>/<n>",
    "speaker": "Zeroth: speaker id; CV22: client_id",
    "speaker_or_sentence": "dev_mix: CV22 utterances by client_id, FLEURS utterances by sentence id",
}
# README가 인용하는 P1 비교(a − b = 끝점 뒤 2차 패스 − 1-pass 스트림). 기록 위치는
# reports/followup/session.json의 runs.<세트>_rule2_0.8.final_pass_vs_stream.
PAIRED = {
    "p1_dev_mix_rule2_0.8": (
        "dev_mix",
        "reports/followup/hyps/dev_mix-session-rule2_0.8-final_pass.jsonl",
        "reports/followup/hyps/dev_mix-session-rule2_0.8-stream.jsonl",
    ),
    "p1_dev_h_rule2_0.8": (
        "dev_h",
        "reports/followup/hyps/dev_h-session-rule2_0.8-final_pass.jsonl",
        "reports/followup/hyps/dev_h-session-rule2_0.8-stream.jsonl",
    ),
}


# 발화 하나의 군집 키. 말뭉치 접두를 붙여 서로 다른 말뭉치의 키가 우연히 같아지지 않게 한다.
# 주의: 정의되지 않은 조합(예: FLEURS 화자)이나 빈 화자는 조용히 한 군집으로 묶이지 않도록 ValueError.
def cluster_key(utt: Utterance, kind: str) -> str:
    if kind == "utterance":
        return utt.id
    if utt.corpus == "zeroth" and kind == "sentence":
        return "zeroth-sentence:" + normalize_score(utt.text)
    if utt.corpus == "zeroth" and kind == "speaker" and utt.speaker:
        return "zeroth-speaker:" + utt.speaker
    if utt.corpus == "fleurs" and kind in ("sentence", "speaker_or_sentence"):
        parts = utt.id.split("/")
        if len(parts) == 4:
            return "fleurs-sentence:" + parts[2]
    if utt.corpus == "cv22" and kind in ("speaker", "speaker_or_sentence") and utt.speaker:
        return "cv22-speaker:" + utt.speaker
    raise ValueError(f"no {kind} cluster key for {utt.id}")


# 가설 파일(JSONL)을 발화 id → 행 사전으로 읽는다.
def _load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


# 발화 단위 구간이 기록된 값과 같은지 확인한다. 같은 가설·같은 채점·같은 순서면 같은 연산이라 비트 단위로
# 같아야 하지만, 플랫폼 사이의 부동소수점 차이에 대비해 1e-12까지 허용한다. 다르면 멈춘다.
def _check_recorded(label: str, n: int, ci: list[float], recorded_n: int, recorded_ci: list[float]) -> None:
    if n != recorded_n or not np.allclose(ci, recorded_ci, rtol=0.0, atol=1e-12):
        raise SystemExit(
            f"{label}: utterance-level interval {ci} (n {n}) differs from the record {recorded_ci}"
        )


# 봉인 표 한 행: 매니페스트 순서로 한글 전용 발화를 채점하고(evaluate.score와 같은 순서·같은 거름), 등록
# 구간과 군집 구간을 계산한 뒤 7단계 결과 파일의 기록과 대조한다.
def sealed_row(name: str, mode: str) -> dict:
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    hyps_rel = f"reports/final/{name}-{mode}.hyps.jsonl"
    hyps = _load(ROOT / hyps_rel)
    if set(hyps) != {u.id for u in utts}:
        raise SystemExit(f"{hyps_rel} does not cover the {name} manifest")
    edits, lengths, kept = [], [], []
    for u in utts:
        if hyps[u.id]["ref"] != u.text:
            raise SystemExit(f"{hyps_rel}: reference of {u.id} differs from the manifest")
        if not u.hangul_only:
            continue
        s = score_ns(u.text, hyps[u.id]["hyp"])
        if s is None:
            continue
        edits.append(s.edits)
        lengths.append(s.length)
        kept.append(u)
    e, ln = (np.array(x, dtype=np.int64) for x in (edits, lengths))
    clusters, ci = {}, {}
    for kind in ("utterance", *CLUSTER_KINDS[name]):
        keys = [cluster_key(u, kind) for u in kept]
        clusters[kind] = len(set(keys))
        ci[kind] = list(cluster_interval(e, ln, keys))
    recorded = json.loads((ROOT / "reports" / "final" / f"{name}-{mode}.json").read_text(encoding="utf-8"))
    rec = recorded["hangul_only"]["cer_ns"]
    _check_recorded(f"{name}-{mode}", len(kept), ci["utterance"], rec["n"], rec["ci"])
    return {
        "hypotheses": hyps_rel,
        "subset": "hangul_only",
        "n": len(kept),
        "rate": error_rate(e, ln),
        "edits": int(e.sum()),
        "length": int(ln.sum()),
        "clusters": clusters,
        "ci95": ci,
    }


# P1 비교 한 개: scripts/compare_runs.py의 compare와 같은 순서(id 정렬)·같은 거름(가설 파일의 hangul_only, 빈
# 정답 제외)으로 짝을 만들고, 짝 차의 등록 구간과 군집 구간을 계산한 뒤 reports/followup/session.json의 기록과
# 대조한다.
def paired_row(key: str) -> dict:
    name, path_a, path_b = PAIRED[key]
    utts = {u.id: u for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl")}
    a, b = _load(ROOT / path_a), _load(ROOT / path_b)
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b) or not set(ids) <= set(utts):
        raise SystemExit(f"{key}: the two files cover different utterances")
    edits_a, edits_b, lengths, kept = [], [], [], []
    for i in ids:
        if a[i]["ref"] != b[i]["ref"]:
            raise SystemExit(f"{key}: reference mismatch for {i}")
        if not a[i]["hangul_only"]:
            continue
        sa, sb = score_ns(a[i]["ref"], a[i]["hyp"]), score_ns(b[i]["ref"], b[i]["hyp"])
        if sa is None:
            continue
        edits_a.append(sa.edits)
        edits_b.append(sb.edits)
        lengths.append(sa.length)
        kept.append(utts[i])
    ea, eb, ln = (np.array(x, dtype=np.int64) for x in (edits_a, edits_b, lengths))
    clusters, ci = {}, {}
    for kind in ("utterance", *CLUSTER_KINDS[name]):
        keys = [cluster_key(u, kind) for u in kept]
        clusters[kind] = len(set(keys))
        ci[kind] = list(cluster_difference_interval(ea, eb, ln, keys))
    session = json.loads((ROOT / "reports" / "followup" / "session.json").read_text(encoding="utf-8"))
    rec = session["runs"][f"{name}_rule2_0.8"]["final_pass_vs_stream"]
    _check_recorded(key, len(kept), ci["utterance"], rec["n"], rec["diff_ci95"])
    rate_a, rate_b = error_rate(ea, ln), error_rate(eb, ln)
    return {
        "set": name,
        "a": path_a,
        "b": path_b,
        "a_minus_b": "final pass after the endpoint - 1-pass stream",
        "subset": "hangul_only",
        "n": len(kept),
        "rate_a": rate_a,
        "rate_b": rate_b,
        "diff_a_minus_b": rate_a - rate_b,
        "clusters": clusters,
        "ci95": ci,
    }


# 보고서 전체(dict). 출력 파일과 테스트가 같은 함수를 쓴다.
def compute() -> dict:
    return {
        "note": (
            "post-hoc supplementary intervals from saved hypotheses; no new decoding; the registered "
            "utterance-level intervals, recorded numbers and verdicts are unchanged"
        ),
        "metric": "CER_ns (score_ns) on hangul-only utterances",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": 0.95,
            "method": (
                "percentile; clusters drawn with replacement using cluster totals of edits and reference "
                "characters; utterances in the order of the registered report, clusters in order of first "
                "appearance"
            ),
            "check": (
                "the utterance column equals the intervals recorded in reports/final/test_*.json and "
                "reports/followup/session.json"
            ),
        },
        "cluster_keys": KIND_NOTES,
        "sealed": {name: {mode: sealed_row(name, mode) for mode in MODES} for name in SEALED_SETS},
        "paired": {key: paired_row(key) for key in PAIRED},
    }


# 명령행: 보고서를 계산해 저장하고(UTF-8, LF), 행마다 비율과 군집별 구간(%)을 한 줄로 찍는다.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "저장된 가설로 README 봉인 표 행과 P1 비교의 문장·화자 군집 부트스트랩 보조 구간을 계산한다(사후 "
            "보조, 새 디코딩 없음, 판정 아님). 발화 단위 열은 기록된 등록 구간과 같아야 하며 다르면 멈춘다."
        )
    )
    p.add_argument(
        "--out",
        default="reports/final/cluster_ci.json",
        help="결과 JSON 경로(저장소 기준, 기본 reports/final/cluster_ci.json)",
    )
    args = p.parse_args(argv)
    report = compute()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    def fmt(ci: dict) -> str:
        return " | ".join(f"{k} {lo * 100:+.2f}~{hi * 100:+.2f}" for k, (lo, hi) in ci.items())

    for name, modes in report["sealed"].items():
        for mode, row in modes.items():
            print(f"{name} {mode}: n {row['n']} rate {row['rate'] * 100:.2f} | {fmt(row['ci95'])}")
    for key, row in report["paired"].items():
        print(f"{key}: n {row['n']} diff {row['diff_a_minus_b'] * 100:+.2f} | {fmt(row['ci95'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
