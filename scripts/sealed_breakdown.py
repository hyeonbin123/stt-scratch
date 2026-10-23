"""봉인된 Zeroth 공식 test의 사후 분석(새 디코딩 없음, 판정 아님).

7단계에서 저장한 test_zeroth 발화별 가설만 읽어 두 가지를 계산한다. 새로 디코딩하지 않으므로 봉인 test를 다시
재는 것이 아니고, 이 분석으로 모델·규칙을 고르지 않는다.

1. 기본(인자 없음): 정답 문장(채점 정규화 기준)이 우리 학습 매니페스트에도 있는가로 두 집단을 나누고, 두 집단
   각각의 CER_ns와 부트스트랩 구간을 낸다 → reports/final/zeroth_seen_unseen.json
2. --anchor-report와 --digit-list를 함께 주면: whisper-ko-ft의 발화별 turbo 제로샷 보고서와 같은 발화로 짝지어
   같은 score_ns로 다시 채점하고, whisper 가설에 아라비아 숫자가 든 발화를 뺀 부분집합에서도 비교한다
   → reports/final/zeroth_anchor_notation.json

    uv run python scripts/sealed_breakdown.py
    uv run python scripts/sealed_breakdown.py \
        --anchor-report ../whisper-ko-ft/reports/whisper-large-v3-turbo/zeroth-test.json \
        --digit-list ../whisper-ko-ft/reports/digit_utterances/zeroth-test.json
"""

# [왜 필요한가 — 학습 문장 분할] Zeroth 공식 test 3.67 %는 whisper 앵커 사이(H7.1 충족)지만, 공식 test 문장
# 대부분이 학습 세트에도 있다(docs/data.md). 이 분할로 학습 세트에 있는 문장 397발화 1.85 %, 없는 문장 60발화
# 16.01 %(스트리밍 3.31 % / 19.25 %)를 보여, H7.1 충족이 문장 암기와 일관된다는 해석의 근거로
# 썼다(docs/experiments.md 7단계 결과, README 결과 표의 사후 분할 행). 문장 난도 차이도 섞여 있을 수 있어
# 암기의 기여 비율은 분리하지 않았다.
# [왜 필요한가 — 앵커 표기] 채점 정규화(normalize_score)는 whisper-ko-ft와 같지만 숫자를 남긴다. Zeroth 정답은
# 숫자를 한글로 풀어 쓰고(docs/data.md "표기 차이") 이 모델 어휘에는 숫자가 없는데 whisper는 아라비아 숫자를
# 내므로, '3'과 '삼' 같은 표기 차이가 앵커(turbo 제로샷 4.48 %) 쪽 오류로 센다. 주 분석은 whisper-ko-ft가 자기
# 1단계에 고정한 숫자 발화 목록(whisper-small 또는 turbo 가설에 [0-9]가 든 발화)을 뺀 부분집합, 민감도는 turbo
# 보고서의 발화별 has_digit(turbo 자신이 숫자를 낸 발화)만 뺀 부분집합이다. 숫자 목록은 whisper 가설로 정해진
# 것이라(정답에는 숫자가 없다) 결과에 그 정의를 함께 남긴다(docs/experiments.md 7단계 결과의 사후 분석).
# [결합] 이 저장소의 Zeroth id는 parquet 행 번호(zeroth/test/<n>, scripts/prepare_data.py의 iter_zeroth)이고
# whisper-ko-ft는 원래 id(예: 105_003_0029)를 쓴다. 0단계가 내려받은 parquet의 n번째 행 id 열로 잇되, 그 행의
# 전사·화자가 매니페스트와 같은지, 결합이 일대일인지, 짝마다 정규화 정답이 같은지 확인한 뒤에만 계산한다. 잘못
# 이은 짝은 그럴듯하지만 틀린 수치를 내므로 하나라도 어긋나면 멈춘다.
# [출처] 다른 저장소의 정답·가설 문자열은 결과에 복사하지 않고, 수치와 출처(저장소 이름, 그 저장소 안의 경로,
# 그 파일을 마지막으로 바꾼 커밋, 파일 SHA-256)만 남긴다. whisper-ko-ft의 '맞춘 채점'(그 저장소의 숫자 표기
# 변환 코드)은 다른 저장소 코드를 불러와야 재현되므로 여기서 하지 않는다.
# [순서] 짝지은 발화는 두 저장소가 공유하는 원래 id 순으로 정렬해 재표본한다. 1,000회 백분위 구간의 끝값은
# 발화 순서에 따라 소수 둘째 자리(%)가 달라질 수 있어(재표본 오차) 순서를 고정해야 같은 값이 재현된다.
# [연결] 입력: scripts/stage7.py가 쓴 reports/final/test_zeroth-{full,streaming}.hyps.jsonl,
# manifests/train.jsonl, manifests/test_zeroth.jsonl, (앵커 분석만)
# data/raw/zeroth_korean/data/test-*.parquet와 whisper-ko-ft의 두 보고서.
# 출력: reports/final/zeroth_seen_unseen.json, reports/final/zeroth_anchor_notation.json.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# numpy: 발화별 편집·글자 수를 배열로 모아 부트스트랩 함수에 넘긴다. pyarrow.parquet: 원본 parquet에서
# id·화자·전사 열만 읽는다(오디오 열은 읽지 않는다).
import numpy as np
import pyarrow.parquet as pq

# 프로젝트 내부 모듈: 매니페스트 읽기, 평가와 같은 CER_ns 채점(score_ns)·요약(summarize)·부트스트랩 구간
# (발화 단위 interval·difference_interval, 화자 군집 cluster_difference_interval), 채점·학습 정규화. 봉인
# test 보고서와 같은 함수로 계산해야 부분집합의 합이 보고된 전체 수치와 맞는다.
from stt_scratch.data import Utterance, read_manifest
from stt_scratch.metrics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    cluster_difference_interval,
    difference_interval,
    error_rate,
    interval,
    score_ns,
    summarize,
)
from stt_scratch.text import normalize_score, normalize_train

# 저장소 루트. 매니페스트와 reports/final/을 이 기준으로 찾고, 명령행의 상대 경로도 이 기준으로 푼다.
ROOT = Path(__file__).resolve().parents[1]
# 0단계(scripts/download_data.py)가 Zeroth-Korean parquet를 내려받는 곳(저장소 기준).
DEFAULT_PARQUET_DIR = "data/raw/zeroth_korean/data"
# 앵커 비교에 쓰는 봉인 가설(full = 전체 문맥 greedy; whisper 앵커도 전체 문맥 디코딩이다).
FULL_HYPS = "reports/final/test_zeroth-full.hyps.jsonl"
# whisper-ko-ft의 has_digit 정의(가설에 아라비아 숫자가 하나라도 있음)와 같은 검사. 보고서의 표시가 가설
# 내용과 맞는지 확인하는 데 쓴다.
_DIGIT = re.compile(r"[0-9]")


# 학습 매니페스트(Zeroth train − 홀드아웃 H + FLEURS + CV)의 문장 집합. 채점 정규화(공백·문장부호 제거)로 맞춰
# 띄어쓰기 차이로 같은 문장을 놓치지 않게 한다. 홀드아웃 H로 빼낸 문장(55개)은 학습에 없으므로 '없는 문장'
# 쪽에 들어간다.
def train_sentences() -> set[str]:
    return {normalize_score(u.text) for u in read_manifest(ROOT / "manifests" / "train.jsonl")}


# 봉인 test 가설 파일 한 모드(full·streaming)의 행(id·ref·hyp) 목록. 파일 순서 = 매니페스트 순서.
def load_hyps(mode: str) -> list[dict]:
    path = ROOT / "reports" / "final" / f"test_zeroth-{mode}.hyps.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# 기본 분석(학습 문장 여부 분할). full과 스트리밍 두 디코딩 모드 각각에 대해 두 집단의 CER과 발화 부트스트랩
# 95 % 구간(1,000회, seed 0)을 계산한다. 결과 모양은 커밋된 reports/final/zeroth_seen_unseen.json과 같다.
def seen_unseen(train: set[str] | None = None) -> dict:
    train = train_sentences() if train is None else train
    out: dict = {"note": "post-hoc split of the saved sealed-test hypotheses; no new decoding"}
    for mode in ("full", "streaming"):
        groups = {"sentence_in_train": [], "sentence_not_in_train": []}
        for r in load_hyps(mode):
            # 정답 문장이 학습 문장 집합에 있으면 seen, 없으면 unseen 집단. Zeroth 공식 test는 모두 한글 전용
            # 발화라 한글 전용 부분집합을 따로 거르지 않는다.
            key = "sentence_in_train" if normalize_score(r["ref"]) in train else "sentence_not_in_train"
            # 채점 정규화 뒤 정답이 비면 None이라 뺀다.
            s = score_ns(r["ref"], r["hyp"])
            if s is not None:
                groups[key].append(s)
        out[mode] = {k: summarize(v) for k, v in groups.items()}
    return out


# parquet 행 번호 → 원래 Zeroth id. prepare_data.iter_zeroth와 같게 test-*.parquet를 이름순으로 이어 읽고
# n번째 행을 zeroth/test/<n>에 맞춘다. 그 행의 화자(speaker_id)와 전사(학습 정규화 뒤)가 매니페스트와 다르면
# 다른 파일·다른 순서를 읽은 것이므로 멈춘다.
# 입력: parquet 폴더, test_zeroth 매니페스트 발화 목록.
# 출력: {"zeroth/test/<n>": 원래 id}.
def parquet_row_ids(parquet_dir: Path, utts: list[Utterance]) -> dict[str, str]:
    files = sorted(parquet_dir.glob("test-*.parquet"))
    if not files:
        raise SystemExit(f"no Zeroth test parquet (test-*.parquet) under {parquet_dir}")
    source_ids: list[str] = []
    speakers: list = []
    texts: list = []
    for f in files:
        table = pq.read_table(f, columns=["id", "speaker_id", "text"])
        source_ids += table.column("id").to_pylist()
        speakers += table.column("speaker_id").to_pylist()
        texts += table.column("text").to_pylist()
    manifest = {u.id: u for u in utts}
    if len(source_ids) != len(manifest):
        raise SystemExit(
            f"parquet has {len(source_ids)} rows but the manifest has {len(manifest)} utterances"
        )
    out: dict[str, str] = {}
    for n, (source, speaker, text) in enumerate(zip(source_ids, speakers, texts, strict=True)):
        utt = manifest.get(f"zeroth/test/{n}")
        if utt is None:
            raise SystemExit(f"manifest has no zeroth/test/{n}")
        # prepare_data와 같은 변환(앞뒤 공백 제거 → 학습 정규화)으로 맞춰 본다.
        if str(speaker) != utt.speaker or normalize_train((text or "").strip()) != utt.text:
            raise SystemExit(f"parquet row {n} ({source}) does not match manifest utterance {utt.id}")
        out[utt.id] = source
    return out


# 앵커와 짝지은 발화 하나. 문자열은 이 저장소 정답만 남기고(학습 문장 분할용), 앵커 쪽은 편집 수와 숫자 표시만
# 둔다(다른 저장소의 가설 문자열을 들고 다니지 않는다). dataclass 대신 NamedTuple을 쓰는 것은, 테스트가 이
# 스크립트를 파일 경로로 불러올 때(sys.modules에 등록하지 않음) dataclass가 모듈을 찾지 못하기 때문이다.
class AnchorPair(NamedTuple):
    source_id: str  # 원래 Zeroth id(두 저장소 공통 키, 정렬 기준)
    utt_id: str  # 이 저장소 id(zeroth/test/<n>)
    speaker: str  # 화자 군집 키(이 저장소 매니페스트)
    reference: str  # 이 저장소 정답(매니페스트 전사)
    ours_edits: int
    anchor_edits: int
    length: int  # 정규화 정답 글자 수(두 저장소가 같아야 한다)
    anchor_has_digit: bool  # 앵커 가설에 아라비아 숫자가 있는가


# 저장된 봉인 가설과 앵커 보고서의 발화를 원래 id로 짝짓는다. 하나라도 어긋나면(앵커 쪽 중복 id, 두 행이 같은
# 원래 id로 이어짐, 한쪽에만 있는 발화, 정규화 정답이 다름, has_digit 표시가 가설 내용과 다름) 계산하지 않고
# 멈춘다.
# 입력: rows(id·ref·hyp), row_to_source(parquet_row_ids 결과), speakers(이 저장소 id → 화자),
# anchor_utts(앵커 보고서의 utterances: id·reference·hypothesis·has_digit).
# 출력: 원래 id 순으로 정렬한 AnchorPair 목록. 두 쪽 정답이 모두 비는 발화는 등록 채점처럼 뺀다.
def pair_with_anchor(
    rows: list[dict], row_to_source: dict[str, str], speakers: dict[str, str], anchor_utts: list[dict]
) -> list[AnchorPair]:
    anchor: dict[str, dict] = {}
    for u in anchor_utts:
        if u["id"] in anchor:
            raise SystemExit(f"anchor report lists {u['id']} twice")
        anchor[u["id"]] = u
    used: dict[str, str] = {}
    pairs: list[AnchorPair] = []
    for r in rows:
        source = row_to_source.get(r["id"])
        speaker = speakers.get(r["id"])
        if source is None or speaker is None:
            raise SystemExit(f"no original id or speaker for {r['id']}")
        if source in used:
            raise SystemExit(f"{used[source]} and {r['id']} both map to {source}")
        used[source] = r["id"]
        a = anchor.get(source)
        if a is None:
            raise SystemExit(f"anchor report has no utterance {source} ({r['id']})")
        ours, theirs = score_ns(r["ref"], r["hyp"]), score_ns(a["reference"], a["hypothesis"])
        if ours is None and theirs is None:
            continue
        # 정규화 정답이 글자까지 같아야 같은 발화다. 길이만 같은 다른 문장도 잘못 이은 짝이다.
        if ours is None or theirs is None or ours.reference != theirs.reference:
            got = [None if s is None else s.length for s in (ours, theirs)]
            raise SystemExit(
                f"normalized references differ for {r['id']} / {source} (lengths {got[0]} vs {got[1]})"
            )
        has_digit = bool(_DIGIT.search(a["hypothesis"]))
        if has_digit != bool(a.get("has_digit")):
            raise SystemExit(f"has_digit of {source} does not match its hypothesis")
        pairs.append(
            AnchorPair(source, r["id"], speaker, r["ref"], ours.edits, theirs.edits, ours.length, has_digit)
        )
    unmatched = set(anchor) - set(used)
    if unmatched:
        raise SystemExit(
            f"anchor report has {len(unmatched)} utterances that are not in the saved hypotheses"
        )
    return sorted(pairs, key=lambda p: p.source_id)


# 이 모델만의 요약(학습 문장 분할용). 모양은 metrics.summarize와 같다: {n, rate, ci, edits, length}.
def _ours_summary(pairs: list[AnchorPair]) -> dict:
    edits = np.array([p.ours_edits for p in pairs], dtype=np.int64)
    lengths = np.array([p.length for p in pairs], dtype=np.int64)
    if not pairs:
        return {"n": 0, "rate": None, "ci": None, "edits": 0, "length": 0}
    return {
        "n": len(pairs),
        "rate": error_rate(edits, lengths),
        "ci": list(interval(edits, lengths)),
        "edits": int(edits.sum()),
        "length": int(lengths.sum()),
    }


# 짝지은 부분집합 하나의 요약: 두 시스템의 CER, 같은 발화 짝 차 rate(이 모델) − rate(앵커)와 그 발화 단위·화자
# 군집 95 % 구간(1,000회, seed 0). 구간이 0을 제외하는지도 적는다.
# 두 시스템 각각의 구간은 싣지 않는다. 이 순서(원래 id 순)로 다시 뽑은 이 모델의 457발화 구간은 매니페스트
# 순서로 잰 등록 구간(README 결과 표)과 끝값이 조금 달라 같은 수치가 두 값으로 보이기 때문이다. 판단에 쓰는
# 것은 짝 차다.
# 주의: 화자 군집은 이 세트에서 10명뿐이라 끝값의 재표본 오차가 크다(사후 보조).
def paired_summary(pairs: list[AnchorPair]) -> dict:
    if not pairs:
        raise SystemExit("empty subset")
    ours = np.array([p.ours_edits for p in pairs], dtype=np.int64)
    anchor = np.array([p.anchor_edits for p in pairs], dtype=np.int64)
    lengths = np.array([p.length for p in pairs], dtype=np.int64)
    speakers = [p.speaker for p in pairs]
    rate_ours, rate_anchor = error_rate(ours, lengths), error_rate(anchor, lengths)
    diff = {
        "utterance": list(difference_interval(ours, anchor, lengths)),
        "speaker": list(cluster_difference_interval(ours, anchor, lengths, speakers)),
    }
    return {
        "n": len(pairs),
        "speakers": len(set(speakers)),
        "edits_ours": int(ours.sum()),
        "edits_anchor": int(anchor.sum()),
        "length": int(lengths.sum()),
        "rate_ours": rate_ours,
        "rate_anchor": rate_anchor,
        "diff_ours_minus_anchor": rate_ours - rate_anchor,
        "diff_ci95": diff,
        "diff_ci_excludes_zero": {k: v[0] > 0 or v[1] < 0 for k, v in diff.items()},
    }


# 네 부분집합의 짝 비교: 전체, 숫자 목록 안, 숫자 목록 밖(주 분석), 앵커 가설에 숫자가 든 발화 제외(민감도).
# 주 분석 부분집합은 이 모델의 CER을 학습 문장 여부로도 나눈다(문장 암기 단서가 이 부분집합에서도 그대로인지).
# 입력: pair_with_anchor 결과, 숫자 목록의 원래 id 집합, 학습 문장 집합(채점 정규화 기준).
# 주의: 숫자 목록에 짝에 없는 id가 있으면 다른 세트의 목록이므로 멈춘다.
def anchor_notation(pairs: list[AnchorPair], digit_ids: set[str], train: set[str]) -> dict:
    missing = digit_ids - {p.source_id for p in pairs}
    if missing:
        raise SystemExit(f"{len(missing)} ids of the digit list are not among the paired utterances")
    inside = [p for p in pairs if p.source_id in digit_ids]
    outside = [p for p in pairs if p.source_id not in digit_ids]
    out = {
        "all": paired_summary(pairs),
        "digit_list": paired_summary(inside),
        "excluding_digit_list": paired_summary(outside),
        "excluding_anchor_digit_hypotheses": paired_summary([p for p in pairs if not p.anchor_has_digit]),
    }
    out["excluding_digit_list"]["split_by_train_sentence"] = {
        "sentence_in_train": _ours_summary([p for p in outside if normalize_score(p.reference) in train]),
        "sentence_not_in_train": _ours_summary(
            [p for p in outside if normalize_score(p.reference) not in train]
        ),
    }
    return out


# 입력 파일의 출처: 저장소 이름(git 최상위 폴더 이름), 그 저장소 안의 경로, 그 파일을 마지막으로 바꾼 커밋,
# 커밋 뒤 수정 여부, 파일 SHA-256. 로컬 절대 경로는 남기지 않는다. git이 없거나 저장소 밖 파일이면
# 저장소·커밋은 null로 두고 경고한다(SHA-256은 항상 남는다).
def file_provenance(path: Path) -> dict:
    path = path.resolve()
    info = {
        "repository": None,
        "path": path.name,
        "commit": None,
        "uncommitted_changes": None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    git = shutil.which("git")
    try:
        if git is None:
            raise FileNotFoundError("git")

        def run(*args: str) -> str:
            return subprocess.run([git, *args], capture_output=True, text=True, check=True).stdout.strip()

        top = Path(run("-C", str(path.parent), "rev-parse", "--show-toplevel")).resolve()
        rel = path.relative_to(top).as_posix()
        commit = run("-C", str(top), "log", "-1", "--format=%H", "--", rel)
        dirty = bool(run("-C", str(top), "status", "--porcelain", "--", rel))
    except (OSError, subprocess.CalledProcessError, ValueError):
        print(f"warning: no git provenance for {path.name}", file=sys.stderr)
        return info
    if dirty:
        print(
            f"warning: {top.name}/{rel} has uncommitted changes; the sha256 identifies the content",
            file=sys.stderr,
        )
    info.update(repository=top.name, path=rel, commit=commit or None, uncommitted_changes=dirty)
    return info


# 앵커 표기 분석 전체. 두 보고서가 Zeroth test 것인지, 숫자 목록에 중복이 없는지 확인하고, parquet 결합 →
# 짝짓기 → 부분집합 비교를 한 뒤 출처와 함께 돌려준다.
# 입력: 앵커 보고서(whisper-ko-ft의 reports/whisper-large-v3-turbo/zeroth-test.json), 숫자 목록
# (whisper-ko-ft의 reports/digit_utterances/zeroth-test.json), Zeroth parquet 폴더. 경로는 받은 그대로 쓴다
# (명령행에서는 main이 저장소 루트 기준으로 바꿔 넘긴다).
# 출력: reports/final/zeroth_anchor_notation.json에 쓸 dict.
def anchor_analysis(anchor_report: Path, digit_list: Path, parquet_dir: Path) -> dict:
    report = json.loads(anchor_report.read_text(encoding="utf-8"))
    digits = json.loads(digit_list.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    if summary.get("set") != "zeroth-test" or digits.get("set") != "zeroth-test":
        raise SystemExit("both whisper-ko-ft inputs must describe the Zeroth test set (set == zeroth-test)")
    if summary.get("utterances") != len(report["utterances"]):
        raise SystemExit("anchor report summary and utterance list disagree")
    ids = digits["ids"]
    if len(set(ids)) != len(ids) or digits.get("count") != len(ids):
        raise SystemExit("digit list has duplicate ids or a wrong count")
    utts = read_manifest(ROOT / "manifests" / "test_zeroth.jsonl")
    row_to_source = parquet_row_ids(parquet_dir, utts)
    pairs = pair_with_anchor(
        load_hyps("full"), row_to_source, {u.id: u.speaker for u in utts}, report["utterances"]
    )
    subsets = anchor_notation(pairs, set(ids), train_sentences())
    return {
        "note": (
            "post-hoc analysis of the saved sealed-test hypotheses against a per-utterance anchor report; "
            "no new decoding, not a verdict; H7.1, its anchors and the sealed numbers are unchanged"
        ),
        "hypotheses": FULL_HYPS,
        "metric": "CER_ns (score_ns, the same normalization as whisper-ko-ft) on the same utterances",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": 0.95,
            "units": "utterance; speaker cluster (speaker of the test_zeroth manifest)",
            "order": "pairs sorted by the original Zeroth id; speakers in order of first appearance",
        },
        "id_join": {
            "files": [f.name for f in sorted(parquet_dir.glob("test-*.parquet"))],
            "rule": "zeroth/test/<n> is row n of the Zeroth-Korean test parquet files read in name order",
            "checks": (
                "speaker and normalized transcript of every row equal the manifest; the join is one-to-one; "
                "the normalized references of every pair are identical"
            ),
            "utterances": len(pairs),
        },
        "sources": {
            "anchor_report": {
                **file_provenance(anchor_report),
                "model": summary.get("model"),
                "adapter": summary.get("adapter"),
                "reported_error_rate": summary.get("error_rate"),
                "hypotheses_with_digit": sum(p.anchor_has_digit for p in pairs),
            },
            "digit_list": {
                **file_provenance(digit_list),
                "rule": digits.get("rule"),
                "per_model": digits.get("per_model"),
                "count": len(ids),
                "definition": (
                    "fixed by whisper-ko-ft in its stage 1 from the hypotheses of its two base models and "
                    "not changed afterwards; the Zeroth references spell numbers in Hangul"
                ),
            },
        },
        "subsets": subsets,
    }


# 명령행 인자. 경로는 저장소 루트 기준(절대 경로도 된다).
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "봉인된 Zeroth 공식 test의 저장된 가설로 사후 분석을 한다(새 디코딩 없음, 판정 아님). "
            "기본은 학습 문장 여부 분할(reports/final/zeroth_seen_unseen.json)이고, --anchor-report와 "
            "--digit-list를 함께 주면 whisper 앵커와 같은 발화로 짝지은 숫자 표기 분석"
            "(reports/final/zeroth_anchor_notation.json)도 쓴다."
        ),
        epilog=(
            "앵커 분석에는 0단계가 내려받은 Zeroth parquet(행 번호 id를 원래 id로 잇는 데 씀)와 "
            "whisper-ko-ft 체크아웃이 필요하다. 다른 저장소의 정답·가설 문자열은 결과에 복사하지 않고 "
            "수치와 출처(커밋, SHA-256)만 남긴다."
        ),
    )
    p.add_argument(
        "--anchor-report",
        help=(
            "whisper-ko-ft의 발화별 보고서"
            "(예: ../whisper-ko-ft/reports/whisper-large-v3-turbo/zeroth-test.json)"
        ),
    )
    p.add_argument(
        "--digit-list",
        help=(
            "whisper-ko-ft의 숫자 발화 목록(예: ../whisper-ko-ft/reports/digit_utterances/zeroth-test.json). "
            "--anchor-report와 함께 준다"
        ),
    )
    p.add_argument(
        "--zeroth-parquet-dir",
        default=DEFAULT_PARQUET_DIR,
        help=f"Zeroth-Korean test parquet 폴더(기본 {DEFAULT_PARQUET_DIR})",
    )
    p.add_argument("--out-dir", default="reports/final", help="결과 JSON을 쓸 폴더(기본 reports/final)")
    return p


# 기본 분석을 저장하고 출력한다. 앵커 인자가 있으면 앵커 표기 분석도 저장하고 부분집합별 한 줄 요약을 찍는다.
def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.anchor_report) != bool(args.digit_list):
        parser.error("--anchor-report and --digit-list go together")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = seen_unseen()
    # 기존 출력과 바이트 단위로 같도록 이 파일은 지금까지처럼 플랫폼 기본 줄바꿈으로 쓴다(커밋된 파일은
    # Windows에서 만든 것이다).
    (out_dir / "zeroth_seen_unseen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 모드·집단별 발화 수, CER(%), 구간(%)을 찍는다.
    for mode in ("full", "streaming"):
        for k, s in out[mode].items():
            print(mode, k, s["n"], round(s["rate"] * 100, 2), [round(c * 100, 2) for c in s["ci"]])
    if not args.anchor_report:
        return
    result = anchor_analysis(
        ROOT / args.anchor_report, ROOT / args.digit_list, ROOT / args.zeroth_parquet_dir
    )
    path = out_dir / "zeroth_anchor_notation.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    # 부분집합별: 발화 수, 이 모델·앵커 CER(%), 짝 차(포인트)와 발화·화자 군집 95 % 구간.
    for name, s in result["subsets"].items():
        utt, spk = ([round(c * 100, 2) for c in s["diff_ci95"][k]] for k in ("utterance", "speaker"))
        print(
            f"anchor {name}: n {s['n']} ours {s['rate_ours'] * 100:.2f} anchor {s['rate_anchor'] * 100:.2f} "
            f"diff {s['diff_ours_minus_anchor'] * 100:+.2f} utterance {utt} speaker {spk}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
