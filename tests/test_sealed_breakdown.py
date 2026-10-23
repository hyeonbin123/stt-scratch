"""봉인 Zeroth 공식 test 사후 분석(scripts/sealed_breakdown.py) 테스트.

확인하는 성질:
(a) 합성 몇 발화로 whisper 앵커와의 짝 비교 경로(원래 id 순 정렬, 짝 차, 발화·화자 군집 구간, 부분집합)를
    계산하고, 숫자 발화 목록 안·밖 두 부분집합의 편집·글자 합이 전체와 같다.
(b) 결합이 일대일이 아니거나(중복·누락·남는 id) 짝의 정규화 정답이 다르거나(길이가 달라도, 길이는 같고 글자만
    달라도) 숫자 목록에 없는 id가 있으면 계산하지 않고 멈춘다(잘못 이은 짝으로 수치를 내지 않게).
(c) 인자 없는 기본 동작은 그대로다: 학습 문장 여부 분할이 커밋된 reports/final/zeroth_seen_unseen.json과 같고,
    앵커 인자는 둘을 함께 줘야 한다.
(d) 실데이터(data/raw의 Zeroth parquet와 형제 폴더의 whisper-ko-ft 체크아웃)가 있으면: 행 번호 결합이
    일대일(457발화), 두 부분집합 합 = 전체(707 / 19,288), 기대 재현값, 커밋된 결과 파일 재현. 없으면(CI)
    건너뛴다.
"""

# [왜 필요한가] 공개 문서가 whisper 제로샷 앵커(4.48 %)와 이 모델(3.67 %)을 나란히 놓는데, Zeroth 정답은
# 숫자를 한글로 풀어 쓰고 whisper는 아라비아 숫자를 내므로 그 표기 차이가 앵커 쪽 오류로 센다. 이 분석은
# 저장된 봉인 가설과 whisper-ko-ft의 발화별 보고서를 같은 발화로 짝지어 숫자 발화를 뺀 부분집합에서 다시
# 비교한다. 두 저장소의 발화 id가 달라(이 저장소는 parquet 행 번호) 결합을 잘못하면 그럴듯하지만 틀린 수치가
# 나오므로 결합 검사를 테스트로 고정한다.
# [방식] scripts/는 패키지가 아니라서 파일 경로로 모듈을 불러온다(tests/test_stage7_guards.py와 같은 방식).

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from stt_scratch.metrics import cluster_difference_interval, difference_interval, score_ns

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sealed_breakdown", REPO / "scripts" / "sealed_breakdown.py")
sealed_breakdown = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sealed_breakdown)

# (d)에 쓰는 실데이터 위치: 0단계가 내려받은 Zeroth parquet와, 이 저장소 옆에 체크아웃한 whisper-ko-ft의
# 보고서.
PARQUET_DIR = REPO / "data" / "raw" / "zeroth_korean" / "data"
WHISPER_KO_FT = REPO.parent / "whisper-ko-ft"
ANCHOR = WHISPER_KO_FT / "reports" / "whisper-large-v3-turbo" / "zeroth-test.json"
DIGITS = WHISPER_KO_FT / "reports" / "digit_utterances" / "zeroth-test.json"
REAL = pytest.mark.skipif(
    not (sorted(PARQUET_DIR.glob("test-*.parquet")) and ANCHOR.exists() and DIGITS.exists()),
    reason="needs data/raw Zeroth parquet and a sibling whisper-ko-ft checkout",
)

# (a)·(b)의 합성 발화 6개. 이 저장소 id(행 번호)와 원래 id의 순서를 일부러 다르게 둬 정렬 기준을 시험한다.
# 화자는 3명. 원래 id "103_…"인 발화 둘은 앵커 가설에 아라비아 숫자가 들어 있다.
_ROWS = [
    ("zeroth/test/0", "105_001_0002", "105", "가나다라마", "가나다라마", "가나다라마", False),
    (
        "zeroth/test/1",
        "103_001_0001",
        "103",
        "이천 십 팔 년에 왔다",
        "이천 십 팔 년에 왔다",
        "2018년에 왔다",
        True,
    ),
    ("zeroth/test/2", "104_001_0005", "104", "마바사아", "마바사", "마바사아", False),
    ("zeroth/test/3", "105_001_0001", "105", "아자차카타", "아자차타타", "아자차카", False),
    ("zeroth/test/4", "103_001_0004", "103", "세 시 반", "세 시 반", "3시 반", True),
    ("zeroth/test/5", "104_001_0003", "104", "하나 둘 셋", "하나 둘", "하나 둘 셋", False),
]


# 합성 입력 묶음: 이 저장소 가설 행, 행 번호 → 원래 id, 행 번호 → 화자, 앵커 보고서의 발화 목록.
def _synthetic():
    rows = [{"id": r[0], "ref": r[3], "hyp": r[4]} for r in _ROWS]
    row_to_source = {r[0]: r[1] for r in _ROWS}
    speakers = {r[0]: r[2] for r in _ROWS}
    anchor = [
        {"id": r[1], "speaker": r[2], "reference": r[3], "hypothesis": r[5], "has_digit": r[6]} for r in _ROWS
    ]
    return rows, row_to_source, speakers, anchor


# (a) 짝 비교 경로: 원래 id 순으로 정렬되고, 비율·짝 차가 손으로 센 값과 같고, 구간이 같은 순서의 배열에 등록
# 함수(difference_interval)와 군집 함수(화자)를 쓴 값과 같다.
def test_paired_summary_on_synthetic_pairs():
    pairs = sealed_breakdown.pair_with_anchor(*_synthetic())
    assert [p.source_id for p in pairs] == sorted(r[1] for r in _ROWS)
    ours = np.array([score_ns(r[3], r[4]).edits for r in sorted(_ROWS, key=lambda r: r[1])], dtype=np.int64)
    anchor = np.array([score_ns(r[3], r[5]).edits for r in sorted(_ROWS, key=lambda r: r[1])], dtype=np.int64)
    lengths = np.array(
        [score_ns(r[3], r[3]).length for r in sorted(_ROWS, key=lambda r: r[1])], dtype=np.int64
    )
    speakers = [r[2] for r in sorted(_ROWS, key=lambda r: r[1])]

    s = sealed_breakdown.paired_summary(pairs)
    assert s["n"] == 6 and s["speakers"] == 3
    assert (s["edits_ours"], s["edits_anchor"], s["length"]) == (ours.sum(), anchor.sum(), lengths.sum())
    assert math.isclose(s["rate_ours"], ours.sum() / lengths.sum())
    assert math.isclose(s["rate_anchor"], anchor.sum() / lengths.sum())
    assert math.isclose(s["diff_ours_minus_anchor"], s["rate_ours"] - s["rate_anchor"])
    assert s["diff_ci95"]["utterance"] == list(difference_interval(ours, anchor, lengths))
    assert s["diff_ci95"]["speaker"] == list(cluster_difference_interval(ours, anchor, lengths, speakers))


# (a) 부분집합: 숫자 목록(원래 id 두 개) 밖·안, 앵커 가설에 숫자가 든 발화 제외. 목록 안·밖의 편집·글자 합은
# 전체와 같고, 학습 문장 분할은 목록 밖 발화만 나눈다.
def test_anchor_notation_subsets_add_up():
    pairs = sealed_breakdown.pair_with_anchor(*_synthetic())
    digit_ids = {"103_001_0001", "103_001_0004"}
    train = {"가나다라마", "마바사아"}
    out = sealed_breakdown.anchor_notation(pairs, digit_ids, train)
    whole, inside, outside = out["all"], out["digit_list"], out["excluding_digit_list"]
    assert (whole["n"], inside["n"], outside["n"]) == (6, 2, 4)
    for key in ("edits_ours", "edits_anchor", "length"):
        assert inside[key] + outside[key] == whole[key]
    assert out["excluding_anchor_digit_hypotheses"]["n"] == 4
    split = out["excluding_digit_list"]["split_by_train_sentence"]
    assert (split["sentence_in_train"]["n"], split["sentence_not_in_train"]["n"]) == (2, 2)


# (b) 결합 실패: 각 경우에 SystemExit로 멈춰야 한다.
def test_join_must_be_one_to_one_with_equal_references():
    rows, row_to_source, speakers, anchor = _synthetic()
    with pytest.raises(SystemExit):  # 앵커 보고서에 같은 id가 두 번
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, anchor + [dict(anchor[0])])
    with pytest.raises(SystemExit):  # 앵커 보고서에 한 발화가 없음
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, anchor[1:])
    with pytest.raises(SystemExit):  # 앵커 보고서에만 있는 발화
        extra = {**anchor[0], "id": "199_001_0001"}
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, anchor + [extra])
    with pytest.raises(SystemExit):  # 두 행 번호가 같은 원래 id로 이어짐
        sealed_breakdown.pair_with_anchor(
            rows, {**row_to_source, "zeroth/test/1": "105_001_0002"}, speakers, anchor
        )
    with pytest.raises(SystemExit):  # 정규화 정답의 길이가 다름
        changed = [dict(a) for a in anchor]
        changed[2]["reference"] = "마바사아자"
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, changed)
    with pytest.raises(SystemExit):  # 길이는 같고 글자가 다름(잘못 이은 짝)
        changed = [dict(a) for a in anchor]
        changed[2]["reference"] = "마바사자"
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, changed)
    with pytest.raises(SystemExit):  # has_digit 표시가 가설 내용과 다름
        changed = [dict(a) for a in anchor]
        changed[0]["has_digit"] = True
        sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, changed)
    pairs = sealed_breakdown.pair_with_anchor(rows, row_to_source, speakers, anchor)
    with pytest.raises(SystemExit):  # 숫자 목록에 짝에 없는 id가 있음
        sealed_breakdown.anchor_notation(pairs, {"103_001_0001", "999_999_9999"}, set())


# 두 JSON 값이 같은지 비교한다. 실수는 1e-12 안이면 같다고 본다(같은 연산이라 실제로는 비트 단위로 같아야
# 하지만, 커밋된 값은 Windows에서 만든 것이라 다른 플랫폼의 CI에서 부동소수점 끝자리가 흔들려도 깨지지 않게).
def _same(a, b, path="") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


# (c) 기본 동작: 학습 문장 여부 분할이 커밋된 결과와 같다(리팩터링으로 바뀌지 않았다는 확인).
def test_default_seen_unseen_is_unchanged():
    committed = json.loads(
        (REPO / "reports" / "final" / "zeroth_seen_unseen.json").read_text(encoding="utf-8")
    )
    _same(sealed_breakdown.seen_unseen(), committed)


# (c) 명령행: 인자가 없으면 seen/unseen 파일 하나만 쓰고(내용은 커밋된 것과 같음), 앵커 인자 하나만 주면
# argparse 오류로 멈춘다.
def test_main_without_anchor_arguments(tmp_path):
    sealed_breakdown.main(["--out-dir", str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zeroth_seen_unseen.json"]
    committed = (REPO / "reports" / "final" / "zeroth_seen_unseen.json").read_text(encoding="utf-8")
    written = (tmp_path / "zeroth_seen_unseen.json").read_text(encoding="utf-8")
    _same(json.loads(written), json.loads(committed))
    with pytest.raises(SystemExit):
        sealed_breakdown.main(["--out-dir", str(tmp_path), "--anchor-report", "x.json"])


# (d) 실데이터: 행 번호 결합이 일대일이고 두 부분집합의 합이 봉인 보고 수치(707 / 19,288)와 같으며,
# docs/experiments.md 7단계 결과의 사후 분석에 적은 값(소수 둘째 자리, %·포인트)이 그대로 나온다.
@REAL
def test_real_anchor_notation_reproduces_expected_values():
    result = sealed_breakdown.anchor_analysis(ANCHOR, DIGITS, PARQUET_DIR)
    subsets = result["subsets"]
    whole = subsets["all"]
    assert (whole["n"], whole["edits_ours"], whole["length"]) == (457, 707, 19288)
    for key in ("edits_ours", "edits_anchor", "length"):
        assert subsets["digit_list"][key] + subsets["excluding_digit_list"][key] == whole[key]

    def pct(x: float) -> float:
        return round(x * 100, 2)

    expected = {
        "all": (457, 3.67, 4.48, -0.81, (-1.54, -0.13), (-1.43, -0.20)),
        "excluding_digit_list": (315, 3.69, 3.29, 0.39, (-0.39, 1.25), (-0.21, 1.02)),
        "excluding_anchor_digit_hypotheses": (384, 3.64, 3.17, 0.47, (-0.22, 1.18), (-0.03, 1.01)),
    }
    for name, (n, ours, anchor, diff, utt, spk) in expected.items():
        s = subsets[name]
        assert s["n"] == n, name
        assert (pct(s["rate_ours"]), pct(s["rate_anchor"])) == (ours, anchor), name
        assert pct(s["diff_ours_minus_anchor"]) == diff, name
        assert tuple(pct(x) for x in s["diff_ci95"]["utterance"]) == utt, name
        assert tuple(pct(x) for x in s["diff_ci95"]["speaker"]) == spk, name
    assert subsets["digit_list"]["n"] == 142 and pct(subsets["digit_list"]["rate_ours"]) == 3.63
    split = subsets["excluding_digit_list"]["split_by_train_sentence"]
    assert (split["sentence_in_train"]["n"], pct(split["sentence_in_train"]["rate"])) == (276, 2.07)
    assert (split["sentence_not_in_train"]["n"], pct(split["sentence_not_in_train"]["rate"])) == (39, 15.29)


# (d) 실데이터: 커밋된 reports/final/zeroth_anchor_notation.json의 수치를 다시 만들 수 있다(출처의 커밋은 다른
# 저장소가 나중에 바뀔 수 있어 수치 부분만 비교).
@REAL
def test_real_committed_anchor_report_is_reproducible():
    committed = json.loads(
        (REPO / "reports" / "final" / "zeroth_anchor_notation.json").read_text(encoding="utf-8")
    )
    result = json.loads(json.dumps(sealed_breakdown.anchor_analysis(ANCHOR, DIGITS, PARQUET_DIR)))
    _same(result["subsets"], committed["subsets"])
    assert committed["sources"]["anchor_report"]["sha256"] == result["sources"]["anchor_report"]["sha256"]
