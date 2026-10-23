"""군집(문장·화자) 부트스트랩 구간과 그 사후 보조 보고서(scripts/cluster_ci.py) 테스트.

확인하는 성질:
(a) 군집 = 발화(키가 모두 다름)이면 등록된 발화 단위 부트스트랩(interval·difference_interval)과 비트 단위로
    같다. 군집 함수가 같은 Generator(seed 0)와 같은 모양의 재표본 표를 쓰기 때문이다.
(b) 군집 안 상관이 큰 합성 데이터에서는 군집 구간이 발화 구간보다 넓다(보조 구간을 따로 내는 이유).
(c) 같은 입력이면 같은 값(결정성)이고, 군집 순서는 키가 처음 나온 순서다.
(d) 잘못된 입력(길이 불일치, None 키, 빈 입력)은 조용히 넘어가지 않고 ValueError.
(e) scripts/cluster_ci.py: 발화 단위 구간이 이미 기록된 구간(reports/final/*.json,
    reports/followup/session.json)과 같고, 커밋된 reports/final/cluster_ci.json을 저장된 가설로 다시 만든다.
"""

# [왜 필요한가] 등록 규칙의 구간은 발화를 서로 독립으로 다시 뽑는다(docs/experiments.md "지표"). 평가 세트는
# 같은 문장을 여러 화자가 읽거나(test_H 1,280발화 167문장) 소수 화자가 많은 발화를 낸다(Zeroth 공식 test 화자
# 10명, CV22 test 74명). 이런 묶음을 무시하면 '다른 문장·화자로 일반화할 때'의 불확실성이 작게 보이므로, 등록
# 구간은 그대로 두고 군집 단위로 다시 뽑은 사후 보조 구간을 함께 낸다. (a)가 성립해야 보조 구간이 등록 구간과
# 같은 방식(같은 재표본 표)의 일반화임을 보장할 수 있다.
# [방식] scripts/는 패키지가 아니라서 파일 경로로 모듈을 불러온다(tests/test_stage7_guards.py와 같은 방식).
# (e)의 입력(매니페스트·저장된 가설·기록된 결과)은 모두 저장소에 커밋돼 있어 CI에서도 돈다.

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from stt_scratch.metrics import cluster_difference_interval, cluster_interval, difference_interval, interval

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cluster_ci", REPO / "scripts" / "cluster_ci.py")
cluster_ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cluster_ci)


# 발화 수 n의 합성 채점 결과(편집 수 ≤ 글자 수). seed를 받은 Generator만 써서 매번 같은 값이 나온다.
def _synthetic(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(5, 60, size=n).astype(np.int64)
    edits_a = rng.binomial(lengths, 0.12).astype(np.int64)
    edits_b = rng.binomial(lengths, 0.15).astype(np.int64)
    return edits_a, edits_b, lengths


# (a) 키가 모두 다르면(군집 = 발화) 군집 합이 원래 배열과 같아 기존 함수와 비트 단위로 같아야 한다. 키의 모양
# (문자열 id, 정수, 섞인 순서의 고유 키)은 결과에 영향을 주지 않는다.
@pytest.mark.parametrize("make_keys", [lambda n: [f"u{i}" for i in range(n)], lambda n: list(range(n))])
def test_unit_clusters_equal_utterance_bootstrap(make_keys):
    edits_a, edits_b, lengths = _synthetic(240)
    keys = make_keys(len(lengths))
    assert cluster_interval(edits_a, lengths, keys) == interval(edits_a, lengths)
    assert cluster_difference_interval(edits_a, edits_b, lengths, keys) == difference_interval(
        edits_a, edits_b, lengths
    )
    # 고유 키를 거꾸로 붙여도 군집 j = 발화 j라 같은 값이다(순서는 키 값이 아니라 처음 나온 위치로 정해진다).
    reversed_keys = [f"k{len(lengths) - i}" for i in range(len(lengths))]
    assert cluster_interval(edits_a, lengths, reversed_keys) == interval(edits_a, lengths)


# (b) 군집마다 오류율이 크게 다르고 군집 안 발화는 비슷한 데이터: 군집 30개 × 발화 20개. 군집 단위로 다시
# 뽑으면 군집 사이 변동이 구간에 들어가 발화 단위보다 넓어야 한다(이 seed에서 약 3배). 짝지은 차이도 군집마다
# 다른 차이를 주면 같은 방향이다.
def test_correlated_clusters_widen_the_interval():
    rng = np.random.default_rng(7)
    clusters, per_cluster = 30, 20
    rate = rng.uniform(0.02, 0.40, size=clusters)
    groups = np.repeat(np.arange(clusters), per_cluster)
    lengths = rng.integers(10, 40, size=clusters * per_cluster).astype(np.int64)
    edits_a = rng.binomial(lengths, rate[groups]).astype(np.int64)
    shift = rng.uniform(-0.15, 0.15, size=clusters)
    edits_b = rng.binomial(lengths, np.clip(rate[groups] + shift[groups], 0.0, 1.0)).astype(np.int64)

    low_u, high_u = interval(edits_a, lengths)
    low_c, high_c = cluster_interval(edits_a, lengths, groups)
    assert high_c - low_c > 2 * (high_u - low_u)
    # 비율 점추정은 두 구간 모두에 들어 있어야 한다(같은 통계량을 다른 단위로 재표본했을 뿐이다).
    rate_all = edits_a.sum() / lengths.sum()
    assert low_c < rate_all < high_c and low_u < rate_all < high_u

    low_ud, high_ud = difference_interval(edits_a, edits_b, lengths)
    low_cd, high_cd = cluster_difference_interval(edits_a, edits_b, lengths, groups)
    assert high_cd - low_cd > 2 * (high_ud - low_ud)


# (c) 결정성과 군집 순서: 같은 입력 두 번은 같은 값이고, 섞여 나오는 키는 처음 나온 순서(s, t, u)로 합쳐진
# 배열에 기존 함수를 쓴 것과 같다.
def test_deterministic_and_first_appearance_order():
    edits_a, edits_b, lengths = _synthetic(90, seed=11)
    keys = [("s", "t", "u", "v", "w")[(i * 7) % 5] for i in range(90)]
    first = cluster_interval(edits_a, lengths, keys)
    assert first == cluster_interval(edits_a, lengths, keys)
    assert cluster_difference_interval(edits_a, edits_b, lengths, keys) == cluster_difference_interval(
        edits_a, edits_b, lengths, keys
    )

    order = list(dict.fromkeys(keys))
    index = np.array([order.index(k) for k in keys])
    totals = [
        np.bincount(index, weights=x, minlength=len(order)).astype(np.int64) for x in (edits_a, lengths)
    ]
    assert first == interval(*totals)

    small_keys = ["s", "t", "s", "u", "t"]
    small_edits = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    small_lengths = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    expected = interval(np.array([4, 7, 4], dtype=np.int64), np.array([22, 25, 13], dtype=np.int64))
    assert cluster_interval(small_edits, small_lengths, small_keys) == expected


# (d) 잘못된 입력: 군집 키 수가 발화 수와 다르거나, 키가 None이거나(서로 다른 발화가 한 군집으로 조용히 묶임),
# 발화가 없으면 ValueError.
def test_invalid_inputs_raise():
    edits_a, edits_b, lengths = _synthetic(4)
    with pytest.raises(ValueError):
        cluster_interval(edits_a, lengths, ["a", "b", "c"])
    with pytest.raises(ValueError):
        cluster_difference_interval(edits_a, edits_b, lengths, ["a", None, "b", "c"])
    with pytest.raises(ValueError):
        cluster_interval(edits_a, lengths[:3], ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        cluster_difference_interval(edits_a, edits_b[:3], lengths, ["a", "b", "c", "d"])
    empty = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):
        cluster_interval(empty, empty, [])


# (e)용 도우미: 두 JSON 값이 같은지 비교한다. 실수는 1e-12 안이면 같다고 본다(플랫폼 사이의 부동소수점 차이에
# 대비; 실제로는 같은 연산이라 비트 단위로 같아야 한다).
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


# (e) 저장된 가설에서 다시 계산한 보고서: (1) 봉인 표 8행과 P1 비교 2개의 발화 단위 구간이 기록된 값과 같다
# (같은 가설·같은 채점·같은 순서라는 확인), (2) 화자 군집(FLEURS는 문장 군집) 구간이 모든 봉인 표 행에서 발화
# 구간보다 넓다(README 표 머리의 서술), (3) 커밋된 reports/final/cluster_ci.json과 같다.
def test_cluster_ci_report_matches_records_and_committed_file():
    report = cluster_ci.compute()
    for name in cluster_ci.SEALED_SETS:
        for mode in cluster_ci.MODES:
            row = report["sealed"][name][mode]
            recorded = json.loads(
                (REPO / "reports" / "final" / f"{name}-{mode}.json").read_text(encoding="utf-8")
            )
            assert row["n"] == recorded["hangul_only"]["cer_ns"]["n"]
            _same(row["ci95"]["utterance"], recorded["hangul_only"]["cer_ns"]["ci"])
            widest = "sentence" if name == "test_fleurs" else "speaker"
            low, high = row["ci95"][widest]
            low_u, high_u = row["ci95"]["utterance"]
            assert high - low > high_u - low_u, (name, mode)
    session = json.loads((REPO / "reports" / "followup" / "session.json").read_text(encoding="utf-8"))
    for key, row in report["paired"].items():
        recorded = session["runs"][f"{row['set']}_rule2_0.8"]["final_pass_vs_stream"]
        assert row["n"] == recorded["n"], key
        _same(row["ci95"]["utterance"], recorded["diff_ci95"], key)
    committed = json.loads((REPO / "reports" / "final" / "cluster_ci.json").read_text(encoding="utf-8"))
    _same(report, committed)


# (e) 명령행: --out으로 준 경로에 커밋된 보고서와 같은 내용을 쓴다(커밋된 파일 자체는 건드리지 않는다). 위
# 테스트가 compute() = 커밋된 파일을 확인하므로 여기서는 compute()를 다시 부르지 않는다.
def test_cluster_ci_main_writes_the_report(tmp_path):
    out = tmp_path / "cluster_ci.json"
    cluster_ci.main(["--out", str(out)])
    committed = json.loads((REPO / "reports" / "final" / "cluster_ci.json").read_text(encoding="utf-8"))
    _same(json.loads(out.read_text(encoding="utf-8")), committed)
