"""같은 발화에서 두 시스템을 짝지어 비교한다(docs/experiments.md "지표": 짝지은 부트스트랩).

scripts/stage2_report.py가 쓴 두 모델(a, b)의 발화별 가설 파일을 읽어 같은 발화끼리 짝짓고, 한글 전용
부분집합과 전체 발화 각각에 대해 두 모델의 CER_ns, 상대 비율 a ÷ b, rate(a) − rate(b) 차이와 그 차이의
짝지은 부트스트랩 95 % 구간(재표본 1,000회, seed 0)을 낸다.

사용법:
  uv run python scripts/compare_runs.py --a reports/stage2/hyps/runA-avg10-dev_mix-full.jsonl \
      --b reports/stage2/hyps/runB-avg10-dev_mix-full.jsonl --name H2.1-dev_mix --out-dir reports/stage2
"""

# [왜 필요한가] 사전 등록 규칙은 두 구성의 비교를 "같은 발화를 다시 뽑는 짝지은 구간"으로 하고, 상대 기준으로
# 개선·채택을 주장하는 가설은 그 구간이 0을 제외해야 한다(docs/experiments.md "지표", "선택 규칙", 규칙 변경
# 23). 두 모델의 구간을 따로 구해 겹치는지 보는 것과 달리, 짝지은 차이는 발화마다 다른 난이도가 두 모델에
# 똑같이 들어가 상쇄되므로 모델 차이만 본다.
# [연결] 입력은 scripts/stage2_report.py가 쓰는 reports/<단계>/hyps/*.jsonl이다. H2.1(Run A vs Run B),
# H2.2(청크 vs full), 3단계 후보 vs Run A 비교에 썼고, scripts/stage3.py가 하위 프로세스로 부른다. 결과는
# <out-dir>/compare-<name>.json.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# numpy: 발화별 편집 수·글자 수를 배열로 모아 부트스트랩 계산에 넘긴다.
import numpy as np

# 프로젝트 채점 모듈: score_ns(whisper-ko-ft와 같은 정규화의 공백 제외 CER 채점), error_rate(묶음 전체 편집 합
# ÷ 글자 합), difference_interval(같은 재표본 인덱스로 두 모델의 차이를 구하는 짝지은 부트스트랩, 1,000회·seed
# 0). 평가 보고서와 같은 함수를 써야 수치가 서로 맞는다.
from stt_scratch.metrics import difference_interval, error_rate, score_ns

# 저장소 루트. --a·--b·--out-dir를 저장소 기준 상대 경로로 받는다.
ROOT = Path(__file__).resolve().parents[1]


# 가설 파일(JSONL, 한 줄에 id·ref·hyp·hangul_only)을 읽어 발화 id → 행 사전으로 돌려준다.
def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


# 두 가설 묶음을 같은 발화끼리 짝지어 CER과 짝지은 차이 구간을 계산한다.
# 입력: a·b(load 결과), hangul_only(True면 정답에 숫자·라틴이 없는 발화만).
# 출력: n(채점한 발화 수), rate_a·rate_b, ratio_a_over_b, diff_a_minus_b, diff_ci95, ci_excludes_zero.
# 주의: 두 파일의 발화 집합이나 정답이 하나라도 다르면 비교가 무의미하므로 계산하지 않고 SystemExit로 멈춘다.
def compare(a: dict[str, dict], b: dict[str, dict], hangul_only: bool) -> dict:
    # 공통 발화 id를 정렬해 고정된 순서로 쓴다. 순서가 고정돼야 부트스트랩 재표본 인덱스가 매번 같은 발화를
    # 가리켜 결과가 재현된다.
    ids = sorted(set(a) & set(b))
    # 한쪽에만 있는 발화가 있으면 같은 세트를 평가한 것이 아니므로 거부한다.
    if len(ids) != len(a) or len(ids) != len(b):
        raise SystemExit(
            f"the two files cover different utterances ({len(a)} vs {len(b)}, common {len(ids)})"
        )
    edits_a, edits_b, lengths = [], [], []
    for i in ids:
        # id가 같아도 정답 문장이 다르면(매니페스트 버전 차이 등) 거부한다.
        if a[i]["ref"] != b[i]["ref"]:
            raise SystemExit(f"reference mismatch for {i}")
        # 한글 전용 부분집합: 모델 어휘로 낼 수 없는 숫자·라틴이 정답에 든 발화를 뺀다(docs/experiments.md
        # "지표"의 두 수치 보고 규칙).
        if hangul_only and not a[i]["hangul_only"]:
            continue
        sa, sb = score_ns(a[i]["ref"], a[i]["hyp"]), score_ns(b[i]["ref"], b[i]["hyp"])
        # 채점 정규화 뒤 정답이 비는 발화는 score_ns가 None을 돌려주므로 뺀다. 정답이 같으므로 b도 똑같이
        # None이다.
        if sa is None:
            continue
        edits_a.append(sa.edits)
        edits_b.append(sb.edits)
        lengths.append(sa.length)
    # 발화별 편집 수(a, b)와 정답 글자 수를 정수 배열로 만든다. 글자 수는 정답에서 나오므로 두 모델이
    # 공유한다.
    ea, eb, ln = (np.array(x, dtype=np.int64) for x in (edits_a, edits_b, lengths))
    ra, rb = error_rate(ea, ln), error_rate(eb, ln)
    # 발화를 1,000번 복원 추출하되 a와 b에 같은 추출을 써서 rate(a) − rate(b)의 2.5·97.5 백분위를 구한다.
    low, high = difference_interval(ea, eb, ln)
    return {
        "n": len(ln),
        "rate_a": ra,
        "rate_b": rb,
        "ratio_a_over_b": ra / rb if rb else None,
        "diff_a_minus_b": ra - rb,
        "diff_ci95": [low, high],
        # 구간이 0을 포함하지 않으면 두 모델의 차이가 재표본 변동보다 크다고 본다. 개선·채택 조건은 이 값이
        # True여야 한다.
        "ci_excludes_zero": low > 0 or high < 0,
    }


# 인자를 읽어 한글 전용·전체 두 비교를 JSON으로 저장하고 한 줄 요약을 출력한다.
# 인자: --a·--b(가설 파일), --name(결과 파일 이름에 붙일 비교 이름, 예: H2.1-dev_mix), --out-dir(기본
# reports/stage2).
# 출력의 차이는 퍼센트 포인트(pp)로, a/b는 비율로 찍는다.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", default="reports/stage2")
    args = p.parse_args(argv)
    a, b = load(ROOT / args.a), load(ROOT / args.b)
    result = {
        "a": args.a,
        "b": args.b,
        "hangul_only": compare(a, b, hangul_only=True),
        "all": compare(a, b, hangul_only=False),
    }
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare-{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for subset in ("hangul_only", "all"):
        r = result[subset]
        print(
            f"{args.name} {subset}: a {r['rate_a'] * 100:.2f} % vs b {r['rate_b'] * 100:.2f} % "
            f"(a/b {r['ratio_a_over_b']:.3f}), a-b {r['diff_a_minus_b'] * 100:+.2f} pp, "
            f"95 % [{r['diff_ci95'][0] * 100:+.2f}, {r['diff_ci95'][1] * 100:+.2f}], n {r['n']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
