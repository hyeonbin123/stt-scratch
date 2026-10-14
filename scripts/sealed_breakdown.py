"""봉인된 Zeroth 공식 test의 사후 분할 분석(새 디코딩 없음, 판정 아님).

7단계에서 저장한 test_zeroth 발화별 가설을 '정답 문장(채점 정규화 기준)이 우리 학습 매니페스트에도
있는가'로 두 집단으로 나누고, 두 집단 각각의 CER_ns와 부트스트랩 구간을 낸다. 새로 디코딩하지 않으므로
봉인 test를 다시 재는 것이 아니다.

    uv run python scripts/sealed_breakdown.py   # 결과: reports/final/zeroth_seen_unseen.json
"""

# [왜 필요한가] Zeroth 공식 test 3.67 %는 whisper 앵커 사이(H7.1 충족)지만, 공식 test 문장 대부분이 학습
# 세트에도 있다(docs/data.md). 이 분할로 학습 세트에 있는 문장 397발화 1.85 %, 없는 문장 60발화 16.01
# %(스트리밍 3.31 % / 19.25 %)를 보여, H7.1 충족이 문장 암기와 일관된다는 해석의 근거로
# 썼다(docs/experiments.md 7단계 결과, README 결과 표의 사후 분할 행). 문장 난도 차이도 섞여 있을 수 있어
# 암기의 기여 비율은 분리하지 않았다.
# [연결] 입력은 scripts/stage7.py가 쓴 reports/final/test_zeroth-{full,streaming}.hyps.jsonl과
# manifests/train.jsonl. 출력은 reports/final/zeroth_seen_unseen.json.

from __future__ import annotations

import json
from pathlib import Path

# 프로젝트 내부 모듈: 학습 매니페스트 읽기, 평가와 같은 CER_ns 채점(score_ns)과 요약·부트스트랩
# 구간(summarize), 채점 정규화(normalize_score). 봉인 test 보고서와 같은 함수로 계산해야 두 집단의 합이 보고된
# 전체 수치와 맞는다.
from stt_scratch.data import read_manifest
from stt_scratch.metrics import score_ns, summarize
from stt_scratch.text import normalize_score

# 저장소 루트. 매니페스트와 reports/final/을 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# full과 스트리밍 두 디코딩 모드 각각에 대해 두 집단의 CER을 계산해 저장하고 출력한다.
def main() -> None:
    # 학습 매니페스트(Zeroth train − 홀드아웃 H + FLEURS + CV)의 문장 집합. 채점 정규화(공백·문장부호 제거)로
    # 맞춰 띄어쓰기 차이로 같은 문장을 놓치지 않게 한다. 홀드아웃 H로 빼낸 문장(55개)은 학습에 없으므로 '없는
    # 문장' 쪽에 들어간다.
    train = {normalize_score(u.text) for u in read_manifest(ROOT / "manifests" / "train.jsonl")}
    out: dict = {"note": "post-hoc split of the saved sealed-test hypotheses; no new decoding"}
    for mode in ("full", "streaming"):
        path = ROOT / "reports" / "final" / f"test_zeroth-{mode}.hyps.jsonl"
        rows = [json.loads(x) for x in path.open(encoding="utf-8")]
        groups = {"sentence_in_train": [], "sentence_not_in_train": []}
        for r in rows:
            # 정답 문장이 학습 문장 집합에 있으면 seen, 없으면 unseen 집단. Zeroth 공식 test는 모두 한글 전용
            # 발화라 한글 전용 부분집합을 따로 거르지 않는다.
            key = "sentence_in_train" if normalize_score(r["ref"]) in train else "sentence_not_in_train"
            # 채점 정규화 뒤 정답이 비면 None이라 뺀다.
            s = score_ns(r["ref"], r["hyp"])
            if s is not None:
                groups[key].append(s)
        # 집단별 CER(편집 합 ÷ 글자 합)과 발화 부트스트랩 95 % 구간(1,000회, seed 0).
        out[mode] = {k: summarize(v) for k, v in groups.items()}
    path = ROOT / "reports" / "final" / "zeroth_seen_unseen.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 모드·집단별 발화 수, CER(%), 구간(%)을 찍는다.
    for mode in ("full", "streaming"):
        for k, s in out[mode].items():
            print(mode, k, s["n"], round(s["rate"] * 100, 2), [round(c * 100, 2) for c in s["ci"]])


if __name__ == "__main__":
    main()
