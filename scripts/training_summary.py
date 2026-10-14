"""outputs/ 아래 모든 학습 런의 fp16 안정성과 GPU 시간을 한 표로 모은다(7단계 최종 보고).

런(outputs/<단계>/<런>)마다: 완료 epoch 수, epoch 계산 시간 합(학습만, dev 평가와 일시 중지 제외),
GradScaler가 건너뛴 step 수, NaN/inf 손실 횟수, 최저 loss scale, 가장 긴 저 scale 연속 길이, 등록된 발산
판정(있으면). 런 로그(train.jsonl, epochs.jsonl)만 읽으며 GPU는 쓰지 않는다. 결과는
reports/final/training_summary.json과 .md에 쓴다.

    uv run python scripts/training_summary.py [--out reports/final/training_summary]
"""

# [왜 필요한가] 7단계 최종 보고에 'fp16 발산 횟수 표'와 'GPU 시간 사용 내역'을 넣기로
# 등록했다(docs/experiments.md 7단계 보고 항목). 단계별 문서의 시간 값은 기준이 서로 달라서(정상 속도 환산,
# 부하 포함 벽시계, 실행 시간), 같은 기준(epoch 학습 시간 합)으로 다시 센 값이 필요했다. 결과: 유지한 17런의
# 학습 계산 합 53.52 h(예: Run A 8.92 h), 규칙 변경 13 이후 비유한 손실 0회, step 300 이후 최저 scale 32~512,
# 발산 판정 3런은 모두 dev 손실 규칙(정렬 시작 안 됨)이었다.
# [연결] 입력은 stt_scratch.train이 쓴 epochs.jsonl·train.jsonl·meta.json. README의 학습·평가 명령 목록에
# 있다.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# 저장소 루트. outputs/와 보고서 경로를 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# JSONL을 읽어 행 목록으로 돌려준다(파일이 없으면 빈 목록). Python json은 비유한 실수를 NaN·Infinity라는
# 글자로 쓰는데, 이 글자를 null로 바꿔 읽으므로 그런 값은 None이 된다.
# 주의: 단순 문자열 치환이라 -Infinity는 -null이 되어 읽기에 실패한다. 이번 17런의 로그는 모두 읽혔으므로 해당
# 값은 없었다.
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line.replace("NaN", "null").replace("Infinity", "null")))
    return rows


# 런 하나의 요약: LR, 이어 학습 여부(--init), 완료 epoch 수, 도중 멈춤 여부, 학습 계산 시간(h), GradScaler가
# 건너뛴 step 수, 업데이트 수, NaN/Inf 손실 횟수, 최저 loss scale(전체와 step 300 이후), 가장 긴 저 scale 연속
# 길이, 발산 판정 사유, 정밀도.
def summarize(run: Path) -> dict:
    # epoch 시간이 기록된 행만 쓴다. epoch 도중에 멈춘 런(NaN·scale 규칙이나 --max-steps)은 그 부분 epoch의
    # 시간과 최종 카운터를 담은 "stopped_early" 행을 쓴다. 이 행은 계산 시간과 카운터에는 넣고 완료 epoch
    # 수에서는 뺀다(규칙 변경 25).
    rows = [r for r in read_jsonl(run / "epochs.jsonl") if "epoch_time_s" in r]
    epochs = [r for r in rows if not r.get("stopped_early")]
    steps = read_jsonl(run / "train.jsonl")
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    # 가장 마지막에 기록된 발산 판정을 쓴다.
    divergence = next((r["divergence"] for r in reversed(rows) if r.get("divergence")), None)
    # train.jsonl의 loss scale은 로그 간격(--log-every)마다 남긴 표본이다. 숫자인 값만 모은다.
    scales = [r["scale"] for r in steps if isinstance(r.get("scale"), (int, float))]
    scales_set = set(scales)
    # step 번호 300 이상의 기록만 본다. 시작 직후에는 GradScaler가 큰 시작 scale(기본 65,536)에서 맞는 값까지
    # 일부러 반씩 줄이며 step을 건너뛰므로, 이 구간을 빼야 학습 중의 실제 최저 scale이
    # 보인다(docs/experiments.md 7단계 발산 표 설명).
    late_scales = [r["scale"] for r in steps if r.get("step", 0) >= 300 and r.get("scale") in scales_set]
    last = rows[-1] if rows else {}
    args = meta.get("args", {})
    return {
        "run": run.relative_to(ROOT / "outputs").as_posix(),
        "lr": args.get("lr"),
        "init": bool(args.get("init")),
        "epochs": len(epochs),
        "stopped_mid_epoch": bool(last.get("stopped_early")),
        # 학습 계산 시간 = epoch 학습 시간의 합(dev 평가·일시 중지·재개 때 버린 구간 제외). 마지막 행의
        # 카운터가 그 런의 최종 누적값이다.
        "compute_h": sum(r.get("epoch_time_s", 0.0) for r in rows) / 3600,
        "skipped_steps": last.get("skipped"),
        "updates": last.get("updates"),
        "nan_count": max((r.get("nan_count", 0) for r in rows), default=0),
        "min_scale": min(scales) if scales else None,
        "min_scale_after_300": min(late_scales) if late_scales else None,
        "max_low_scale_streak": max((r.get("low_scale_streak", 0) for r in rows), default=0),
        "divergence": divergence.get("reason") if isinstance(divergence, dict) else divergence,
        "precision": "fp32" if args.get("fp32") else "fp16",
    }


# outputs/<단계>/<런> 가운데 epochs.jsonl이 있는 폴더를 모두 요약해 JSON과 Markdown 표로 쓴다. 인자:
# --out(확장자 없는 출력 경로, 기본 reports/final/training_summary).
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/final/training_summary")
    args = p.parse_args(argv)
    runs = sorted(d for d in (ROOT / "outputs").glob("*/*") if (d / "epochs.jsonl").exists())
    rows = [summarize(r) for r in runs]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["compute_h"] for r in rows)
    diverged = [r["run"] for r in rows if r["divergence"]]
    report = {"runs": rows, "compute_h_total": total, "diverged": diverged}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # 표 칸 형식: 없거나 NaN이면 "-".
    def fmt(v, spec=""):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        return format(v, spec) if spec else str(v)

    lines = [
        "| run | LR | 이어 학습 | epoch | 학습 계산 h | 건너뛴 step | NaN "
        "| 최저 scale(step 300 이후) | 발산 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {fmt(r['lr'])} | {'예' if r['init'] else ''} | {r['epochs']} | "
            f"{r['compute_h']:.2f} | "
            f"{fmt(r['skipped_steps'])} | {r['nan_count']} | {fmt(r['min_scale_after_300'], 'g')} | "
            f"{fmt(r['divergence'])} |"
        )
    # 합계 줄에 시간 기준(epoch 학습 시간만)을 함께 적는다. docs/experiments.md는 이 값(53.52 h)을 기준이 섞인
    # GPU 시간 표(약 65.3 h)와 따로 적었다.
    note = "epoch 학습 시간만; dev 평가·일시 중지·측정·재개로 버린 구간 제외"
    lines += ["", f"학습 계산 합계 {total:.2f} h ({note}), 발산 {len(diverged)}런"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
