"""4단계 방출 지연 제어 이어 학습 실행기(docs/experiments.md 4단계, 규칙 변경 16·18·20).

시작 모델(Run A의 평균 모델, 또는 3단계 승자의 평균 모델과 그 후보의 구조·손실 옵션)에서 Peak-First λ를
보정하고, 같은 처방으로 세 런(통제, TrimTail 200 ms, Peak-First)을 15 epoch씩 이어 학습한다. 각 런을
평균하고(15개 중 dev_mix 상위 10개) scripts/stage4_latency.py로 지연과 CER을 잰 뒤 판정을
reports/stage4/summary.json에 쓴다.
  주 판정: 서비스 청크 320 ms의 dev_mix. 통제 대비 CER_ns 상대 증가 ≤ 3 %이고 dev_H가 0.3포인트보다 더
  나빠지지 않으면 자격이 있다. 자격 있는 변형 가운데 FTD P50 감소(> 0)가 가장 큰 것을 채택하며, 그 감소가
  80 ms 이상이면 H4 충족이다. 그런 변형이 없으면 통제를 유지한다. 640 ms 수치와, 두 모델 모두 첫 어절을
  맞힌 발화에서의 FTD는 보고만 하고 판정을 바꾸지 않는다.
예전 시작 모델로 만든 파일은 재사용하지 않는다(모든 출력에 시작 체크포인트의 SHA-256을 기록한다).

사용법: uv run python scripts/stage4.py --start runA      (3단계 승자에서 시작하면 --start a|b|d)
"""

# [왜 필요한가] 스트리밍 자막은 토큰이 늦게 나오면 체감 지연이 커진다. 지연 감소 기법 두 가지를 같은 조건에서
# 통제군과 비교해, CER을 거의 잃지 않으면서 첫 어절 지연(FTD)을 가장 많이 줄이는 것을 고르는
# 단계다(docs/experiments.md 4단계, 규칙 변경 16·18·20). 결과: TrimTail 채택(FTD P50 −10 ms, CER 상대 +0.28
# %), H4(80 ms 이상 감소) 미충족, Peak-First는 −50 ms였지만 CER 상대 +3.16 %로 탈락. 최종 모델이
# outputs/stage4/trimtail/avg10.pt다.
# [보호 장치] 시작 체크포인트 SHA-256 기록과 재사용 금지는 결과 파일뿐 아니라 런 폴더에도 적용한다(Codex
# 2·3단계 검토, 규칙 변경 20·23). --start는 필수다.
# [연결] 하위 프로세스로 scripts/calibrate_peak_first.py, stt_scratch.train(--init 이어 학습),
# scripts/average_checkpoints.py, scripts/stage4_latency.py를 부른다. 3단계 후보 정의는 scripts/stage3.py의
# ARMS에서 가져온다. 로그는 work/stage4.log.

from __future__ import annotations

# 표준 라이브러리만 쓴다: hashlib(시작 체크포인트 SHA-256), statistics(공통 발화 FTD 중앙값), subprocess(하위
# 실행).
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# src/와 scripts/를 import 경로에 넣어 3단계 후보 정의(ARMS)를 그대로 가져온다. 시작 모델이 3단계 후보이면 그
# 후보의 구조·손실 옵션을 이어 학습에도 똑같이 써야 하기 때문이다(규칙 변경 20·22).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stage3 import ARMS  # noqa: E402

# 하위 프로세스 환경: UTF-8 로그, 3단계와 같은 PyTorch 할당기 설정.
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}
# RECIPE: 세 변형이 공유하는 이어 학습 처방(규칙 변경 16). --init으로 가중치만 가져오고 옵티마이저·스케줄은
# 새로 시작한다. 이미 정렬된 모델이므로 증강을 처음부터 켜고, warmup 500, 15 epoch 코사인, peak LR 1e-3(2단계
# 2e-3의 절반)이며 dev 손실 발산 규칙은 쓰지 않는다. 세 변형의 차이는 TrimTail·Peak-First 옵션 하나뿐이다.
RECIPE = [
    "--epochs", "15", "--lr", "1e-3", "--warmup", "500", "--seed", "1",
    "--augment-start-epoch", "0", "--perturb-start-epoch", "0", "--cuda-memory-fraction", "0.8",
]  # fmt: skip


# 파일의 SHA-256(1 MiB씩 읽어 큰 체크포인트도 메모리에 다 올리지 않음). 시작 체크포인트의 지문으로 쓴다.
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# 명령을 저장소 루트에서 실행하고 출력을 work/stage4.log에 덧붙인다(명령과 시각을 먼저 적음). 종료 코드를
# 돌려준다.
def sh(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


# runA면 2단계 Run A 평균(추가 옵션 없음). 3단계 후보(a, b, d)면 reports/stage3/status.json에서 완료된 런을
# 찾아 그 평균을 쓰고, 후보 옵션을 가져온다. 실제 4단계는 --start a(InterCTC 단독)로 돌았다(규칙 변경 22).
def start_model(start: str) -> tuple[Path, list[str]]:
    """시작 모델의 평균 체크포인트 경로와 그 모델의 구조·손실 옵션을 돌려준다."""
    if start == "runA":
        return ROOT / "outputs" / "stage2" / "runA" / "avg10.pt", []
    status = json.loads((ROOT / "reports" / "stage3" / "status.json").read_text(encoding="utf-8"))
    if status.get(start, {}).get("state") != "done":
        raise SystemExit(f"Stage 3 arm {start} is not done")
    run = status[start]["run"]
    args = list(ARMS[start]["args"])
    # (d)의 SpecAugment 시작 step 옵션은 빼서 이어 학습이 처음부터 증강하게 한다(규칙 변경 16).
    if "--augment-start-step" in args:
        i = args.index("--augment-start-step")
        del args[i : i + 2]
    return ROOT / "outputs" / "stage3" / run / "avg10.pt", args


# 다른 시작 모델의 파일은 지우지 않고 .stale-<시각>.json으로 이름을 바꿔 치운 뒤 False를 돌려준다(그러면
# 호출자가 다시 계산한다).
def fresh(path: Path, init_sha: str) -> bool:
    """`path`가 있고 같은 시작 체크포인트(SHA-256)로 만들어졌으면 True. 오래된 파일은 이름을 바꿔 둔다."""
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf-8")).get("init_sha256") == init_sha:
        return True
    path.rename(path.with_suffix(f".stale-{int(time.time())}.json"))
    return False


# λ 보정 → 세 변형 학습·평균·측정 → 판정. 인자: --start(runA 또는 끝난 3단계 후보 a·b·d).
# 중간에 멈춰도 다시 실행하면 끝난 학습·평균·측정은 대조 뒤 재사용하고 나머지만 한다.
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--start", required=True, help="시작 모델: runA 또는 끝난 3단계 후보(a, b, d). 규칙 변경 22"
    )
    args = p.parse_args()
    py = sys.executable
    init, model_args = start_model(args.start)
    init_rel, init_sha = str(init.relative_to(ROOT)), sha256(init)
    out_root, rep = ROOT / "outputs" / "stage4", ROOT / "reports" / "stage4"
    out_root.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    log = ROOT / "work" / "stage4.log"

    # Peak-First λ: 같은 시작 모델로 이미 계산한 파일이 있으면 쓰고, 없으면 보정 스크립트를 돌린다. (d)처럼
    # 배치 크기가 다른 후보는 그 max-duration으로 보정해 학습 조건을 맞춘다. 결과 파일에 시작 체크포인트
    # SHA-256을 덧붙인다(실제 값 λ = 1.62e-3).
    lam_path = rep / "peak_first_lambda.json"
    lambda_was_fresh = fresh(lam_path, init_sha)
    if not lambda_was_fresh:
        cmd = [py, "scripts/calibrate_peak_first.py", "--checkpoint", init_rel]
        if "--max-duration" in model_args:
            cmd += ["--max-duration", model_args[model_args.index("--max-duration") + 1]]
        if sh(cmd, log) != 0:
            raise SystemExit("lambda calibration failed")
        data = json.loads(lam_path.read_text(encoding="utf-8"))
        data["init_sha256"] = init_sha
        lam_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    lam = json.loads(lam_path.read_text(encoding="utf-8"))["lambda"]
    # 세 변형의 추가 옵션. TrimTail(arXiv 2211.00522)은 발화 끝을 0~200 ms(mel 0~20프레임) 무작위로 잘라
    # 학습해, 뒤쪽 음성 없이도 토큰을 내도록 유도한다. Peak-First(arXiv 2211.03284)는 각 프레임 분포를 다음
    # 프레임 쪽으로 당기는 정규화 항으로 CTC 피크를 앞당긴다(규칙 변경 18, model/features.py trim_tail,
    # model/ctc.py).
    variants = {
        "control": [],
        "trimtail": ["--trimtail-ms", "200"],
        "peakfirst": ["--peak-first-lambda", f"{lam:.6g}"],
    }
    summary = {"start": args.start, "init": init_rel, "init_sha256": init_sha, "lambda": lam, "variants": {}}
    sys.path.insert(0, str(ROOT / "src"))
    from stt_scratch.train import RECIPE_KEYS, build_parser, completed_epochs, expected_recipe

    # 조건을 어기면 SystemExit로 멈춘다(Codex 3단계 검토 P1). 체크포인트가 있는데 meta.json이 없는 런도
    # 거부한다.
    def check_existing(run: Path, argv: list[str]) -> None:
        """기존 런은 바로 이 시작 체크포인트로 학습됐고 기록된 처방이 등록 처방과 같을 때만 재사용·재개한다.

        시작 체크포인트는 런 폴더의 SHA-256 파일(init_sha256.txt)로 확인한다. 그 파일이 생기기 전에 시작한
        예전 런은 이 시작 모델의 λ 파일이 이번 실행에서 새로 만들어지지 않았을 때(= 같은 시작 모델로 보정된
        상태)만 허용한다.
        """
        meta = run / "meta.json"
        if not meta.exists():
            if any(run.glob("*.pt")):
                raise SystemExit(f"{run} has checkpoints but no meta.json")
            return
        sidecar = run / "init_sha256.txt"
        if sidecar.exists():
            if sidecar.read_text(encoding="utf-8").strip() != init_sha:
                raise SystemExit(f"{run} was trained from a different starting checkpoint")
        elif not lambda_was_fresh:
            raise SystemExit(f"{run} predates the current starting checkpoint; move it away first")
        recorded = json.loads(meta.read_text(encoding="utf-8"))["args"]
        parser, want = build_parser(), expected_recipe(argv)
        diff = {
            k: (recorded.get(k, parser.get_default(k)), want[k])
            for k in RECIPE_KEYS
            if recorded.get(k, parser.get_default(k)) != want[k]
            and not (isinstance(want[k], float) and float(recorded.get(k, 0) or 0) == want[k])
        }
        if diff:
            raise SystemExit(f"{run} holds a run with a different recipe: {diff}")

    # 변형마다: 학습 인자 = 런 폴더 + 공통 처방 + 시작 모델 옵션 + 변형 옵션 + --init(시작 가중치).
    for name, extra in variants.items():
        run = out_root / name
        argv = ["--run", str(run.relative_to(ROOT)), *RECIPE, *model_args, *extra, "--init", init_rel]
        check_existing(run, argv)
        # 15 epoch가 다 끝나지 않았으면 학습한다. 완료 epoch 체크포인트가 있으면 --init을 그대로 둔 채
        # --resume로 이어 간다(규칙 변경 20). 시작 체크포인트 SHA-256을 런 폴더에 먼저 남겨 나중의 재사용
        # 검사에 쓴다.
        if completed_epochs(run) < 15:
            resume = ["--resume"] if (run / "last.pt").exists() else []
            run.mkdir(parents=True, exist_ok=True)
            (run / "init_sha256.txt").write_text(init_sha + "\n", encoding="utf-8")
            cmd = [py, "-m", "stt_scratch.train", *argv, *resume]
            if sh(cmd, log) != 0 or completed_epochs(run) < 15:
                raise SystemExit(f"variant {name} did not finish")
        # 15 epoch 중 dev_mix 상위 10개 평균을 만들고(규칙 변경 18: 평가 모델은 15 epoch 중 상위 10개 평균),
        # 지연·CER 측정 결과가 같은 시작 모델의 것이 아니면 다시 잰다. 측정 파일에도 시작 체크포인트 SHA-256을
        # 덧붙인다.
        avg = run / "avg10.pt"
        if not avg.exists() and sh([py, "scripts/average_checkpoints.py", "--run", str(run.relative_to(ROOT)),
                                    "--k", "10", "--out", str(avg.relative_to(ROOT))], log) != 0:  # fmt: skip
            raise SystemExit(f"averaging {name} failed")
        lat = rep / f"latency-{name}.json"
        if not fresh(lat, init_sha):
            if sh(
                [py, "scripts/stage4_latency.py", "--checkpoint", str(avg.relative_to(ROOT)), "--name", name],
                log,
            ):
                raise SystemExit(f"latency measurement of {name} failed")
            data = json.loads(lat.read_text(encoding="utf-8"))
            data["init_sha256"] = init_sha
            lat.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["variants"][name] = json.loads(lat.read_text(encoding="utf-8"))

    # 한 변형의 측정 결과에서 (세트, 청크 ms)에 해당하는 행을 꺼낸다.
    def pick(variant: str, s: str, chunk_ms: int) -> dict:
        return next(
            r for r in summary["variants"][variant]["results"] if r["set"] == s and r["chunk_ms"] == chunk_ms
        )

    # 320·640 ms 각각, 변형마다 통제 대비 수치를 계산한다(dev_mix 기준, dev_H는 가드):
    # - ftd_p50_reduction_ms: 통제 FTD P50 − 변형 FTD P50. 각 모델이 첫 어절을 맞힌 발화 집단끼리의 P50
    #   차이다.
    # - ftd_p50_reduction_common_ms: 두 모델 모두 첫 어절을 맞힌 공통 발화에서의 FTD 중앙값 차이(보조 보고).
    #   TrimTail의 −10 ms가 공통 발화에서는 0 ms였다.
    # - cer_relative_change: dev_mix CER_ns 상대 변화, dev_h_cer_change_pp: dev_H CER 변화(퍼센트 포인트).
    # - eligible: CER 상대 증가 ≤ 3 %이고 dev_H 악화 ≤ 0.3포인트.
    table = {}
    for chunk_ms in (320, 640):
        c_mix, c_h = pick("control", "dev_mix", chunk_ms), pick("control", "dev_h", chunk_ms)
        for name in ("trimtail", "peakfirst"):
            v_mix, v_h = pick(name, "dev_mix", chunk_ms), pick(name, "dev_h", chunk_ms)
            common = sorted(set(c_mix["ftd_by_utt_ms"]) & set(v_mix["ftd_by_utt_ms"]))
            common_gain = (
                statistics.median(c_mix["ftd_by_utt_ms"][u] for u in common)
                - statistics.median(v_mix["ftd_by_utt_ms"][u] for u in common)
                if common
                else None
            )
            dev_h_change = (v_h["cer_ns"] - c_h["cer_ns"]) * 100
            cer_rel = v_mix["cer_ns"] / c_mix["cer_ns"] - 1
            table[f"{name}@{chunk_ms}ms"] = {
                "ftd_p50_reduction_ms": c_mix["ftd_ms"]["p50"] - v_mix["ftd_ms"]["p50"],
                "ltd_p50_reduction_ms": c_mix["ltd_ms"]["p50"] - v_mix["ltd_ms"]["p50"],
                "ftd_n_control": c_mix["ftd_ms"]["n"],
                "ftd_n_variant": v_mix["ftd_ms"]["n"],
                "ftd_p50_reduction_common_ms": common_gain,
                "ftd_common_n": len(common),
                "cer_relative_change": cer_rel,
                "dev_h_cer_change_pp": dev_h_change,
                "eligible": cer_rel <= 0.03 and dev_h_change <= 0.3,
            }
    # 판정(규칙 변경 20): 서비스 청크 320 ms의 dev_mix에서 조건을 통과하고 FTD P50 감소가 0보다 큰 변형 가운데
    # 감소가 가장 큰 것을 채택하고, 없으면 통제를 유지한다. 채택 변형의 감소가 80 ms 이상이면 H4 충족. 640
    # ms와 공통 발화 수치는 보고만 하고 판정을 바꾸지 않는다.
    candidates = [
        (table[f"{n}@320ms"]["ftd_p50_reduction_ms"], n)
        for n in ("trimtail", "peakfirst")
        if table[f"{n}@320ms"]["eligible"] and table[f"{n}@320ms"]["ftd_p50_reduction_ms"] > 0
    ]
    adopted = max(candidates)[1] if candidates else "control"
    summary["table"] = table
    summary["verdict"] = {
        "adopted": adopted,
        "h4_met": bool(candidates) and max(candidates)[0] >= 80,
        "basis": "service chunk 320 ms, dev_mix FTD P50; eligibility CER +3 % relative and dev_H +0.3 points",
    }
    # 변형별 측정 결과 전체, 비교 표, 판정을 reports/stage4/summary.json에 쓰고 표와 판정을 출력한다.
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"table": table, "verdict": summary["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
