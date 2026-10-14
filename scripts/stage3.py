"""3단계 동일 예산 ablation 실행기(docs/experiments.md 3단계, 규칙 변경 17·19). GPU 한 장에서 돈다.

등록된 후보 (a) InterCTC 단독, (b) E-Branchformer, (c) 폭 384, (d) CR-CTC를 우선순위 순서로 하나씩 돌린다.
각 후보는 의도한 차이 하나만 빼고 Run A와 같은 처방으로 처음부터 학습하고, dev_mix full 기준 상위 10개를
평균한 뒤 scripts/stage2_report.py로 평가해 reports/stage3/에 쓰고, 같은 발화에서 Run A 평균 모델과
짝지어 비교한다(scripts/compare_runs.py). 진행 상태는 reports/stage3/status.json에 남긴다.
보호 장치(Codex 2단계 검토):
- 완료 여부는 epoch 로그가 아니라 런의 last.pt로 판단한다. 이미 있는 런(끝났든 중간이든)은 meta.json의
  처방이 등록된 후보 처방과 같을 때만 재사용한다
- 다른 프로세스가 이미 학습 중인 런은 끝날 때까지 기다리며, 같은 런을 두 번 시작하지 않는다
- 평가 단계는 모두 성공하고 산출물 파일을 남겨야 한다. 그렇지 않으면 그 후보를 failed로 표시한다
- 예산: 시작한 후보의 실제 시간 + 남은 후보의 추정 시간. 예산을 넘으면 등록한 대로 남은 후보 가운데
  우선순위가 가장 낮은 것부터 뺀다(d, 그다음 c, b, a)
- (c)는 Run A의 학습셋/dev_H CER 비가 있고 0.5 이상일 때만 돈다
- 발산한 후보는 LR × 0.5로 <후보>-lr<값> 폴더에서 최대 두 번 다시 시작하고(fp16 규칙), 그 LR을 보고한다

사용법: uv run python scripts/stage3.py --arms a,b,c,d --budget-hours 45 [--dry-run]
"""

# [왜 필요한가] 후보마다 Run A와 같은 처방(LR 2e-3, 시드 1, 같은 step 수)에서 의도한 차이 하나만 바꿔야 동일
# 예산 비교가 된다(docs/experiments.md 3단계, 규칙 변경 17). GPU 시간 예산 45 h를 넘으면 우선순위의 끝부터(d →
# c → b → a) 빼고, (c)는 Run A에 과적합 징후가 없을 때만 도는 등 판정 전에 등록한 규칙을 사람이 아니라 코드가
# 집행하게 했다.
# [보호 장치] docstring의 보호 장치 목록은 Codex 2·3단계 검토를 반영한 것이다(규칙 변경 19, 23).
# [결과] (a) 7.9 h, (b) 16.5 h, (d) 2e-3·1e-3 발산 뒤 5e-4로 7.8 h, (c)는 과적합 조건으로 제외. 합계 33.14
# h(발산 시도 포함). 판정은 (a) 채택(InterCTC로 교체), 나머지 불채택.
# [연결] 학습은 stt_scratch.train, 평균은 scripts/average_checkpoints.py, 보고서는 scripts/stage2_report.py,
# 비교는 scripts/compare_runs.py를 하위 프로세스로 부른다. 로그는 work/stage3_<후보>.log.

from __future__ import annotations

# 표준 라이브러리만 쓴다: subprocess(학습·평가 하위 프로세스), os(환경 변수 복사), time(대기·로그 시각).
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# src/를 import 경로 앞에 넣어 이 저장소의 stt_scratch.train에서 처방 키(RECIPE_KEYS), 인자 파서, 완료 epoch
# 수 계산, 등록 처방 계산 함수를 가져온다. train.py와 같은 함수로 처방을 대조해야 두 곳의 정의가 어긋나지
# 않는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stt_scratch.train import RECIPE_KEYS, build_parser, completed_epochs, expected_recipe  # noqa: E402

# 실행 순서이자 우선순위. 예산을 넘으면 목록의 끝(d)부터 뺀다.
PRIORITY = ["a", "b", "c", "d"]
# 모든 후보의 공통 처방: Run A와 같은 peak LR 2e-3, 시드 1, GPU 메모리 상한 0.8(Windows에서 할당기 조각으로
# 메모리가 넘치지 않게 하는 설정; README, 규칙 변경 17).
BASE_LR = 2e-3
BASE = ["--seed", "1", "--cuda-memory-fraction", "0.8"]
# ARMS: 후보별 정의(규칙 변경 17). epochs: 학습 epoch 수, args: Run A와 다른 train.py 인자, factor: Run A
# epoch 대비 예상 시간 배율(예산 추정용, b 1.15, c 1.6), epoch_scale: Run A epoch 대비 epoch 시간 비(epoch당
# step 수 비. d는 배치가 절반이라 epoch당 697 step이 아니라 1,418 step이므로 1,418 ÷ 697).
ARMS = {
    # (a) InterCTC 단독: 6층에 중간 CTC 헤드 하나, 가중치 0.3, softmax 피드백 없음
    "a": {
        "epochs": 100,
        "args": ["--sc-layers", "6", "--sc-weight", "0.3", "--no-sc-feedback"],
        "factor": 1.0,
        "epoch_scale": 1.0,
    },
    # (b) E-Branchformer 블록, 파라미터 수는 Run A의 5 % 안(21.32 M vs 20.79 M)
    "b": {
        "epochs": 100,
        "args": ["--block", "ebranchformer", "--ffn-dim", "768", "--cgmlp-units", "896"],
        "factor": 1.15,
        "epoch_scale": 1.0,
    },
    # (c) 폭 384(FFN 4배 = 1536, 44.5 M). Run A에 과적합 징후가 없을 때만 돈다
    "c": {
        "epochs": 100,
        "args": ["--d-model", "384", "--ffn-dim", "1536"],
        "factor": 1.6,
        "epoch_scale": 1.0,
    },
    # (d) CR-CTC: 두 뷰, alpha 0.2, 시간 마스크 2.5배(5 → 12), max-duration 절반, 49 epoch × 1,418 step
    #     = 69,482 step(Run A 69,700), SpecAugment는 같은 step 2,091부터
    "d": {
        "epochs": 49,
        "args": [
            "--cr-ctc-alpha",
            "0.2",
            "--time-masks",
            "12",
            "--max-duration",
            "150",
            "--augment-start-step",
            "2091",
        ],
        "factor": 1.0,
        "epoch_scale": 1418 / 697,
    },
}
# 하위 프로세스 환경: 로그 인코딩을 UTF-8로 고정하고, PyTorch 할당기가 안 쓰는 캐시 블록을 조금씩 회수하게
# 한다(2단계 Run A가 GPU 메모리 페이징으로 느려진 뒤 정한 설정; docs/experiments.md 2단계 진행 기록).
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256",
}
# --dry-run이면 명령만 찍고 실행하지 않는다(main에서 설정). 실행기 논리를 GPU 없이 확인하는 데 썼다.
DRY_RUN = False


# 후보 하나의 train.py 인자 목록을 만든다: 런 폴더(저장소 기준 상대 경로), epoch 수, LR, 공통 처방, 후보 고유
# 인자. 학습 명령과 처방 대조가 같은 인자 목록을 쓴다.
def arm_argv(arm: str, run: Path, lr: float) -> list[str]:
    spec = ARMS[arm]
    return [
        "--run",
        str(run.relative_to(ROOT)),
        "--epochs",
        str(spec["epochs"]),
        "--lr",
        f"{lr:g}",
        *BASE,
        *spec["args"],
    ]


# 반환: (같은가, 다른 키 → (기록값, 기대값)).
# 왜: 이미 있는 런을 이어 쓰거나 재사용할 때, 다른 처방으로 학습된 런이 섞이면 비교가 무효가 된다(Codex 2단계
# 검토 P1).
def recipe_matches(run: Path, argv: list[str]) -> tuple[bool, dict]:
    """런에 기록된 처방(meta.json의 args)과 argv의 처방이 같은지 대조한다.

    나중에 추가된 처방 키가 기록에 없으면 파서 기본값으로 본다.
    """
    meta = run / "meta.json"
    # meta.json이 없을 때: 체크포인트도 없으면 아직 시작 안 한 빈 런이라 통과, 체크포인트만 있으면 처방을
    # 확인할 수 없으므로 불일치로 본다(Codex 3단계 검토 반영).
    if not meta.exists():
        has_ckpt = run.exists() and any(run.glob("*.pt"))
        return (not has_ckpt), ({"meta.json": ("missing", "required")} if has_ckpt else {})
    recorded = json.loads(meta.read_text(encoding="utf-8"))["args"]
    parser = build_parser()
    want = expected_recipe(argv)
    diff = {}
    # 나중에 추가된 처방 키는 기록에 없을 수 있으므로 파서 기본값으로 본다. 부동소수 인자(LR 등)는 문자열 표기
    # 차이를 피하려고 float로 바꿔 비교한다.
    for k in RECIPE_KEYS:
        have = recorded.get(k, parser.get_default(k))
        same = have == want[k] or (isinstance(want[k], float) and have is not None and float(have) == want[k])
        if not same:
            diff[k] = (have, want[k])
    return not diff, diff


# 이 런 폴더로 stt_scratch.train을 돌리는 python 프로세스가 지금 있는지 PowerShell(Win32_Process의 명령줄)로
# 확인한다. Windows 전용이다.
# 왜: (a)는 이 실행기보다 먼저 직접 시작했다(규칙 변경 19). 같은 런을 두 프로세스가 동시에 학습하지 않도록,
# 살아 있으면 끝날 때까지 기다린다. 확인 자체가 실패하면 '없음'으로 본다.
def training_alive(run: Path) -> bool:
    rel = str(run.relative_to(ROOT)).replace("\\", "/")
    query = (
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
        "$_.CommandLine -match 'stt_scratch.train' -and "
        f"(($_.CommandLine -replace '\\\\','/') -match '{rel}( |$)') }}).Count"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=60
        ).stdout.strip()
        return int(out or "0") > 0
    except Exception:  # noqa: BLE001
        return False


# 일시 중지·부하 구간이 포함된 실제 사용 시간으로, 등록된 예산 계산의 '이미 쓴 시간'이다(규칙 변경 19).
def actual_hours(run: Path) -> float:
    """런 시작(meta.json 생성 시각)부터 가장 최근 epoch 체크포인트까지의 벽시계 시간(h, 일시 중지 포함)."""
    pts = list(run.glob("epoch-*.pt"))
    if not pts or not (run / "meta.json").exists():
        return 0.0
    return (max(p.stat().st_mtime for p in pts) - (run / "meta.json").stat().st_mtime) / 3600


# 체크포인트 저장 시각 간격의 중앙값(h) = 정상 속도의 epoch 시간. 일시 중지나 느린 구간에 끌려가지 않게 평균이
# 아니라 중앙값을 쓴다. Run A 기준값으로 남은 후보의 시간을 추정한다.
def median_epoch_hours(run: Path) -> float:
    stamps = sorted(p.stat().st_mtime for p in run.glob("epoch-*.pt"))
    if len(stamps) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:], strict=False))
    return gaps[len(gaps) // 2] / 3600


# 후보의 예상 학습 시간 = Run A epoch 시간 × epoch 수 × epoch 길이 비 × 후보 계수.
def estimate_hours(arm: str, ref_epoch_h: float) -> float:
    spec = ARMS[arm]
    return ref_epoch_h * spec["epochs"] * spec["epoch_scale"] * spec["factor"]


# 명령을 저장소 루트에서 ENV로 실행하고 출력을 후보별 로그 파일에 덧붙인다(명령과 시각을 먼저 적음). 반환값은
# 종료 코드. dry-run이면 명령만 찍고 0을 돌려준다.
def sh(cmd: list[str], log: Path) -> int:
    if DRY_RUN:
        print("DRY", " ".join(cmd), flush=True)
        return 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


# epoch 기록에 발산 판정이나 조기 종료가 하나라도 있으면 True. 발산한 런은 train.py가 재개를 거부하므로 새
# 이름(낮춘 LR)으로 다시 시작해야 한다.
def diverged(run: Path) -> bool:
    f = run / "epochs.jsonl"
    if not f.exists():
        return False
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    return any(r.get("divergence") or r.get("stopped_early") for r in rows)


# (c) 폭 384 후보를 학습하기 전에 쓴다. 8.0 GB를 넘으면 메모리 예산 초과로 (c)를 빼기로 등록했다(규칙 변경
# 17). 실제로는 과적합 조건으로 먼저 빠져 실행되지 않았다.
# 벤치 폴더는 매번 지우고 새로 시작한다(train.py는 기존 로그가 있는 폴더에서 --resume 없이 시작하지 않으므로).
def bench_peak_gb(spec: dict, log: Path) -> float | None:
    """후보를 스크래치 폴더(work/stage3_bench_c)에서 300 step만 학습해 피크 할당 GPU 메모리(GB)를 돌려준다."""
    bench = ROOT / "work" / "stage3_bench_c"
    if bench.exists():
        import shutil

        shutil.rmtree(bench)
    cmd = [
        sys.executable,
        "-m",
        "stt_scratch.train",
        "--run",
        str(bench.relative_to(ROOT)),
        "--epochs",
        "1",
        "--max-steps",
        "300",
        "--log-every",
        "50",
        "--lr",
        f"{BASE_LR:g}",
        *BASE,
        *spec["args"],
    ]
    if sh(cmd, log) != 0:
        return None
    if DRY_RUN:
        return 0.0
    rows = [json.loads(x) for x in (bench / "train.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return max((r.get("mem_gb") or 0.0) for r in rows) if rows else None


# 처음은 LR 2e-3이고, 재시작은 <후보>-lr<값> 폴더에서 한다(fp16 규칙의 발산 처리, 규칙 변경 19).
def train_arm(arm: str, status: dict, log: Path) -> Path | None:
    """후보 하나를 학습(또는 재사용)한다. 발산하면 LR × 0.5로 최대 두 번 다시 시작한다.

    반환: 끝까지 학습한 시도의 런 폴더, 실패하면 None.
    """
    lr = BASE_LR
    for attempt in range(3):
        name = arm if attempt == 0 else f"{arm}-lr{lr:g}"
        run = ROOT / "outputs" / "stage3" / name
        argv = arm_argv(arm, run, lr)
        # 폴더에 다른 처방의 런이 있으면 덮어쓰지 않고 멈춘다.
        ok, diff = recipe_matches(run, argv)
        if not ok:
            raise SystemExit(f"{run} holds a run with a different recipe: {diff}")
        # 다른 프로세스가 이 런을 학습 중이면(예: 이 실행기 밖에서 직접 시작한 (a)) 2분 간격으로 끝날 때까지
        # 기다린다.
        while not DRY_RUN and training_alive(run):
            time.sleep(120)
        # 체크포인트 기준으로 이미 다 학습된 런이면 그대로 쓴다.
        if completed_epochs(run) >= ARMS[arm]["epochs"]:
            status[arm].update({"run": name, "lr": lr})
            return run
        # 발산 기록이 없으면 학습한다. 완료 epoch 체크포인트(last.pt)가 있으면 --resume로 이어서 한다(데이터
        # 순서·난수·옵티마이저까지 복원되므로 처방이 같다).
        if not diverged(run):
            resume = ["--resume"] if (run / "last.pt").exists() else []
            code = sh([sys.executable, "-m", "stt_scratch.train", *argv, *resume], log)
            if DRY_RUN:
                status[arm].update({"run": name, "lr": lr})
                return run
            if code == 0 and completed_epochs(run) >= ARMS[arm]["epochs"]:
                status[arm].update({"run": name, "lr": lr})
                return run
        # 끝나지도 않았고 발산 기록도 없으면(예: 프로세스 오류) 원인을 모르므로 재시작하지 않고 실패로 남긴다.
        if not diverged(run):
            status[arm]["error"] = f"{name} ended without finishing and without a divergence record"
            return None
        # 발산한 시도를 기록하고(예산 합계에도 들어간다) LR을 절반으로 낮춰 다음 시도로 간다.
        status[arm].setdefault("diverged_attempts", []).append({"run": name, "lr": lr})
        lr /= 2
    return None


# 끝난 후보를 평가한다: 상위 10개 평균 → 2단계 보고서 스크립트(청크 8·16만, 스트리밍 점검 없음, 학습셋 표본
# 1,000) → Run A 평균 모델과 dev_mix·dev_h full 가설의 짝지은 비교. 이미 만든 산출물은 건너뛴다.
# 각 단계는 종료 코드 0이고 기대한 산출물 파일이 생겨야 성공이다. 하나라도 실패하면 False(Codex 2단계 검토:
# 평가 실패를 완료로 적지 않음).
def evaluate_arm(run: Path, reference: str, log: Path) -> bool:
    py, rel = sys.executable, str(run.relative_to(ROOT))
    name = f"{run.name}-avg10"
    rep = ROOT / "reports" / "stage3"
    steps = []
    if not (run / "avg10.pt").exists():
        steps.append(
            (
                [py, "scripts/average_checkpoints.py", "--run", rel, "--k", "10", "--out", f"{rel}/avg10.pt"],
                run / "avg10.pt",
            )
        )
    if not (rep / f"{name}.json").exists():
        steps.append(
            (
                [
                    py,
                    "scripts/stage2_report.py",
                    "--checkpoint",
                    f"{rel}/avg10.pt",
                    "--name",
                    name,
                    "--report-dir",
                    "reports/stage3",
                    "--chunks",
                    "8,16",
                    "--streaming-check",
                    "0",
                    "--train-sample",
                    "1000",
                ],
                rep / f"{name}.json",
            )
        )
    for s in ("dev_mix", "dev_h"):
        steps.append(
            (
                [
                    py,
                    "scripts/compare_runs.py",
                    "--a",
                    f"reports/stage3/hyps/{name}-{s}-full.jsonl",
                    "--b",
                    f"reports/stage2/hyps/{reference}-{s}-full.jsonl",
                    "--name",
                    f"{run.name}-vs-runA-{s}",
                    "--out-dir",
                    "reports/stage3",
                ],
                rep / f"compare-{run.name}-vs-runA-{s}.json",
            )
        )
    for cmd, expected in steps:
        if sh(cmd, log) != 0 or (not DRY_RUN and not expected.exists()):
            return False
    return True


# 후보를 우선순위대로 돌린다. 인자: --arms(돌릴 후보), --budget-hours(등록 예산 45 h), --reference(비교 기준
# 2단계 보고서 이름, runA-avg10), --dry-run.
# 상태 파일을 읽고 이어서 진행하므로, 중간에 멈춰도 다시 실행하면 끝난 후보는 대조만 하고 건너뛴다.
def main() -> None:
    global DRY_RUN
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="a,b,c,d")
    p.add_argument("--budget-hours", type=float, default=45.0)
    p.add_argument("--reference", default="runA-avg10", help="비교 기준 모델의 2단계 보고서 이름")
    p.add_argument(
        "--dry-run", action="store_true", help="명령만 출력하고 실행하지 않는다. 상태 파일은 work/에 쓴다"
    )
    args = p.parse_args()
    DRY_RUN = args.dry_run
    rep = ROOT / "reports" / "stage3"
    (ROOT / "outputs" / "stage3").mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    # dry-run 상태는 커밋되지 않는 work/에, 실제 상태는 reports/stage3/status.json에 쓴다.
    status_path = (ROOT / "work" / "stage3_status_dry.json") if DRY_RUN else rep / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    ref_path = ROOT / "reports" / "stage2" / f"{args.reference}.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    # 예산 추정의 기준: Run A의 정상 속도 epoch 시간.
    ref_epoch_h = median_epoch_hours(ROOT / "outputs" / "stage2" / "runA")
    requested = [a for a in PRIORITY if a in args.arms.split(",")]

    def save() -> None:
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")

    for arm in requested:
        status.setdefault(arm, {})
    # (c)의 등록 조건: Run A의 학습셋 CER ÷ dev_H CER 비가 있고 0.5 이상일 때만(과적합 징후 없음) 돈다. 실측
    # 비는 0.008이라 (c)는 빠졌다.
    ratio = ref.get("train_sample", {}).get("ratio_to_dev_h")
    if "c" in requested and status["c"].get("state") != "done" and (ratio is None or ratio < 0.5):
        status["c"] = {"state": "skipped", "reason": f"overfitting condition: train/dev_H CER ratio {ratio}"}
    save()

    # 발산 시도도 예산에 넣는다(Codex 3단계 검토 반영).
    def used_hours() -> float:
        """이 단계가 시작한 모든 런(발산한 시도 포함)의 실제 사용 시간 합(h)."""
        runs = set()
        for s in status.values():
            if isinstance(s, dict):
                if s.get("run"):
                    runs.add(s["run"])
                runs.update(a["run"] for a in s.get("diverged_attempts", []))
        return sum(actual_hours(ROOT / "outputs" / "stage3" / r) for r in runs)

    for arm in requested:
        # 완료로 적힌 후보도 다시 실행할 때마다 처방과 완료 epoch를 대조해, 기록과 실제 런이 다르면 멈춘다.
        if status[arm].get("state") == "done":
            run = ROOT / "outputs" / "stage3" / status[arm]["run"]
            ok, diff = recipe_matches(run, arm_argv(arm, run, status[arm]["lr"]))
            if not ok or completed_epochs(run) < ARMS[arm]["epochs"]:
                raise SystemExit(f"arm {arm} is marked done but {run} does not match: {diff}")
            continue
        if status[arm].get("state") in ("skipped", "failed"):
            continue
        # 예산 판정(규칙 변경 19): 이미 쓴 시간 + 아직 남은 후보들의 추정 시간이 예산을 넘으면, 남은 후보
        # 가운데 우선순위가 가장 낮은 것부터 하나씩 빼며 예산 안에 들 때까지 반복한다.
        used = used_hours()
        pending = [a for a in requested if status[a].get("state") not in ("done", "skipped", "failed")]
        while pending:
            estimate = sum(estimate_hours(a, ref_epoch_h) for a in pending)
            if used + estimate <= args.budget_hours:
                break
            dropped = pending.pop()  # 우선순위가 낮은 것부터: d, 그다음 c, b, a
            status[dropped] = {
                "state": "skipped",
                "reason": f"budget: used {used:.1f} h + estimate {estimate:.1f} h > {args.budget_hours} h",
            }
        status["budget"] = {"limit_h": args.budget_hours, "used_h": used, "ref_epoch_h": ref_epoch_h}
        save()
        if status[arm].get("state") == "skipped":
            continue
        log = ROOT / "work" / f"stage3_{arm}.log"
        # (c)는 학습 전에 300 step 메모리 벤치를 먼저 돌려 8.0 GB 조건을 확인한다.
        if arm == "c" and not (ROOT / "outputs" / "stage3" / "c").exists():
            peak = bench_peak_gb(ARMS["c"], log)
            status["c"]["bench_peak_gb"] = peak
            if peak is None or peak > 8.0:
                status["c"].update(
                    {"state": "skipped", "reason": f"memory: 300-step bench peak {peak} GB > 8.0"}
                )
                save()
                continue
        # 학습 → 실제 시간 기록 → 평가. 어느 단계든 실패하면 failed로 남기고 다음 후보로 간다.
        run = train_arm(arm, status, log)
        if run is None:
            status[arm]["state"] = "failed"
            save()
            continue
        status[arm]["actual_hours"] = actual_hours(run)
        if not evaluate_arm(run, args.reference, log):
            status[arm].update({"state": "failed", "error": "evaluation step failed"})
            save()
            continue
        status[arm]["state"] = "done"
        save()
        print(f"arm {arm} done ({run.name})", flush=True)
    # 모든 후보를 처리한 뒤 최종 사용 시간을 다시 계산해 남긴다(docs/experiments.md의 3단계 합계 33.14 h가 이
    # 값이다).
    status["budget"] = {"limit_h": args.budget_hours, "used_h": used_hours(), "ref_epoch_h": ref_epoch_h}
    save()


if __name__ == "__main__":
    main()
