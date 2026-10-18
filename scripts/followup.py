"""7단계 봉인 test 뒤의 후속 측정 도구(docs/experiments.md "7단계 뒤 후속 측정", 규칙 변경 26).

봉인 test 수치(12층·greedy·1-pass)와 무관한 dev 전용 후속 측정이다. 하위 명령마다 결과를
<out-dir>/<명령>.json(기본 reports/followup/)에 쓰고, 이미 있으면 --overwrite 없이는 거부한다.
  gate      P3 출력 동일성 관문: fast·graph step 경로 vs eager(6·7단계 경로). dev_H 앞 30발화의 청크별
            log-prob 차이(GPU fp16)와 CPU fp32 5발화, dev_H 전수의 실제 스트리밍 전사 일치
  timing    P3·P4 청크당 계산 시간: {eager, fast, graph} × {12층, 6층} GPU fp16(dev_H 앞 30발화),
            {eager, fast} × {12층, 6층} CPU fp32 1스레드(앞 10발화). 부하 조건 기록
  exit6     P4 6층 조기 종료(보고만): simulated 320 ms(6층 중간 헤드 vs 12층, 짝지은 구간)와 실제 캐시
            스트리밍 CER(dev_mix·dev_H)
  session   P1 끝점 뒤 전체 문맥 2차 패스 + P2 캐시 유지 상태 rule2 0.8 s: dev_mix 세션(rule2 1.2·0.8 s,
            캐시 유지) → rule2 선택 → dev_H 세션(선택한 rule2). 1-pass(320 ms 스트리밍) vs 2-pass 확정
            자막의 CER·짝지은 구간, EP50/EP90, 2차 패스 계산 시간, 확정 때 바뀐 세그먼트 비율
  final-cpu P1 2차 패스의 CPU fp32 1스레드 계산 시간(dev_mix 앞 50발화 세션)
  summary   위 결과 파일을 모아 등록된 판정을 다시 계산해 summary.json·summary.md로 쓴다

사용법(등록한 순서: gate → timing → exit6 → session → final-cpu → summary; session은 gate 결과를 읽는다):
  uv run --no-sync python scripts/followup.py gate --checkpoint outputs/stage4/trimtail/avg10.pt
CPU 연기 시험: --device cpu --limit 3 --out-dir <임시 폴더> (코드 확인용이며 판정에 쓰지 않는다)
"""

# [왜 필요한가] 봉인 test는 이미 한 번 썼으므로(docs/experiments.md 7단계) 그 뒤의 개선은 dev에서만 주장할 수
# 있고, 다른 단계와 같이 후보·판정 규칙을 먼저 커밋한 뒤 잰다(규칙 변경 26). 측정 코드는 서비스와 같은
# StreamSession·StreamingEncoder·decode_streaming을 그대로 써서 측정값이 곧 서비스 동작을 대표하게 한다.
# [재사용] 세션 잡음(-60 dBFS 3 s), 패킷 규약, MFA 마지막 단어 끝, UPWR/UPSR 정의는 6단계 도구
# (scripts/stage6.py)의 함수를, 짝지은 부트스트랩은 scripts/compare_runs.py의 compare를 그대로 쓴다.
# [봉인 보호] dev_mix·dev_h 매니페스트만 연다(DEV_SETS). 봉인 test 매니페스트는 이 스크립트에서 열 수 없다.

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# numpy: 백분위. torch: 모델 실행·동기화·스레드 수.
import numpy as np
import torch

# 프로젝트 모듈: 평가 경로(오프라인·실제 스트리밍 디코딩, 채점, autocast, 모델 로드), 특징, 청크 설정,
# 스트리밍 인코더(경로·조기 종료), greedy, 세션·끝점·예열, 어휘.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import (
    autocast,
    batch_features,
    decode_streaming,
    duration_batches,
    load_model,
    score,
)
from stt_scratch.model import streaming
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_runs import compare  # noqa: E402
from stage6 import SR, last_word_end, noise, packets, rate, stability  # noqa: E402

DEV_SETS = ("dev_mix", "dev_h")  # 봉인 test 매니페스트는 열지 않는다
SERVICE_SPEC = ChunkSpec(8, 128)  # 6단계 서비스 구성: 320 ms 청크, 좌측 128프레임
STAGE6_SESSION = ROOT / "reports" / "stage6" / "session.json"

# 등록 값(docs/experiments.md 규칙 변경 26). 숫자를 여기 한곳에만 두고 판정 코드가 읽는다.
GATE_MAX_ABS_LOGPROB = 1e-3  # 관문 (가): 청크별 log-prob 최대 절대차(기대값 0, 비트 단위 동일 설계)
THRESHOLD = 0.0186  # 채택 임계 max(0.3, 2s) = 1.86포인트(1단계 s = 0.93)
GUARD = 0.003  # dev_H 가드 0.3포인트
RULE2_MAX_INCREASE = 0.002  # 6단계 rule2 조건: 끝점 끈 기준 대비 증가 ≤ 0.2포인트
FINAL_PASS_GPU_P90_MS = 250.0  # 2차 패스 계산 p90 상한(GPU fp16)
FINAL_PASS_CPU_P90_MS = 500.0  # CPU fp32 1스레드(넘으면 CPU 장치에서는 기본 끔)
REPRO_TOL = 0.0002  # rule2 1.2 s 캐시 유지 1-pass 재측정이 6단계 값과 0.02포인트 안이어야 측정 유효


# --- 공통 도우미 ---------------------------------------------------------------------------------------


def manifest(name: str, limit: int):
    if name not in DEV_SETS:
        raise SystemExit(f"followup measures dev sets only, not {name!r}")
    utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
    return (utts[:limit] if limit else utts), PcmStore(ROOT / "data" / "pcm", name)


# 결과 파일 경로. 이미 있으면(등록 측정 결과를 실수로 덮지 않게) --overwrite 없이는 멈춘다.
def out_file(args, name: str) -> Path:
    path = ROOT / args.out_dir / name
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


# 짝지은 비교(scripts/compare_runs.py)가 읽는 가설 파일 형식: id·ref·hyp·hangul_only.
def hyp_rows(utts, hyps: dict[str, str]) -> dict[str, dict]:
    return {u.id: {"id": u.id, "ref": u.text, "hyp": hyps[u.id], "hangul_only": u.hangul_only} for u in utts}


def write_hyps(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentiles(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


# 결과마다 남기는 출처: 체크포인트 SHA-256, 코드 커밋, 장치, torch 버전, 시각.
def provenance(args, device: torch.device) -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(),
        "git_head": head,
        "git_dirty": dirty,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "limit": args.limit,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --- 부하 조건 기록 -------------------------------------------------------------------------------------
# 지연 측정은 다른 프로그램이 GPU·CPU를 쓰면 느려진다(docs/experiments.md "예산 규칙"의 부하 조건). 등록
# 규칙(규칙 변경 26): 쉬는 상태를 최대 10분 기다리고, 그래도 아니면 그대로 재되 GPU·CPU 사용률과 돌던
# 프로그램을 함께 기록한다(문서에는 프로그램을 일반명으로 적는다).


def _cpu_times() -> tuple[int, int] | None:
    """(유휴, 전체) CPU 시간. Windows는 GetSystemTimes, 리눅스는 /proc/stat. 둘 다 아니면 None."""
    if platform.system() == "Windows":
        idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        return idle.value, kernel.value + user.value  # 커널 시간에 유휴가 들어 있다
    stat = Path("/proc/stat")
    if stat.exists():
        fields = [int(x) for x in stat.read_text().split("\n")[0].split()[1:]]
        return fields[3] + fields[4], sum(fields)
    return None


def cpu_percent(interval: float = 2.0) -> float | None:
    a = _cpu_times()
    time.sleep(interval)
    b = _cpu_times()
    if a is None or b is None or b[1] == a[1]:
        return None
    return 100.0 * (1 - (b[0] - a[0]) / (b[1] - a[1]))


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_state() -> dict:
    """nvidia-smi로 GPU 사용률·메모리와 GPU 계산 프로세스 이름(없으면 빈 목록)을 읽는다."""
    util = _run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    out: dict = {"utilization_pct": None, "memory_used_mb": None, "compute_apps": []}
    if util:
        u, used, total = (v.strip() for v in util.strip().split("\n")[0].split(","))
        out.update(utilization_pct=float(u), memory_used_mb=float(used), memory_total_mb=float(total))
    if apps:
        for line in apps.strip().split("\n"):
            pid, _, name = line.partition(",")
            if pid.strip().isdigit():
                out["compute_apps"].append({"pid": int(pid), "name": Path(name.strip()).name})
    return out


def top_cpu_processes(n: int = 8) -> list[str] | None:
    """2초 동안 CPU 시간을 많이 쓴 프로세스 이름과 그 CPU 초(Windows). 다른 OS나 실패면 None.

    성능 카운터(Get-Counter)는 카운터 이름이 OS 언어에 따라 달라 한국어 Windows에서 실패하므로, 프로세스별
    누적 CPU 시간을 2초 간격으로 두 번 읽어 차이를 낸다(여러 코어를 쓰면 2초를 넘을 수 있다).
    """
    if platform.system() != "Windows":
        return None
    script = (
        "$a=@{}; Get-Process | ForEach-Object { $a[$_.Id]=$_.CPU }; Start-Sleep -Seconds 2; "
        "Get-Process | ForEach-Object { if ($a.ContainsKey($_.Id) -and $_.CPU) "
        "{ [pscustomobject]@{N=$_.ProcessName; D=($_.CPU - $a[$_.Id])} } } | Sort-Object D -Descending | "
        f"Select-Object -First {n} | ForEach-Object {{ '{{0}} {{1:N2}}' -f $_.N, $_.D }}"
    )
    text = _run(["powershell", "-NoProfile", "-Command", script])
    return [line.strip() for line in text.splitlines() if line.strip()] if text else None


def load_snapshot() -> dict:
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu_state(),
        "cpu_pct": cpu_percent(),
        "top_cpu_processes": top_cpu_processes(),
    }


# 다른 학습·평가 작업: GPU 프로세스 목록 가운데 이 프로세스가 아닌 python. Windows(WDDM)의 nvidia-smi는
# 탐색기·브라우저 같은 화면 프로세스도 이 목록에 보여 주므로(실측) 목록이 비었는지로는 판단할 수 없다.
def other_python_gpu_apps(gpu: dict) -> list[dict]:
    return [
        a for a in gpu["compute_apps"] if a["name"].lower().startswith("python") and a["pid"] != os.getpid()
    ]


# GPU 사용률은 순간값이 출렁이므로 1초 간격 5번의 평균을 쓴다.
def gpu_utilization_mean(samples: int = 5) -> float | None:
    values = []
    for _ in range(samples):
        u = gpu_state()["utilization_pct"]
        if u is not None:
            values.append(u)
        time.sleep(1)
    return float(np.mean(values)) if values else None


# 쉬는 상태 = GPU 사용률(5초 평균) ≤ 10 %, 다른 python GPU 프로세스 없음, CPU 전체 사용률 ≤ 25 %. 30초마다
# 보며 최대 minutes분 기다린다. 결과에 '쉬는 상태에서 쟀는지'와 마지막 기록(사용률·돌던 프로그램)을 남긴다.
def wait_for_idle(minutes: float) -> dict:
    start = time.time()
    while True:
        snap = load_snapshot()
        snap["gpu_utilization_mean_pct"] = gpu_utilization_mean()
        snap["other_python_gpu_apps"] = other_python_gpu_apps(snap["gpu"])
        util = snap["gpu_utilization_mean_pct"]
        idle = (
            (util is None or util <= 10)
            and not snap["other_python_gpu_apps"]
            and (snap["cpu_pct"] is None or snap["cpu_pct"] <= 25)
        )
        waited = time.time() - start
        if idle or waited >= minutes * 60:
            return {"idle": idle, "waited_s": round(waited, 1), "snapshot": snap}
        print(
            f"waiting for an idle PC ({waited / 60:.1f} min): GPU {util} %, CPU {snap['cpu_pct']} %, "
            f"python on GPU {snap['other_python_gpu_apps']}",
            flush=True,
        )
        time.sleep(30)


# 요약에 쓰는 부하 한 줄: 쉬는 상태였는지, GPU·CPU 사용률, 다른 python GPU 프로세스, CPU를 많이 쓴 프로그램.
def load_line(load: dict) -> str:
    snap = load["snapshot"]
    others = snap.get("other_python_gpu_apps", [])
    others = len(others) if isinstance(others, list) else others  # 공개 결과 파일에는 개수만 남긴다
    return (
        f"쉬는 상태 {load['idle']}(대기 {load['waited_s']} s), GPU {snap.get('gpu_utilization_mean_pct')} %, "
        f"CPU {fmt(snap['cpu_pct'], digits=0)} %, 다른 python GPU 프로세스 {others}개, "
        f"CPU 상위 {snap['top_cpu_processes']}"
    )


# --- gate: 출력 동일성 관문(P3) ---------------------------------------------------------------------------


# 발화 하나를 두 경로로 청크씩 돌려 청크별 log-prob을 비교한다. mel은 오프라인 LogMel(두 경로에 같은 입력).
@torch.no_grad()
def compare_paths_on_utterance(model, wav, spec, device, path: str, amp: bool) -> dict:
    logmel = LogMel().to(device)
    mel, _ = logmel(wav[None].to(device), torch.tensor([wav.numel()], device=device))
    eager, other = StreamingEncoder(model, spec), StreamingEncoder(model, spec, path=path)
    s_eager, s_other = eager.new_state(), other.new_state()
    step = 4 * spec.chunk_size
    stats = {"steps": 0, "bit_identical_steps": 0, "frames": 0, "argmax_agree": 0, "max_abs": 0.0}
    with autocast(device, amp):
        for i in range(0, mel.shape[1], step):
            a = eager.step(mel[0, i : i + step], s_eager)
            b = other.step(mel[0, i : i + step], s_other)
            stats["steps"] += 1
            stats["bit_identical_steps"] += int(torch.equal(a, b))
            stats["frames"] += a.shape[0]
            stats["argmax_agree"] += int((a.argmax(-1) == b.argmax(-1)).sum())
            stats["max_abs"] = max(stats["max_abs"], float((a - b).abs().max()))
    return stats


def merge_stats(rows: list[dict]) -> dict:
    total = {k: sum(r[k] for r in rows) for k in ("steps", "bit_identical_steps", "frames", "argmax_agree")}
    total["max_abs"] = max((r["max_abs"] for r in rows), default=0.0)
    total["argmax_agreement"] = total["argmax_agree"] / max(1, total["frames"])
    return total


def cmd_gate(args, model, vocab, device) -> None:
    path_out = out_file(args, "gate.json")
    spec = SERVICE_SPEC
    utts, store = manifest("dev_h", args.limit)
    amp = device.type == "cuda"
    result: dict = {"provenance": provenance(args, device), "spec": "chunk8 left128", "paths": {}}
    # (가) 청크별 log-prob: 서비스 장치(GPU fp16)에서 앞 30발화, CPU fp32에서 앞 5발화(CPU의 graph는 같은
    # 고정 버퍼 코드를 그래프 없이 돈다).
    cpu_model = load_model(ROOT / args.checkpoint, torch.device("cpu"))[0] if device.type == "cuda" else None
    for path in ("fast", "graph"):
        rows = [
            compare_paths_on_utterance(model, store.waveform(u.id), spec, device, path, amp)
            for u in utts[: args.logprob_utts]
        ]
        entry: dict = {"logprob": merge_stats(rows)}
        if cpu_model is not None:
            # CPU 비교는 1스레드로 돈다. 다른 프로그램이 CPU를 쓰는 중에 여러 스레드로 돌리면 step 하나가 수
            # 초씩 걸렸다(CPU 연기 시험에서 확인). 값은 스레드 수와 무관하게 두 경로가 같은지만 본다.
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            cpu_rows = [
                compare_paths_on_utterance(
                    cpu_model, store.waveform(u.id), spec, torch.device("cpu"), path, False
                )
                for u in utts[: args.cpu_logprob_utts]
            ]
            torch.set_num_threads(threads)
            entry["logprob_cpu_fp32"] = merge_stats(cpu_rows)
        result["paths"][path] = entry
        print(path, json.dumps(entry), flush=True)
    stepper = streaming._static_stepper(StreamingEncoder(model, spec, path="graph"))
    result["graph_captured"] = stepper.graph is not None
    result["graph_failed"] = stepper.graph_failed
    # (나) dev_H 전수 실제 스트리밍 전사: eager(기준, 6층 비교의 12층 가설로도 쓴다) vs graph, graph가
    # 관문을 통과하지 못하면 fast도.
    logmel = LogMel().to(device)
    t0 = time.perf_counter()
    eager_hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel, amp=True, path="eager")
    result["eager_seconds"] = round(time.perf_counter() - t0, 1)
    result["eager_cer_ns"] = rate(score(utts, eager_hyps))
    write_hyps(ROOT / args.out_dir / "hyps" / "dev_h-stream-l12-eager.jsonl", hyp_rows(utts, eager_hyps))
    for path in ("graph", "fast"):
        entry = result["paths"][path]
        t0 = time.perf_counter()
        hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel, amp=True, path=path)
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        entry["identical_transcripts"] = sum(hyps[u.id] == eager_hyps[u.id] for u in utts)
        entry["utterances"] = len(utts)
        entry["cer_ns"] = rate(score(utts, hyps))
        entry["passed"] = (
            entry["identical_transcripts"] == len(utts)
            and entry["logprob"]["max_abs"] <= GATE_MAX_ABS_LOGPROB
            and (
                "logprob_cpu_fp32" not in entry
                or entry["logprob_cpu_fp32"]["max_abs"] <= GATE_MAX_ABS_LOGPROB
            )
        )
        print(
            path,
            {k: v for k, v in entry.items() if k.startswith(("ident", "cer", "pass", "sec"))},
            flush=True,
        )
        if path == "graph" and entry["passed"]:
            # 등록 순서: graph가 통과하면 fast 전사는 따로 재지 않는다. graph의 정상 상태가 아닌 step이 fast와
            # 같은 코드이므로, fast는 (가)의 log-prob 비교(GPU·CPU)만 통과하면 통과로 본다.
            fast = result["paths"]["fast"]
            fast["transcripts"] = "skipped: graph passed (registered order)"
            fast["passed"] = fast["logprob"]["max_abs"] <= GATE_MAX_ABS_LOGPROB and (
                "logprob_cpu_fp32" not in fast or fast["logprob_cpu_fp32"]["max_abs"] <= GATE_MAX_ABS_LOGPROB
            )
            break
    write_json(path_out, result)


# --- timing: 청크당 계산 시간(P3·P4) ----------------------------------------------------------------------


# 6단계 time_steps와 같은 정의(발화마다 오프라인 mel을 미리 계산하고 step 전후 동기화)에 경로·조기 종료를
# 더했다. 첫 발화를 한 번 미리 돌려(가중치 복사본·그래프 캡처·cuDNN 준비) 1회성 비용을 빼고 잰다. 모든 청크와
# 정상 상태 청크(캐시가 찬 17번째 청크부터, graph 경로가 그래프를 쓰는 청크)를 따로 요약한다.
@torch.no_grad()
def time_paths(model, utts, store, spec, device, amp: bool, path: str, exit_layer: int | None) -> dict:
    logmel = LogMel().to(device)
    enc = StreamingEncoder(model, spec, exit_layer=exit_layer, path=path)
    steady_from = spec.left_context // spec.chunk_size
    step = 4 * spec.chunk_size
    # 예열 1: 정상 상태까지 가는 긴 무작위 mel(24청크)로 가중치 복사본 만들기·그래프 캡처를 미리 끝낸다.
    warm = torch.randn(step * 24, 80, generator=torch.Generator().manual_seed(0)).to(device)
    state = enc.new_state()
    with autocast(device, amp):
        for i in range(0, warm.shape[0], step):
            enc.step(warm[i : i + step], state)
    all_ms, steady_ms, audio_s = [], [], 0.0
    for k, u in enumerate([utts[0], *utts]):
        wav = store.waveform(u.id).to(device)
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        state = enc.new_state()
        for c, i in enumerate(range(0, mel.shape[1], step)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device, amp):
                enc.step(mel[0, i : i + step], state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if k == 0:
                continue  # 예열 2: 첫 발화를 한 번 미리 돈다(시간은 버린다)
            ms = (time.perf_counter() - t0) * 1000
            all_ms.append(ms)
            if c >= steady_from and i + step <= mel.shape[1]:
                steady_ms.append(ms)
        if k:
            audio_s += u.duration_s
    return {
        "path": path,
        "layers": exit_layer or len(model.blocks),
        "all_chunks": percentiles(all_ms),
        "steady_chunks": percentiles(steady_ms),
        "rtf": float(sum(all_ms) / 1000 / audio_s),
    }


def cmd_timing(args, model, vocab, device) -> None:
    path_out = out_file(args, "timing.json")
    spec = SERVICE_SPEC
    utts, store = manifest("dev_h", args.limit)
    result: dict = {"provenance": provenance(args, device), "load_before": wait_for_idle(args.wait_idle_min)}
    rows = []
    if device.type == "cuda":
        for exit_layer in (None, 6):
            for path in ("eager", "fast", "graph"):
                row = time_paths(model, utts[: args.timing_utts], store, spec, device, True, path, exit_layer)
                row["device"] = "cuda fp16"
                print(json.dumps(row), flush=True)
                rows.append(row)
    cpu_model = load_model(ROOT / args.checkpoint, torch.device("cpu"))[0]
    torch.set_num_threads(1)  # 6단계와 같은 CPU 1스레드 조건(프로세스 전체에 남는다)
    for exit_layer in (None, 6):
        for path in ("eager", "fast"):
            row = time_paths(
                cpu_model, utts[: args.cpu_utts], store, spec, torch.device("cpu"), False, path, exit_layer
            )
            row["device"] = "cpu fp32 1 thread"
            print(json.dumps(row), flush=True)
            rows.append(row)
    result["rows"] = rows
    result["load_after"] = load_snapshot()
    write_json(path_out, result)


# --- exit6: 6층 조기 종료(P4, 보고만) ---------------------------------------------------------------------


# simulated 320 ms(전체 발화 + 청크 마스크) 배치 디코딩에서 최종 헤드(12층)와 exit_layer층 중간 헤드의 가설을
# 함께 낸다. 최종 모델은 InterCTC 6층·피드백 없음이라 6층 중간 헤드 = 6층 조기 종료 출력이다
# (tests/test_fast_streaming.py test_exit_layer_matches_intermediate_head).
@torch.no_grad()
def decode_offline_heads(model, vocab, utts, store, spec, device, exit_layer: int):
    if exit_layer not in model.cfg.sc_layers:
        raise SystemExit(f"layer {exit_layer} has no intermediate head in this checkpoint")
    head = sorted(model.cfg.sc_layers).index(exit_layer)
    logmel = LogMel().to(device)
    final, early = {}, {}
    for batch in duration_batches(utts, 400.0):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        with autocast(device, True):
            logits, inter, lengths = model(mel, frames, spec)
        for j, a, b in zip(
            batch,
            greedy_batch(logits.float(), lengths),
            greedy_batch(inter[head].float(), lengths),
            strict=True,
        ):
            final[utts[j].id] = vocab.decode(a)
            early[utts[j].id] = vocab.decode(b)
    return final, early


def cmd_exit6(args, model, vocab, device) -> None:
    path_out = out_file(args, "exit6.json")
    spec = SERVICE_SPEC
    logmel = LogMel().to(device)
    result: dict = {"provenance": provenance(args, device), "exit_layer": 6, "sets": {}}
    for name in DEV_SETS:
        utts, store = manifest(name, args.limit)
        final, early = decode_offline_heads(model, vocab, utts, store, spec, device, 6)
        rows12, rows6 = hyp_rows(utts, final), hyp_rows(utts, early)
        write_hyps(ROOT / args.out_dir / "hyps" / f"{name}-sim-l12.jsonl", rows12)
        write_hyps(ROOT / args.out_dir / "hyps" / f"{name}-sim-l6.jsonl", rows6)
        t0 = time.perf_counter()
        real6 = decode_streaming(model, vocab, utts, store, spec, device, logmel, amp=True, exit_layer=6)
        real_rows = hyp_rows(utts, real6)
        write_hyps(ROOT / args.out_dir / "hyps" / f"{name}-stream-l6.jsonl", real_rows)
        entry = {
            # a = 6층, b = 12층: diff는 6층 − 12층(음수면 6층이 낫다)
            "simulated_l6_vs_l12": compare(rows6, rows12, hangul_only=True),
            "real_streaming_l6_cer_ns": rate(score(utts, real6)),
            "real_minus_simulated_l6_pp": (rate(score(utts, real6)) - rate(score(utts, early))) * 100,
            "real_streaming_seconds": round(time.perf_counter() - t0, 1),
        }
        eager12 = ROOT / args.out_dir / "hyps" / f"{name}-stream-l12-eager.jsonl"
        if eager12.exists():  # dev_H는 gate가 남긴 12층 실제 스트리밍 가설과 짝지어 비교한다
            rows = [json.loads(line) for line in eager12.read_text(encoding="utf-8").splitlines() if line]
            ref12 = {r["id"]: r for r in rows}
            if set(ref12) == set(real_rows):
                entry["real_l6_vs_l12"] = compare(real_rows, ref12, hangul_only=True)
        result["sets"][name] = entry
        print(name, json.dumps(entry), flush=True)
    # 보고용 표현(판정 아님): dev_mix 6층 − 12층(simulated) ≤ 0.3포인트이고 dev_H도 ≤ 0.3포인트면 '비열등'.
    diffs = [result["sets"][n]["simulated_l6_vs_l12"]["diff_a_minus_b"] for n in DEV_SETS]
    result["descriptive_noninferior"] = all(d <= GUARD for d in diffs)
    write_json(path_out, result)


# --- session: 2차 패스(P1)와 rule2(P2) -------------------------------------------------------------------


# 발화마다 StreamSession(2차 패스 켬, 끝점 뒤 캐시 유지)을 돌려, 같은 실행에서 1-pass(스트리밍) 가설과 2-pass
# 확정 가설을 모두 얻는다. 2차 패스는 스트리밍 상태를 바꾸지 않으므로 1-pass 쪽은 2차 패스를 끈 세션과 같다
# (tests/test_final_pass.py). 6단계 run_sessions와 같은 잡음·패킷·첫 끝점 정의를 쓴다.
def run_sessions_two_pass(model, vocab, utts, store, endpoint, device, tail_s: float, step_path: str) -> dict:
    logmel = LogMel().to(device)
    out: dict = {
        "hyp_stream": {},
        "hyp_final": {},
        "first_endpoint_s": {},
        "first_close_pass_ms": {},
        "pass_ms": [],
        "step_ms": [],
        "rules": Counter(),
        "segments": 0,
        "text_segments": 0,
        "changed_segments": 0,
        "changed_segments_ns": 0,
        "runs_stream": [],
        "runs_final": [],
    }
    for k, u in enumerate(utts):
        wav = torch.cat([store.waveform(u.id), noise(tail_s, seed=k)]).to(device)
        session = StreamSession(
            model, vocab, SERVICE_SPEC, logmel, endpoint, False, step_path=step_path, final_pass="full"
        )
        closes, partials = [], {}
        for pos, piece in packets(wav):
            for ev in session.push_samples(piece):
                if ev.kind == "partial":
                    partials.setdefault(ev.segment, []).append(ev.text)
                    continue
                closes.append(ev)
                if u.id not in out["first_endpoint_s"]:
                    out["first_endpoint_s"][u.id] = (pos + piece.numel()) / SR
                    out["first_close_pass_ms"][u.id] = ev.final_pass_ms
        closes.extend(session.flush())
        out["step_ms"].extend(session.stats.compute_ms)
        out["hyp_stream"][u.id] = " ".join(e.stream_text for e in closes if (e.stream_text or "").strip())
        out["hyp_final"][u.id] = " ".join(e.text for e in closes if e.kind == "final")
        for e in closes:
            stream_text = e.stream_text or ""
            out["pass_ms"].append(e.final_pass_ms)
            out["rules"][e.rule] += 1
            out["segments"] += 1
            if stream_text.strip() or e.text.strip():
                out["text_segments"] += 1
                out["changed_segments"] += stream_text.strip() != e.text.strip()
                out["changed_segments_ns"] += stream_text.replace(" ", "") != e.text.replace(" ", "")
            if stream_text.strip():
                out["runs_stream"].append((partials.get(e.segment, []), stream_text))
            if e.kind == "final":
                out["runs_final"].append((partials.get(e.segment, []), e.text))
    return out


def summarize_session(utts, run: dict, ends: dict[str, float], name: str, rule2: float) -> dict:
    stream_rows, final_rows = hyp_rows(utts, run["hyp_stream"]), hyp_rows(utts, run["hyp_final"])
    delays = [run["first_endpoint_s"][u] - ends[u] for u in run["first_endpoint_s"] if u in ends]
    final_delays = [
        (run["first_endpoint_s"][u] - ends[u]) * 1000 + run["first_close_pass_ms"][u]
        for u in run["first_endpoint_s"]
        if u in ends
    ]
    return (
        {
            "set": name,
            "rule2_s": rule2,
            "cache_policy": "keep",
            "cer_ns_stream": rate(score(utts, run["hyp_stream"])),
            "cer_ns_final_pass": rate(score(utts, run["hyp_final"])),
            # a = 2-pass, b = 1-pass(스트리밍): diff = 2-pass − 1-pass(음수면 2차 패스가 낫다)
            "final_pass_vs_stream": compare(final_rows, stream_rows, hangul_only=True),
            "ep_ms": percentiles(d * 1000 for d in delays),
            "ep_plus_final_pass_ms": percentiles(final_delays),
            "final_pass_ms": percentiles(run["pass_ms"]),
            "step_ms": percentiles(run["step_ms"]),
            "segments": run["segments"],
            "rules": dict(run["rules"]),
            "text_segments": run["text_segments"],
            "changed_at_final_rate": run["changed_segments"] / max(1, run["text_segments"]),
            "changed_at_final_rate_ignoring_spaces": run["changed_segments_ns"]
            / max(1, run["text_segments"]),
            "stability_stream": stability(run["runs_stream"]),
            "stability_final_pass": stability(run["runs_final"]),
        },
        stream_rows,
        final_rows,
    )


def gate_status(args) -> dict:
    path = ROOT / args.out_dir / "gate.json"
    if not path.exists():
        return {}
    paths = json.loads(path.read_text(encoding="utf-8"))["paths"]
    return {p: bool(v.get("passed")) for p, v in paths.items()}


def cmd_session(args, model, vocab, device) -> None:
    path_out = out_file(args, "session.json")
    passed = gate_status(args)
    # 등록: 관문을 통과한 graph 경로가 있으면 세션 측정도 그 경로로 돈다(같은 전사가 관문으로 확인됐으므로).
    step_path = "graph" if passed.get("graph") else "eager"
    stage6 = json.loads(STAGE6_SESSION.read_text(encoding="utf-8"))
    baseline = stage6["baseline_no_endpoint"]["cer_ns"]
    stage6_keep = stage6["keep_cache"]["cer_ns"]
    result: dict = {
        "provenance": provenance(args, device),
        "step_path": step_path,
        "stage6_baseline_no_endpoint_cer_ns": baseline,
        "stage6_keep_rule2_1.2_cer_ns": stage6_keep,
        "tail_noise_s": args.tail_s,
        "load_before": wait_for_idle(args.wait_idle_min),
        "runs": {},
    }
    # 예열: 2차 패스 길이 묶음·부분 청크 길이·graph 캡처의 1회성 비용을 측정 전에 치른다(서버 시작과 같다).
    result["warm_up_s"] = warm_up(
        StreamSession(
            model,
            vocab,
            SERVICE_SPEC,
            LogMel().to(device),
            EndpointConfig(),
            False,
            step_path=step_path,
            final_pass="full",
        )
    )
    hyp_dir = ROOT / args.out_dir / "hyps"
    # dev_mix: 6단계 선택(1.2 s)을 먼저, 그다음 0.8 s. 둘 다 캐시 유지, 2차 패스 켬.
    utts, store = manifest("dev_mix", args.limit)
    ends = last_word_end("dev_mix")
    for rule2 in (1.2, 0.8):
        t0 = time.perf_counter()
        run = run_sessions_two_pass(
            model, vocab, utts, store, EndpointConfig(rule2_s=rule2), device, args.tail_s, step_path
        )
        summary, stream_rows, final_rows = summarize_session(utts, run, ends, "dev_mix", rule2)
        summary["seconds"] = round(time.perf_counter() - t0, 1)
        write_hyps(hyp_dir / f"dev_mix-session-rule2_{rule2}-stream.jsonl", stream_rows)
        write_hyps(hyp_dir / f"dev_mix-session-rule2_{rule2}-final_pass.jsonl", final_rows)
        result["runs"][f"dev_mix_rule2_{rule2}"] = summary
        print("dev_mix", rule2, json.dumps(summary), flush=True)
    keep12 = result["runs"]["dev_mix_rule2_1.2"]["cer_ns_stream"]
    result["reproduces_stage6"] = abs(keep12 - stage6_keep) <= REPRO_TOL or bool(args.limit)
    # P2: 캐시 유지 상태에서 rule2 0.8 s의 1-pass CER 증가(끝점 끈 6단계 기준 대비) ≤ 0.2포인트면 0.8 s.
    increase = result["runs"]["dev_mix_rule2_0.8"]["cer_ns_stream"] - baseline
    chosen = 0.8 if increase <= RULE2_MAX_INCREASE else 1.2
    result["p2"] = {"increase_pp": increase * 100, "chosen_rule2_s": chosen}
    # P1 가드용 dev_H 세션(선택한 rule2).
    utts, store = manifest("dev_h", args.limit)
    t0 = time.perf_counter()
    run = run_sessions_two_pass(
        model, vocab, utts, store, EndpointConfig(rule2_s=chosen), device, args.tail_s, step_path
    )
    summary, stream_rows, final_rows = summarize_session(utts, run, last_word_end("dev_h"), "dev_h", chosen)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    write_hyps(hyp_dir / f"dev_h-session-rule2_{chosen}-stream.jsonl", stream_rows)
    write_hyps(hyp_dir / f"dev_h-session-rule2_{chosen}-final_pass.jsonl", final_rows)
    result["runs"][f"dev_h_rule2_{chosen}"] = summary
    print("dev_h", chosen, json.dumps(summary), flush=True)
    result["p1"] = p1_verdict(result, chosen)
    result["load_after"] = load_snapshot()
    write_json(path_out, result)


# P1 판정(등록 규칙): 선택한 rule2에서 dev_mix 2-pass − 1-pass ≤ −1.86포인트이고 짝지은 95 % 구간 상한 < 0,
# dev_H 2-pass − 1-pass ≤ +0.3포인트, GPU 2차 패스 계산 p90 ≤ 250 ms(dev_mix 실행). CPU 상한은 final-cpu
# 결과로 summary에서 따로 본다.
def p1_verdict(result: dict, chosen: float) -> dict:
    mix = result["runs"][f"dev_mix_rule2_{chosen}"]
    dev_h = result["runs"][f"dev_h_rule2_{chosen}"]
    diff = mix["final_pass_vs_stream"]
    guard = dev_h["final_pass_vs_stream"]["diff_a_minus_b"]
    p90 = mix["final_pass_ms"]["p90"]
    checks = {
        "dev_mix_improvement": diff["diff_a_minus_b"] <= -THRESHOLD,
        "dev_mix_ci_below_zero": diff["diff_ci95"][1] < 0,
        "dev_h_guard": guard <= GUARD,
        "gpu_p90": p90 is not None and p90 <= FINAL_PASS_GPU_P90_MS,
    }
    return {
        "dev_mix_diff_pp": diff["diff_a_minus_b"] * 100,
        "dev_mix_ci95_pp": [x * 100 for x in diff["diff_ci95"]],
        "dev_h_diff_pp": guard * 100,
        "final_pass_p90_ms": p90,
        "checks": checks,
        "adopt": all(checks.values()) and bool(result.get("reproduces_stage6")),
    }


# --- final-cpu: 2차 패스의 CPU 계산 시간(P1) --------------------------------------------------------------


def cmd_final_cpu(args, model, vocab, device) -> None:
    path_out = out_file(args, "final_cpu.json")
    cpu = torch.device("cpu")
    cpu_model = load_model(ROOT / args.checkpoint, cpu)[0]
    torch.set_num_threads(1)
    session_file = ROOT / args.out_dir / "session.json"
    chosen = 1.2
    if session_file.exists():
        chosen = json.loads(session_file.read_text(encoding="utf-8"))["p2"]["chosen_rule2_s"]
    utts, store = manifest("dev_mix", args.limit or args.cpu_session_utts)
    result: dict = {
        "provenance": provenance(args, cpu),
        "rule2_s": chosen,
        "load_before": wait_for_idle(args.wait_idle_min),
    }
    endpoint = EndpointConfig(rule2_s=chosen)
    logmel = LogMel()
    result["warm_up_s"] = warm_up(
        StreamSession(cpu_model, vocab, SERVICE_SPEC, logmel, endpoint, False, amp=False, final_pass="full")
    )
    run = run_sessions_two_pass(cpu_model, vocab, utts, store, endpoint, cpu, args.tail_s, "eager")
    result["final_pass_ms"] = percentiles(run["pass_ms"])
    result["step_ms"] = percentiles(run["step_ms"])
    result["segments"] = run["segments"]
    result["within_cpu_limit"] = (result["final_pass_ms"]["p90"] or 0) <= FINAL_PASS_CPU_P90_MS
    result["load_after"] = load_snapshot()
    write_json(path_out, result)


# --- summary --------------------------------------------------------------------------------------------


def fmt(x, scale: float = 1.0, digits: int = 2) -> str:
    """요약 표시용 숫자 서식(None이면 '-'). 표본이 적어 값이 없을 때도 요약이 멈추지 않게 한다."""
    return "-" if x is None else f"{x * scale:.{digits}f}"


# 등록 규칙(규칙 변경 26)의 서버 기본 step 경로: 관문을 통과한 경로 가운데 그 장치에서 모든 청크 p90이
# eager보다 낮은 것 중 가장 빠른 것, 없으면 eager. GPU는 graph·fast, CPU는 fast만 후보다.
def choose_step_path(rows: list[dict], passed: dict, device_prefix: str, candidates: tuple[str, ...]) -> str:
    p90 = {
        r["path"]: r["all_chunks"]["p90"]
        for r in rows
        if r["device"].startswith(device_prefix) and r["layers"] == 12 and r["all_chunks"]["p90"] is not None
    }
    ok = [p for p in candidates if passed.get(p) and p in p90 and "eager" in p90 and p90[p] < p90["eager"]]
    return min(ok, key=p90.get) if ok else "eager"


def cmd_summary(args, model, vocab, device) -> None:
    base = ROOT / args.out_dir
    load = {
        n: json.loads((base / f"{n}.json").read_text(encoding="utf-8"))
        for n in ("gate", "timing", "exit6", "session", "final_cpu")
        if (base / f"{n}.json").exists()
    }
    summary: dict = {"files": sorted(load)}
    lines = ["# 후속 측정 요약 (자동 생성, 판정 규칙은 docs/experiments.md 규칙 변경 26)", ""]
    if "gate" in load:
        gate = load["gate"]
        summary["gate"] = {p: v.get("passed") for p, v in gate["paths"].items()}
        lines.append(
            f"- 관문: {summary['gate']}, 그래프 캡처 {gate['graph_captured']} ({gate['graph_failed']})"
        )
    if "timing" in load:
        rows = load["timing"]["rows"]
        for r in rows:
            a, s = r["all_chunks"], r["steady_chunks"]
            lines.append(
                f"- {r['device']} {r['path']} {r['layers']}층: 모든 청크 p50 {fmt(a['p50'], digits=1)} / "
                f"p90 {fmt(a['p90'], digits=1)} ms, 정상 상태 p90 {fmt(s['p90'], digits=1)} ms, "
                f"RTF {fmt(r['rtf'], digits=3)}"
            )
        passed = summary.get("gate", {})
        summary["server_step_path_gpu"] = choose_step_path(rows, passed, "cuda", ("graph", "fast"))
        summary["server_step_path_cpu"] = choose_step_path(rows, passed, "cpu", ("fast",))
        lines.append(
            f"- 서버 step 경로(등록 규칙): GPU {summary['server_step_path_gpu']}, "
            f"CPU {summary['server_step_path_cpu']}"
        )
        lines.append(f"- 부하(timing 측정 전): {load_line(load['timing']['load_before'])}")
    if "exit6" in load:
        for name, e in load["exit6"]["sets"].items():
            d = e["simulated_l6_vs_l12"]
            ci = d["diff_ci95"]
            lines.append(
                f"- 6층 {name}: simulated 6층 {fmt(d['rate_a'], 100)} vs 12층 {fmt(d['rate_b'], 100)} "
                f"(차 {fmt(d['diff_a_minus_b'], 100)}, 95 % [{fmt(ci[0], 100)}, {fmt(ci[1], 100)}]), "
                f"실제 스트리밍 6층 {fmt(e['real_streaming_l6_cer_ns'], 100)}"
            )
    if "session" in load:
        s = load["session"]
        summary.update(p2=s["p2"], p1=s["p1"], reproduces_stage6=s["reproduces_stage6"])
        lines.append(
            f"- 6단계 재현(1.2 s 캐시 유지 1-pass): {s['reproduces_stage6']}, step 경로 {s['step_path']}"
        )
        lines.append(
            f"- P2 rule2: 증가 {fmt(s['p2']['increase_pp'], digits=3)}포인트 → {s['p2']['chosen_rule2_s']} s"
        )
        for key, r in s["runs"].items():
            fp = r["final_pass_ms"]
            lines.append(
                f"- {key}: 1-pass {fmt(r['cer_ns_stream'], 100, 3)} / "
                f"2-pass {fmt(r['cer_ns_final_pass'], 100, 3)}, "
                f"EP50 {fmt(r['ep_ms']['p50'], digits=0)} ms, 2차 패스 p50/p90 {fmt(fp['p50'], digits=1)}/"
                f"{fmt(fp['p90'], digits=1)} ms, "
                f"확정 때 바뀐 세그먼트 {fmt(r['changed_at_final_rate'], digits=3)}"
            )
        lines.append(f"- 부하(session 측정 전): {load_line(s['load_before'])}")
        cpu_ok = load.get("final_cpu", {}).get("within_cpu_limit")
        summary["p1"]["cpu_within_limit"] = cpu_ok
        lines.append(f"- P1 판정: 채택 {s['p1']['adopt']} {s['p1']['checks']}, CPU 상한 안 {cpu_ok}")
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="7단계 봉인 test 뒤의 dev 전용 후속 측정(docs/experiments.md 규칙 변경 26). "
        "봉인 test 세트는 "
        "열지 않는다."
    )
    p.add_argument("command", choices=["gate", "timing", "exit6", "session", "final-cpu", "summary"])
    p.add_argument("--checkpoint", default="outputs/stage4/trimtail/avg10.pt", help="측정 모델(최종 모델)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir", default="reports/followup", help="결과 폴더(저장소 기준). 연기 시험은 work/ 아래로"
    )
    p.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어쓴다")
    p.add_argument("--limit", type=int, default=0, help="앞 N발화만(연기 시험용; 0 = 전수, 판정은 전수로만)")
    p.add_argument("--logprob-utts", type=int, default=30, help="gate: GPU log-prob 비교 발화 수(dev_H 앞)")
    p.add_argument("--cpu-logprob-utts", type=int, default=5, help="gate: CPU fp32 log-prob 비교 발화 수")
    p.add_argument("--timing-utts", type=int, default=30, help="timing: GPU 계산 시간 발화 수(6단계와 같음)")
    p.add_argument("--cpu-utts", type=int, default=10, help="timing: CPU 1스레드 발화 수(6단계와 같음)")
    p.add_argument("--cpu-session-utts", type=int, default=50, help="final-cpu: dev_mix 앞 N발화 세션")
    p.add_argument(
        "--tail-s", type=float, default=3.0, help="세션: 발화 뒤에 붙이는 -60 dBFS 잡음 길이(6단계와 같음)"
    )
    p.add_argument(
        "--wait-idle-min", type=float, default=10.0, help="지연 측정 전 쉬는 상태를 기다리는 최대 분"
    )
    args = p.parse_args(argv)
    # Windows 콘솔·파이프의 기본 인코딩(cp949)으로 한국어 요약을 찍으면 로그가 깨지므로 표준 출력을 UTF-8로
    # 둔다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device(args.device)
    if args.command == "summary":
        cmd_summary(args, None, None, device)
        return
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    commands = {
        "gate": cmd_gate,
        "timing": cmd_timing,
        "exit6": cmd_exit6,
        "session": cmd_session,
        "final-cpu": cmd_final_cpu,
    }
    commands[args.command](args, model, vocab, device)


if __name__ == "__main__":
    main(sys.argv[1:])
