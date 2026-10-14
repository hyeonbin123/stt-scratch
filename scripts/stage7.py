"""7단계 봉인 test(docs/experiments.md 7단계): 2~5단계에서 고정한 구성으로 단 한 번만 잰다.

Zeroth 공식 test(학습에 있는 문장), Zeroth test_H(처음 보는 문장), FLEURS test, CV22 test 네 세트를 고정
구성(4단계 채택 모델, greedy, GPU fp16 autocast)으로 잰다: full context greedy CER_ns/CER_ws(전체와 한글
전용, 부트스트랩 구간), 서비스 구성(320 ms, 좌측 128)의 실제 캐시 스트리밍 greedy, 세트마다 출력 불가능
문자 하한(어휘 밖 음절 등). 결과와 H7.1·H7.2는 reports/final/sealed_test.json과 .md에 쓰고 잠금 파일을
남긴다. 두 번째 실행은 거부한다(봉인 test는 한 번만 재고, 그 뒤에는 모델과 규칙을 바꾸지 않는다).

사용법: uv run python scripts/stage7.py --checkpoint <최종 avg10.pt> --confirm-sealed-test
"""

# [왜 필요한가] 사전 등록 측정의 핵심은 '봉인 test는 마지막에 정해 둔 구성으로 한 번만 잰다'이다. test 결과를
# 보고 모델·규칙을 고치면 test가 dev가 되어 버리기 때문이다(docs/experiments.md 7단계: test 뒤에는 모델·규칙을
# 바꾸지 않는다). 그래서 사람이 실수로 두 번 돌리거나 다른 구성으로 돌리는 것을 코드가 막는다.
# [보호 장치](규칙 변경 24·25, Codex 4·6단계 검토 반영) 확인 플래그 필수, 봉인 잠금 파일(한 번 만들면 지우지
# 않음), 사전 검사(채택 체크포인트·청크 8·좌측 128·GPU·6단계 H6.1 통과와 그 출처), 실행 중 잠금(동시 실행
# 방지), 중단 시 이미 잰 세트는 다시 재지 않고 --resume-sealed로 마무리(봉인 구성과 같을 때만).
# [결과] 2026-10-14 1회, 재개 없음: Zeroth 공식 test 3.67 %(H7.1 충족), test_H 15.55 %, FLEURS 한글 전용 14.62
# %, CV22 28.19 %(H7.2 범위 안). 보호 장치는 tests/test_stage7_guards.py가 임시 ROOT에서 검사한다.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# torch: GPU 확인과 모델 실행.
import torch

# 프로젝트 내부 모듈: dev 측정과 같은 평가 함수(evaluate_manifest: full·실제 스트리밍 디코딩과 채점, 발화별
# 가설 저장), 체크포인트 로더, log-mel, 청크 설정, 어휘와 채점 정규화. dev와 같은 코드 경로여야 봉인 수치를
# dev 수치와 비교할 수 있다.
from stt_scratch.data import read_manifest
from stt_scratch.evaluate import evaluate_manifest, load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.text import HANGUL, Vocab, normalize_score

# SETS: 봉인 test 네 세트(0단계에서 고정한 매니페스트). ANCHORS: 같은 Zeroth 공식 test에서 별도 프로젝트
# whisper-ko-ft가 같은 채점 정규화로 잰 whisper-large-v3-turbo 제로샷 4.48 %와 LoRA 파인튜닝 1.96 %(CER_ns
# %, H7.1의 두 경계).
ROOT = Path(__file__).resolve().parents[1]
SETS = ("test_zeroth", "test_h", "test_fleurs", "test_cv")
ANCHORS = {"whisper-turbo zero-shot": 4.48, "whisper-turbo + LoRA": 1.96}


# 파일 SHA-256(1 MiB씩 읽음). 봉인 구성의 체크포인트 지문으로 쓴다.
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# 모델이 낼 수 없는 글자는 숫자, 라틴, 어휘에 없는 음절이다. 이 글자들은 반드시 오류가 되므로 CER의 하한이다.
# 전체·한글 전용 두 부분집합으로 낸다(규칙 변경 24에서 'OOV 하한'을 이 이름으로 바로잡음).
# 예: FLEURS 전체 4.80 %는 숫자·라틴 문자 때문이고 한글 전용은 0.14 %다.
def oov_bound(name: str, vocab: Vocab) -> dict:
    """출력 불가능 문자 하한: 채점 정규화(공백 제외) 뒤 정답 글자 가운데 어휘로 낼 수 없는 글자의 비율."""
    known = set(vocab.tokens)
    out = {}
    for subset in ("all", "hangul_only"):
        total = oov = 0
        for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl"):
            if subset == "hangul_only" and not u.hangul_only:
                continue
            chars = normalize_score(u.text)
            total += len(chars)
            oov += sum(1 for c in chars if not (HANGUL.fullmatch(c) and c in known))
        out[subset] = oov / max(1, total)
    return out


# 판정 파일은 reports/stage4/summary.json이다. 봉인 test에 넘긴 체크포인트가 이것과 같아야 한다.
def adopted_checkpoint() -> str:
    """2~5단계에서 고정한 최종 모델, 즉 4단계에서 채택한 변형의 평균 체크포인트 경로를 돌려준다."""
    summary = json.loads((ROOT / "reports" / "stage4" / "summary.json").read_text(encoding="utf-8"))
    adopted = summary["verdict"]["adopted"]
    return f"outputs/stage4/{adopted}/avg10.pt"


# 하나라도 다르면 SystemExit로 거부한다. 최초 실행과 재개 모두 거친다(규칙 변경 25).
# 검사: (1) 체크포인트 = 4단계 채택 모델, (2) 청크 8·좌측 128(2단계에서 정한 서비스 구성), (3) CUDA 있음(모든
# dev 측정과 같은 GPU fp16), (4) 6단계 H6.1 충족이고 그 결과가 같은 체크포인트 SHA-256·같은
# 구성·GPU·전수(limit 0)로 잰 것, (5) 네 매니페스트 존재.
def preflight(args, ckpt: Path) -> None:
    """봉인을 쓰기 전에 구성을 검사한다. test 전에 고정한 구성과 다르면 봉인을 쓰지 않고 거부한다."""
    want = adopted_checkpoint()
    if Path(args.checkpoint).as_posix() != want:
        raise SystemExit(f"checkpoint {args.checkpoint} is not the adopted final model {want}")
    if (args.chunk, args.left) != (8, 128):
        raise SystemExit("the service configuration fixed in Stage 2 is chunk 8 (320 ms), left 128")
    if not torch.cuda.is_available():
        raise SystemExit("the sealed test runs on the GPU with fp16 autocast, as every dev measurement")
    h61_path = ROOT / "reports" / "stage6" / "h61.json"
    h61 = json.loads(h61_path.read_text(encoding="utf-8")) if h61_path.exists() else {}
    if not h61.get("h61_met"):
        raise SystemExit("Stage 6 H6.1 (real streaming == simulated) must be met before the sealed test")
    want_h61 = {"checkpoint_sha256": sha256(ckpt), "spec": "chunk8 left128", "device": "cuda", "limit": 0}
    got_h61 = {k: h61.get(k) for k in want_h61}
    if got_h61 != want_h61:
        raise SystemExit(f"Stage 6 H6.1 was not measured with the sealed configuration: {got_h61}")
    for name in SETS:
        if not (ROOT / "manifests" / f"{name}.jsonl").exists():
            raise SystemExit(f"manifest {name} missing")


# 봉인 test 진입점. 인자: --checkpoint, --chunk·--left(8·128만 허용), --confirm-sealed-test(필수 확인 플래그),
# --resume-sealed(중단된 봉인 test 마무리).
# 파일 세 개로 상태를 관리한다: sealed_test.lock(봉인; 최초 실행 때 배타적으로 만들고 끝나면 finished 줄을
# 덧붙임, 지우지 않음), sealed_config.json(봉인 구성), sealed_test.running(실행 중 잠금; 끝나면 지움).
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--confirm-sealed-test", action="store_true")
    p.add_argument(
        "--resume-sealed",
        action="store_true",
        help="중단된 봉인 test를 마무리한다. 결과가 저장된 세트는 다시 재지 않는다",
    )
    args = p.parse_args(argv)
    out_dir = ROOT / "reports" / "final"
    lock = out_dir / "sealed_test.lock"  # 봉인: 한 번 만들고 지우지 않는다
    config_path = out_dir / "sealed_config.json"
    run_lock = out_dir / "sealed_test.running"  # 최초 실행이든 재개든 한 번에 한 프로세스만
    # 확인 플래그 없이 실수로 실행해도 봉인을 쓰지 않게 한다.
    if not args.confirm_sealed_test:
        raise SystemExit("pass --confirm-sealed-test: the sealed test is measured once and never again")
    ckpt = ROOT / args.checkpoint
    preflight(args, ckpt)  # 재개할 때도 검사한다: 구성이 여전히 봉인 구성이어야 한다
    # 봉인 구성: 체크포인트 경로·SHA-256, 청크·좌측, 정밀도, 장치. 최초 실행에서 sealed_config.json으로
    # 남기고, 재개는 이것과 같아야 한다.
    config = {
        "checkpoint": Path(args.checkpoint).as_posix(),
        "checkpoint_sha256": sha256(ckpt),
        "spec": f"chunk{args.chunk} left{args.left}",
        "precision": "fp16-autocast",
        "device": "cuda",
    }
    # 재개는 봉인이 시작됐지만 끝나지 않은 경우(lock에 finished가 없음)에만, 그리고 봉인 구성이 같을 때만
    # 허용한다. 재개가 아니면 봉인이 이미 있을 때 거부한다(두 번째 봉인 test 금지).
    if args.resume_sealed:
        if not lock.exists() or "finished" in lock.read_text(encoding="utf-8"):
            raise SystemExit("nothing to resume: no unfinished sealed test")
        sealed = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        if sealed != config:
            raise SystemExit(f"resume refused: sealed configuration {sealed} differs from {config}")
    elif lock.exists():
        raise SystemExit(f"the sealed test was already started ({lock.read_text(encoding='utf-8').strip()})")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 실행 중 잠금을 배타적 생성("x" 모드)으로 잡는다. 이미 있으면 다른 프로세스가 돌고 있거나 비정상 종료로
    # 남은 것이므로, 사람이 확인해 지운 뒤에만 재개하게 한다(규칙 변경 25).
    try:
        run_fd = open(run_lock, "x", encoding="utf-8")  # 배타적 생성
    except FileExistsError as err:
        raise SystemExit(
            f"another sealed-test process holds {run_lock}; if none is running (e.g. after a crash), delete "
            "that file and resume with --resume-sealed"
        ) from err
    with run_fd:
        run_fd.write(f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # 본 측정. 성공하든 예외가 나든 실행 중 잠금은 지운다(봉인 잠금은 남는다).
    try:
        run_sealed(args, ckpt, out_dir, lock, config_path, config)
    finally:
        run_lock.unlink(missing_ok=True)


# 세트 이름·모드·정밀도·체크포인트 경로·SHA-256·발화 수가 봉인 구성과 같고 발화별 가설 파일도 있어야 하며,
# 아니면 멈춘다. 봉인 test를 다시 재지 않으면서도 다른 구성의 결과가 섞이지 않게 하는 장치다(Codex 6단계 검토
# P1).
def check_saved(path: Path, name: str, mode: str, config: dict) -> dict:
    """중단 전에 저장된 세트 결과는 봉인 구성으로 잰 것일 때만 재사용한다."""
    r = json.loads(path.read_text(encoding="utf-8"))
    want_mode = "full" if mode == "full" else f"streaming {config['spec']}"
    want = {
        "manifest": name,
        "mode": want_mode,
        "precision": config["precision"],
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "n": len(read_manifest(ROOT / "manifests" / f"{name}.jsonl")),
    }
    got = {k: r.get(k) for k in want}
    if got != want or not path.with_suffix(".hyps.jsonl").exists():
        raise SystemExit(f"saved result {path} does not match the sealed configuration: {got} vs {want}")
    return r


# 봉인 측정 본체: (최초 실행이면) 봉인 잠금과 구성 파일을 만들고 → 세트마다 full·스트리밍을 재거나(재개면
# 저장된 결과를 검사해 재사용) → H7.1·H7.2 → JSON·Markdown 보고서 → 봉인 잠금에 finished를 덧붙인다.
def run_sealed(args, ckpt: Path, out_dir: Path, lock: Path, config_path: Path, config: dict) -> None:
    # 최초 실행인데 reports/final/에 test 결과 파일이 이미 있으면 거부한다(봉인 전에 누가 test를 잰
    # 흔적이므로). 봉인 잠금은 배타적 생성이라 두 번째 봉인은 만들어질 수 없다.
    if not args.resume_sealed:
        stale = sorted(p.name for p in out_dir.glob("test_*.json*"))
        if stale:
            raise SystemExit(f"results exist before the seal was started: {stale}")
        with open(lock, "x", encoding="utf-8") as f:  # 배타적 생성: 두 번째 봉인 test는 시작될 수 없다
            f.write(
                f"started {time.strftime('%Y-%m-%d %H:%M:%S')}, checkpoint {args.checkpoint} "
                f"sha256 {config['checkpoint_sha256']}\n"
            )
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    device = torch.device("cuda")
    model, state = load_model(ckpt, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    spec = ChunkSpec(args.chunk, args.left)
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "averaged_epochs": state.get("averaged"),
        "service_spec": config["spec"],
        "lock": lock.read_text(encoding="utf-8").strip(),
        "resumed": args.resume_sealed,
        "sets": {},
    }
    for name in SETS:
        results = {}
        # 두 디코딩 모드: full(전체 문맥, 앵커 비교용)과 서비스 구성의 실제 캐시 스트리밍(끝점 없음; 6단계
        # 판정 요약에 등록된 측정). 결과 파일에 체크포인트 SHA-256을 함께 남긴다. 이미 저장된 모드는 다시 재지
        # 않는다.
        for mode, mode_spec, streaming in (("full", ChunkSpec(0, -1), False), ("streaming", spec, True)):
            path = out_dir / f"{name}-{mode}.json"
            if path.exists():  # 중단 전에 잰 결과: 구성이 맞으면 재사용(다시 재지 않음)
                results[mode] = check_saved(path, name, mode, config)
                continue
            results[mode] = evaluate_manifest(
                model,
                vocab,
                name,
                mode_spec,
                device,
                logmel=logmel,
                streaming=streaming,
                out_path=path,
                checkpoint=config["checkpoint"],
                extra={"checkpoint_sha256": config["checkpoint_sha256"]},
            )
        report["sets"][name] = {**results, "unproducible_char_lower_bound": oov_bound(name, vocab)}
        print(
            name,
            "full",
            round(results["full"]["cer_ns"] * 100, 2),
            "streaming",
            round(results["streaming"]["cer_ns"] * 100, 2),
            flush=True,
        )
    # H7.1: Zeroth 공식 test full CER_ns가 1.96 % < x < 4.48 %(이 세트는 모두 한글 전용이라 전체 = 한글 전용).
    # H7.2: 판정 없는 범위 보고(FLEURS 한글 전용 12~30 %, CV22 15~40 %, test_H > 공식 test). 구간이 아니라
    # 점추정으로 본다.
    rate = {n: r["full"]["hangul_only"]["cer_ns"]["rate"] * 100 for n, r in report["sets"].items()}
    z = report["sets"]["test_zeroth"]["full"]["all"]["cer_ns"]["rate"] * 100
    report["h71"] = {
        "zeroth_full_cer_ns": z,
        "anchors": ANCHORS,
        "met": ANCHORS["whisper-turbo + LoRA"] < z < 4.48,
    }
    report["h72"] = {  # 판정 없는 범위 보고(등록된 대로)
        "fleurs_hangul_only": rate["test_fleurs"],
        "fleurs_in_12_30": 12 <= rate["test_fleurs"] <= 30,
        "cv_hangul_only": rate["test_cv"],
        "cv_in_15_40": 15 <= rate["test_cv"] <= 40,
        "test_h_above_official": rate["test_h"] > rate["test_zeroth"],
    }
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "sealed_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    def cell(s: dict) -> str:
        """표 칸 문자열: "비율 (95 % 부트스트랩 구간)"을 백분율로 쓴다."""
        lo, hi = s["ci"]
        return f"{s['rate'] * 100:.2f} ({lo * 100:.2f}-{hi * 100:.2f})"

    # README·docs/experiments.md 7단계 표로 옮긴 Markdown 보고서. 봉인 잠금 내용(시작 시각·체크포인트)과 끝난
    # 시각을 머리에 적는다.
    lines = [
        f"# 봉인 test ({report['lock']}; finished {report['finished']})",
        "",
        f"checkpoint `{report['checkpoint']}` (SHA-256 {report['checkpoint_sha256'][:12]}...), greedy, "
        f"fp16 autocast; streaming = real cache streaming {report['service_spec']}",
        "",
        "| set | n | full CER_ns all | full CER_ns 한글 전용 | full CER_ws 한글 전용 "
        "| streaming CER_ns 한글 전용 | streaming CER_ws 한글 전용 "
        "| 출력 불가능 문자 하한 all / 한글 전용 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in report["sets"].items():
        f_, s_, b_ = r["full"], r["streaming"], r["unproducible_char_lower_bound"]
        lines.append(
            f"| {name} | {f_['n']} | {cell(f_['all']['cer_ns'])} | {cell(f_['hangul_only']['cer_ns'])} | "
            f"{cell(f_['hangul_only']['cer_ws'])} | {cell(s_['hangul_only']['cer_ns'])} | "
            f"{cell(s_['hangul_only']['cer_ws'])} | {b_['all'] * 100:.2f} / {b_['hangul_only'] * 100:.2f} |"
        )
    lines += ["", f"H7.1: {report['h71']}", f"H7.2: {report['h72']}"]
    (out_dir / "sealed_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 봉인 잠금에 finished를 덧붙여 '끝난 봉인'으로 표시한다. 이후에는 재개도 거부된다.
    with open(lock, "a", encoding="utf-8") as f:
        f.write(f"finished {report['finished']}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
