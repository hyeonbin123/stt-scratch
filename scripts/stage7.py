"""Stage 7 sealed test (docs/experiments.md 7단계): measured ONCE with the configuration fixed by Stages 2-5.

For Zeroth official test (seen sentences), Zeroth test_H (unseen sentences), FLEURS test and CV22 test:
full-context greedy CER_ns/CER_ws (all and Hangul-only, bootstrap intervals) and real cache streaming
greedy with the service configuration, plus the OOV-syllable lower bound of every set. Writes
reports/final/sealed_test.json and .md and a lock file; a second run is refused (the sealed test is
measured once and the model and rules are not changed afterwards).

Usage: uv run python scripts/stage7.py --checkpoint <final avg10.pt> --confirm-sealed-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from stt_scratch.data import read_manifest
from stt_scratch.evaluate import evaluate_manifest, load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.text import HANGUL, Vocab, normalize_score

ROOT = Path(__file__).resolve().parents[1]
SETS = ("test_zeroth", "test_h", "test_fleurs", "test_cv")
ANCHORS = {"whisper-turbo zero-shot": 4.48, "whisper-turbo + LoRA": 1.96}  # Zeroth official test, CER_ns %


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def oov_bound(name: str, vocab: Vocab) -> dict:
    """Share of scored reference characters that the vocabulary cannot produce (a CER lower bound)."""
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


def adopted_checkpoint() -> str:
    """The final model fixed by Stages 2-5: the Stage 4 adopted variant's averaged checkpoint."""
    summary = json.loads((ROOT / "reports" / "stage4" / "summary.json").read_text(encoding="utf-8"))
    adopted = summary["verdict"]["adopted"]
    return f"outputs/stage4/{adopted}/avg10.pt"


def preflight(args, ckpt: Path) -> None:
    """Refuse to consume the seal with a configuration other than the one fixed before the test."""
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--confirm-sealed-test", action="store_true")
    p.add_argument(
        "--resume-sealed",
        action="store_true",
        help="finish an interrupted sealed test: sets with saved results are NOT measured again",
    )
    args = p.parse_args(argv)
    out_dir = ROOT / "reports" / "final"
    lock = out_dir / "sealed_test.lock"  # the seal: created once, never removed
    config_path = out_dir / "sealed_config.json"
    run_lock = out_dir / "sealed_test.running"  # one process at a time, first run and resume alike
    if not args.confirm_sealed_test:
        raise SystemExit("pass --confirm-sealed-test: the sealed test is measured once and never again")
    ckpt = ROOT / args.checkpoint
    preflight(args, ckpt)  # also on resume: the configuration must still be the sealed one
    config = {
        "checkpoint": Path(args.checkpoint).as_posix(),
        "checkpoint_sha256": sha256(ckpt),
        "spec": f"chunk{args.chunk} left{args.left}",
        "precision": "fp16-autocast",
        "device": "cuda",
    }
    if args.resume_sealed:
        if not lock.exists() or "finished" in lock.read_text(encoding="utf-8"):
            raise SystemExit("nothing to resume: no unfinished sealed test")
        sealed = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
        if sealed != config:
            raise SystemExit(f"resume refused: sealed configuration {sealed} differs from {config}")
    elif lock.exists():
        raise SystemExit(f"the sealed test was already started ({lock.read_text(encoding='utf-8').strip()})")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_fd = open(run_lock, "x", encoding="utf-8")  # exclusive
    except FileExistsError as err:
        raise SystemExit(
            f"another sealed-test process holds {run_lock}; if none is running (e.g. after a crash), delete "
            "that file and resume with --resume-sealed"
        ) from err
    with run_fd:
        run_fd.write(f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        run_sealed(args, ckpt, out_dir, lock, config_path, config)
    finally:
        run_lock.unlink(missing_ok=True)


def check_saved(path: Path, name: str, mode: str, config: dict) -> dict:
    """A result saved before an interruption is reused only if it was measured with the sealed config."""
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


def run_sealed(args, ckpt: Path, out_dir: Path, lock: Path, config_path: Path, config: dict) -> None:
    if not args.resume_sealed:
        stale = sorted(p.name for p in out_dir.glob("test_*.json*"))
        if stale:
            raise SystemExit(f"results exist before the seal was started: {stale}")
        with open(lock, "x", encoding="utf-8") as f:  # exclusive: a second sealed test cannot start
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
        for mode, mode_spec, streaming in (("full", ChunkSpec(0, -1), False), ("streaming", spec, True)):
            path = out_dir / f"{name}-{mode}.json"
            if path.exists():  # measured before an interruption: reuse (never re-measure) if it matches
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
    rate = {n: r["full"]["hangul_only"]["cer_ns"]["rate"] * 100 for n, r in report["sets"].items()}
    z = report["sets"]["test_zeroth"]["full"]["all"]["cer_ns"]["rate"] * 100
    report["h71"] = {
        "zeroth_full_cer_ns": z,
        "anchors": ANCHORS,
        "met": ANCHORS["whisper-turbo + LoRA"] < z < 4.48,
    }
    report["h72"] = {  # range report, no verdict (registered)
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
        """rate (95 % bootstrap interval) in percent"""
        lo, hi = s["ci"]
        return f"{s['rate'] * 100:.2f} ({lo * 100:.2f}-{hi * 100:.2f})"

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
    with open(lock, "a", encoding="utf-8") as f:
        f.write(f"finished {report['finished']}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
