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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--confirm-sealed-test", action="store_true")
    args = p.parse_args(argv)
    out_dir = ROOT / "reports" / "final"
    lock = out_dir / "sealed_test.lock"
    if lock.exists():
        raise SystemExit(f"the sealed test was already measured ({lock.read_text(encoding='utf-8').strip()})")
    if not args.confirm_sealed_test:
        raise SystemExit("pass --confirm-sealed-test: the sealed test is measured once and never again")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / args.checkpoint
    model, state = load_model(ckpt, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    spec = ChunkSpec(args.chunk, args.left)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    lock.write_text(
        f"started {started}, checkpoint {args.checkpoint} sha256 {sha256(ckpt)}\n", encoding="utf-8"
    )
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(ckpt),
        "averaged_epochs": state.get("averaged"),
        "service_spec": f"chunk{args.chunk} left{args.left}",
        "started": started,
        "sets": {},
    }
    for name in SETS:
        full = evaluate_manifest(
            model,
            vocab,
            name,
            ChunkSpec(0, -1),
            device,
            logmel=logmel,
            out_path=out_dir / f"{name}-full.json",
            checkpoint=args.checkpoint,
        )
        stream = evaluate_manifest(
            model,
            vocab,
            name,
            spec,
            device,
            logmel=logmel,
            streaming=True,
            out_path=out_dir / f"{name}-streaming.json",
            checkpoint=args.checkpoint,
        )
        report["sets"][name] = {"full": full, "streaming": stream, "oov_lower_bound": oov_bound(name, vocab)}
        print(
            name,
            "full",
            round(full["cer_ns"] * 100, 2),
            "streaming",
            round(stream["cer_ns"] * 100, 2),
            flush=True,
        )
    z = report["sets"]["test_zeroth"]["full"]["all"]["cer_ns"]["rate"] * 100
    report["h71"] = {
        "zeroth_full_cer_ns": z,
        "anchors": ANCHORS,
        "met": ANCHORS["whisper-turbo + LoRA"] < z < 4.48,
    }
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "sealed_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lines = [
        f"# 봉인 test ({report['started']} ~ {report['finished']}, {args.checkpoint})",
        "",
        "| set | full CER_ns all | full CER_ns 한글 전용 | streaming CER_ns 한글 전용 | OOV 하한 all |",
        "|---|---|---|---|---|",
    ]
    for name, r in report["sets"].items():
        lines.append(
            f"| {name} | {r['full']['all']['cer_ns']['rate'] * 100:.2f} | {r['full']['cer_ns'] * 100:.2f} | "
            f"{r['streaming']['cer_ns'] * 100:.2f} | {r['oov_lower_bound']['all'] * 100:.2f} |"
        )
    (out_dir / "sealed_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock.write_text(lock.read_text(encoding="utf-8") + f"finished {report['finished']}\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
