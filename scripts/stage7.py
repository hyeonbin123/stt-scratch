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
    h61 = ROOT / "reports" / "stage6" / "h61.json"
    if not h61.exists() or not json.loads(h61.read_text(encoding="utf-8")).get("h61_met"):
        raise SystemExit("Stage 6 H6.1 (real streaming == simulated) must be met before the sealed test")
    if json.loads(h61.read_text(encoding="utf-8")).get("checkpoint_sha256") not in (None, sha256(ckpt)):
        raise SystemExit("Stage 6 was measured with another checkpoint")
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
    lock = out_dir / "sealed_test.lock"
    if not args.confirm_sealed_test:
        raise SystemExit("pass --confirm-sealed-test: the sealed test is measured once and never again")
    ckpt = ROOT / args.checkpoint
    if args.resume_sealed:
        if not lock.exists() or "finished" in lock.read_text(encoding="utf-8"):
            raise SystemExit("nothing to resume: no unfinished sealed test")
    else:
        preflight(args, ckpt)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(lock, "x", encoding="utf-8") as f:  # exclusive: a second run cannot start
                f.write(
                    f"started {time.strftime('%Y-%m-%d %H:%M:%S')}, checkpoint {args.checkpoint} "
                    f"sha256 {sha256(ckpt)}\n"
                )
        except FileExistsError as err:
            raise SystemExit(
                f"the sealed test was already started ({lock.read_text(encoding='utf-8').strip()})"
            ) from err
    device = torch.device("cuda")
    model, state = load_model(ckpt, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    spec = ChunkSpec(args.chunk, args.left)
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(ckpt),
        "averaged_epochs": state.get("averaged"),
        "service_spec": f"chunk{args.chunk} left{args.left}",
        "lock": lock.read_text(encoding="utf-8").strip(),
        "resumed": args.resume_sealed,
        "sets": {},
    }
    for name in SETS:
        results = {}
        for mode, mode_spec, streaming in (("full", ChunkSpec(0, -1), False), ("streaming", spec, True)):
            path = out_dir / f"{name}-{mode}.json"
            if path.exists():  # measured before an interruption: reuse, never re-measure
                results[mode] = json.loads(path.read_text(encoding="utf-8"))
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
                checkpoint=args.checkpoint,
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
    lines = [
        f"# 봉인 test ({report['lock']}; finished {report['finished']})",
        "",
        "| set | full CER_ns all | full 한글 전용 | streaming 한글 전용 | 출력 불가능 문자 하한 all |",
        "|---|---|---|---|---|",
    ]
    for name, r in report["sets"].items():
        lines.append(
            f"| {name} | {r['full']['all']['cer_ns']['rate'] * 100:.2f} | {r['full']['cer_ns'] * 100:.2f} | "
            f"{r['streaming']['cer_ns'] * 100:.2f} | {r['unproducible_char_lower_bound']['all'] * 100:.2f} |"
        )
    lines += ["", f"H7.1: {report['h71']}", f"H7.2: {report['h72']}"]
    (out_dir / "sealed_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(lock, "a", encoding="utf-8") as f:
        f.write(f"finished {report['finished']}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
