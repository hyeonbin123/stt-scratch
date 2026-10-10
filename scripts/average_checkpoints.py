"""Average the top-k epoch checkpoints of a run by dev_mix full-context CER_ns (docs/experiments.md: top 10).

Usage: uv run python scripts/average_checkpoints.py --run outputs/stage2/runA --k 10 --out <path.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--dev", default="dev_mix")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run = ROOT / args.run
    epochs = [
        json.loads(line) for line in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    scored = [
        (e["dev"][args.dev]["full"], e["epoch"]) for e in epochs if "dev" in e and e["dev"].get(args.dev)
    ]
    chosen = sorted(scored)[: args.k]
    paths = [run / f"epoch-{ep:03d}.pt" for _, ep in chosen]
    avg = None
    config = None
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if avg is None:
            avg = {
                k: v.clone().double() if v.is_floating_point() else v.clone()
                for k, v in state["model"].items()
            }
            config = state["config"]
        else:
            for k, v in state["model"].items():
                if v.is_floating_point():
                    avg[k] += v.double()
    assert avg is not None, "no checkpoints found"
    for k, v in avg.items():
        if v.is_floating_point():
            avg[k] = (v / len(paths)).float()
    out = ROOT / args.out
    torch.save({"model": avg, "config": config, "averaged": [ep for _, ep in chosen], "dev": args.dev}, out)
    print("averaged epochs", [ep for _, ep in chosen], "->", out)


if __name__ == "__main__":
    main()
