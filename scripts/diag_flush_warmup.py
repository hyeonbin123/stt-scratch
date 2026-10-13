"""Exploratory (not a Stage 6 verdict): is the slow flush right after server start a first-use cost of each
new partial-last-chunk length? Times session.flush() (and its encoder step) for the first 50 dev_H utterances
in one process, in the order the client sent them, then repeats the same 50 in a second pass.

    uv run python scripts/diag_flush_warmup.py
"""

import json
import statistics as st
import time
from pathlib import Path

import torch

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda")
model, _ = load_model(ROOT / "outputs/stage4/trimtail/avg10.pt", device)
vocab = Vocab.load(ROOT / "manifests/vocab.json")
logmel = LogMel().to(device)
utts = read_manifest(ROOT / "manifests/dev_h.jsonl")[:50]
store = PcmStore(ROOT / "data/pcm", "dev_h")
endpoint = EndpointConfig(rule2_s=1.2)
rows = []
for rep in range(2):
    for k, u in enumerate(utts):
        offset, length = store.index[u.id]
        pcm = bytes(memoryview(store.bin[offset : offset + length]).cast("B"))
        s = StreamSession(model, vocab, ChunkSpec(8, 128), logmel, endpoint, False)
        i, size = 0, 5360 * 2
        while i < len(pcm):
            s.push_pcm16(pcm[i : i + size])
            i += size
            size = 5120 * 2
        pending = int(s.pending.shape[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev = s.flush()
        torch.cuda.synchronize()
        rows.append(
            {
                "pass": rep,
                "k": k,
                "pending_mel": pending,
                "flush_ms": (time.perf_counter() - t0) * 1000,
                "step_ms": ev[0].compute_ms if ev and hasattr(ev[0], "compute_ms") else None,
            }
        )
seen = set()
for r in rows:
    r["new_length"] = r["pending_mel"] not in seen
    seen.add(r["pending_mel"])
out = ROOT / "reports/stage6/diag-flush_warmup.json"
out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
for r in rows[:15]:
    print(r)

for rep in (0, 1):
    a = [r["flush_ms"] for r in rows if r["pass"] == rep]
    new = [r["flush_ms"] for r in rows if r["pass"] == rep and r["new_length"]]
    old = [r["flush_ms"] for r in rows if r["pass"] == rep and not r["new_length"]]
    print(
        "pass",
        rep,
        "median",
        round(st.median(a), 1),
        "max",
        round(max(a), 1),
        "new-length n",
        len(new),
        "median",
        round(st.median(new), 1) if new else None,
        "seen-length n",
        len(old),
        "median",
        round(st.median(old), 1) if old else None,
    )
