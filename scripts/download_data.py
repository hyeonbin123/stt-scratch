"""Download the public Korean speech corpora used by this project into data/raw/.

Sources (all ungated on the Hugging Face Hub, checked at runtime before downloading):
- kresnik/zeroth_korean          Zeroth-Korean, CC BY 4.0, parquet (train 6 shards + test)
- google/fleurs  (ko_kr)         FLEURS Korean, CC BY 4.0, parquet from the refs/convert/parquet branch
- fsicoli/common_voice_17_0 (ko) Common Voice 17 Korean mirror, CC0 1.0, audio tar + transcript tsv

Usage: uv run python scripts/download_data.py [--dest data/raw]
Writes <dest>/manifest.json with the exact revision (commit sha) and file list of every download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCES = [
    {
        "name": "zeroth_korean",
        "repo": "kresnik/zeroth_korean",
        "revision": None,
        "allow": ["README.md", "data/*.parquet"],
        "license": "CC BY 4.0",
    },
    {
        "name": "fleurs_ko",
        "repo": "google/fleurs",
        "revision": "refs/convert/parquet",
        "allow": ["ko_kr/*/*.parquet"],
        "license": "CC BY 4.0",
    },
    {
        "name": "common_voice_17_ko",
        "repo": "fsicoli/common_voice_17_0",
        "revision": None,
        "allow": ["README.md", "audio/ko/*/*.tar", "transcript/ko/*.tsv"],
        "license": "CC0 1.0",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw")
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = []
    for src in SOURCES:
        info = api.dataset_info(src["repo"], revision=src["revision"])
        if info.gated:
            raise SystemExit(
                f"{src['repo']} is gated ({info.gated}); refusing to download without the user's consent"
            )
        local = dest / src["name"]
        print(
            f"[{src['name']}] {src['repo']} rev={src['revision'] or 'main'} sha={info.sha} -> {local}",
            flush=True,
        )
        snapshot_download(
            repo_id=src["repo"],
            repo_type="dataset",
            revision=src["revision"],
            allow_patterns=src["allow"],
            local_dir=str(local),
            max_workers=4,
        )
        files = sorted(p for p in local.rglob("*") if p.is_file() and ".cache" not in p.parts)
        manifest.append(
            {
                "name": src["name"],
                "repo": src["repo"],
                "revision": src["revision"] or "main",
                "commit_sha": info.sha,
                "license": src["license"],
                "gated": info.gated,
                "files": [
                    {"path": str(p.relative_to(local)).replace("\\", "/"), "bytes": p.stat().st_size}
                    for p in files
                ],
            }
        )
        total = sum(f["bytes"] for f in manifest[-1]["files"])
        print(f"[{src['name']}] {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("manifest written:", dest / "manifest.json")


if __name__ == "__main__":
    main()
