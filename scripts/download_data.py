"""이 프로젝트가 쓰는 공개 한국어 음성 코퍼스를 Hugging Face Hub에서 data/raw/ 아래로 내려받는다.

출처(모두 Hugging Face Hub에서 게이트가 없는 저장소이며, 받기 전에 실행 중에 다시 확인한다):
- kresnik/zeroth_korean          Zeroth-Korean, CC BY 4.0, parquet(train 6조각 + test)
- google/fleurs  (ko_kr)         FLEURS 한국어, CC BY 4.0, refs/convert/parquet 브랜치의 parquet
- fsicoli/common_voice_17_0 (ko) Common Voice 17 한국어 미러, CC0 1.0, 오디오 tar + 전사 tsv
- fsicoli/common_voice_22_0 (ko) Common Voice 22 한국어 미러, CC0 1.0(실제로 씀, 17은 폴백으로 둠)

0단계 데이터 준비의 첫 명령이다(README).

사용법: uv run python scripts/download_data.py [--dest data/raw]
받은 모든 저장소의 정확한 revision(커밋 sha)과 파일 목록·크기를 <dest>/manifest.json에 쓴다.
"""

# [왜 필요한가] 데이터 원칙은 "계정이나 약관 동의 없이 받을 수 있는 공개 한국어 음성만
# 쓴다"이다(docs/data.md). 그래서 받기 전에 저장소가 게이트(접근 승인·약관 동의 필요)인지 확인하고, 게이트면
# 받지 않고 멈춘다. 커밋 sha를 남기는 이유는 나중에 같은 데이터를 다시 받거나 결과를 재현할 때 버전을 특정하기
# 위해서다(기록본: reports/data/download-manifest-2026-10-11.json).
# [흐름] 출처 목록(SOURCES)을 돌며 → 게이트 확인 → 허용 패턴에 맞는 파일만 snapshot_download → 받은 파일
# 목록과 크기를 manifest에 추가 → 마지막에 manifest.json 저장.
# [연결] 받은 원본은 scripts/inventory_data.py(재고 조사)와 scripts/prepare_data.py(정규화·분할·PCM 저장소)가
# 읽는다.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# huggingface_hub: HfApi.dataset_info로 저장소의 게이트 여부와 현재 커밋 sha를 조회하고, snapshot_download로
# 필요한 파일만 골라 받는다(datasets 라이브러리로 전체를 불러오지 않고 원본 파일 그대로 받는다).
from huggingface_hub import HfApi, snapshot_download

# 내려받을 출처 목록. 각 항목: name(로컬 폴더 이름), repo(HF 데이터셋 저장소), revision(None이면 main),
# allow(받을 파일 패턴), license(docs/data.md 표와 같은 라이선스 표기).
# - Zeroth-Korean: README와 parquet(16 kHz FLAC 음성 포함)만.
# - FLEURS: HF가 parquet로 변환해 둔 refs/convert/parquet 브랜치에서 한국어(ko_kr) parquet만
#   받는다(docs/data.md 출처 표).
# - Common Voice 17·22: 공식 저장소는 계정이 필요해 CC0 미러(fsicoli)에서 ko 오디오 tar와 전사 tsv만 받는다.
#   실제로 쓰는 것은 22이고 17은 22를 못 쓰게 될 때의 폴백이다(docs/data.md).
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
    {
        "name": "common_voice_22_ko",
        "repo": "fsicoli/common_voice_22_0",
        "revision": None,
        "allow": ["README.md", "audio/ko/*/*.tar", "transcript/ko/*.tsv"],
        "license": "CC0 1.0",
    },
]


# 출처를 차례로 내려받고 manifest.json을 갱신한다.
# 인자: --dest(저장 폴더, 기본 data/raw; 현재 작업 폴더 기준이므로 저장소 루트에서 실행한다), --only(일부 출처
# 이름만 받기).
# 기존 manifest.json이 있으면 읽어 두고, 이번에 받은 출처의 항목만 새것으로 바꾼다. 그래서 --only로 일부만
# 다시 받아도 다른 출처의 기록은 남는다.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument(
        "--only", nargs="*", help="내려받을 출처 이름(SOURCES의 name) 일부. 주지 않으면 전부 받는다"
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    for src in SOURCES:
        if args.only and src["name"] not in args.only:
            continue
        # 이 출처의 예전 기록을 지우고 아래에서 새 기록으로 넣는다(같은 이름이 두 번 들어가지 않게).
        manifest = [m for m in manifest if m["name"] != src["name"]]
        # 저장소 메타데이터를 조회한다. gated가 참("auto"·"manual")이면 접근 승인이나 약관 동의가 필요한
        # 저장소이므로, 사용자의 동의 없이 받지 않도록 여기서 중단한다. info.sha는 이 시점의 커밋으로, 아래
        # 기록에 쓴다(다운로드는 같은 revision 이름으로 받으므로 그 사이에 저장소가 바뀌면 기록과 어긋날 수
        # 있다).
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
        # 허용 패턴에 맞는 파일만 local_dir에 실제 파일로 받는다. 이미 받은 파일은 다시 받지 않으므로 중간에
        # 끊겨도 다시 실행하면 이어서 받는다. 동시 다운로드 4개.
        snapshot_download(
            repo_id=src["repo"],
            repo_type="dataset",
            revision=src["revision"],
            allow_patterns=src["allow"],
            local_dir=str(local),
            max_workers=4,
        )
        # 받은 파일 목록을 정렬해 기록한다. huggingface_hub가 local_dir 안에 만드는 .cache 폴더(다운로드
        # 메타데이터)는 데이터가 아니므로 뺀다. 경로는 OS와 무관하게 '/'로 적는다.
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
        # 출처별 파일 수와 총 용량(GB)을 찍어 받은 양을 확인한다(전체 약 5.5 GB, docs/data.md).
        total = sum(f["bytes"] for f in manifest[-1]["files"])
        print(f"[{src['name']}] {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    # 모든 출처를 처리한 뒤 한 번에 저장한다. 중간에 실패하면 이번 실행의 기록은 저장되지 않으므로 다시
    # 실행해야 한다.
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("manifest written:", dest / "manifest.json")


if __name__ == "__main__":
    main()
