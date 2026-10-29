"""데이터 내려받기 스크립트(scripts/download_data.py)의 커밋 고정과 게이트 검사 테스트.

확인하는 성질:
(a) 출처마다 HfApi.dataset_info로 조회한 커밋 sha(info.sha)를 snapshot_download의 revision으로 그대로 넘긴다.
    조회는 SOURCES의 revision 이름(None = main, refs/convert/parquet)으로 하고 내려받기는 그 조회가 돌려준
    커밋으로 하므로, 조회와 내려받기 사이에 브랜치가 움직여도 manifest의 commit_sha와 실제로 받은 파일이
    어긋나지 않는다.
(b) manifest.json의 revision은 사람이 읽는 이름(main, refs/convert/parquet) 그대로이고 commit_sha가 실제로
    받은 커밋이다. 파일 목록은 local_dir에 실제로 생긴 파일과 크기다(huggingface_hub의 .cache 폴더는 뺀다).
(c) 게이트 저장소(gated가 "auto"·"manual")면 받지 않고 SystemExit로 멈추며 manifest.json도 쓰지 않는다.
(d) 조회 결과에 커밋 sha가 없으면 움직이는 이름으로 받는 대신 멈춘다(고정할 커밋이 없으므로).

네트워크는 쓰지 않는다: HfApi와 snapshot_download를 가짜로 바꿔 호출 인자만 기록한다.
"""

# [왜 필요한가] 이 스크립트는 받은 저장소의 커밋 sha를 manifest에 남겨 나중에 같은 데이터를 특정할 수 있게
# 한다(기록본 reports/data/download-manifest-2026-10-11.json). 그런데 내려받기를 움직이는 참조(main 브랜치,
# refs/convert/parquet)로 하면 조회 뒤 저장소가 바뀌었을 때 기록된 sha와 다른 파일이 들어와도 알 수 없다(네
# 출처 가운데 Common Voice 두 개는 제3자 미러라 내용이 바뀔 수 있다). 데이터는 이미 받아 두었으므로 스크립트를
# 실제로 다시 돌리지 않고, 호출 인자를 가짜 객체로 가로채 커밋이 고정되는지 확인한다.
# [방식] scripts/는 패키지가 아니라서 파일 경로로 모듈을 불러온다(tests/test_sealed_breakdown.py와 같은 방식).
# main()은 sys.argv를 읽으므로 monkeypatch로 명령줄 인자를 넣는다.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("download_data", REPO / "scripts" / "download_data.py")
download_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_data)

# 가짜 조회 결과가 돌려줄 커밋 sha(40자 16진수). 저장소마다 값이 달라 어느 출처의 sha가 어느 내려받기로
# 갔는지 구별된다.
SHAS = {
    "kresnik/zeroth_korean": "1" * 40,
    "google/fleurs": "2" * 40,
}


class FakeHub:
    """HfApi.dataset_info와 snapshot_download를 대신하는 가짜.

    gated: 조회 결과의 gated 값(False면 공개 저장소). shas: 저장소 이름 → 조회 결과의 sha(없으면 None).
    info_calls에는 (저장소, revision) 조회 인자를, download_calls에는 snapshot_download의 키워드 인자를
    부른 순서대로 쌓는다. 내려받기 흉내로 local_dir에 작은 데이터 파일 하나와 huggingface_hub가 만드는 것과
    같은 자리의 .cache 메타데이터 파일 하나를 만든다(manifest에서 .cache가 빠지는지 보려고).
    """

    def __init__(self, gated: bool | str = False, shas: dict[str, str] | None = None) -> None:
        self.gated = gated
        self.shas = SHAS if shas is None else shas
        self.info_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[dict] = []

    def api(self):
        # main()이 HfApi()로 만드는 객체 자리에 들어간다. main은 dataset_info만 쓴다.
        hub = self

        class _Api:
            def dataset_info(self, repo_id, revision=None, **kwargs):
                hub.info_calls.append((repo_id, revision))
                return SimpleNamespace(sha=hub.shas.get(repo_id), gated=hub.gated)

        return _Api()

    def snapshot_download(self, **kwargs):
        self.download_calls.append(kwargs)
        local = Path(kwargs["local_dir"])
        (local / "data").mkdir(parents=True, exist_ok=True)
        (local / "data" / "part-0.parquet").write_bytes(b"x" * 7)
        (local / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (local / ".cache" / "huggingface" / "meta").write_bytes(b"m")
        return str(local)


def run_main(monkeypatch, hub: FakeHub, dest: Path, only: list[str]) -> None:
    """가짜 허브를 끼우고 `download_data.py --dest <dest> --only <only...>`를 실행한다."""
    monkeypatch.setattr(download_data, "HfApi", hub.api)
    monkeypatch.setattr(download_data, "snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--dest", str(dest), "--only", *only])
    download_data.main()


def test_downloads_the_commit_that_was_looked_up(monkeypatch, tmp_path):
    hub = FakeHub()
    run_main(monkeypatch, hub, tmp_path, ["zeroth_korean", "fleurs_ko"])

    # 조회는 SOURCES의 revision 이름으로 하고(main은 None), 내려받기는 그 조회가 돌려준 커밋으로 한다.
    assert hub.info_calls == [("kresnik/zeroth_korean", None), ("google/fleurs", "refs/convert/parquet")]
    assert [c["repo_id"] for c in hub.download_calls] == ["kresnik/zeroth_korean", "google/fleurs"]
    assert [c["revision"] for c in hub.download_calls] == [
        SHAS["kresnik/zeroth_korean"],
        SHAS["google/fleurs"],
    ]
    # 받는 파일 패턴·저장소 종류·저장 위치는 바뀌지 않는다.
    fleurs = hub.download_calls[1]
    assert fleurs["repo_type"] == "dataset"
    assert fleurs["allow_patterns"] == ["ko_kr/*/*.parquet"]
    assert Path(fleurs["local_dir"]) == tmp_path / "fleurs_ko"

    # manifest: revision은 이름 그대로, commit_sha는 실제로 받은 커밋, 파일 목록에는 .cache가 없다.
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [(m["name"], m["revision"], m["commit_sha"]) for m in manifest] == [
        ("zeroth_korean", "main", SHAS["kresnik/zeroth_korean"]),
        ("fleurs_ko", "refs/convert/parquet", SHAS["google/fleurs"]),
    ]
    assert all(m["commit_sha"] == c["revision"] for m, c in zip(manifest, hub.download_calls, strict=True))
    assert manifest[0]["files"] == [{"path": "data/part-0.parquet", "bytes": 7}]


@pytest.mark.parametrize("gated", ["auto", "manual"])
def test_gated_repository_stops_before_download(monkeypatch, tmp_path, gated):
    hub = FakeHub(gated=gated)
    with pytest.raises(SystemExit, match="gated"):
        run_main(monkeypatch, hub, tmp_path, ["zeroth_korean"])
    assert hub.info_calls == [("kresnik/zeroth_korean", None)]
    assert hub.download_calls == []
    assert not (tmp_path / "manifest.json").exists()


def test_missing_commit_sha_stops_instead_of_following_the_branch(monkeypatch, tmp_path):
    hub = FakeHub(shas={})
    with pytest.raises(SystemExit, match="commit sha"):
        run_main(monkeypatch, hub, tmp_path, ["fleurs_ko"])
    assert hub.info_calls == [("google/fleurs", "refs/convert/parquet")]
    assert hub.download_calls == []
    assert not (tmp_path / "manifest.json").exists()
