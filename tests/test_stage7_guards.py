"""봉인 test 보호 장치(scripts/stage7.py) 테스트. 임시 ROOT에서 돌리므로 실제 봉인은 건드리지 않는다.

두 번째 시작 거부, 다른 구성으로의 재개 거부, 끝나지 않은 봉인만 재개, 실행 중 잠금(동시 실행 방지)과 해제,
저장된 세트 결과를 같은 출처일 때만 재사용하는지 확인한다.
"""

# [왜 중요한가] 봉인 test는 사전 등록 구성으로 단 한 번만 재고, 잰 뒤에는 모델·규칙을 바꾸지 않는 것이 측정
# 규칙의 핵심이다(docs/experiments.md 7단계). Codex 6단계 검토가 '재개가 사전 검사·구성 대조 없이 기존 결과를
# 섞을 수 있다', '재개에 배타적 잠금이 없다'는 P1 두 건을 지적해 보호 장치를 넣었고(규칙 변경 25), 이 테스트가
# 그것을 고정한다.
# [방식] stage7 모듈의 ROOT를 임시 폴더로 바꿔 실행하므로 실제 봉인 기록(reports/final/)은 건드리지 않는다.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# scripts/stage7.py를 파일 경로로 불러온다(scripts/는 패키지가 아님). 모듈 정의만 실행되고 main은 테스트가
# 직접 부른다.
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage7", REPO / "scripts" / "stage7.py")
stage7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage7)

# stage7이 시작할 때 sealed_config.json에 남기는 '봉인 구성'과 같은 모양의 값(SHA-256은 가짜 고정값). 재개 시
# 이것과 지금 구성이 같아야 한다.
CONFIG = {
    "checkpoint": "outputs/stage4/trimtail/avg10.pt",
    "checkpoint_sha256": "a" * 64,
    "spec": "chunk8 left128",
    "precision": "fp16-autocast",
    "device": "cuda",
}


# 임시 저장소 fixture: 최소 매니페스트와 가짜 체크포인트 파일을 만들고 monkeypatch로 (1) ROOT를 임시 폴더로,
# (2) GPU·6단계 결과가 필요한 preflight를 아무것도 안 하게, (3) 체크포인트 SHA-256을 고정값으로, (4) 실제 측정
# 함수 run_sealed를 호출되면 실패하도록 바꾼다. (4) 덕분에 거부되어야 할 경로가 측정까지 가면 테스트가
# "ran"으로 실패한다.
@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "test_zeroth.jsonl").write_text(
        json.dumps({"id": "u1", "text": "가", "duration_s": 1.0, "hangul_only": True, "corpus": "zeroth"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "final").mkdir(parents=True)
    (tmp_path / "outputs" / "stage4" / "trimtail").mkdir(parents=True)
    (tmp_path / "outputs" / "stage4" / "trimtail" / "avg10.pt").write_bytes(b"x")
    monkeypatch.setattr(stage7, "ROOT", tmp_path)
    monkeypatch.setattr(stage7, "preflight", lambda args, ckpt: None)
    monkeypatch.setattr(stage7, "sha256", lambda path: "a" * 64)
    monkeypatch.setattr(stage7, "run_sealed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    return tmp_path


# 봉인 test는 --confirm-sealed-test 확인 플래그 없이는 시작하지 않으므로 모든 호출에 넣는다.
ARGS = ["--checkpoint", "outputs/stage4/trimtail/avg10.pt", "--confirm-sealed-test"]


# 확인: 봉인 잠금 파일(sealed_test.lock)이 이미 있으면 새 시작이 거부된다. 봉인 test를 두 번 재는 일을 막는다.
def test_second_start_is_refused(root) -> None:
    (root / "reports" / "final" / "sealed_test.lock").write_text("started x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already started"):
        stage7.main(ARGS)


# 확인: 중단된 봉인을 재개할 때, 저장된 봉인 구성과 다르면(청크 16) 거부하고, 저장된 구성이 맞아도 재개 명령이
# 다른 청크를 요구하면 거부한다. 서로 다른 구성의 결과가 한 보고서에 섞이는 것을 막는다.
def test_resume_refuses_another_configuration(root) -> None:
    final = root / "reports" / "final"
    (final / "sealed_test.lock").write_text("started x\n", encoding="utf-8")
    (final / "sealed_config.json").write_text(
        json.dumps({**CONFIG, "spec": "chunk16 left128"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="resume refused"):
        stage7.main([*ARGS, "--resume-sealed"])
    with pytest.raises(SystemExit, match="resume refused"):  # 재개 명령이 청크 16을 요구
        (final / "sealed_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        stage7.main([*ARGS, "--resume-sealed", "--chunk", "16"])


# 확인: 잠금이 없거나(시작한 적 없음) 이미 finished가 적힌 봉인은 재개할 것이 없다며 거부한다.
def test_resume_needs_an_unfinished_seal(root) -> None:
    with pytest.raises(SystemExit, match="nothing to resume"):
        stage7.main([*ARGS, "--resume-sealed"])
    lock = root / "reports" / "final" / "sealed_test.lock"
    lock.write_text("started x\nfinished y\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing to resume"):
        stage7.main([*ARGS, "--resume-sealed"])


# 확인: 실행 중 잠금(sealed_test.running)이 있으면 다른 프로세스가 돌고 있다고 보고 재개를 거부한다. 잠금을
# 지운 뒤 재개하면 측정 함수가 도는 동안 실행 중 잠금이 존재하고, 끝나면 지워진다(배타적 생성 + finally 해제).
def test_a_running_process_blocks_resume_and_the_run_lock_is_released(root, monkeypatch) -> None:
    final = root / "reports" / "final"
    (final / "sealed_test.lock").write_text("started x\n", encoding="utf-8")
    (final / "sealed_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (final / "sealed_test.running").write_text("pid 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="another sealed-test process"):
        stage7.main([*ARGS, "--resume-sealed"])
    (final / "sealed_test.running").unlink()
    calls = []
    monkeypatch.setattr(
        stage7, "run_sealed", lambda *a: calls.append((final / "sealed_test.running").exists())
    )
    stage7.main([*ARGS, "--resume-sealed"])
    assert calls == [True] and not (final / "sealed_test.running").exists()


# 확인: 중단 전에 저장된 세트 결과는 세트·모드·정밀도·체크포인트·SHA-256·발화 수가 봉인 구성과 모두 같을 때만
# 재사용되고, 모드(청크 16), SHA-256, 발화 수 중 하나라도 다르면 거부된다. 다시 재지 않고 저장값을 쓰는 재개
# 경로가 다른 출처의 숫자를 섞지 않게 한다.
def test_saved_results_are_reused_only_with_the_sealed_provenance(root) -> None:
    path = root / "reports" / "final" / "test_zeroth-streaming.json"
    good = {
        "manifest": "test_zeroth",
        "mode": "streaming chunk8 left128",
        "precision": "fp16-autocast",
        "checkpoint": CONFIG["checkpoint"],
        "checkpoint_sha256": CONFIG["checkpoint_sha256"],
        "n": 1,
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    path.with_suffix(".hyps.jsonl").write_text("{}\n", encoding="utf-8")
    assert stage7.check_saved(path, "test_zeroth", "streaming", CONFIG)["n"] == 1
    for key, value in (("mode", "streaming chunk16 left128"), ("checkpoint_sha256", "b" * 64), ("n", 2)):
        path.write_text(json.dumps({**good, key: value}), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not match"):
            stage7.check_saved(path, "test_zeroth", "streaming", CONFIG)
