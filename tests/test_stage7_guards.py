"""Sealed-test guards (scripts/stage7.py), run against a temporary ROOT: the real seal is never touched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage7", REPO / "scripts" / "stage7.py")
stage7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage7)

CONFIG = {
    "checkpoint": "outputs/stage4/trimtail/avg10.pt",
    "checkpoint_sha256": "a" * 64,
    "spec": "chunk8 left128",
    "precision": "fp16-autocast",
    "device": "cuda",
}


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


ARGS = ["--checkpoint", "outputs/stage4/trimtail/avg10.pt", "--confirm-sealed-test"]


def test_second_start_is_refused(root) -> None:
    (root / "reports" / "final" / "sealed_test.lock").write_text("started x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already started"):
        stage7.main(ARGS)


def test_resume_refuses_another_configuration(root) -> None:
    final = root / "reports" / "final"
    (final / "sealed_test.lock").write_text("started x\n", encoding="utf-8")
    (final / "sealed_config.json").write_text(
        json.dumps({**CONFIG, "spec": "chunk16 left128"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="resume refused"):
        stage7.main([*ARGS, "--resume-sealed"])
    with pytest.raises(SystemExit, match="resume refused"):  # chunk 16 requested on resume
        (final / "sealed_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        stage7.main([*ARGS, "--resume-sealed", "--chunk", "16"])


def test_resume_needs_an_unfinished_seal(root) -> None:
    with pytest.raises(SystemExit, match="nothing to resume"):
        stage7.main([*ARGS, "--resume-sealed"])
    lock = root / "reports" / "final" / "sealed_test.lock"
    lock.write_text("started x\nfinished y\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing to resume"):
        stage7.main([*ARGS, "--resume-sealed"])


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
