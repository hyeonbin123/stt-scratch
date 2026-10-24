"""학습 체크포인트 저장(train.save_checkpoint)의 원자성 테스트.

epoch 끝에는 epochs.jsonl 요약 줄을 먼저 쓰고 epoch-NNN.pt와 last.pt(각 약 246 MB, D:는 HDD)를 차례로 쓴다.
저장 도중 멈추면(epoch 경계에서 멈추고 --resume하는 운영에서는 요약 줄이 생긴 것을 보고 멈출 수 있다) 제자리에
쓰던 last.pt가 깨져 --resume의 torch.load와 3·4단계 실행기의 completed_epochs가 zip 손상 예외로 멈춘다. 그래서
임시 파일(last.pt.tmp)에 다 쓴 뒤 os.replace로 바꿔 넣고, 실패하면 임시 파일을 지운다. 여기서는 torch.save를
'일부만 쓰고 예외'로 바꿔 끼워 이전 last.pt가 바이트 단위로 그대로인지 본다. GPU·실제 모델은 쓰지 않는다.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch import nn

from stt_scratch import train
from stt_scratch.model.config import ModelConfig

# 체크포인트의 "config"는 asdict(cfg)로 들어가므로 실제 ModelConfig를 작은 값으로 쓴다(모델 자체는 Linear
# 하나로 충분하다: save_checkpoint는 state_dict만 읽는다).
CFG = ModelConfig(vocab_size=12, d_model=8, n_layers=1, n_heads=2, ffn_dim=16, subsampling_channels=4)


@pytest.fixture
def parts() -> tuple[nn.Module, torch.optim.Optimizer, torch.amp.GradScaler]:
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # train.py가 CPU에서 쓰는 설정과 같다
    return model, opt, scaler


# train.py의 epoch 끝 호출과 같은 인자 모양으로 last.pt를 쓴다(완료 epoch, 처방 포함).
def save(run: Path, epoch: int, model, opt, scaler) -> None:
    train.save_checkpoint(
        run / "last.pt",
        model,
        opt,
        scaler,
        CFG,
        epoch,
        step=10 * (epoch + 1),
        counters={"updates": 3},
        dev_loss=[1.5, 1.2],
        rng=random.Random(epoch),
        device=torch.device("cpu"),
        complete=True,
        recipe={"lr": 1e-3},
    )


# 확인: 정상 저장은 torch.save를 임시 파일 last.pt.tmp에 하고(대상 last.pt에 직접 쓰지 않음) 끝나면 last.pt만
# 남으며, --resume과 같은 torch.load(weights_only=True)로 읽히고 completed_epochs가 epoch + 1을 돌려준다.
def test_save_checkpoint_writes_through_a_temporary_file(tmp_path, parts, monkeypatch) -> None:
    model, opt, scaler = parts
    targets: list[Path] = []
    original = torch.save

    def recording_save(obj, f, *args, **kwargs):
        targets.append(Path(f))
        original(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)
    save(tmp_path, 0, model, opt, scaler)
    assert targets == [tmp_path / "last.pt.tmp"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert state["epoch"] == 0 and state["epoch_complete"] and state["recipe"] == {"lr": 1e-3}
    assert torch.equal(state["model"]["weight"], model.weight)
    assert train.completed_epochs(tmp_path) == 1


# 확인: 저장 중 예외(디스크 오류 흉내: 일부 바이트만 쓰고 OSError)나 중단(KeyboardInterrupt)이 나도 이전
# last.pt는 바이트 단위로 그대로이고 그대로 읽히며, 임시 파일은 남지 않고, 예외는 그대로 올라간다. 왜: 재개와
# 단계 실행기가 읽는 파일이 깨진 채 남는 것이 원래 결함이다.
@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, parts, monkeypatch, error) -> None:
    model, opt, scaler = parts
    save(tmp_path, 0, model, opt, scaler)
    before = (tmp_path / "last.pt").read_bytes()

    def failing_save(obj, f, *args, **kwargs):
        Path(f).write_bytes(b"partial")
        raise error

    monkeypatch.setattr(torch, "save", failing_save)
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(type(error)):
        save(tmp_path, 1, model, opt, scaler)
    assert (tmp_path / "last.pt").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]
    assert torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)["epoch"] == 0
    assert train.completed_epochs(tmp_path) == 1
