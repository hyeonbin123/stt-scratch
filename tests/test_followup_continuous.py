"""연속 음성 측정(scripts/followup.py continuous)과 드리프트 측정(scripts/stage6.py drift)의 새 인자 테스트.

docs/experiments.md 규칙 변경 27의 (B)·(C)를 재는 도구다. 확인하는 것: (1) continuous_streams가 발화를
매니페스트 순서 그대로 두고 발화 경계에서 자른다(발화와 뒤 간격을 더한 길이가 목표에 닿는 발화에서 스트림을
닫고, 남은 발화는 마지막 스트림), (2) stream_audio가 발화마다 뒤에 seed가 고정된 −60 dBFS 잡음 간격을 붙이고
정답을 공백으로 잇는다, (3) cmd_continuous가 작은 무작위 모델로 끝까지 돌아 조건마다 스트림 수, 1-pass
(stream_text 이음)·2-pass(확정 텍스트 이음) 가설, 규칙별 세그먼트 수를 맞게 기록하고 서버 명령행 기본 구성을
쓴다, (4) drift의 --step-path·--final-pass·--warm-up이 세션에 전달되고 기본값은 이전 동작(eager, 2차 패스
없음, 예열 없음, rule2 1.2 s, 캐시 초기화)이다.
"""

# [방식] tests/test_service.py와 같은 작은 무작위 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 돈다. 두 스크립트의
# ROOT를 tmp_path로 바꿔 가짜 매니페스트·PCM 저장소를 읽게 하고, 부하 기록(nvidia-smi·PowerShell 호출)은 가짜
# 함수로 바꾼다. 정확도(CER) 값 자체는 무작위 모델이라 보지 않고 기록의 구조와 정의만 본다.

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from stt_scratch.data import PcmStore, Utterance, read_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

REPO = Path(__file__).resolve().parents[1]
# scripts/는 패키지가 아니므로 경로를 더해 불러온다. followup이 stage6을 같은 이름으로 불러 쓰므로 둘은 같은
# 모듈 객체다(한쪽의 ROOT를 바꾸면 그 모듈의 함수에만 적용된다).
sys.path.insert(0, str(REPO / "scripts"))
followup = importlib.import_module("followup")
stage6 = importlib.import_module("stage6")

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))


def tiny_model(seed: int = 0) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(1,),
    )
    return ChunkConformerCTC(cfg).eval()


def utt(i: int, seconds: float, hangul: bool = True) -> Utterance:
    return Utterance(id=f"u{i}", corpus="cv22", text=f"가나 {i}", duration_s=seconds, hangul_only=hangul)


# 도우미: root 아래 manifests/<name>.jsonl과 data/pcm/<name>.{bin,index.jsonl}을 만든다. rows는 (id, 정답, 초,
# 한글 전용) 목록이고 PCM은 seed 고정 난수(표준편차 2000의 int16)라 작은 모델도 토큰을 낸다.
def fake_set(root: Path, name: str, rows: list[tuple[str, str, float, bool]]) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pcm_dir = root / "data" / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name))
    pieces, offset = [], 0
    with (
        open(root / "manifests" / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as manifest,
        open(pcm_dir / f"{name}.index.jsonl", "w", encoding="utf-8", newline="\n") as index,
    ):
        for uid, text, seconds, hangul in rows:
            n = int(seconds * 16000)
            line = {"id": uid, "corpus": "cv22", "text": text, "duration_s": seconds, "hangul_only": hangul}
            manifest.write(json.dumps(line, ensure_ascii=False) + "\n")
            index.write(json.dumps({"id": uid, "offset": offset, "length": n}) + "\n")
            pieces.append(np.clip(rng.standard_normal(n) * 2000, -32768, 32767).astype("<i2"))
            offset += n
    np.concatenate(pieces).tofile(pcm_dir / f"{name}.bin")


# 가짜 dev 두 세트: dev_mix는 1.5 s 발화 6개(그중 하나는 라틴 문자가 든 발화라 한글 전용이 아님), dev_H는 2 s
# 발화 10개(이어 붙이면 25 s라 드리프트에서 rule3 20 s 끝점이 한 번은 걸린다). 체크포인트 m.pt도 둔다.
@pytest.fixture
def fake_root(tmp_path) -> Path:
    syllables = ["가나", "다라", "마가", "나다", "라마", "가다", "나라", "다마", "라가", "마나"]
    mix = [(f"cv/{i}", syllables[i], 1.5, True) for i in range(6)]
    mix[2] = ("cv/2", "abc 가나", 1.5, False)
    fake_set(tmp_path, "dev_mix", mix)
    fake_set(tmp_path, "dev_h", [(f"zeroth/{i}", syllables[i], 2.0, True) for i in range(10)])
    model = tiny_model()
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, tmp_path / "m.pt")
    return tmp_path


# 확인: 스트림은 매니페스트 순서를 지키고 발화 경계에서만 잘리며, '발화 + 뒤 간격' 누적이 목표(초)에 닿는
# 발화에서 닫힌다(그래서 마지막을 뺀 스트림은 목표 이상). 간격이 누적에 들어가므로 간격에 따라 자르는 자리가
# 달라질 수 있고, 목표보다 긴 발화 하나는 혼자 스트림이 되며, 남은 발화는 짧아도 마지막 스트림이 된다.
def test_streams_are_cut_at_utterance_boundaries() -> None:
    streams = followup.continuous_streams
    assert streams([utt(0, 50), utt(1, 50), utt(2, 50), utt(3, 30)], 0.5, 120) == [[0, 1, 2], [3]]
    assert streams([utt(0, 59.6), utt(1, 59.6), utt(2, 10)], 0.5, 120) == [[0, 1], [2]]
    assert streams([utt(0, 59.6), utt(1, 59.6), utt(2, 10)], 0.3, 120) == [[0, 1, 2]]
    assert streams([utt(0, 130), utt(1, 5)], 0.5, 120) == [[0], [1]]
    assert streams([], 0.5, 120) == []


# 확인: stream_audio의 파형 = 발화마다 [그 발화 PCM, 간격 잡음]을 순서대로 이은 것이고, 간격 잡음은
# stage6.noise(간격, seed = CONTINUOUS_SEED + 그 발화의 세트 안 인덱스)와 같다(스트림이 세트 중간에서 시작해도
# 인덱스는 세트 기준). 같은 입력이면 같은 파형이 나오고(seed 고정), 정답은 발화 정답을 공백 하나로 이은
# 것이다.
def test_stream_audio_appends_seeded_gaps_and_joins_references(fake_root) -> None:
    utts = read_manifest(fake_root / "manifests" / "dev_h.jsonl")
    store = PcmStore(fake_root / "data" / "pcm", "dev_h")
    wav, ref = followup.stream_audio(utts, [3, 4, 5], store, 0.5)
    again, _ = followup.stream_audio(utts, [3, 4, 5], store, 0.5)
    assert torch.equal(wav, again) and ref == " ".join(utts[i].text for i in (3, 4, 5))
    pos = 0
    for i in (3, 4, 5):
        speech = store.waveform(utts[i].id)
        assert torch.equal(wav[pos : pos + speech.numel()], speech)
        pos += speech.numel()
        gap = stage6.noise(0.5, seed=followup.CONTINUOUS_SEED + i)
        assert gap.numel() == 8000 and torch.equal(wav[pos : pos + gap.numel()], gap)
        pos += gap.numel()
    assert pos == wav.numel()
    assert float(gap.abs().max()) < 10 ** (-60 / 20) * 6  # −60 dBFS 가우시안(진폭 1e-3)


def continuous_args(**overrides) -> argparse.Namespace:
    values = {
        "checkpoint": "m.pt",
        "out_dir": "out",
        "overwrite": False,
        "limit": 0,
        "gaps": "0.5,1.0",
        "dev_h_gaps": "0.5",
        "stream_minutes": 0.05,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# 확인: cmd_continuous가 등록 조건 세 개(dev_mix 간격 0.5·1.0 s, dev_H 0.5 s)를 모두 돌려 continuous.json과
# 조건별 가설 파일을 쓰고, 한글 전용 발화만 쓰며(dev_mix의 라틴 발화 제외), 스트림 수가 continuous_streams와
# 같고, 규칙별 세그먼트 수의 합 = 세그먼트 수, rule3 비율이 0~1이다. 구성은 서버 명령행 기본값(rule2 0.8 s,
# 끝점 뒤 캐시 유지, CPU fast 경로, 2차 패스 full)이다. 첫 스트림을 같은 구성의 세션으로 따로 흘려 얻은
# 1-pass(비지 않은 stream_text 이음)·2-pass(final 텍스트 이음)가 가설 파일과 같다. 결과 파일이 있으면
# --overwrite 없이 거부한다.
def test_continuous_command_records_every_condition(fake_root, monkeypatch) -> None:
    monkeypatch.setattr(followup, "ROOT", fake_root)
    monkeypatch.setattr(followup, "load_snapshot", lambda: {"stub": True})
    model = tiny_model()
    args = continuous_args()
    followup.cmd_continuous(args, model, VOCAB, torch.device("cpu"))
    result = json.loads((fake_root / "out" / "continuous.json").read_text(encoding="utf-8"))
    config = result["config"]
    assert (config["rule2_s"], config["reset_cache_on_endpoint"]) == (0.8, False)
    assert (config["step_path"], config["final_pass"], config["chunk"], config["left"]) == (
        "fast",
        "full",
        8,
        128,
    )
    assert set(result["conditions"]) == {"dev_mix_gap_0.5", "dev_mix_gap_1.0", "dev_h_gap_0.5"}
    for key, cond in result["conditions"].items():
        name = cond["set"]
        utts = [u for u in read_manifest(fake_root / "manifests" / f"{name}.jsonl") if u.hangul_only]
        streams = followup.continuous_streams(utts, cond["gap_s"], 3.0)
        assert cond["utterances"] == len(utts) == (5 if name == "dev_mix" else 10)
        assert cond["streams"] == len(streams) == cond["final_pass_vs_stream"]["n"]
        assert sum(cond["rules"].values()) == cond["segments"] >= cond["text_segments"]
        assert 0.0 <= cond["rule3_cut_rate"] <= 1.0
        assert cond["expression"] in ("gain_kept", "worse", "neither")
        assert cond["cer_ns_final_pass"] == cond["final_pass_vs_stream"]["rate_a"]
        hyps = fake_root / "out" / "hyps"
        stream_rows = [
            json.loads(x) for x in (hyps / f"continuous-{key}-stream.jsonl").open(encoding="utf-8")
        ]
        final_rows = [
            json.loads(x) for x in (hyps / f"continuous-{key}-final_pass.jsonl").open(encoding="utf-8")
        ]
        assert [r["utts"] for r in stream_rows] == [[utts[i].id for i in s] for s in streams]
        assert [r["ref"] for r in final_rows] == [" ".join(utts[i].text for i in s) for s in streams]
    # 첫 dev_mix 스트림을 따로 흘려 가설 정의를 확인한다.
    utts = [u for u in read_manifest(fake_root / "manifests" / "dev_mix.jsonl") if u.hangul_only]
    first = followup.continuous_streams(utts, 0.5, 3.0)[0]
    wav, _ = followup.stream_audio(utts, first, PcmStore(fake_root / "data" / "pcm", "dev_mix"), 0.5)
    session = StreamSession(
        model,
        VOCAB,
        ChunkSpec(8, 128),
        LogMel(),
        EndpointConfig(rule2_s=0.8),
        False,
        step_path="fast",
        final_pass="full",
    )
    closes = []
    for _pos, piece in stage6.packets(wav):
        closes.extend(e for e in session.push_samples(piece) if e.kind != "partial")
    closes.extend(session.flush())
    hyps = fake_root / "out" / "hyps"
    stream_row = json.loads(
        (hyps / "continuous-dev_mix_gap_0.5-stream.jsonl").open(encoding="utf-8").readline()
    )
    final_row = json.loads(
        (hyps / "continuous-dev_mix_gap_0.5-final_pass.jsonl").open(encoding="utf-8").readline()
    )
    assert stream_row["hyp"] == " ".join(e.stream_text for e in closes if e.stream_text.strip())
    assert final_row["hyp"] == " ".join(e.text for e in closes if e.kind == "final")
    with pytest.raises(SystemExit):
        followup.cmd_continuous(args, model, VOCAB, torch.device("cpu"))


# 확인: 간격 인자는 쉼표로 여러 값을 받고, 0 이하나 숫자가 아닌 값은 측정 전에 거부한다.
def test_gap_list_parsing() -> None:
    assert followup.parse_gaps("0.5,1.0") == [0.5, 1.0] and followup.parse_gaps(" 1 ") == [1.0]
    for bad in ("", "0", "-0.5", "a", "0.5,,1"):
        with pytest.raises(SystemExit):
            followup.parse_gaps(bad)


def drift_args(**overrides) -> argparse.Namespace:
    values = vars(stage6.build_parser().parse_args(["drift", "--checkpoint", "m.pt"]))
    values.update({"left": 64, "minutes": 0.4})
    values.update(overrides)
    return argparse.Namespace(**values)


# 확인: drift의 새 인자 기본값은 이전 동작(eager, 2차 패스 없음, 예열 없음; rule2 1.2 s·캐시 초기화도
# 그대로)이고 그때 결과의 2차 패스 시간·예열·그래프 상태는 null이다. --step-path fast --final-pass full
# --warm-up --policy keep --rule2 0.8(서버 기본 구성과 같은 성격)을 주면 세션이 2차 패스를 실제로 돌려 닫힌
# 세그먼트마다 2차 패스 시간이 기록되고(25 s 오디오라 rule3 20 s가 적어도 한 번 걸림) 예열 시간이 남는다. 왜:
# 10-18 기본값의 장시간 드리프트를 재려면 서버와 같은 step 경로·2차 패스·예열로 돌아야 한다(규칙 변경 27 (B)).
def test_drift_passes_the_step_path_and_final_pass(fake_root, monkeypatch) -> None:
    monkeypatch.setattr(stage6, "ROOT", fake_root)
    defaults = stage6.build_parser().parse_args(["drift", "--checkpoint", "m.pt"])
    assert (defaults.step_path, defaults.final_pass, defaults.warm_up) == ("eager", "none", False)
    assert (defaults.rule2, defaults.policy) == (1.2, "reset")
    model = tiny_model()
    cpu = torch.device("cpu")
    base = stage6.cmd_drift(drift_args(), model, VOCAB, cpu)
    assert (base["step_path"], base["final_pass"], base["warm_up_s"], base["graph"]) == (
        "eager",
        "none",
        None,
        None,
    )
    assert base["final_pass_ms"] is None and base["audio_s"] >= 24.0 and base["per_minute"]
    assert sum(base["closes_by_rule"].values()) >= base["segments_closed"] >= 0
    served = drift_args(step_path="fast", final_pass="full", warm_up=True, policy="keep", rule2=0.8)
    out = stage6.cmd_drift(served, model, VOCAB, cpu)
    assert (out["step_path"], out["final_pass"], out["policy"], out["rule2_s"]) == (
        "fast",
        "full",
        "keep",
        0.8,
    )
    assert out["warm_up_s"] is not None and out["warm_up_s"] > 0
    closes = sum(out["closes_by_rule"].values())
    assert closes >= 1 and out["final_pass_ms"]["n"] == closes and out["final_pass_ms"]["p50"] > 0
