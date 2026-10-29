"""측정 클라이언트(scripts/stream_client.py)의 기록·요약 테스트(docs/experiments.md 규칙 변경 27).

확인하는 것: (1) 서버 이벤트에서 발화 하나의 확정 기록을 모으는 close_record가 2차 패스를 켠 서버의
stream_text(endpoint 이벤트 것 포함)를 잇고, 그 밖(2차 패스를 끈 서버, stream_text를 보내기 전 서버, 설정을
모름)에는 None으로 둔다, (2) summarize가 기록의 서버 설정(-meta.json의 server_info.final_pass)에 따라 텍스트
비교를 고른다: full이면 stream_text를 simulated 가설과(공백 포함·공백 무시 두 수), 확정 텍스트를 오프라인
재생과 비교해 유효성 조건을 계산하고, none(또는 그 키가 없는 2차 패스 이전 서버)이면 6단계처럼 확정 텍스트를
simulated 가설과 비교하며, 판별할 수 없으면 null과 경고를 남긴다, (3) --replay-check의 오프라인 재생이 실제
서버 경로(TestClient WebSocket)와 같은 확정 기록을 내고 체크포인트가 다르면 거부한다, (4) 커밋된 6단계
클라이언트 기록을 지금 코드로 다시 요약해도 커밋된 요약 값이 그대로 나온다(로컬 data/pcm이 없으면 건너뜀).
"""

# [방식] 서버는 실제 포트에 띄우지 않고 FastAPI TestClient로 앱을 직접 부른다. 모델은 tests/test_service.py와
# 같은 작은 무작위 모델(d=32, 2층)과 8토큰 어휘라 학습된 가중치 없이 CPU에서 돈다. 매니페스트·PCM 저장소·가설
# 파일은 tmp_path에 만든 가짜이고, summarize·replay_expected에 root·out_dir로 그 폴더를 넘긴다.

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.text import Vocab

REPO = Path(__file__).resolve().parents[1]
# scripts/는 패키지가 아니므로 경로를 더해 모듈로 불러온다(측정할 때와 같은 파일을 그대로 시험한다).
sys.path.insert(0, str(REPO / "scripts"))
stream_client = importlib.import_module("stream_client")

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))
SIMULATED = "reports/stage4/hyps/trimtail-avg10-dev_h-chunk8_320ms.jsonl"
# (4)에 쓰는 로컬 실데이터: 0단계가 만든 dev_H PCM 색인(마지막 패킷 길이 계산에 필요). CI에는 없다.
REAL = pytest.mark.skipif(
    not (REPO / "data" / "pcm" / "dev_h.index.jsonl").exists(), reason="needs the local data/pcm dev_h store"
)


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


# 도우미: tmp_path 아래 manifests/<manifest>.jsonl과 data/pcm/<manifest>.{bin,index.jsonl}을 만든다. utts는
# (id, 정답, 표본 수) 목록이고 PCM은 seed 고정 난수(표준편차 2000의 int16)라 작은 모델도 토큰을 낸다.
def fake_store(root: Path, utts: list[tuple[str, str, int]], manifest: str = "dev_h") -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pcm_dir = root / "data" / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pieces, offset, lines, index = [], 0, [], []
    for uid, text, n in utts:
        lines.append(
            {"id": uid, "corpus": "zeroth", "text": text, "duration_s": n / 16000, "hangul_only": True}
        )
        index.append({"id": uid, "offset": offset, "length": n})
        pieces.append(np.clip(rng.standard_normal(n) * 2000, -32768, 32767).astype("<i2"))
        offset += n
    write_jsonl(root / "manifests" / f"{manifest}.jsonl", lines)
    write_jsonl(pcm_dir / f"{manifest}.index.jsonl", index)
    np.concatenate(pieces).tofile(pcm_dir / f"{manifest}.bin")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 도우미: stream_one이 쓰는 한 줄과 같은 모양의 클라이언트 기록. 지연 값은 요약 계산이 돌 만큼만 채운다.
def client_row(uid: str, events: list[dict], final_pass: str | None) -> dict:
    return {
        "id": uid,
        **stream_client.close_record(events, final_pass),
        "n_partials": 1,
        "first_partial_s": 0.5,
        "partial_lag_ms": [60.0, 61.0],
        "final_after_last_audio_s": 0.2,
        "final_after_flush_s": 0.05,
        "paced": True,
        "chunks": 7,
        "compute_ms_p50": 8.0,
        "compute_ms_p90": 9.0,
        "compute_ms_max": 30.0,
    }


def close(
    kind: str, segment: int, text: str, rule: str, stream_text: str | None = None, ms: float = 0.0
) -> dict:
    event = {"type": kind, "segment": segment, "text": text, "rule": rule, "frames": 20, "final_pass_ms": ms}
    if stream_text is not None:
        event["stream_text"] = stream_text
    return event


# 확인: close_record가 final 텍스트를 이어 확정 텍스트로 쓰고(6단계와 같은 정의), 2차 패스를 켠 서버면 닫힘
# 이벤트(final·endpoint)의 비지 않은 stream_text를 공백으로 이어 스트리밍 텍스트로 쓴다. 2차 패스 결과가 비어
# endpoint로 닫힌 세그먼트의 stream_text도 들어가야 한다(followup.py 세션 측정의 1-pass 정의와 같음). 2차
# 패스를 끈 서버, 설정을 모를 때, 닫힘 이벤트 하나라도 stream_text가 없을 때(필드를 보내기 전 서버)는
# None이다.
def test_close_record_joins_stream_text_of_every_close() -> None:
    events = [
        {"type": "partial", "segment": 0, "text": "가"},
        close("final", 0, "가나", "rule2", "가 나", 5.0),
        close("endpoint", 1, "", "rule2", "다", 4.0),
        close("endpoint", 2, "", "rule1", "", 3.0),
        close("final", 3, "라마", "eos", "라마", 6.0),
        {"type": "stats", "chunks": 9},
    ]
    record = stream_client.close_record(events, "full")
    assert (record["text"], record["stream_text"]) == ("가나 라마", "가 나 다 라마")
    assert (record["n_finals"], record["rules"]) == (2, ["rule2", "eos"])
    assert (record["n_endpoints"], record["endpoint_rules"]) == (2, ["rule2", "rule1"])
    assert [e["segment"] for e in record["closes"]] == [0, 1, 2, 3]
    assert stream_client.close_record(events, None)["stream_text"] is None
    assert stream_client.close_record(events, "none")["stream_text"] is None
    missing = [close("final", 0, "가나", "eos")]
    assert stream_client.close_record(missing, "full")["stream_text"] is None
    assert stream_client.close_record([], "full")["stream_text"] == ""


# 도우미: 요약 시험용 가짜 저장소(발화 2개)와 simulated 가설 파일, 클라이언트 기록을 만든다. u1은
# stream_text가 simulated와 정확히 같고 u2는 공백만 다르다(rule2 0.8 s에서 세그먼트가 늘면 이음 자리의 공백이
# 다를 수 있다).
def summary_case(tmp_path: Path, final_pass: str | None, info: dict | None, replay: dict | None) -> dict:
    fake_store(tmp_path, [("u1", "가나다", 20000), ("u2", "라마", 12000)])
    write_jsonl(tmp_path / "sim.jsonl", [{"id": "u1", "hyp": "가나 다"}, {"id": "u2", "hyp": "라마"}])
    rows = [
        client_row("u1", [close("final", 0, "가나다라", "eos", "가나 다", 7.0)], final_pass),
        client_row(
            "u2",
            [close("final", 0, "라", "rule2", "라", 6.0), close("endpoint", 1, "", "rule2", "마", 5.0)],
            final_pass,
        ),
    ]
    out_dir = tmp_path / "reports"
    shutil.rmtree(out_dir, ignore_errors=True)  # 앞 호출이 남긴 -meta.json·-replay.jsonl을 지운다
    write_jsonl(out_dir / "c.jsonl", rows)
    if info is not None:
        (out_dir / "c-meta.json").write_text(json.dumps({"server_info": info}), encoding="utf-8")
    if replay is not None:
        write_jsonl(out_dir / "c-replay.jsonl", [{"id": k, "text": v} for k, v in replay.items()])
    return stream_client.summarize("dev_h", "sim.jsonl", "c", root=tmp_path, out_dir=out_dir)


# 확인: 2차 패스를 켠 서버의 기록이면 확정 텍스트(2차 패스 결과)를 simulated와 비교하지 않고(null),
# stream_text 이음을 simulated와 비교해 공백 포함 1·공백 무시 2를 세며, 확정 텍스트는 오프라인 재생과
# 비교한다. 두 조건이 모두 2/2면 유효성 조건 충족, 재생이 하나라도 다르면 미충족, 재생 기록이 없으면 판정
# 불가(null)다. 2차 패스 시간은 닫힘 이벤트마다의 값으로, endpoint 규칙은 따로 센다. 요약 파일도 같은 내용으로
# 쓴다.
def test_summary_of_a_final_pass_server_compares_stream_text_and_replay(tmp_path) -> None:
    out = summary_case(tmp_path, "full", {"final_pass": "full"}, {"u1": "가나다라", "u2": "라"})
    assert (out["final_pass"], out["same_text_as_simulated"]) == ("full", None)
    assert out["same_stream_text_as_simulated"] == {"exact": 1, "ignoring_spaces": 2}
    assert (out["same_final_text_as_replay"], out["warnings"]) == (2, [])
    assert out["validity"] == {
        "stream_text_matches_simulated_ignoring_spaces": True,
        "final_text_matches_replay": True,
        "met": True,
    }
    assert out["final_pass_ms"]["n"] == 3 and out["final_pass_ms"]["max"] == 7.0
    assert out["endpoint_rules"]["rule2"] == 1 and out["final_rules"]["rule2"] == 1
    assert out["cer_ns_hangul_only_stream_text"] is not None
    saved = json.loads((tmp_path / "reports" / "c-summary.json").read_text(encoding="utf-8"))
    assert saved["validity"]["met"] is True
    out = summary_case(tmp_path, "full", {"final_pass": "full"}, {"u1": "가나다라", "u2": "라마"})
    assert out["same_final_text_as_replay"] == 1 and out["validity"]["met"] is False
    out = summary_case(tmp_path, "full", {"final_pass": "full"}, None)
    assert out["same_final_text_as_replay"] is None and out["validity"]["met"] is None
    # 재생 기록이 발화 일부만 덮으면 비교하지 않고 경고한다.
    out = summary_case(tmp_path, "full", {"final_pass": "full"}, {"u1": "가나다라"})
    assert out["same_final_text_as_replay"] is None and out["warnings"]


# 확인: 2차 패스를 끈 서버(final_pass none)와 그 키가 없는 이전 서버(6단계 기록)는 확정 텍스트가 곧 스트리밍
# 텍스트이므로 6단계처럼 확정 텍스트를 simulated와 비교한다. 서버 설정을 알 수 없는 기록(-meta.json 없음)과,
# 2차 패스 서버인데 stream_text가 없는 기록(필드를 보내기 전 서버나 이전 클라이언트)은 텍스트 비교를 null로
# 두고 경고를 남긴다. 왜: 2차 패스 서버의 확정 텍스트를 simulated와 비교하면 구조적으로 거의 0이 되어 서비스
# 경로 버그처럼 보였다.
def test_summary_picks_the_comparison_from_the_server_setting(tmp_path) -> None:
    out = summary_case(tmp_path, None, {"final_pass": "none"}, None)
    assert (out["final_pass"], out["same_text_as_simulated"]) == ("none", 0)
    assert out["same_stream_text_as_simulated"] == {"exact": 0, "ignoring_spaces": 0}
    assert out["final_pass_ms"] is None and out["warnings"] == []
    legacy = summary_case(tmp_path, None, {"chunk_ms": 320}, None)
    assert legacy["final_pass"] == "none" and "predates" in legacy["final_pass_basis"]
    unknown = summary_case(tmp_path, None, None, None)
    assert unknown["final_pass"] is None and unknown["warnings"]
    assert unknown["same_text_as_simulated"] is None and unknown["same_stream_text_as_simulated"] is None
    assert unknown["validity"]["met"] is None
    old_server = summary_case(tmp_path, None, {"final_pass": "full"}, None)
    assert old_server["same_stream_text_as_simulated"] is None and old_server["warnings"]
    assert old_server["same_text_as_simulated"] is None


# 도우미: flush를 보낸 뒤 서버 메시지를 stats가 올 때까지(최대 200개) 읽는다(tests/test_service.py와 같음).
def read_until_stats(ws) -> list[dict]:
    got: list[dict] = []
    for _ in range(200):
        got.append(json.loads(ws.receive_text()))
        if got[-1]["type"] == "stats":
            break
    return got


# 확인: --replay-check의 오프라인 재생(replay_expected: /info 설정으로 세션을 만들어 같은 패킷을 push_pcm16 →
# flush)이 실제 서버 경로(TestClient WebSocket으로 같은 패킷 송신)와 발화마다 같은 확정 텍스트·stream_text·
# 세그먼트(종류·번호·규칙·프레임 수)를 낸다. 서버는 기본값과 같은 성격(2차 패스 full, rule2 0.8 s, 끝점 뒤
# 캐시 유지, fast 경로)이다. /info의 체크포인트 SHA-256과 다른 파일로는 재생하지 않는다. 왜: 2차 패스 서버의
# 확정 텍스트는 simulated 가설과 비교할 수 없으므로, 같은 코드·설정·입력의 오프라인 실행이 기대값이 된다(규칙
# 변경 27 (A)의 유효성 조건).
def test_replay_matches_the_server_path(tmp_path) -> None:
    from stt_scratch.service import server

    model = tiny_model()
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    fake_store(tmp_path, [("u1", "가나", 2 * 16000 + 1234), ("u2", "다라", 3 * 16000)])
    VOCAB.save(tmp_path / "manifests" / "vocab.json")
    original = server.ROOT
    server.ROOT = tmp_path
    try:
        app = server.build_app(
            ckpt,
            8,
            64,
            "cpu",
            EndpointConfig(rule2_s=0.8),
            False,
            warm=False,
            step_path="fast",
            final_pass="full",
            allowed_hosts=["testserver"],
        )
    finally:
        server.ROOT = original
    client = TestClient(app)
    info = client.get("/info").json()
    utts = read_manifest(tmp_path / "manifests" / "dev_h.jsonl")
    store = PcmStore(tmp_path / "data" / "pcm", "dev_h")
    served = {}
    for u in utts:
        offset, length = store.index[u.id]
        pcm = np.asarray(store.bin[offset : offset + length])
        with client.websocket_connect("/ws") as ws:
            for start, end in stream_client.packet_bounds(len(pcm)):
                ws.send_bytes(pcm[start:end].astype("<i2").tobytes())
            ws.send_text("flush")
            served[u.id] = stream_client.close_record(read_until_stats(ws), info["final_pass"])
    replay, provenance = stream_client.replay_expected(info, utts, store, root=tmp_path)

    def key(record: dict) -> tuple:
        segments = [
            (c["type"], c["segment"], c["text"], c["stream_text"], c["rule"], c["frames"])
            for c in record["closes"]
        ]
        return record["text"], record["stream_text"], segments

    assert all(served[u.id]["closes"] and served[u.id]["stream_text"] is not None for u in utts)
    assert all(key(replay[u.id]) == key(served[u.id]) for u in utts)
    assert (provenance["checkpoint_sha256"], provenance["final_pass"]) == (info["checkpoint_sha256"], "full")
    assert provenance["step_path"] == "fast" and provenance["warm_up_s"] >= 0
    with pytest.raises(SystemExit):
        stream_client.replay_expected({**info, "checkpoint_sha256": "0" * 64}, utts, store, root=tmp_path)


# 확인: 커밋된 6단계 클라이언트 기록(서버 /info에 final_pass 키가 없던 1-pass 서버)을 지금 코드로 다시
# 요약해도 커밋된 요약의 모든 값이 그대로 나오고, 새 필드는 1-pass 서버로 판별해 stream 비교도 50/50이다. 왜:
# 요약 방식을 서버 설정에 따라 나누면서 6단계 기록(H6.1·서비스 경로 검증의 근거)의 해석이 바뀌지 않아야 한다.
@REAL
def test_summary_of_the_stage6_record_is_unchanged(tmp_path) -> None:
    source = REPO / "reports" / "stage6"
    for suffix in (".jsonl", "-meta.json"):
        shutil.copy(source / f"client-dev_h-warmup{suffix}", tmp_path / f"client-dev_h-warmup{suffix}")
    out = stream_client.summarize("dev_h", SIMULATED, "client-dev_h-warmup", root=REPO, out_dir=tmp_path)
    committed = json.loads((source / "client-dev_h-warmup-summary.json").read_text(encoding="utf-8"))
    assert {k: out[k] for k in committed} == committed
    assert out["final_pass"] == "none" and out["warnings"] == []
    assert out["same_stream_text_as_simulated"] == {"exact": 50, "ignoring_spaces": 50}
