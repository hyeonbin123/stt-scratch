"""끝점 뒤 전체 문맥 2차 패스(StreamSession final_pass="full") 테스트(docs/experiments.md 규칙 변경 26).

2차 패스는 세그먼트가 닫힐 때 그 세그먼트에서 인코더에 넣은 mel 전체를 같은 모델의 전체 문맥 모드로 다시
디코딩해 확정 자막으로 쓴다. 확인하는 것: (1) 확정 텍스트 = 그 세그먼트 mel을 오프라인 전체 문맥으로 디코딩한
결과, (2) 세그먼트들의 mel을 이으면 음성 전체가 빠짐없이 들어 있음, (3) 2차 패스를 켜도 스트리밍 쪽(부분 자막,
끝점, stream_text)은 끈 세션과 같음, (4) 길이 묶음용 0 덧붙이기가 결과를 바꾸지 않음, (5) 예열 뒤 세션이 새
세션과 같음, (6) 서버 옵션과 /info.
"""

# [방식] test_service.py와 같은 작은 무작위 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 돈다. 학습된 가중치 없이
# 구조적 성질만 본다. 정확도 이득(CER)은 GPU 측정(scripts/followup.py session)이 dev에서 잰다.

from __future__ import annotations

import json

import torch
from fastapi.testclient import TestClient

from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))
SPEC = ChunkSpec(8, 64)
NO_ENDPOINT = EndpointConfig(rule1_s=1e9, rule2_s=1e9, rule3_s=1e9)


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


def noise_wav(seconds: float, seed: int) -> torch.Tensor:
    return torch.randn(int(16000 * seconds), generator=torch.Generator().manual_seed(seed)) * 0.05


# 세션에 320 ms(5,120표본) 조각으로 흘리고 flush까지 한 이벤트 목록.
def run(session: StreamSession, wav: torch.Tensor) -> list:
    events = []
    for i in range(0, wav.numel(), 5120):
        events.extend(session.push_samples(wav[i : i + 5120]))
    events.extend(session.flush())
    return events


# 오프라인 기준: mel (T, 80)을 전체 문맥으로 한 번에 디코딩한 텍스트(덧붙이기 없음).
def offline_full(model: ChunkConformerCTC, mel: torch.Tensor) -> str:
    with torch.no_grad():
        logits, _, lengths = model(mel[None], torch.tensor([mel.shape[0]]), ChunkSpec(0, -1))
    return VOCAB.decode(greedy_batch(logits.float(), lengths)[0])


# 확인: 끝점을 끄면 세그먼트가 하나라 확정 자막 = 발화 전체 mel의 오프라인 전체 문맥 디코딩이고, stream_text는
# 2차 패스를 끈 세션의 확정 자막과 같다. 왜: 2차 패스가 '같은 모델의 전체 문맥 모드'라는 정의를 고정한다.
def test_final_pass_without_endpoints_equals_offline_full_context() -> None:
    model = tiny_model()
    logmel = LogMel()
    wav = noise_wav(3.1, seed=5)
    events = run(StreamSession(model, VOCAB, SPEC, logmel, NO_ENDPOINT, final_pass="full"), wav)
    baseline = run(StreamSession(model, VOCAB, SPEC, logmel, NO_ENDPOINT), wav)
    closing = [e for e in events if e.kind in ("final", "endpoint")]
    assert len(closing) == 1 and closing[0].rule == "eos"
    with torch.no_grad():
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()]))
    expected = offline_full(model, mel[0])
    assert expected.strip() and closing[0].kind == "final" and closing[0].text == expected
    assert closing[0].stream_text == " ".join(e.text for e in baseline if e.kind == "final")
    # 부분 자막은 2차 패스와 무관하게 같다.
    partials = [e.text for e in events if e.kind == "partial"]
    assert partials == [e.text for e in baseline if e.kind == "partial"]


# 확인: rule3을 0.6 s로 줄여 청크 2개(16프레임)마다 세그먼트를 닫으면, 각 확정 자막이 그 세그먼트 mel 구간의
# 오프라인 전체 문맥 디코딩과 같고, 세그먼트들이 음성 전체(마지막 부분 청크 포함)를 빠짐없이 덮으며,
# 스트리밍 쪽 이벤트(종류 외 텍스트·세그먼트·규칙)와 stream_text는 2차 패스를 끈 세션과 같다(캐시 유지·초기화
# 모두). 왜: 끝점에서 버퍼를 넘기는 규칙(규칙 변경 14)과 2차 패스의 세그먼트 경계가 맞아야 음성을 잃지 않는다.
def test_final_pass_segments_cover_the_audio_and_match_offline() -> None:
    model = tiny_model(seed=1)
    logmel = LogMel()
    wav = noise_wav(2.3, seed=6)
    with torch.no_grad():
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()]))
    mel = mel[0]
    for reset in (False, True):
        endpoint = EndpointConfig(rule3_s=0.6)
        session = StreamSession(model, VOCAB, SPEC, logmel, endpoint, reset, final_pass="full")
        seen: list[torch.Tensor] = []
        original = session._final_pass_decode

        def record(original=original, session=session, seen=seen) -> tuple[str, float]:
            seen.append(torch.cat(session.segment_mel) if session.segment_mel else torch.zeros(0, 80))
            return original()

        session._final_pass_decode = record
        events = run(session, wav)
        baseline = run(StreamSession(model, VOCAB, SPEC, logmel, endpoint, reset), wav)
        # 세그먼트를 이으면 음성 전체다(빠짐·겹침 없음). 스트리밍 fbank와 오프라인 mel은 1e-4 안에서 같다
        # (tests/test_model.py test_streaming_fbank_matches_offline과 같은 허용 오차).
        joined = torch.cat(seen)
        assert joined.shape == mel.shape and torch.allclose(joined, mel, atol=1e-4)
        closing = [e for e in events if e.kind in ("final", "endpoint")]
        # 끝점으로 닫힌 세그먼트는 모두 이벤트가 있고, 스트림 끝 세그먼트는 두 텍스트가 모두 비면 이벤트가
        # 없다.
        assert len(closing) in (len(seen), len(seen) - 1)
        for event, seg in zip(closing, seen, strict=False):
            expected = offline_full(model, seg)  # 그 세그먼트 mel의 오프라인 전체 문맥 디코딩
            assert event.text == (expected if expected.strip() else "")
        streamed = [e.stream_text for e in closing if e.stream_text]
        assert streamed == [e.text for e in baseline if e.kind == "final"]
        assert [e.text for e in events if e.kind == "partial"] == [
            e.text for e in baseline if e.kind == "partial"
        ]


# 확인: 길이 묶음(final_pass_bucket)으로 0을 덧붙여도 전체 문맥 출력의 유효 프레임과 텍스트가 같다(fp32 1e-5).
# 왜: GPU에서 새 입력 길이의 1회성 비용을 피하려고 길이를 묶는데, 그것이 결과를 바꾸면 안 된다.
def test_bucket_padding_does_not_change_the_result() -> None:
    model = tiny_model(seed=2)
    session = StreamSession(model, VOCAB, SPEC, LogMel(), final_pass="full", final_pass_bucket=128)
    for t, seed in ((37, 1), (128, 2), (300, 3)):
        mel = torch.randn(t, 80, generator=torch.Generator().manual_seed(seed))
        text, _ = session.decode_full(mel)
        assert text == offline_full(model, mel)
        padded = torch.cat([mel, torch.zeros(384 - t, 80)])
        with torch.no_grad():
            a, _, la = model(mel[None], torch.tensor([t]), ChunkSpec(0, -1))
            b, _, lb = model(padded[None], torch.tensor([t]), ChunkSpec(0, -1))
        n = int(la[0])
        assert int(lb[0]) == n and float((a[0, :n] - b[0, :n]).abs().max()) <= 1e-5


# 확인: 2차 패스·graph 경로로 예열한 세션이 새 세션과 같은 이벤트를 내고 통계·세그먼트 번호가 처음 상태다.
# 예열은 길이 묶음마다 전체 문맥 디코딩을 한 번씩 한다(rule3 20 s + 청크 2개 → 128프레임 묶음 17개).
def test_warm_up_with_final_pass_and_graph_leaves_a_fresh_session() -> None:
    model = tiny_model(seed=3)
    logmel = LogMel()
    options = {"step_path": "graph", "final_pass": "full"}
    session = StreamSession(model, VOCAB, SPEC, logmel, EndpointConfig(), False, **options)
    lengths: list[int] = []
    original = session.decode_full
    session.decode_full = lambda mel: lengths.append(mel.shape[0]) or original(mel)
    warm_up(session)
    session.decode_full = original
    assert set(range(128, 128 * 17 + 1, 128)) <= set(lengths)
    assert session.stats.chunks == 0 and session.segment == 0
    wav = noise_wav(2.0, seed=8)

    def summary(events):
        return [(e.kind, e.text, e.segment, e.frames, e.rule, e.stream_text) for e in events]

    fresh = StreamSession(model, VOCAB, SPEC, logmel, EndpointConfig(), False, **options)
    assert summary(run(session, wav)) == summary(run(fresh, wav))


# 확인: 서버가 --step-path·--final-pass 설정을 세션에 넘기고 /info에 보이며, 확정 자막 메시지에 2차 패스
# 시간이 들어간다. 기본값은 eager·none(판정 전)이다.
def test_server_passes_the_followup_options(tmp_path) -> None:
    from stt_scratch.service import server

    model = tiny_model()
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    (tmp_path / "manifests").mkdir()
    VOCAB.save(tmp_path / "manifests" / "vocab.json")
    original = server.ROOT
    server.ROOT = tmp_path
    try:
        default = server.build_app(ckpt, 8, 64, "cpu", EndpointConfig(), False, warm=False)
        app = server.build_app(
            ckpt, 8, 64, "cpu", EndpointConfig(), False, warm=False, step_path="fast", final_pass="full"
        )
    finally:
        server.ROOT = original
    info = TestClient(default).get("/info").json()
    assert (info["step_path"], info["final_pass"]) == ("eager", "none")
    client = TestClient(app)
    info = client.get("/info").json()
    assert (info["step_path"], info["final_pass"]) == ("fast", "full")
    pcm = (noise_wav(1.0, seed=9) * 40000).to(torch.int16).numpy().tobytes()
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(pcm)
        ws.send_text("flush")
        got = []
        for _ in range(200):
            got.append(json.loads(ws.receive_text()))
            if got[-1]["type"] == "stats":
                break
    finals = [m for m in got if m["type"] == "final"]
    assert finals and all("final_pass_ms" in m for m in finals)
