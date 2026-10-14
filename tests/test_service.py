"""서비스 게이트 테스트: 끝점 규칙, 세션 == offline 스트리밍 디코딩, flush, WebSocket 왕복.

스트리밍 서비스(service/endpoint, session, server)의 동작을 확인한다: 끝점 규칙 시점, 끝점을 끈 세션 결과 ==
배치 청크 마스크 greedy, 끝점이 버퍼 음성을 잃지 않는지, fp16 정밀도, WebSocket 왕복과 보안 헤더·Origin
검사, CSP 해시, 서버 예열.
"""

# [방식] 작은 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 돌린다. 서버는 실제 포트에 띄우지 않고 FastAPI
# TestClient로 앱을 직접 호출한다. 모든 테스트가 학습된 가중치 없이 '구조적 성질'만 본다.

from __future__ import annotations

import json

# FastAPI TestClient: 프로세스 안에서 HTTP·WebSocket 요청을 앱에 바로 보내는 시험용 클라이언트. starlette
# WebSocketDisconnect: 서버가 연결을 거부했는지 확인할 때 쓴다.
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel, n_mel_frames
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.subsampling import subsampled_length
from stt_scratch.service.endpoint import EndpointConfig, Endpointer
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

# 시험용 어휘: 특수 토큰 3개 + 음절 5개.
VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))


# 시드를 고정한 작은 무작위 모델(eval 모드, dropout 0, 1층 뒤 중간 헤드 하나).
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


# 끝점 시험용 가짜 log-prob: silence는 모든 프레임이 blank(확률 1), speech는 모든 프레임이 음절 id 3(blank
# 확률 사실상 0).
def silence(n: int, v: int = 8) -> torch.Tensor:
    lp = torch.full((n, v), -20.0)
    lp[:, 0] = 0.0
    return lp


def speech(n: int, v: int = 8) -> torch.Tensor:
    lp = torch.full((n, v), -20.0)
    lp[:, 3] = 0.0
    return lp


# 확인: rule2는 토큰이 나온 뒤 묵음 1.2 s(30프레임)에서 정확히 켜지고 1.16 s에서는 안 켜진다. rule1은 토큰
# 없이 2.4 s(60프레임)에서, rule3은 세그먼트 20 s(500프레임)에서 켜진다. 처음의 1.0 s 묵음은 토큰이 없으므로
# rule2를 켜지 않는다. 왜: 6단계에서 고른 끝점 시간과 실제 동작이 프레임 단위로 같아야 EP 측정이 의미가 있다.
def test_endpoint_rules() -> None:
    ep = Endpointer(EndpointConfig(rule1_s=2.4, rule2_s=1.2, rule3_s=20.0))
    assert ep.update(silence(25), 0) is None  # 묵음 1.0 s, 아직 디코딩된 것 없음
    assert ep.update(speech(5), 1) is None
    assert ep.update(silence(29), 0) is None  # 토큰 뒤 1.16 s
    assert ep.update(silence(1), 0) == "rule2"  # 1.2 s
    ep.reset()
    assert ep.update(silence(59), 0) is None
    assert ep.update(silence(1), 0) == "rule1"  # 토큰 없이 2.4 s
    ep.reset()
    assert ep.update(speech(499), 1) is None
    assert ep.update(speech(1), 0) == "rule3"  # 세그먼트 20 s


# 끝점은 규칙 시간을 1e9 s로 두어 끈다. 세션에는 5,120표본씩 흘린다. 부분 자막은 항상 다음 부분 자막의
# 앞부분이고(덧붙기만 함), 세션이 센 청크 수가 '전체 mel ÷ 32'인지도 본다. 왜: 서비스 경로가 평가 경로와 같은
# 결과를 내야 dev에서 잰 CER이 서비스 품질이 된다(6단계 '서비스 경로 CER' 조건).
def test_session_matches_offline_streaming_decode() -> None:
    """끝점을 끄면 세션의 최종 텍스트가 같은 파형의 배치 청크 마스크 greedy 디코딩과 같다."""
    model = tiny_model()
    logmel = LogMel()
    spec = ChunkSpec(8, 64)
    g = torch.Generator().manual_seed(5)
    wav = torch.randn(16000 * 3 + 777, generator=g) * 0.05
    no_endpoint = EndpointConfig(rule1_s=1e9, rule2_s=1e9, rule3_s=1e9)
    session = StreamSession(model, VOCAB, spec, logmel, no_endpoint)
    events = []
    for i in range(0, wav.numel(), 5120):
        events.extend(session.push_samples(wav[i : i + 5120]))
    events.extend(session.flush())
    finals = [e for e in events if e.kind == "final"]
    partials = [e for e in events if e.kind == "partial"]
    # 덧붙이기만 함: 모든 부분 자막은 다음 부분 자막의 앞부분이다.
    for a, b in zip(partials, partials[1:], strict=False):
        assert b.text.startswith(a.text)
    with torch.no_grad():
        mel, frames = logmel(wav[None], torch.tensor([wav.numel()]))
        logits, _, lengths = model(mel, frames, spec)
    ref = VOCAB.decode(greedy_batch(log_probs_fp32(logits), lengths)[0])
    got = " ".join(f.text for f in finals)
    assert got.replace(" ", "") == ref.replace(" ", "")
    assert (
        session.stats.chunks == n_mel_frames(wav.numel()) // 32
    )  # 온전한 청크만 센다. 나머지는 flush로 처리됐다


# 도우미: encoder.step을 감싸 인코더에 들어간 mel 프레임 수를 기록한다.
def _run_aligned_packets(session: StreamSession, wav: torch.Tensor) -> tuple[list[int], list[int]]:
    """첫 패킷 5,360표본, 이후 5,120표본씩 넣고 (step별 mel 프레임 수, 패킷별 청크 수)를 돌려준다."""
    fed: list[int] = []
    original = session.encoder.step
    session.encoder.step = lambda mel, state: fed.append(mel.shape[0]) or original(mel, state)
    per_packet: list[int] = []
    pos, packet = 0, 5120 + FIRST_PACKET_EXTRA
    while pos < wav.numel():
        before = len(fed)
        session.push_samples(wav[pos : pos + packet])
        per_packet.append(len(fed) - before)
        pos += packet
        packet = 5120
    session.flush()
    return fed, per_packet


# 확인: 5번째·12번째 청크에서 끝점을 강제로 일으켜도 스트림의 모든 mel 프레임이 인코더에 들어가고(합계 ==
# 전체), 끝점 직후를 포함해 마지막을 뺀 모든 패킷이 청크 하나씩을 내며, 세그먼트가 3개(끝점 2 + flush)가 된다.
# 캐시 초기화·유지 두 정책 모두. 왜: 처음 구현은 끝점에서 버퍼까지 지워 세그먼트마다 음성 약 300 ms를 잃고
# 패킷 격자가 어긋났다(적대적 검토의 P1, 규칙 변경 14). 그 회귀를 막는다.
@pytest.mark.parametrize("reset_cache", [True, False])
def test_endpoint_keeps_buffered_audio_and_packet_grid(reset_cache: bool) -> None:
    """5·12번째 청크에서 강제로 낸 끝점이 쌓인 mel 프레임과 fbank 이월 표본을 버리면 안 된다.

    스트림의 모든 mel 프레임이 인코더에 들어가고, 첫 패킷 뒤의 모든 패킷이 여전히 청크 하나씩을 내야 한다.
    """
    model = tiny_model()
    session = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel(), EndpointConfig(), reset_cache)
    session.endpointer.update = lambda lp, n: "rule2" if session.stats.chunks in (5, 12) else None
    g = torch.Generator().manual_seed(8)
    wav = torch.randn(16000 * 5 + 1234, generator=g) * 0.05
    fed, per_packet = _run_aligned_packets(session, wav)
    n_mel = n_mel_frames(wav.numel())
    assert sum(fed) == n_mel
    assert all(n == 1 for n in per_packet[:-1])  # 끝점 바로 뒤 패킷도 포함. 마지막 패킷만 짧을 수 있다
    assert session.stats.chunks == n_mel // 32
    assert sum(subsampled_length(n) for n in fed) == subsampled_length(n_mel)
    assert session.segment == 3  # 끝점 두 번 + flush


# 확인: CUDA에서 세션의 기본 정밀도가 fp16 autocast(out_proj 출력이 fp16)이고, amp=False면 fp32다. 왜:
# 서비스가 평가와 다른 정밀도로 돌던 것을 고쳤으므로(규칙 변경 8·14) 그 상태를 고정한다.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_session_runs_under_fp16_autocast_on_cuda() -> None:
    """서비스는 evaluate.decode_streaming과 같은 정밀도로 디코딩한다(규칙 변경 8/12)."""
    model = tiny_model().cuda()
    seen: list[torch.dtype] = []
    model.out_proj.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    session = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel().cuda(), EndpointConfig())
    g = torch.Generator().manual_seed(9)
    session.push_samples((torch.randn(5360, generator=g) * 0.05).cuda())
    assert seen and all(d == torch.float16 for d in seen)
    seen.clear()
    fp32 = StreamSession(model, VOCAB, ChunkSpec(8, 64), LogMel().cuda(), EndpointConfig(), amp=False)
    fp32.push_samples((torch.randn(5360, generator=g) * 0.05).cuda())
    assert seen and all(d == torch.float32 for d in seen)


# 확인: 임시 체크포인트·어휘로 서버 앱을 만들어 /info 설정(320 ms, CPU라 fp32), 1초 PCM을 10,240바이트(320 ms)
# 프레임으로 보낸 뒤 flush하면 이벤트들 다음에 "stats"가 오고 그 청크 수가 맞는지, final이 하나 이상 나오는지
# 본다. 이어서 보안 점검(OWASP ZAP) 반영 사항을 확인한다.
# server.ROOT를 임시 폴더로 바꾸는 이유: build_app이 ROOT 아래 manifests/vocab.json을 읽으므로, 실제 저장소
# 어휘 대신 시험용 어휘를 쓰게 한다. finally로 원래 값을 되돌린다.
def test_websocket_round_trip(tmp_path) -> None:
    from stt_scratch.service import server

    model = tiny_model()
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    vocab_path = tmp_path / "vocab.json"
    VOCAB.save(vocab_path)
    original = server.ROOT
    server.ROOT = tmp_path  # 어휘를 이 폴더에서 찾게 한다
    (tmp_path / "manifests").mkdir()
    VOCAB.save(tmp_path / "manifests" / "vocab.json")
    try:
        app = server.build_app(ckpt, 8, 64, "cpu", EndpointConfig(), True)
    finally:
        server.ROOT = original
    client = TestClient(app)
    assert client.get("/info").json()["chunk_ms"] == 320
    g = torch.Generator().manual_seed(1)
    pcm = (torch.randn(16000, generator=g) * 2000).to(torch.int16).numpy().tobytes()
    assert client.get("/info").json()["precision"] == "fp32"
    with client.websocket_connect("/ws") as ws:
        for i in range(0, len(pcm), 10240):
            ws.send_bytes(pcm[i : i + 10240])
        ws.send_text("flush")
        # flush에는 항상 이벤트들 다음에 stats 한 개로 답하므로, stats가 올 때까지만 읽는다(final이 없으면
        # 영원히 기다리던 이전 테스트를 유한 루프로 바꿈).
        got = []
        for _ in range(200):
            got.append(json.loads(ws.receive_text()))
            if got[-1]["type"] == "stats":
                break
    assert got[-1]["type"] == "stats" and got[-1]["chunks"] == n_mel_frames(16000) // 32
    assert any(m["type"] == "final" for m in got)
    # 보안(OWASP ZAP 점검 결과): 모든 응답의 보안 헤더, 페이지의 해시 기반 CSP, API 문서 없음, /info에 로컬
    # 디렉터리 구조 없음, 교차 사이트 WebSocket 거부.
    page = client.get("/")
    csp = page.headers["content-security-policy"]
    assert "'sha256-" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    for r in (page, client.get("/info"), client.get("/openapi.json")):
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
        assert r.headers["cross-origin-resource-policy"] == "same-origin"
        assert "content-security-policy" in r.headers and "permissions-policy" in r.headers
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404
    assert client.get("/info").json()["checkpoint"] == "m.pt"  # 저장소 밖 경로는 파일 이름만
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as bad:
            bad.receive_text()
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as good:
        good.send_text("flush")
        assert json.loads(good.receive_text())["type"] == "stats"


# 확인: page_csp가 인라인 스타일·스크립트 블록 내용의 SHA-256을 CSP 해시 형식으로 넣는지, same_origin이 Origin
# 없음(스크립트 클라이언트)과 같은 출처는 통과시키고 다른 사이트는 막는지.
def test_page_csp_hashes_match_the_inline_blocks() -> None:
    from stt_scratch.service import server

    html = "<style>a{}</style><script>let x = 1;</script>"
    csp = server.page_csp(html)
    import base64
    import hashlib

    for block in ("a{}", "let x = 1;"):
        digest = base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    assert server.same_origin(None, "127.0.0.1:8765") and server.same_origin(
        "http://127.0.0.1:8765", "127.0.0.1:8765"
    )
    assert not server.same_origin("http://evil.example", "127.0.0.1:8765")


# 확인: rule3을 0.32 s(청크 하나)로 줄이고 무음을 넣으면 아무것도 디코딩하지 않아도 끝점마다 세그먼트 번호가
# 넘어가고, 이벤트 종류는 partial·final·endpoint 중 하나다.
@pytest.mark.parametrize("reset_cache", [True, False])
def test_endpoint_closes_segment(reset_cache: bool) -> None:
    model = tiny_model()
    session = StreamSession(
        model, VOCAB, ChunkSpec(8, 64), LogMel(), EndpointConfig(rule3_s=0.32), reset_cache
    )
    wav = torch.zeros(16000)
    events = []
    for i in range(0, wav.numel(), 5120):
        events.extend(session.push_samples(wav[i : i + 5120]))
    # rule3 0.32 s는 청크마다 한 번씩 켜지므로, 아무것도 디코딩하지 않아도 세그먼트가 넘어간다.
    assert session.segment >= 2
    assert all(e.kind in ("partial", "final", "endpoint") for e in events)


# 같은 이벤트는 종류·텍스트·세그먼트·프레임·규칙이 모두 같다는 뜻이다. 왜: 예열이 첫 사용자의 지연을 없애는
# 대신 세션 상태를 오염시키면 안 된다.
def test_warm_up_covers_every_partial_chunk_length_and_leaves_a_fresh_session() -> None:
    """warm_up은 마지막 부분 청크 길이 1..31 mel 프레임을 한 번씩 flush한다.

    그 뒤 세션은 새 세션과 똑같이 동작한다(같은 이벤트, 빈 통계).
    """
    model = tiny_model()
    logmel = LogMel()
    spec = ChunkSpec(8, 64)
    session = StreamSession(model, VOCAB, spec, logmel, EndpointConfig())
    fed: list[int] = []
    original = session.encoder.step
    session.encoder.step = lambda mel, state: fed.append(mel.shape[0]) or original(mel, state)
    warm_up(session)
    session.encoder.step = original
    assert sorted(n for n in fed if n != 32) == list(range(1, 32))
    assert session.stats.chunks == 0 and not session.stats.compute_ms and session.segment == 0
    g = torch.Generator().manual_seed(7)
    wav = torch.randn(16000 * 2 + 333, generator=g) * 0.05

    def run(s: StreamSession) -> list[tuple]:
        events = []
        for i in range(0, wav.numel(), 5120):
            events.extend(s.push_samples(wav[i : i + 5120]))
        events.extend(s.flush())
        return [(e.kind, e.text, e.segment, e.frames, e.rule) for e in events]

    assert run(session) == run(StreamSession(model, VOCAB, spec, logmel, EndpointConfig()))
