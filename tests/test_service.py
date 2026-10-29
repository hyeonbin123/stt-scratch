"""서비스 게이트 테스트: 끝점 규칙, 세션 == offline 스트리밍 디코딩, flush, WebSocket 왕복.

스트리밍 서비스(service/endpoint, session, server)의 동작을 확인한다: 끝점 규칙 시점, 끝점을 끈 세션 결과 ==
배치 청크 마스크 greedy, 끝점이 버퍼 음성을 잃지 않는지, fp16 정밀도, WebSocket 왕복과 보안 헤더·Origin
검사, Host 허용 목록(DNS 리바인딩 방어), WebSocket 입력·자원 상한(홀수 바이트 프레임, 큰 프레임, 동시 세션
수), 명령행 실행이 uvicorn에 넘기는 전송 상한과 경고, CSP 해시, 서버 예열, 서버 상태 보고(/info의 어휘
SHA-256·torch 스레드 수·CUDA Graph 캡처 상태, 캡처 실패 시 fast 폴백), 어휘 크기 검사, CPU 장치의 기본
1스레드, 2차 패스를 켰을 때만 확정 이벤트에 붙는 stream_text.
"""

# [방식] 작은 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 돌린다. 서버는 실제 포트에 띄우지 않고 FastAPI
# TestClient로 앱을 직접 호출한다. 모든 테스트가 학습된 가중치 없이 '구조적 성질'만 본다. TestClient는 요청의
# Host를 'testserver'로 보내므로, 기본 Host 허용 목록(루프백 이름만) 자체를 보는 테스트가 아니면 앱을
# allowed_hosts=["testserver"]로 만든다. TestClient에는 전송 계층의 메시지 크기 상한이 없어서 크기 상한은 앱
# 처리기 검사로 확인하고, uvicorn에 넘기는 값(ws_max_size·limit_concurrency)은 명령행 실행 테스트에서 본다.

from __future__ import annotations

import hashlib
import json
import logging

# FastAPI TestClient: 프로세스 안에서 HTTP·WebSocket 요청을 앱에 바로 보내는 시험용 클라이언트. starlette
# WebSocketDisconnect: 서버가 연결을 닫았는지(종료 코드 포함) 확인할 때 쓴다. WebSocketDenialResponse: 수락
# 전에 HTTP 거부 응답(Host 허용 목록 밖이면 400)으로 막힌 WebSocket을 TestClient가 알리는 예외다.
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
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
    """서비스는 evaluate.decode_streaming과 같은 정밀도로 디코딩한다(규칙 변경 8·14)."""
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
        # TestClient의 Host(testserver)를 허용 목록에 넣는다. 기본 목록(루프백 이름만)은 아래 Host 테스트가
        # 본다.
        app = server.build_app(ckpt, 8, 64, "cpu", EndpointConfig(), True, allowed_hosts=["testserver"])
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


# 도우미: 시험용 작은 모델 체크포인트와 어휘를 tmp_path에 만들고 server.build_app으로 앱을 만든다(기본 CPU,
# 청크 8, 좌측 64, 끝점 기본값, 끝점에서 캐시 초기화). build_app이 server.ROOT 아래 manifests/vocab.json을
# 읽으므로 ROOT를 잠시 tmp_path로 바꾸고 finally로 되돌린다. device는 build_app에 넘기는 장치 문자열(CUDA 전용
# 테스트가 "cuda"를 준다), vocab은 저장할 어휘(어휘 크기 검사 테스트가 모델과 다른 크기를 준다).
# kwargs(allowed_hosts·max_frame_bytes·max_sessions·warm·step_path 등)는 그대로 넘긴다. 같은 tmp_path로
# 여러 번 불러도 된다(파일을 덮어쓴다).
def make_app(tmp_path, device: str = "cpu", vocab: Vocab = VOCAB, **kwargs):
    from stt_scratch.service import server

    model = tiny_model()
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    (tmp_path / "manifests").mkdir(exist_ok=True)
    vocab.save(tmp_path / "manifests" / "vocab.json")
    original = server.ROOT
    server.ROOT = tmp_path
    try:
        return server.build_app(ckpt, 8, 64, device, EndpointConfig(), True, **kwargs)
    finally:
        server.ROOT = original


# 도우미: server.StreamSession을 생성 인자(step_path·final_pass)를 기록하는 하위 클래스로 바꿔 끼운다. 서버가
# 세션에 실제로 넘기는 옵션을 /info 값과 별개로 확인할 때 쓴다(예열 세션과 연결마다의 세션이 모두 순서대로
# 기록된다). monkeypatch가 테스트 끝에 원래 클래스로 되돌린다.
def record_sessions(monkeypatch) -> list[tuple]:
    from stt_scratch.service import server

    seen: list[tuple] = []

    class Recording(server.StreamSession):
        def __init__(self, *args, **kwargs) -> None:
            seen.append((kwargs.get("step_path"), kwargs.get("final_pass")))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server, "StreamSession", Recording)
    return seen


# 도우미: flush를 보낸 뒤 서버 메시지를 stats가 올 때까지(최대 200개) 읽어 돌려준다. 서버는 flush에 항상
# 이벤트들 다음 stats 한 개로 답하므로 유한 루프로 충분하다(무작위 작은 모델은 무음에도 토큰을 낼 수 있어
# final이 먼저 올 수 있다).
def read_until_stats(ws) -> list[dict]:
    got: list[dict] = []
    for _ in range(200):
        got.append(json.loads(ws.receive_text()))
        if got[-1]["type"] == "stats":
            break
    return got


# 확인: 기본 Host 허용 목록(127.0.0.1·localhost·[::1])은 루프백 이름(포트 유무, IPv6 대괄호 표기)으로 온
# 요청만 받는다. 다른 이름으로 온 HTTP 요청은 400이고 그 거부 응답에도 보안 헤더가 붙으며, Host·Origin이 모두
# 공격자 도메인인 WebSocket은 수락 전에 400 거부 응답으로 막힌다. 허용된 Host에서 같은 출처 페이지가 여는
# 연결은 그대로 된다. 왜: 공격자가 자기 도메인의 DNS를 127.0.0.1로 다시 가리키면(DNS 리바인딩) 브라우저는
# Host와 Origin에 모두 그 도메인을 실어 보내므로 'Origin == Host' 검사만으로는 막지 못했다(보안 검토 재현: 그
# 연결이 수락되고 flush에 stats가 돌아왔으며, 같은 Host의 GET /info도 200이었다).
def test_host_allow_list_blocks_dns_rebinding(tmp_path) -> None:
    from stt_scratch.service import server

    client = TestClient(make_app(tmp_path, warm=False))  # allowed_hosts 기본값
    bad = client.get("/info", headers={"host": "attacker.example:8765"})
    assert bad.status_code == 400
    assert bad.headers["x-content-type-options"] == "nosniff" and bad.headers["x-frame-options"] == "DENY"
    assert bad.headers["content-security-policy"] == server.API_CSP
    assert client.get("/info").status_code == 400  # TestClient 기본 Host(testserver)도 기본 목록 밖이다
    for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "127.0.0.1"):
        assert client.get("/info", headers={"host": host}).status_code == 200
    rebound = {"host": "attacker.example:8765", "origin": "http://attacker.example:8765"}
    with pytest.raises(WebSocketDenialResponse) as denied:
        with client.websocket_connect("/ws", headers=rebound):
            pass
    assert denied.value.status_code == 400
    local = {"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:8765"}
    with client.websocket_connect("/ws", headers=local) as ws:
        ws.send_text("flush")
        assert read_until_stats(ws)[-1]["type"] == "stats"


# 확인: 명령행의 Host 허용 목록 계산. 기본은 루프백 이름 셋이고, --host가 와일드카드(0.0.0.0·::)가 아니면 그
# 값을 브라우저가 Host 헤더에 쓰는 모양(소문자, IPv6는 대괄호)으로 더하며, --allowed-host 값을 중복 없이 뒤에
# 붙인다. 루프백 판정(경고 여부)과, 잘못된 패턴(빈 값, 맨 앞 '*.' 밖의 '*')을 서버가 뜨기 전에 거부하는 것도
# 본다. 왜: Starlette TrustedHostMiddleware는 패턴을 첫 요청 때 검사하므로(assert) 잘못된 값이면 서버가 뜬
# 뒤에야 실패한다.
def test_allowed_hosts_for_the_command_line() -> None:
    from stt_scratch.service import server

    default = ["127.0.0.1", "localhost", "[::1]"]
    assert server.allowed_hosts_for("127.0.0.1") == default
    assert server.allowed_hosts_for("::1") == default and server.allowed_hosts_for("localhost") == default
    assert server.allowed_hosts_for("0.0.0.0") == default and server.allowed_hosts_for("::") == default
    assert server.allowed_hosts_for("192.168.0.7") == [*default, "192.168.0.7"]
    assert server.allowed_hosts_for("FE80::1") == [*default, "[fe80::1]"]
    extra = ["Host.Docker.Internal", "localhost", "*.lan"]
    assert server.allowed_hosts_for("0.0.0.0", extra) == [*default, "host.docker.internal", "*.lan"]
    assert all(server.is_loopback_host(h) for h in ("127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"))
    assert not any(server.is_loopback_host(h) for h in ("0.0.0.0", "::", "192.168.0.7", "stt.lan"))
    assert server.check_allowed_hosts(["testserver", "*.lan", "*"]) == ["testserver", "*.lan", "*"]
    for bad in ([], [""], ["stt*.lan"], ["*lan"]):
        with pytest.raises(ValueError):
            server.check_allowed_hosts(bad)


# 확인: 명령행 실행(main)이 계산한 Host 허용 목록을 build_app에 넘기고, uvicorn에 WebSocket 메시지 상한(1
# MiB)과 동시 처리 상한(16)을 넘기며, --host가 루프백이 아니면 '인증 없는 서버를 외부에 연다' 경고를,
# --allowed-host '*'면 Host 검사가 꺼진다는 경고를 남긴다. 모델을 올리거나 포트를 열지 않도록 build_app과
# uvicorn.run을 기록용 가짜로 바꾼다.
def test_main_passes_the_allow_list_and_transport_limits(monkeypatch, caplog) -> None:
    from stt_scratch.service import server

    seen: dict = {}
    monkeypatch.setattr(server, "build_app", lambda *a, **k: seen.update(build=k) or "app")
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **k: seen.update(app=app, run=k))
    # main은 CPU 장치에서 torch 스레드 수를 1로 두므로(아래 스레드 테스트), 이 테스트 프로세스의 스레드 수가
    # 바뀌지 않게 가짜로 바꾼다.
    monkeypatch.setattr(server.torch, "set_num_threads", lambda n: None)
    caplog.set_level(logging.WARNING, logger="stt_scratch.service.server")

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "stt_scratch.service.server"]

    server.main([])
    assert seen["app"] == "app" and seen["build"]["allowed_hosts"] == ["127.0.0.1", "localhost", "[::1]"]
    assert (seen["run"]["host"], seen["run"]["port"]) == ("127.0.0.1", 8765)
    assert seen["run"]["ws_max_size"] == 1 << 20 and seen["run"]["limit_concurrency"] == 16
    assert not warnings()  # 루프백에서는 경고가 없다
    server.main(["--host", "0.0.0.0", "--allowed-host", "stt.lan", "--allowed-host", "192.168.0.7"])
    assert seen["build"]["allowed_hosts"] == ["127.0.0.1", "localhost", "[::1]", "stt.lan", "192.168.0.7"]
    assert seen["run"]["host"] == "0.0.0.0" and seen["run"]["ws_max_size"] == 1 << 20
    assert len(warnings()) == 1 and "unauthenticated" in warnings()[0]
    caplog.clear()
    server.main(["--allowed-host", "*"])
    assert seen["build"]["allowed_hosts"][-1] == "*"
    assert len(warnings()) == 1 and "disables the Host check" in warnings()[0]


# 확인: 명령행 실행(main)이 torch 연산 스레드 수를 정한다. 기본값은 장치가 cpu면 1(6단계·후속 측정의 CPU
# 1스레드 조건과 같게), cuda면 바꾸지 않음. --threads를 주면 장치와 무관하게 그 값이고, 1보다 작으면 모델을
# 올리기 전에 거부한다. 왜: 서버는 스레드 수를 정하지 않아 CPU 장치에서 기본 스레드 수(이 PC 6코어)로 돌았고,
# 그러면 README·docs의 'CPU 1스레드' 수치가 서버가 실제로 도는 조건의 값이 아니었다. 테스트 프로세스의 스레드
# 수가 바뀌지 않게 torch.set_num_threads를 기록용 가짜로 바꾼다.
def test_main_sets_the_cpu_thread_count(monkeypatch) -> None:
    from stt_scratch.service import server

    seen: dict = {}
    threads: list[int] = []
    monkeypatch.setattr(server, "build_app", lambda *a, **k: seen.update(build=k) or "app")
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **k: seen.update(app=app, run=k))
    monkeypatch.setattr(server.torch, "set_num_threads", threads.append)
    server.main(["--device", "cpu"])
    assert threads == [1]  # CPU 장치 기본값: 측정 조건과 같은 1스레드
    server.main(["--device", "cuda"])
    assert threads == [1]  # CUDA 장치 기본값: 바꾸지 않는다
    server.main(["--device", "cuda", "--threads", "2"])
    server.main(["--device", "cpu", "--threads", "4"])
    assert threads == [1, 2, 4]
    with pytest.raises(SystemExit):
        server.main(["--device", "cpu", "--threads", "0"])
    assert threads == [1, 2, 4] and seen["app"] == "app"


# 확인: /info가 어휘 파일의 SHA-256(vocab_sha256)과 지금 프로세스의 torch 연산 스레드 수(cpu_threads),
# 프레임·세션 상한(max_frame_bytes·max_sessions)을 보이고, CUDA Graph 상태 두 필드(graph_captured·
# graph_failed)는 step 경로가 graph가 아니거나 예열하지 않았으면 null, CPU의 graph 경로는 고정 버퍼
# 코드만 돌고 그래프를 캡처하지 않으므로 예열 뒤 captured false·실패 없음이다. 왜: 측정 도구
# (scripts/stream_client.py)가 /info를 결과 파일의 출처로 남기므로, 어떤 어휘·스레드 수·step
# 경로·상한으로 돌았는지 거기서 드러나야 한다.
def test_info_reports_vocab_hash_threads_and_graph_status(tmp_path) -> None:
    from stt_scratch.model import streaming
    from stt_scratch.service import server

    info = TestClient(make_app(tmp_path, warm=False, allowed_hosts=["testserver"])).get("/info").json()
    vocab_bytes = (tmp_path / "manifests" / "vocab.json").read_bytes()
    assert info["vocab_sha256"] == hashlib.sha256(vocab_bytes).hexdigest() and info["vocab"] == len(VOCAB)
    assert info["cpu_threads"] == torch.get_num_threads() and info["cpu_threads"] >= 1
    assert (info["max_frame_bytes"], info["max_sessions"]) == (server.MAX_FRAME_BYTES, server.MAX_SESSIONS)
    assert (info["step_path"], info["graph_captured"], info["graph_failed"]) == ("eager", None, None)
    graph = make_app(tmp_path, warm=True, step_path="graph", allowed_hosts=["testserver"])
    info = TestClient(graph).get("/info").json()
    assert (info["step_path"], info["graph_captured"], info["graph_failed"]) == ("graph", False, None)
    # 상태 도우미 자체: 아직 step이 없으면 captured false·실패 없음, CPU에서 정상 상태 step을 돌린 뒤에도
    # 그래프는 없고, 실패 사유가 남은 stepper는 그대로 보고하며, 다른 청크 설정의 상태와 섞이지 않는다.
    model = tiny_model()
    spec = ChunkSpec(8, 64)
    assert streaming.graph_status(model, spec) == {"captured": False, "failed": None}
    enc = streaming.StreamingEncoder(model, spec, path="graph")
    state = enc.new_state()
    mel = torch.randn(32 * 10, 80, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        for i in range(0, mel.shape[0], 32):
            enc.step(mel[i : i + 32], state)
    assert streaming.graph_status(model, spec) == {"captured": False, "failed": None}
    streaming._static_stepper(enc).graph_failed = "simulated"
    assert streaming.graph_status(model, spec) == {"captured": False, "failed": "simulated"}
    assert streaming.graph_status(model, ChunkSpec(4, 32)) == {"captured": False, "failed": None}


# 확인: 2차 패스를 켠 서버(final_pass full)가 세그먼트를 닫는 이벤트(final·endpoint)에는 stream_text(2차
# 패스가 바꾸기 전의 320 ms 스트리밍 텍스트)가 붙고 부분 자막(partial)에는 없으며, 2차 패스를 끈 서버(none)의
# 이벤트에는 그 키가 아예 없다. 같은 PCM에서 full 서버의 stream_text 이음(빈 값 제외, 공백으로 이음)은 none
# 서버의 확정 텍스트 이음과 같다(2차 패스는 스트리밍 상태를 바꾸지 않는다: tests/test_final_pass.py). 왜: 서버
# 기본값(full)에서는 확정 텍스트가 2차 패스 결과라 측정 클라이언트(scripts/stream_client.py)가 확정 텍스트로는
# 서비스 경로의 스트리밍 텍스트를 확인할 수 없었다. 이 필드로 6단계와 같은 'simulated 가설과 같은 텍스트'
# 검증을 기본값 서버에서도 한다(docs/experiments.md 규칙 변경 27).
def test_close_events_carry_stream_text_only_with_the_final_pass(tmp_path) -> None:
    hosts = ["testserver"]
    full = TestClient(
        make_app(tmp_path, warm=False, step_path="fast", final_pass="full", allowed_hosts=hosts)
    )
    none = TestClient(make_app(tmp_path, warm=False, step_path="fast", allowed_hosts=hosts))
    g = torch.Generator().manual_seed(4)
    pcm = (torch.randn(3 * 16000, generator=g) * 2000).to(torch.int16).numpy().tobytes()

    def stream(client: TestClient) -> list[dict]:
        with client.websocket_connect("/ws") as ws:
            for i in range(0, len(pcm), 10240):
                ws.send_bytes(pcm[i : i + 10240])
            ws.send_text("flush")
            return read_until_stats(ws)

    got_full, got_none = stream(full), stream(none)
    closes = [m for m in got_full if m["type"] in ("final", "endpoint")]
    assert closes and all(isinstance(m.get("stream_text"), str) for m in closes)
    assert all("stream_text" not in m for m in got_full if m["type"] == "partial")
    assert all("stream_text" not in m for m in got_none)
    streamed = " ".join(m["stream_text"] for m in closes if m["stream_text"].strip())
    assert streamed and streamed == " ".join(m["text"] for m in got_none if m["type"] == "final")


# 확인: 어휘 파일의 토큰 수가 체크포인트의 출력 크기(ModelConfig.vocab_size)와 다르면 evaluate.check_vocab과
# 서버 앱 만들기가 두 값을 적은 SystemExit로 멈춘다(어휘가 크든 작든). 같으면 그대로 지나간다. 왜: 전에는
# 검사가 없어서 어휘가 모델 출력보다 작으면 디코딩 중 IndexError로 연결이 끊기고, 크기가 같고 순서만 다르면
# 조용히 다른 글자를 냈다(순서 차이는 크기 검사로 잡을 수 없으므로 /info의 vocab_sha256으로 출처를 남긴다).
def test_vocab_size_mismatch_refuses_to_start(tmp_path) -> None:
    from stt_scratch import evaluate

    model = tiny_model()
    evaluate.check_vocab(model, VOCAB)  # 8 == 8
    bigger = Vocab((*VOCAB.tokens, "바"))
    with pytest.raises(SystemExit) as exc:
        evaluate.check_vocab(model, bigger)
    assert "9" in str(exc.value) and "8" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        make_app(tmp_path, vocab=bigger, warm=False, allowed_hosts=["testserver"])
    assert "9" in str(exc.value) and "8" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        make_app(tmp_path, vocab=Vocab(VOCAB.tokens[:7]), warm=False, allowed_hosts=["testserver"])
    assert "7" in str(exc.value) and "8" in str(exc.value)


# 확인: 예열 뒤 CUDA Graph 캡처가 실패한 상태(streaming.graph_status를 실패로 바꿔 흉내)면 서버가 경고를
# 남기고 step 경로를 fast로 바꿔 /info에 step_path fast·graph_captured false·graph_failed(사유 앞 200자)를
# 적으며, 이후 연결의 세션이 실제로 fast로 만들어진다(예열 세션은 graph). 왜: 전에는 캡처 실패가 사유만 남기고
# 조용히 고정 버퍼 코드로 폴백해 /info와 측정 결과 파일이 여전히 graph라고 적었다. fast와 graph는 출력
# 동일성 관문에서 같은 출력이 확인된 경로라 바꿔도 자막은 같다. 실제 CUDA 캡처 실패 경로는 아래 CUDA 전용
# 테스트가 본다.
def test_graph_capture_failure_falls_back_to_fast(tmp_path, monkeypatch, caplog) -> None:
    from stt_scratch.model import streaming

    reason = "simulated capture failure: " + "x" * 300
    monkeypatch.setattr(streaming, "graph_status", lambda *a, **k: {"captured": False, "failed": reason})
    seen = record_sessions(monkeypatch)
    caplog.set_level(logging.WARNING, logger="stt_scratch.service.server")
    app = make_app(tmp_path, warm=True, step_path="graph", allowed_hosts=["testserver"])
    assert seen == [("graph", "none")]  # 예열 세션은 graph 경로로 돌았다
    warnings = [r.getMessage() for r in caplog.records if r.name == "stt_scratch.service.server"]
    assert len(warnings) == 1 and "fast" in warnings[0] and reason[:200] in warnings[0]
    client = TestClient(app)
    info = client.get("/info").json()
    assert (info["step_path"], info["graph_captured"]) == ("fast", False)
    assert info["graph_failed"] == reason[:200] and len(info["graph_failed"]) == 200
    with client.websocket_connect("/ws") as ws:
        ws.send_text("flush")
        assert read_until_stats(ws)[-1]["type"] == "stats"
    assert seen[-1] == ("fast", "none")  # 이후 연결은 fast로 돈다


# 확인(GPU 전용, CPU·CI에서는 건너뜀): CUDA에서 graph 경로로 예열한 서버는 /info에 graph_captured true를 적고,
# 캡처 중 torch.cuda.graph가 RuntimeError를 내면(캡처를 지원하지 않는 연산·메모리 부족 등을 흉내) 같은 서버가
# step_path fast·graph_captured false·graph_failed(사유)를 적으며 그 뒤 연결도 정상으로 동작한다. 왜: 위 CPU
# 테스트는 상태 도우미를 바꿔 흉내만 내므로, 실제 캡처 실패가 _StaticStep에 남는 경로를 GPU에서 한 번 본다.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_capture_status_and_failure_report(tmp_path, monkeypatch) -> None:
    app = make_app(tmp_path, device="cuda", step_path="graph", allowed_hosts=["testserver"])
    info = TestClient(app).get("/info").json()
    assert (info["step_path"], info["graph_captured"], info["graph_failed"]) == ("graph", True, None)
    assert info["precision"] == "fp16-autocast"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(torch.cuda, "graph", boom)
    app = make_app(tmp_path, device="cuda", step_path="graph", allowed_hosts=["testserver"])
    client = TestClient(app)
    info = client.get("/info").json()
    assert (info["step_path"], info["graph_captured"]) == ("fast", False)
    assert "simulated capture failure" in info["graph_failed"]
    g = torch.Generator().manual_seed(1)
    pcm = (torch.randn(16000, generator=g) * 2000).to(torch.int16).numpy().tobytes()
    with client.websocket_connect("/ws") as ws:
        for i in range(0, len(pcm), 10240):
            ws.send_bytes(pcm[i : i + 10240])
        ws.send_text("flush")
        assert read_until_stats(ws)[-1]["chunks"] == n_mel_frames(16000) // 32


# 확인: push_pcm16은 홀수 바이트(int16 표본 경계가 깨진 입력)를 원인이 드러나는 ValueError로 거부하고, 빈
# 프레임은 이벤트 없이 지나간다. 왜: 세션을 라이브러리로 직접 쓰는 사람에게 np.frombuffer의 'buffer size must
# be a multiple of element size' 대신 무엇이 잘못됐는지 알린다(서버 처리기는 이 앞에서 close 1007로 답한다).
def test_push_pcm16_rejects_odd_byte_counts() -> None:
    session = StreamSession(tiny_model(), VOCAB, ChunkSpec(8, 64), LogMel(), EndpointConfig())
    with pytest.raises(ValueError, match="even number of bytes"):
        session.push_pcm16(b"\x01\x02\x03")
    assert session.push_pcm16(b"") == []


# 확인: 홀수 바이트 바이너리 프레임은 오류 JSON 한 줄({"type": "error", "reason": ...}) 뒤 close 1007(메시지
# 형식에 맞지 않는 데이터)로 끝난다. 왜: 전에는 np.frombuffer의 ValueError가 처리기 밖으로 나가 트레이스백을
# 남기고 1011로 끊겼다(보안 검토 재현).
def test_odd_length_frame_is_refused_with_1007(tmp_path) -> None:
    client = TestClient(make_app(tmp_path, warm=False, allowed_hosts=["testserver"]))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x01\x02\x03")
        error = json.loads(ws.receive_text())
        assert error["type"] == "error" and "even" in error["reason"]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1007


# 확인: 메시지 상한은 1 MiB(16 kHz int16 약 32.8 s)이고, 그보다 큰 바이너리 프레임은 처리하기 전에 오류 JSON
# 한 줄 뒤 close 1009(메시지가 너무 큼)로 끝난다. 상한과 같은 크기는 처리한다(작은 상한 10,240바이트로 경계
# 확인). 왜: uvicorn 기본 상한 16 MiB면 한 프레임에 약 524 s 오디오가 들어와 인코더 계산이 이벤트 루프를 오래
# 붙잡는다(다른 연결의 자막이 멈춘다). 실제 서버에서는 uvicorn ws_max_size가 같은 상한으로 전송 계층에서 먼저
# 1009로 닫는다.
def test_oversized_frame_is_refused_with_1009(tmp_path) -> None:
    from stt_scratch.service import server

    assert server.MAX_FRAME_BYTES == 1 << 20
    client = TestClient(make_app(tmp_path, warm=False, allowed_hosts=["testserver"]))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(bytes((1 << 20) + 2))
        assert json.loads(ws.receive_text())["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1009
    small = TestClient(make_app(tmp_path, warm=False, allowed_hosts=["testserver"], max_frame_bytes=10240))
    with small.websocket_connect("/ws") as ws:
        ws.send_bytes(bytes(10240))  # 상한과 같은 크기(320 ms)는 처리한다
        ws.send_text("flush")
        assert read_until_stats(ws)[-1]["type"] == "stats"
        ws.send_bytes(bytes(10242))
        assert json.loads(ws.receive_text())["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1009


# 확인: 동시 세션 상한을 넘는 연결은 오류 JSON 한 줄 뒤 close 1013(나중에 다시 시도)으로 끝나고, 이미 열린
# 세션은 그대로 동작하며, 세션이 끝나면 자리가 비어 새 연결을 받는다. 상한을 1로 줄여 확인한다(기본 8). 왜:
# uvicorn의 limit_concurrency는 HTTP 요청에만 503을 내고 WebSocket 업그레이드는 막지 않으므로(uvicorn 0.54
# 소스와 실제 서버로 확인) 앱에서 세션 수를 센다. TestClient를 with로 열어 실제 서버처럼 모든 연결이 한 이벤트
# 루프에서 돈다.
def test_session_cap_refuses_extra_connections_with_1013(tmp_path) -> None:
    from stt_scratch.service import server

    assert server.MAX_SESSIONS == 8
    app = make_app(tmp_path, warm=False, allowed_hosts=["testserver"], max_sessions=1)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as first:
            with client.websocket_connect("/ws") as second:
                assert json.loads(second.receive_text())["type"] == "error"
                with pytest.raises(WebSocketDisconnect) as closed:
                    second.receive_text()
            assert closed.value.code == 1013
            first.send_text("flush")
            assert read_until_stats(first)[-1]["type"] == "stats"
        with client.websocket_connect("/ws") as again:
            again.send_text("flush")
            assert read_until_stats(again)[-1]["type"] == "stats"


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
