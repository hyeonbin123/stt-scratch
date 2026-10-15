"""WebSocket 스트리밍 STT 서버(FastAPI + uvicorn).

  uv run python -m stt_scratch.service.server  # 최종 모델, 320 ms 청크, 좌측 128, rule2 1.2 s, 캐시 유지

기본값은 6단계에서 고른 구성이다(docs/experiments.md). 서버는 시작할 때 가능한 마지막 부분 청크 길이마다
한 번씩 flush해(예열) 첫 클라이언트가 새 텐서 모양의 1회성 준비 시간을 겪지 않게 한다.

프로토콜: 클라이언트는 ws://host:port/ws를 열고 little-endian int16 mono 16 kHz PCM을 바이너리 프레임으로
보낸다(크기는 자유, 320 ms = 10,240바이트 권장). 서버는 JSON 텍스트 프레임으로 답한다.
  {"type": "partial", "segment": n, "text": "..."}      열린 세그먼트의 부분 자막(덧붙기만 함)
  {"type": "final",   "segment": n, "text": "...", "rule": "rule2"}   끝점 규칙으로 닫힌 세그먼트
  {"type": "endpoint", "segment": n, "text": "", "rule": "rule1"}     빈 세그먼트를 닫은 끝점
텍스트 프레임 "flush"는 스트림을 마무리하고(마지막 부분 청크 + final), 이벤트들 뒤에 다음으로 답한다.
  {"type": "stats", "chunks": n, "compute_ms_p50": ..., "compute_ms_p90": ..., "compute_ms_max": ...}
(세션 전체의 청크당 계산 시간). "reset"은 새 세션을 시작한다.
GET /은 web/index.html(마이크 클라이언트 페이지), GET /info는 서버 설정을 준다. 인증이 없으므로 로컬
데모 전용이며 기본으로 127.0.0.1에만 연다. 모든 HTTP 응답에 보안 헤더(페이지에는 해시 기반 CSP)를 붙이고,
대화형 API 문서는 끄며, Origin이 다른 사이트인 WebSocket은 거부한다(OWASP ZAP 점검 결과, docs/security.md).
"""

# [구성] 기본값은 6단계에서 정한 서비스 구성이다: 최종 모델(outputs/stage4/trimtail/avg10.pt), 320 ms
# 청크(8프레임), 좌측 128프레임, GPU fp16 autocast, rule2 1.2 s, 끝점 뒤 캐시 유지(docs/experiments.md 6단계
# 판정 요약).
# [보안] 인증이 없는 로컬 데모라 기본으로 127.0.0.1에만 연다. 공개 전 OWASP ZAP 점검에서 나온 보안 헤더 누락을
# 고치고, ZAP이 보지 않는 WebSocket은 Origin 검사를 따로 넣었다(docs/security.md).
# [연결] 인식 로직은 service.session.StreamSession에 있고, 이 파일은 연결 관리·직렬화·보안 헤더만 맡는다.
# scripts/stream_client.py(실시간 지연 측정)와 scripts/stage6.py가 이 서버를 띄워 /info로 설정을 대조한 뒤
# 쓴다.

from __future__ import annotations

# base64·hashlib: CSP용 인라인 블록 SHA-256 해시와 /info의 체크포인트 SHA-256. re: 페이지에서 인라인 블록을
# 찾고 Origin을 비교. os: STT_CHECKPOINT 환경 변수.
import argparse
import base64
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

# FastAPI: HTTP 경로·WebSocket·미들웨어, uvicorn: ASGI 서버 실행. 프로젝트 모듈: 체크포인트
# 로드(evaluate.load_model), LogMel, 청크 설정, 끝점 설정, 세션과 예열, 어휘.
import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from stt_scratch.evaluate import load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

# ROOT: 저장소 루트(src/stt_scratch/service/server.py에서 세 단계 위). 체크포인트·어휘·페이지 경로의 기준이다.
# 테스트는 이 값을 임시 폴더로 바꿔 쓴다.
ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web" / "index.html"
DEFAULT_CHECKPOINT = "outputs/stage4/trimtail/avg10.pt"  # 최종 모델(4단계 판정)
# 모든 HTTP 응답에 붙이는 보안 헤더(ZAP 기본 스캔의 Medium 2건·Low 5건 대응, docs/security.md): MIME 추측
# 금지, 다른 사이트의 프레임 삽입 금지(클릭재킹), Referer 미전송, 마이크는 같은 출처만·카메라·위치는 금지,
# 교차 출처 격리(COOP·COEP·CORP), 캐시 금지. API_CSP는 JSON 응답용으로 아무것도 불러오거나 프레임에 넣을 수
# 없게 한다.
COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
API_CSP = "default-src 'none'; frame-ancestors 'none'"


# CSP의 해시 출처 형식('sha256-<base64>')으로 인라인 블록 내용의 SHA-256을 만든다.
def _sha256_source(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


# 'unsafe-inline' 없이 페이지 안의 인라인 스크립트·스타일을 해시로만 허용하고, AudioWorklet 모듈을 Blob
# URL로 불러오므로 script-src에 blob:을 연다. 해시는 서버 시작 때 실제 페이지 파일에서 계산하므로
# 페이지를 고쳐도 해시를 손으로 맞출 필요가 없다. 정규식은 속성 없는 여는 태그만 찾으므로 페이지의 인라인
# 블록도 속성 없이 둔다.
def page_csp(html: str) -> str:
    """페이지용 CSP를 만든다.

    페이지 자신의 인라인 스크립트·스타일 블록(해시), AudioWorklet blob, 같은 출처 fetch·WebSocket만
    허용한다. 그 밖에는 아무것도 불러오거나, 이 페이지를 프레임에 넣거나, 폼을 제출할 수 없다.
    """
    scripts = [_sha256_source(s) for s in re.findall(r"<script>(.*?)</script>", html, re.S)]
    styles = [_sha256_source(s) for s in re.findall(r"<style>(.*?)</style>", html, re.S)]
    return (
        f"default-src 'none'; script-src {' '.join(scripts)} blob:; style-src {' '.join(styles)}; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


# 교차 사이트 WebSocket 하이재킹 방어: 브라우저는 다른 사이트의 페이지에서도 ws://127.0.0.1:8765/ws를 열 수
# 있으므로, Origin이 있고 그 호스트:포트가 요청 Host와 다르면 거부한다. Origin을 보내지 않는 스크립트
# 클라이언트(scripts/stream_client.py)는 통과한다. 다른 페이지가 마이크 음성을 훔칠 수는 없지만 이 서버를 계산
# 자원으로 쓸 수 있어서 막는다(docs/security.md).
def same_origin(origin: str | None, host: str | None) -> bool:
    """다른 사이트의 브라우저 WebSocket은 Origin을 싣고 오므로 거부한다. Origin 없는 스크립트는 통과한다."""
    if not origin:
        return True
    return host is not None and re.sub(r"^https?://", "", origin) == host


# 저장소 기준 상대 경로를 주고, 저장소 밖이면 파일 이름만 준다(ZAP 점검 뒤 수정).
def shown_path(path: Path) -> str:
    """/info에 보여 줄 체크포인트 경로. 로컬 절대 경로(디렉터리 구조)를 드러내지 않는다."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


# 모델·어휘·LogMel을 한 번 올리고, 예열을 돌린 뒤 FastAPI 앱을 만들어 돌려준다. 모델은 모든 연결이 공유하고
# 연결마다 세션 상태(캐시·버퍼)만 따로 둔다. 테스트(tests/test_service.py)는 작은 모델과 warm=True 그대로 이
# 함수를 직접 부른다.
def build_app(
    checkpoint: Path,
    chunk: int,
    left: int,
    device: str,
    endpoint: EndpointConfig,
    reset_cache: bool,
    amp: bool | None = None,
    warm: bool = True,
    step_path: str = "eager",
    final_pass: str = "none",
) -> FastAPI:
    dev = torch.device(device)
    model, _ = load_model(checkpoint, dev)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(dev)
    spec = ChunkSpec(chunk, left)
    use_amp = (dev.type == "cuda") if amp is None else amp
    # 연결마다 같은 설정으로 세션을 만든다(step 경로·2차 패스는 7단계 뒤 후속 옵션, 규칙 변경 26).
    options = {"amp": use_amp, "step_path": step_path, "final_pass": final_pass}
    # 예열(session.warm_up): 부분 청크 길이마다 한 번씩 flush해 첫 클라이언트가 새 텐서 모양의 1회성 준비
    # 시간을 겪지 않게 한다. 걸린 시간은 /info의 warm_up_s로 보인다.
    warm_s = (
        warm_up(StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)) if warm else 0.0
    )
    # 대화형 API 문서(/docs, /redoc)는 쓰지 않고 외부 CDN 스크립트를 불러오므로 끈다(docs/security.md).
    app = FastAPI(title="stt-scratch streaming", docs_url=None, redoc_url=None)
    page = WEB.read_text(encoding="utf-8")
    csp = page_csp(page)

    # 모든 HTTP 응답에 공통 보안 헤더를 붙인다. setdefault라 경로가 직접 정한 헤더(페이지의 CSP)는 덮어쓰지
    # 않는다.
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in COMMON_HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers.setdefault("Content-Security-Policy", API_CSP)
        return response

    # /info 내용: 체크포인트와 그 SHA-256, 청크·좌측 문맥, 장치·정밀도, 끝점 설정, 캐시 정책, 어휘 크기, 예열
    # 시간. 페이지가 설정을 표시하고, 측정 스크립트가 서버가 등록한 구성으로 떠 있는지 대조하며 결과 파일에
    # 출처로 남긴다(규칙 변경 24, docs/ai-verification.md 6단계 검토 반영).
    info = {
        "checkpoint": shown_path(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "chunk_frames": chunk,
        "chunk_ms": chunk * 40,
        "left_frames": left,
        "device": device,
        "precision": "fp16-autocast" if use_amp else "fp32",
        "endpoint": asdict(endpoint),
        "reset_cache_on_endpoint": reset_cache,
        "step_path": step_path,
        "final_pass": final_pass,
        "vocab": len(vocab),
        "warm_up_s": round(warm_s, 2),
    }

    @app.get("/", include_in_schema=False)  # 페이지는 API 스키마에 넣지 않는다
    def index() -> HTMLResponse:
        return HTMLResponse(page, headers={"Content-Security-Policy": csp})

    @app.get("/info")
    def get_info() -> JSONResponse:
        return JSONResponse(info)

    # 스트리밍 엔드포인트. Origin을 먼저 검사해 다른 사이트면 수락 전에 1008(정책 위반)로 닫는다. 연결마다 새
    # 세션을 만들고, 메시지마다: 바이너리 → PCM으로 처리, "flush" → 스트림 마무리 후 이벤트와 stats, "reset" →
    # 세션 새로 만들기, 그 밖의 텍스트는 무시한다.
    # 주의: 인코더 계산이 async 처리기 안에서 동기로 돌기 때문에 여러 연결이 동시에 오면 차례로 처리된다(로컬
    # 데모 전제).
    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not same_origin(socket.headers.get("origin"), socket.headers.get("host")):
            await socket.close(code=1008)  # 교차 사이트 WebSocket 하이재킹 방어
            return
        await socket.accept()
        session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                stats = None
                if message.get("bytes") is not None:
                    events = session.push_pcm16(message["bytes"])
                elif message.get("text") == "flush":
                    events = session.flush()
                    stats = session.stats.summary()
                elif message.get("text") == "reset":
                    session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)
                    events = []
                else:
                    events = []
                # 이벤트마다 JSON 텍스트 프레임 하나: 종류, 세그먼트 번호, 텍스트, 규칙, 세그먼트 프레임 수,
                # 계산 시간(ms). ensure_ascii=False로 한글을 그대로 보낸다.
                for ev in events:
                    await socket.send_text(
                        json.dumps(
                            {
                                "type": ev.kind,
                                "segment": ev.segment,
                                "text": ev.text,
                                "rule": ev.rule,
                                "frames": ev.frames,
                                "compute_ms": round(ev.compute_ms, 1),
                                # 2차 패스를 켰을 때 확정 자막 계산에 더 든 시간(끄면 0)
                                "final_pass_ms": round(ev.final_pass_ms, 1),
                            },
                            ensure_ascii=False,
                        )
                    )
                if stats is not None:
                    await socket.send_text(json.dumps({"type": "stats", **stats}))
        except WebSocketDisconnect:
            pass

    return app


# 명령행 실행. 체크포인트는 --checkpoint 또는 STT_CHECKPOINT 환경 변수(기본은 최종 모델). --reset-cache는
# 6단계에서 기각한 '끝점에서 캐시 초기화'를 켠다. --keep-cache는 6단계 전 기본값이 초기화였던 시절의 옵션으로,
# 지금은 기본값과 같아 도움말에서 숨겼다. --fp32는 fp32로 학습한 모델용이다.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.environ.get("STT_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rule2", type=float, default=1.2)
    p.add_argument(
        "--reset-cache",
        action="store_true",
        help="끝점에서 인코더 캐시를 초기화한다(6단계는 유지를 골랐다)",
    )
    p.add_argument("--keep-cache", action="store_true", help=argparse.SUPPRESS)  # 6단계 이후 기본 동작
    p.add_argument("--fp32", action="store_true", help="fp16 autocast를 쓰지 않는다(--fp32로 학습한 모델용)")
    p.add_argument(
        "--step-path",
        choices=["eager", "fast", "graph"],
        default="eager",
        help="인코더 step 경로. eager = 6·7단계 측정에 쓴 경로, fast = 같은 출력으로 연산 수를 줄인 경로, "
        "graph = fast + 정상 상태 step의 CUDA Graph(CPU에서는 고정 버퍼 코드만). fast·graph는 출력 동일성 "
        "관문(docs/experiments.md 규칙 변경 26)을 통과하기 전까지 기본값이 아니다",
    )
    p.add_argument(
        "--final-pass",
        choices=["none", "full"],
        default="none",
        help="full이면 끝점에서 세그먼트를 같은 모델의 전체 문맥으로 다시 디코딩해 확정 자막으로 쓴다(부분 "
        "자막을 대체). 판정 전 기본값은 none(docs/experiments.md 규칙 변경 26)",
    )
    args = p.parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint (or STT_CHECKPOINT) is required")
    app = build_app(
        ROOT / args.checkpoint,
        args.chunk,
        args.left,
        args.device,
        EndpointConfig(rule2_s=args.rule2),
        args.reset_cache,
        amp=False if args.fp32 else None,
        step_path=args.step_path,
        final_pass=args.final_pass,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])
