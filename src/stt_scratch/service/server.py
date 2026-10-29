"""WebSocket 스트리밍 STT 서버(FastAPI + uvicorn).

  uv run python -m stt_scratch.service.server  # 최종 모델, 320 ms 청크, 좌측 128, rule2 0.8 s, 캐시 유지,
                                               # 끝점 뒤 전체 문맥 2차 패스, step 경로 auto(GPU graph·CPU
                                               # fast)

기본값은 6단계에서 고른 구성에 7단계 뒤 후속 측정(docs/experiments.md 규칙 변경 26)의 판정을 더한 것이다.
서버는 시작할 때 가능한 마지막 부분 청크 길이마다 한 번씩 flush하고, 2차 패스 길이 묶음마다 전체 문맥
디코딩을 한 번씩 돌리며, graph 경로면 정상 상태 step의 CUDA Graph를 캡처해(예열) 첫 클라이언트가 1회성 준비
시간을 겪지 않게 한다. 캡처가 실패하면 경고를 남기고 이후 세션의 step 경로를 fast로 바꾼다(출력은 같고 /info에
사유가 남는다). 어휘 파일의 토큰 수가 체크포인트 출력 크기와 다르면 시작하지 않는다. CPU 장치에서는 torch 연산
스레드를 기본 1로 둔다(README·docs의 CPU 수치를 잰 조건, --threads).

프로토콜: 클라이언트는 ws://host:port/ws를 열고 little-endian int16 mono 16 kHz PCM을 바이너리 프레임으로
보낸다(크기는 짝수 바이트로 1 MiB = 약 32.8 s까지 자유, 320 ms = 10,240바이트 권장). 서버는 JSON 텍스트
프레임으로 답한다.
  {"type": "partial", "segment": n, "text": "..."}      열린 세그먼트의 부분 자막(덧붙기만 함)
  {"type": "final",   "segment": n, "text": "...", "rule": "rule2"}   끝점 규칙으로 닫힌 세그먼트
  {"type": "endpoint", "segment": n, "text": "", "rule": "rule1"}     빈 세그먼트를 닫은 끝점
  {"type": "error", "code": 1007, "reason": "..."}      잘못된 입력·상한 초과(이 한 줄 뒤 같은 코드로 닫음)
이벤트(partial·final·endpoint)에는 "frames"(그 세그먼트에서 처리한 인코더 프레임 수), "compute_ms"(그 청크의
인코더 계산 ms), "final_pass_ms"(2차 패스 계산 ms, 끄면 0)도 있다. 2차 패스(--final-pass full, 명령행
기본값)를 켜면 final의 text는 2차 패스 결과이고, final·endpoint에 "stream_text"(같은 세그먼트의 320 ms
스트리밍 텍스트 = 부분 자막이 쌓인 값, 2차 패스가 바꾸기 전)가 더 붙는다. 2차 패스 결과가 비면 endpoint로
닫히지만 그 stream_text는 비어 있지 않을 수 있다. 2차 패스를 끄면 stream_text 키가 없다(확정 텍스트가 곧
스트리밍 텍스트).
텍스트 프레임 "flush"는 스트림을 마무리하고(마지막 부분 청크 + final), 이벤트들 뒤에 다음으로 답한다.
  {"type": "stats", "chunks": n, "compute_ms_p50": ..., "compute_ms_p90": ..., "compute_ms_max": ...}
(세션 전체의 청크당 계산 시간). "reset"은 새 세션을 시작한다.
서버가 연결을 닫는 코드: 1007 = 홀수 바이트 프레임(int16 표본으로 나뉘지 않음), 1009 = 1 MiB를 넘는 프레임
(uvicorn에서는 전송 계층이 오류 줄 없이 먼저 닫는다), 1013 = 동시 세션 상한(8개)이 차 있음(나중에 다시 시도),
1008 = 다른 사이트의 Origin(수락 전에 닫음).
GET /은 web/index.html(마이크 클라이언트 페이지), GET /info는 서버 설정과 상태를 준다(체크포인트·어휘 파일
SHA-256, 실제 step 경로와 CUDA Graph 캡처 상태, torch 스레드 수, 프레임·세션 상한). 인증이 없으므로 로컬
데모 전용이며 기본으로 127.0.0.1에만 연다. Host 헤더가 허용 목록(기본 127.0.0.1·localhost·[::1], --host 값,
--allowed-host로 더한 값) 밖이면 HTTP·WebSocket 모두 400으로 거부해 DNS 리바인딩을 막는다. 모든 HTTP 응답에
보안 헤더(페이지에는 해시 기반 CSP)를 붙이고, 대화형 API 문서는 끄며, Origin이 다른 사이트인 WebSocket은
거부한다(OWASP ZAP 점검 결과, docs/security.md).
"""

# [구성] 기본값: 최종 모델(outputs/stage4/trimtail/avg10.pt), 320 ms 청크(8프레임), 좌측 128프레임, GPU fp16
# autocast, 끝점 뒤 캐시 유지(6단계 판정). 7단계 뒤 후속 측정(2026-10-18, reports/followup/summary.md)의
# 판정으로 rule2를 1.2 s에서 0.8 s로(P2: 캐시 유지 상태에서 끝점 끈 기준 대비 CER 증가 0.000포인트), 확정
# 자막을 끝점 뒤 전체 문맥 2차 패스로(P1: dev_mix 26.86 → 22.85 %, dev_H 18.75 → 16.01 %, GPU 2차 패스 p90 55
# ms), step 경로를 GPU graph·CPU fast로(P3: dev_H 전사 1,113/1,113 동일, GPU 청크 p90 72.5 → 43.0 ms) 바꿨다.
# 모두 dev 수치이며 봉인 test(12층·greedy·1-pass)로는 확인하지 않았다. 서비스 모델은 봉인 test로 잰 12층 모델
# 그대로다.
# [보안] 인증이 없는 로컬 데모라 기본으로 127.0.0.1에만 연다. 공개 전 OWASP ZAP 점검에서 나온 보안 헤더 누락을
# 고치고, ZAP이 보지 않는 WebSocket은 Origin 검사를 따로 넣었다(docs/security.md). 그 뒤 보안 검토에서 Origin
# 검사가 DNS 리바인딩으로 우회되고(Host와 Origin이 둘 다 공격자 도메인이 됨) WebSocket 입력·자원 상한이 없는
# 것이 나와 Host 허용 목록(TrustedHostMiddleware), 메시지 크기·짝수 바이트 검사, 동시 세션 상한, uvicorn 전송
# 상한을 더했다.
# [연결] 인식 로직은 service.session.StreamSession에 있고, 이 파일은 연결 관리·직렬화·보안 헤더만 맡는다.
# scripts/stream_client.py(실시간 지연 측정)와 scripts/stage6.py가 이 서버를 띄워 /info로 설정을 대조한 뒤
# 쓴다.

from __future__ import annotations

# base64·hashlib: CSP용 인라인 블록 SHA-256 해시와 /info의 체크포인트 SHA-256. re: 페이지에서 인라인 블록을
# 찾고 Origin을 비교. os: STT_CHECKPOINT 환경 변수. ipaddress: Host 허용 목록의 IP 표기 정규화와 루프백 판정.
# logging: 루프백 밖에 열 때의 시작 경고.
import argparse
import base64
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

# FastAPI: HTTP 경로·WebSocket·미들웨어, TrustedHostMiddleware: Host 허용 목록(Starlette 것을 FastAPI가 다시
# 내보냄), uvicorn: ASGI 서버 실행. 프로젝트 모듈: 체크포인트 로드와 어휘 크기 검사(evaluate.load_model·
# check_vocab), 그래프 캡처 상태(model.streaming.graph_status; 테스트가 바꿔 끼울 수 있게 모듈로 가져온다),
# LogMel, 청크 설정, 끝점 설정, 세션과 예열, 어휘.
import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from stt_scratch.evaluate import check_vocab, load_model
from stt_scratch.model import streaming
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import Event, StreamSession, warm_up
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
# Host 허용 목록 기본값: 이 PC의 루프백 이름. 브라우저는 주소창의 호스트를 Host 헤더에 그대로 싣고(IPv6는
# 대괄호 표기), Starlette TrustedHostMiddleware는 포트를 뗀 호스트 부분을 이 목록과 정확히 비교해 목록 밖이면
# HTTP 요청과 WebSocket 핸드셰이크 모두 400으로 거부한다. 왜: Origin 검사(same_origin)는 Origin과 요청 Host만
# 비교하는데, DNS 리바인딩(공격자가 자기 도메인의 DNS를 127.0.0.1로 다시 가리킴) 뒤에는 브라우저가 Host와
# Origin에 둘 다 공격자 도메인을 실어 그 검사를 통과했다. Host를 목록으로 묶으면 그 경로가 닫힌다.
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
# 모든 인터페이스에 여는 바인드 주소(Host 헤더 모양으로 정규화한 값). 클라이언트가 Host로 보내는 값이 아니므로
# --host가 이 값이면 허용 목록에 넣지 않는다.
WILDCARD_BIND_HOSTS = ("0.0.0.0", "[::]")
# WebSocket 메시지 상한(바이트): 1 MiB = int16 표본 524,288개 = 16 kHz에서 약 32.8 s(인코더 청크 약 102개
# 분량). 처리기가 이보다 큰 바이너리 프레임을 1009로 닫고, 명령행 실행은 같은 값을 uvicorn ws_max_size로 넘겨
# 전송 계층에서도 막는다. 왜: uvicorn 기본 상한 16 MiB면 한 프레임에 약 524 s 오디오가 들어오고, 인코더 계산이
# async 처리기 안에서 동기로 돌기 때문에 그 계산 동안 다른 연결이 모두 멈춘다. 페이지와 측정 도구(scripts/
# stream_client.py)는 320 ms(10,240바이트)씩 보내므로 여유가 크다.
MAX_FRAME_BYTES = 1 << 20
# 동시 WebSocket 세션 상한. 넘으면 오류 줄 뒤 1013(나중에 다시 시도)으로 닫는다. 왜: 로컬 1인 데모에서 동시에
# 필요한 세션은 한두 개이고(측정 도구는 발화마다 연결 하나를 차례로 연다), 세션마다 인코더 캐시를 따로 들며
# 계산이 이벤트 루프에서 차례로 돌아 세션이 늘수록 모든 자막이 늦어진다. uvicorn limit_concurrency는 새 HTTP
# 요청에만 503을 내고 WebSocket 업그레이드는 막지 않으므로(uvicorn 0.54 소스와 실제 서버로 확인) 앱에서 따로
# 센다.
MAX_SESSIONS = 8
# uvicorn limit_concurrency: 열린 연결(HTTP keep-alive와 WebSocket 포함)이나 처리 중인 작업이 이 수에 닿으면
# 새 HTTP 요청에 503을 낸다. 세션 상한(8)보다 크게 두어 세션이 다 차도 페이지와 /info는 열린다.
LIMIT_CONCURRENCY = 16
# 시작 경고용 로거(인증 없는 서버를 루프백 밖에 열 때, Host 검사를 끌 때). uvicorn이 로깅을 설정하기 전에
# 남기므로 처리기가 없으면 파이썬 기본 처리기(logging.lastResort)가 stderr에 쓴다.
log = logging.getLogger("stt_scratch.service.server")


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
# 자원으로 쓸 수 있어서 막는다(docs/security.md). Host 자체는 이 검사 앞에서 TrustedHostMiddleware가 허용
# 목록으로 제한하므로, Origin == Host이면 Origin의 호스트도 허용 목록 안이다. 그래서 이 함수는 Host 비교만
# 하면 된다(DNS 리바인딩처럼 Host와 Origin이 같은 공격자 도메인인 경우는 미들웨어가 먼저 400으로 막는다).
def same_origin(origin: str | None, host: str | None) -> bool:
    """다른 사이트의 브라우저 WebSocket은 Origin을 싣고 오므로 거부한다. Origin 없는 스크립트는 통과한다."""
    if not origin:
        return True
    return host is not None and re.sub(r"^https?://", "", origin) == host


# Host 허용 목록 도우미 네 개. 명령행 실행(main)이 --host·--allowed-host 값으로 목록을 만들고 검사하며,
# build_app은 받은 목록을 다시 검사해 TrustedHostMiddleware에 넘긴다. normalize_host: 호스트 이름·IP를
# 브라우저가 Host 헤더에 싣는 모양으로 맞춘다. 소문자, IPv6 리터럴은 압축 표기를 대괄호로 감싼 것('FE80::1' →
# '[fe80::1]', '[::1]'은 그대로), IPv4는 그대로, IP가 아닌 이름·패턴('*.lan', '*')은 소문자로만. 왜:
# TrustedHostMiddleware는 Host에서 포트를 뗀 호스트 부분(IP 리터럴의 대괄호 보존)을 목록과 문자열로 정확히
# 비교하므로, `--host ::1`처럼 대괄호 없이 적은 값을 그대로 넣으면 어떤 요청과도 맞지 않는다.
def normalize_host(value: str) -> str:
    """호스트 이름·IP를 Host 헤더에 나타나는 모양(소문자, IPv6는 대괄호 압축 표기)으로 돌려준다."""
    value = value.strip().lower()
    bare = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        return value
    return f"[{ip}]" if ip.version == 6 else str(ip)


# 루프백 판정: 127.0.0.0/8, ::1, localhost. 왜: 이 서버는 인증이 없어서 루프백 밖에 열면 같은 네트워크의
# 누구나 GPU·CPU를 쓸 수 있으므로 main()이 경고를 남긴다. 0.0.0.0·::(모든 인터페이스)도 루프백이 아니다.
def is_loopback_host(value: str) -> bool:
    """`--host` 값이 이 PC 안에서만 닿는 루프백 주소(127.0.0.0/8, ::1, localhost)인지."""
    host = normalize_host(value)
    if host == "localhost":
        return True
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return ipaddress.ip_address(bare).is_loopback
    except ValueError:
        return False


# 명령행 실행의 Host 허용 목록. 기본 루프백 이름 셋(DEFAULT_ALLOWED_HOSTS)에 --host 값과 --allowed-host 값들을
# 정규화해 중복 없이 순서대로 붙인다. 0.0.0.0·::처럼 모든 인터페이스에 여는 바인드 주소는 클라이언트가 Host로
# 보내는 값이 아니므로 넣지 않는다(그 경우 LAN 클라이언트가 쓸 이름·IP는 --allowed-host로 준다). 입력:
# bind_host (--host), extra(--allowed-host 값들, None이면 없음). 출력: TrustedHostMiddleware에 그대로 넘길
# 목록. 모양 검사는 check_allowed_hosts가 따로 한다.
def allowed_hosts_for(bind_host: str, extra: Sequence[str] | None = None) -> list[str]:
    """`--host`·`--allowed-host`로 Host 허용 목록을 만든다(기본 127.0.0.1·localhost·[::1]을 항상 포함)."""
    hosts = list(DEFAULT_ALLOWED_HOSTS)
    bind = normalize_host(bind_host)
    candidates = [] if bind in WILDCARD_BIND_HOSTS else [bind]
    candidates.extend(normalize_host(h) for h in extra or ())
    for host in candidates:
        if host not in hosts:
            hosts.append(host)
    return hosts


# TrustedHostMiddleware가 받는 모양인지 서버가 뜨기 전에 확인한다. 미들웨어도 같은 규칙을 assert로 보지만
# 미들웨어 스택은 첫 요청 때 만들어지므로, 잘못된 값이면 서버가 뜬 뒤 첫 요청에서야 실패한다. 규칙: 비어 있지
# 않고, 빈 항목이 없고, '*'는 단독('*' = Host 검사를 끔) 또는 맨 앞의 하위 도메인 패턴('*.example.com')으로만.
def check_allowed_hosts(hosts: Sequence[str]) -> list[str]:
    """Host 허용 목록을 검사해 list로 돌려준다. 잘못되면 원인을 적은 ValueError."""
    checked = list(hosts)
    if not checked:
        raise ValueError("the Host allow list must not be empty")
    for pattern in checked:
        if not pattern:
            raise ValueError("the Host allow list has an empty entry")
        bare_wildcard = pattern.startswith("*") and pattern != "*" and not pattern.startswith("*.")
        if "*" in pattern[1:] or bare_wildcard:
            raise ValueError(
                f"host pattern {pattern!r} is not supported: use a name, an IP, '*.example.com' or '*'"
            )
    return checked


# WebSocket 처리기가 입력을 거부할 때: 원인을 적은 오류 JSON 한 줄({"type": "error", "code": n, "reason":
# "..."})을 보내고 같은 코드로 닫는다. 왜: 종료 코드만으로는 브라우저 콘솔에 원인이 남지 않고, 전에는 홀수
# 바이트 입력이 처리기 밖으로 예외를 던져 트레이스백과 1011(서버 내부 오류)로 끝났다. 코드: 1007 = 형식에 맞지
# 않는 데이터, 1009 = 메시지가 너무 큼, 1013 = 나중에 다시 시도(세션 상한). 이유 문구는 닫기 프레임의
# 상한(123바이트)보다 짧게 쓴다. 로그에도 경고로 남겨 잘못된 클라이언트나 남용을 서버 쪽에서 알 수 있게 한다.
async def refuse(socket: WebSocket, code: int, reason: str) -> None:
    """오류 JSON 한 줄을 보내고 `code`로 연결을 닫는다(수락된 연결에만 쓴다)."""
    log.warning("websocket from %s closed with %d: %s", socket.client, code, reason)
    await socket.send_text(json.dumps({"type": "error", "code": code, "reason": reason}))
    await socket.close(code=code, reason=reason)


# 세션 이벤트 하나를 WebSocket으로 보낼 JSON 객체로 바꾼다(프로토콜은 이 파일 머리말). 키: 종류, 세그먼트
# 번호, 텍스트, 끝점 규칙, 세그먼트 프레임 수, 그 청크의 계산 ms, 2차 패스 계산 ms(끄면 0; 소수 첫째 자리로
# 반올림). stream_text는 세션이 값을 채운 이벤트(2차 패스를 켠 세션의 final·endpoint)에만 넣는다. 왜: 2차
# 패스를 켜면 확정 텍스트가 2차 패스 결과라, 측정 클라이언트가 서비스 경로의 320 ms 스트리밍 텍스트를
# simulated 가설과 비교하려면 그 텍스트가 따로 필요했다(docs/experiments.md 규칙 변경 27). 키를 더하기만
# 하므로 기존 클라이언트(웹 페이지 포함)는 그대로 동작한다. 측정 클라이언트의 오프라인
# 재생(scripts/stream_client.py --replay-check)도 이 함수로 직렬화해 서버와 같은 모양의 기록을 만든다.
def event_message(ev: Event) -> dict:
    """Event를 서버가 보내는 JSON 객체(dict)로 바꾼다. stream_text는 값이 있을 때만 넣는다."""
    message = {
        "type": ev.kind,
        "segment": ev.segment,
        "text": ev.text,
        "rule": ev.rule,
        "frames": ev.frames,
        "compute_ms": round(ev.compute_ms, 1),
        "final_pass_ms": round(ev.final_pass_ms, 1),
    }
    if ev.stream_text is not None:
        message["stream_text"] = ev.stream_text
    return message


# 저장소 기준 상대 경로를 주고, 저장소 밖이면 파일 이름만 준다(ZAP 점검 뒤 수정).
def shown_path(path: Path) -> str:
    """/info에 보여 줄 체크포인트 경로. 로컬 절대 경로(디렉터리 구조)를 드러내지 않는다."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


# 모델·어휘·LogMel을 한 번 올리고, 예열을 돌린 뒤 FastAPI 앱을 만들어 돌려준다. 모델은 모든 연결이 공유하고
# 연결마다 세션 상태(캐시·버퍼)만 따로 둔다. 테스트(tests/test_service.py)는 작은 모델과 warm=True 그대로 이
# 함수를 직접 부른다. 보안 인자: allowed_hosts = Host 허용 목록(None이면 DEFAULT_ALLOWED_HOSTS; 명령행은
# main()이 --host·--allowed-host로 만든 목록을, 테스트는 TestClient의 Host인 'testserver'를 넘긴다),
# max_frame_bytes = 바이너리 프레임 상한(넘으면 1009), max_sessions = 동시 세션 상한(넘으면 1013). 라이브러리
# 기본값(eager·none)은 측정 도구용 원래 동작 그대로다. step 경로 auto의 장치별 값: 후속 측정 P3 판정(출력
# 동일성 관문을 통과하고 같은 실행의 eager보다 청크 p90이 낮은 것 중 가장 빠른 것). GPU는 graph(43.0 ms <
# eager 72.5 ms), CPU 1스레드는 fast(44.4 ms < 51.3 ms).
AUTO_STEP_PATH = {"cuda": "graph", "cpu": "fast"}


def resolve_step_path(step_path: str, device: str) -> str:
    """'auto'를 장치에 맞는 step 경로로 바꾼다(그 밖의 값은 그대로). 표에 없는 장치는 eager."""
    if step_path != "auto":
        return step_path
    return AUTO_STEP_PATH.get(torch.device(device).type, "eager")


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
    allowed_hosts: Sequence[str] | None = None,
    max_frame_bytes: int = MAX_FRAME_BYTES,
    max_sessions: int = MAX_SESSIONS,
) -> FastAPI:
    # 허용 목록은 모델을 올리기 전에 검사해 잘못된 패턴이면 바로(첫 요청이 아니라) 실패하게 한다.
    hosts = check_allowed_hosts(DEFAULT_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts)
    dev = torch.device(device)
    model, _ = load_model(checkpoint, dev)
    vocab_path = ROOT / "manifests" / "vocab.json"
    vocab = Vocab.load(vocab_path)
    # 어휘 토큰 수 ≠ 체크포인트 출력 크기면 여기서 멈춘다(SystemExit, evaluate.check_vocab). 전에는
    # 검사가 없어 어휘가 작으면 디코딩 중 IndexError로 연결이 끊기고, 크기가 같고 순서만 다르면 조용히
    # 다른 글자를 냈다. 순서 차이는 크기로 잡을 수 없으므로 /info의 vocab_sha256으로 어떤 어휘
    # 파일이었는지 남긴다.
    check_vocab(model, vocab)
    logmel = LogMel().to(dev)
    spec = ChunkSpec(chunk, left)
    use_amp = (dev.type == "cuda") if amp is None else amp
    step_path = resolve_step_path(step_path, device)
    # 연결마다 같은 설정으로 세션을 만든다(step 경로·2차 패스는 7단계 뒤 후속 옵션, 규칙 변경 26). ws 처리기가
    # 이 dict를 그대로 읽으므로 아래에서 step_path를 바꾸면 이후 연결에 바로 적용된다.
    options = {"amp": use_amp, "step_path": step_path, "final_pass": final_pass}
    # 예열(session.warm_up): 부분 청크 길이마다 한 번씩 flush해 첫 클라이언트가 새 텐서 모양의 1회성 준비
    # 시간을 겪지 않게 한다. 걸린 시간은 /info의 warm_up_s로 보인다.
    warm_s = (
        warm_up(StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)) if warm else 0.0
    )
    # CUDA Graph 캡처 상태(graph 경로로 예열했을 때만 값이 있고 그 밖에는 둘 다 None). 정상 상태 stepper는
    # 모델·청크 설정마다 하나를 모든 세션이 공유하므로(model/streaming.py _STEPPERS) 예열 중 캡처가 실패하면
    # 그 뒤 모든 연결이 그래프 없는 고정 버퍼 코드로 돈다. 전에는 stepper가 사유만 남기고 /info는 인자로 받은
    # 'graph'를 그대로 보여 측정 결과 파일(scripts/stream_client.py의 -meta.json)에도 graph로 적혔다. 이제
    # 실패면 경고를 남기고 이후 세션의 step 경로를 fast로 바꿔 /info가 실제 경로를 적게 한다. fast와 graph는
    # 출력 동일성 관문(규칙 변경 26 P3)에서 같은 출력이 확인된 경로라 자막은 같고, 달라지는 것은 청크당 계산
    # 시간(graph 정상 상태 p90 8.5 ms → fast 수준)뿐이다. CPU의 graph 경로는 그래프를 캡처하지 않으므로
    # captured False·failed None 그대로 둔다. 사유는 앞 200자만 싣는다(stepper는 500자까지 보관).
    graph_captured = graph_failed = None
    if warm and step_path == "graph":
        status = streaming.graph_status(model, spec)
        graph_captured = status["captured"]
        if status["failed"] is not None:
            graph_failed = status["failed"][:200]
            log.warning(
                "CUDA Graph capture failed during warm-up; sessions use the fast step path instead (same "
                "output, more kernel launches per chunk): %s",
                graph_failed,
            )
            step_path = options["step_path"] = "fast"
    # 대화형 API 문서(/docs, /redoc)는 쓰지 않고 외부 CDN 스크립트를 불러오므로 끈다(docs/security.md).
    app = FastAPI(title="stt-scratch streaming", docs_url=None, redoc_url=None)
    page = WEB.read_text(encoding="utf-8")
    csp = page_csp(page)

    # Host 허용 목록(DNS 리바인딩 차단). Starlette는 나중에 등록한 미들웨어가 바깥에 오므로, 보안 헤더
    # 미들웨어보다 먼저 등록해 안쪽에 두면 이 미들웨어의 400 거부 응답에도 보안 헤더가 붙는다. http 요청과
    # websocket 핸드셰이크 모두 Host를 검사하며, websocket은 거부 응답(websocket.http.response.*)으로 수락
    # 전에 막힌다(Starlette 1.7, uvicorn은 이 확장을 지원). www_redirect=False: 'www.' 리다이렉트는 이 서버에
    # 쓸 일이 없다.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts, www_redirect=False)

    # 모든 HTTP 응답에 공통 보안 헤더를 붙인다. setdefault라 경로가 직접 정한 헤더(페이지의 CSP)는 덮어쓰지
    # 않는다.
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in COMMON_HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers.setdefault("Content-Security-Policy", API_CSP)
        return response

    # /info 내용: 체크포인트와 그 SHA-256, 청크·좌측 문맥, 장치·정밀도, 끝점 설정, 캐시 정책, 실제 step 경로와
    # CUDA Graph 캡처 상태, 2차 패스, 어휘 크기와 어휘 파일 SHA-256, torch 연산 스레드 수, 프레임·세션 상한,
    # 예열 시간. 페이지가 설정을 표시하고, 측정 스크립트가 서버가 등록한 구성으로 떠 있는지 대조하며 결과
    # 파일에 출처로 남긴다(규칙 변경 24, docs/ai-verification.md 6단계 검토 반영). 값은 앱을 만들 때 한 번
    # 정한다(스레드 수는 main()이 모델을 올리기 전에 정하므로 예열·세션과 같은 값이다).
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
        "graph_captured": graph_captured,
        "graph_failed": graph_failed,
        "final_pass": final_pass,
        "vocab": len(vocab),
        "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "cpu_threads": torch.get_num_threads(),
        "max_frame_bytes": max_frame_bytes,
        "max_sessions": max_sessions,
        "warm_up_s": round(warm_s, 2),
    }

    @app.get("/", include_in_schema=False)  # 페이지는 API 스키마에 넣지 않는다
    def index() -> HTMLResponse:
        return HTMLResponse(page, headers={"Content-Security-Policy": csp})

    @app.get("/info")
    def get_info() -> JSONResponse:
        return JSONResponse(info)

    # 지금 열린 세션 수(동시 세션 상한 검사용). uvicorn은 프로세스 하나·이벤트 루프 하나로 돌고 처리기는 await
    # 사이에서만 바뀌므로 잠금 없이 세어도 된다. 연결이 어떻게 끝나든(클라이언트 종료, 거부, 예외) finally에서
    # 뺀다.
    app.state.active_sessions = 0

    # 스트리밍 엔드포인트. Origin을 먼저 검사해 다른 사이트면 수락 전에 1008(정책 위반)로 닫는다(Host는 그
    # 앞에서 TrustedHostMiddleware가 허용 목록으로 걸렀다). 수락 뒤 동시 세션 상한이 차 있으면 오류 줄 뒤
    # 1013으로 닫고, 아니면 연결마다 새 세션을 만든다. 메시지마다: 바이너리 → 상한(1009)·짝수 바이트(1007)
    # 검사 뒤 PCM으로 처리, "flush" → 스트림 마무리 후 이벤트와 stats, "reset" → 세션 새로 만들기, 그 밖의
    # 텍스트는 무시한다. 주의: 인코더 계산이 async 처리기 안에서 동기로 돌기 때문에 여러 연결이 동시에 오면
    # 차례로 처리된다(로컬 데모 전제). 프레임 상한과 세션 상한은 그 차례가 한 클라이언트 때문에 한없이
    # 길어지는 것을 막는 장치다.
    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not same_origin(socket.headers.get("origin"), socket.headers.get("host")):
            await socket.close(code=1008)  # 교차 사이트 WebSocket 하이재킹 방어
            return
        await socket.accept()
        counted = False
        try:
            if app.state.active_sessions >= max_sessions:
                await refuse(socket, 1013, f"{max_sessions} sessions are already open; try again later")
                return
            app.state.active_sessions += 1
            counted = True
            session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                stats = None
                data = message.get("bytes")
                if data is not None:
                    if len(data) > max_frame_bytes:
                        reason = f"frame of {len(data)} bytes exceeds the {max_frame_bytes}-byte limit"
                        await refuse(socket, 1009, reason)
                        return
                    if len(data) % 2:
                        reason = f"frame of {len(data)} bytes: int16 PCM needs an even number of bytes"
                        await refuse(socket, 1007, reason)
                        return
                    events = session.push_pcm16(data)
                elif message.get("text") == "flush":
                    events = session.flush()
                    stats = session.stats.summary()
                elif message.get("text") == "reset":
                    session = StreamSession(model, vocab, spec, logmel, endpoint, reset_cache, **options)
                    events = []
                else:
                    events = []
                # 이벤트마다 JSON 텍스트 프레임 하나(event_message: 종류, 세그먼트 번호, 텍스트, 규칙,
                # 세그먼트 프레임 수, 계산 시간, 2차 패스 시간, 2차 패스를 켰으면 닫힘 이벤트의 stream_text).
                # ensure_ascii=False로 한글을 그대로 보낸다.
                for ev in events:
                    await socket.send_text(json.dumps(event_message(ev), ensure_ascii=False))
                if stats is not None:
                    await socket.send_text(json.dumps({"type": "stats", **stats}))
        except WebSocketDisconnect:
            pass
        finally:
            if counted:
                app.state.active_sessions -= 1

    return app


# 명령행 인자. 체크포인트는 --checkpoint 또는 STT_CHECKPOINT 환경 변수(기본은 최종 모델). --reset-cache는
# 6단계에서 기각한 '끝점에서 캐시 초기화'를 켠다. --keep-cache는 6단계 전 기본값이 초기화였던 시절의 옵션으로,
# 지금은 기본값과 같아 도움말에서 숨겼다. --fp32는 fp32로 학습한 모델용이다. 테스트가 기본값을 확인할 수
# 있게 파서를 따로 만든다.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="스트리밍 STT 서버(로컬 데모). 기본값은 6단계·후속 측정 판정의 구성"
    )
    p.add_argument("--checkpoint", default=os.environ.get("STT_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="바인드 주소(기본 127.0.0.1 = 이 PC에서만). 루프백이 아니면 인증 없는 서버가 밖에 열리므로 "
        "시작할 때 경고를 남기고, 0.0.0.0·::이 아닌 값은 Host 허용 목록에 더한다",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--allowed-host",
        action="append",
        metavar="HOST",
        help="Host 헤더 허용 목록에 더할 이름이나 IP(여러 번 줄 수 있음). 기본 목록은 "
        "127.0.0.1·localhost·[::1]과 --host 값이고, 목록 밖의 Host로 온 HTTP·WebSocket 요청은 400으로 "
        "거부한다(DNS 리바인딩으로 Origin 검사를 우회하는 것을 막음). '*.example.com' 패턴과 "
        "'*'(Host 검사를 끔, 경고)도 받는다. 예: Docker 컨테이너에서 붙는 점검 도구는 "
        "host.docker.internal",
    )
    p.add_argument(
        "--rule2",
        type=float,
        default=0.8,
        help="토큰 뒤 묵음 끝점(초). 6단계는 캐시 초기화 상태에서 1.2를 골랐고, 캐시 유지 상태로 다시 잰 "
        "후속 측정(P2)에서 0.8이 등록 조건(CER 증가 ≤ 0.2포인트)을 만족해 기본값이 됐다(EP50 1,295 → 905 ms)",
    )
    p.add_argument(
        "--reset-cache",
        action="store_true",
        help="끝점에서 인코더 캐시를 초기화한다(6단계는 유지를 골랐다)",
    )
    p.add_argument("--keep-cache", action="store_true", help=argparse.SUPPRESS)  # 6단계 이후 기본 동작
    p.add_argument("--fp32", action="store_true", help="fp16 autocast를 쓰지 않는다(--fp32로 학습한 모델용)")
    p.add_argument(
        "--step-path",
        choices=["auto", "eager", "fast", "graph"],
        default="auto",
        help="인코더 step 경로. eager = 6·7단계 측정에 쓴 경로, fast = 같은 출력으로 연산 수를 줄인 경로, "
        "graph = fast + 정상 상태 step의 CUDA Graph(CPU에서는 고정 버퍼 코드만). auto = GPU graph, CPU fast"
        "(후속 측정 P3: 두 경로 모두 출력 동일성 관문 통과, docs/experiments.md 규칙 변경 26)",
    )
    p.add_argument(
        "--final-pass",
        choices=["none", "full"],
        default="full",
        help="full이면 끝점에서 세그먼트를 같은 모델의 전체 문맥으로 다시 디코딩해 확정 자막으로 쓴다(부분 "
        "자막을 대체). 후속 측정 P1 채택으로 기본값 full(GPU 2차 패스 p90 55 ms, CPU 1스레드 195 ms). none = "
        "6단계까지의 1-pass 확정 자막",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="torch 연산(intra-op) 스레드 수(1 이상). 기본값은 --device가 cpu면 1(README·docs의 CPU "
        "청크 계산·2차 패스 수치를 잰 1스레드 조건과 같게; 다른 프로그램이 CPU를 쓰는 중에 여러 "
        "스레드로 돌리면 step이 수 초씩 늦어진 적이 있음), cuda면 바꾸지 않음(프로세스 기본값). "
        "실제 값은 /info의 cpu_threads",
    )
    return p


# 명령행 실행: 인자를 읽어 Host 허용 목록을 만들고(잘못된 패턴은 모델을 올리기 전에 거부), torch 연산 스레드
# 수를 정하고(--threads, CPU 장치 기본 1), 인증 없는 서버를 루프백 밖에 열거나 Host 검사를 끄면 경고를 남긴
# 뒤, 앱을 만들어 uvicorn으로 띄운다. uvicorn에는 WebSocket
# 메시지 상한(ws_max_size = MAX_FRAME_BYTES, 넘으면 전송 계층이 오류 줄 없이 1009로 닫음)과 동시 연결·작업
# 상한(limit_concurrency, 넘으면 새 HTTP 요청에 503)을 함께 넘긴다. ping 간격·시간 제한은 uvicorn 기본(20 s)
# 그대로다. 경고는 uvicorn이 로깅을 설정하기 전에 나가므로 파이썬 기본 처리기(stderr)가 찍는다.
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint (or STT_CHECKPOINT) is required")
    try:
        allowed = check_allowed_hosts(allowed_hosts_for(args.host, args.allowed_host))
    except ValueError as e:
        raise SystemExit(f"--allowed-host: {e}") from None
    # torch 연산 스레드 수. --threads가 없으면 장치가 cpu일 때만 1(6단계·후속 측정의 'CPU 1스레드' 조건과
    # 같게: scripts/stage6.py·scripts/followup.py가 torch.set_num_threads(1)로 잰 값이 README·docs의 CPU
    # 수치다), cuda면 프로세스 기본값 그대로. 모델을 올리기 전에 정해야 예열과 /info의 cpu_threads가 세션과
    # 같은 조건이 된다. 여기(main)에서만 바꾸고 build_app은 건드리지 않아 테스트 프로세스의 스레드 수에는
    # 영향이 없다. 왜 기본 1인가: 다른 프로그램이 CPU를 쓰는 중에 여러 스레드로 돌리면 step 하나가 수 초씩
    # 걸린 관찰이 있고(scripts/followup.py 관문 측정 주석), 기본 스레드 수(이 PC 6코어)로 돈 서버의 지연은
    # 기록된 수치가 재는 조건이 아니었다.
    threads = args.threads
    if threads is None and torch.device(args.device).type == "cpu":
        threads = 1
    if threads is not None:
        if threads < 1:
            raise SystemExit(f"--threads must be at least 1, got {threads}")
        torch.set_num_threads(threads)
    if not is_loopback_host(args.host):
        log.warning(
            "--host %s is not a loopback address: this unauthenticated server accepts connections from other "
            "machines. Only requests whose Host header is in the allow list are served (%s); add names or "
            "IPs with --allowed-host",
            args.host,
            ", ".join(allowed),
        )
    if "*" in allowed:
        log.warning(
            "--allowed-host '*' disables the Host check; DNS rebinding can bypass the Origin check again"
        )
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
        allowed_hosts=allowed,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_max_size=MAX_FRAME_BYTES,
        limit_concurrency=LIMIT_CONCURRENCY,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
