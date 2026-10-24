"""브라우저 자막 페이지(web/index.html) 스크립트의 연결·오디오 수명 관리와 2차 패스 표시 테스트.

페이지의 인라인 <script>를 Node.js vm 안에서 가짜 DOM·WebSocket·AudioContext·getUserMedia·가상
타이머로 실행하는 하네스(tests/web_page_harness.js)에 시나리오를 넘기고 관찰값을 본다. 확인하는 것:
(1) 멈춤 뒤 곧바로 다시 시작해도 새 연결은 살아 있고 이전 연결만 flush 뒤 닫히며 이전 연결의 확정 자막은
그대로 붙음, (2) 멈출 때마다 마이크 트랙이 모두 멈추고 worklet에 남은 음성을 먼저 내보냄(drain), (3) 시작
실패(권한 거부, 비보안 출처, createMediaStreamSource 예외, 열리기 전 연결 실패)에서 사유가 상태 줄에
보이고 시작 버튼이 돌아오며 처리되지 않은 거부가 없음, (4) AudioContext가 16 kHz로 열리지 않으면 아무것도
보내지 않고 정리, (5) endpoint 이벤트가 잠정(부분) 자막을 비우고 멈춘 연결의 늦은 부분 자막이 새 연결의
자막을 덮지 않음, (6) 서버의 오류 줄과 종료 코드가 상태 줄에 보이고 권한 창이 떠 있는 동안 끊긴 연결도
마이크를 바로 놓음, (7) flush 뒤 stats가 오면 1.5 s를 기다리지 않고 닫음, (8) /info 줄에 2차 패스·step
경로, 안내 문구, 잠정 자막 표시(class), CSP 호환(인라인 블록 하나씩, style 속성 없음).
"""

# [방식] Node.js(node)가 PATH에 없으면 하네스 테스트는 건너뛴다(CI의 ubuntu-24.04 러너에는 있다). 가짜
# 환경은 페이지가 쓰는 API 표면만 흉내 내므로 실제 브라우저(권한 창, 오디오 스레드, 닫기 핸드셰이크에 걸리는
# 시간)와 다를 수 있다. 그래서 README 서비스 절에 브라우저에서 직접 보는 수동 점검 항목을 함께 둔다. 시간은
# 가상 시계(ms)이고, 단계 사이에 마이크로태스크를 모두 비우므로 관찰값은 결정적이다. 페이지 자체(정적 표시·
# CSP)는 Node 없이 파일을 읽어 본다.

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "index.html"
HARNESS = Path(__file__).with_name("web_page_harness.js")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None, reason="Node.js(node)가 PATH에 없어 페이지 스크립트 시뮬레이션을 건너뜀"
)


# 하네스 실행: 단계 목록과 옵션을 JSON으로 표준 입력에 넘기고 결과 JSON(final·snapshots·log)을 돌려준다.
# 하네스가 실패하면(페이지 구조 변경, 알 수 없는 단계, 끝나지 않는 Promise) 표준 오류를 메시지로 보인다.
# 입력·출력은 UTF-8(한글 상태 줄).
def run_page(steps: list[dict], **options) -> dict:
    request = json.dumps({"options": options, "steps": steps}, ensure_ascii=False)
    proc = subprocess.run(
        [NODE, str(HARNESS)],
        input=request,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


# 단계 도우미(하네스의 op 이름 그대로). socket 인덱스는 만들어진 순서이고 -1이 마지막이다.
def start() -> dict:
    return {"op": "start"}


def stop() -> dict:
    return {"op": "stop"}


def open_socket(index: int = -1) -> dict:
    return {"op": "open", "socket": index}


def fail_socket(index: int = -1, code: int = 1006) -> dict:
    return {"op": "fail", "socket": index, "code": code}


def server_close(index: int = -1, code: int = 1000, reason: str = "") -> dict:
    return {"op": "server_close", "socket": index, "code": code, "reason": reason}


def message(data: dict, index: int = -1) -> dict:
    return {"op": "message", "socket": index, "data": data}


def packet(samples: int = 5360) -> dict:
    return {"op": "packet", "samples": samples}


def advance(ms: int) -> dict:
    return {"op": "advance", "ms": ms}


def snap(label: str) -> dict:
    return {"op": "snapshot", "label": label}


# 서버 메시지 모양(service/server.py 머리말의 프로토콜).
def partial(text: str, segment: int = 0) -> dict:
    return {"type": "partial", "segment": segment, "text": text, "rule": None, "frames": 8, "compute_ms": 5.0}


def final(text: str, segment: int = 0, rule: str = "rule2") -> dict:
    return {"type": "final", "segment": segment, "text": text, "rule": rule, "frames": 40, "compute_ms": 5.0}


def endpoint(segment: int = 0, rule: str = "rule2") -> dict:
    return {"type": "endpoint", "segment": segment, "text": "", "rule": rule, "frames": 40, "compute_ms": 5.0}


def stats(chunks: int = 3) -> dict:
    return {
        "type": "stats",
        "chunks": chunks,
        "compute_ms_p50": 1.0,
        "compute_ms_p90": 2.0,
        "compute_ms_max": 2.5,
    }


def texts(sock: dict) -> list[str]:
    return [m for m in sock["sent"] if isinstance(m, str)]


def binaries(sock: dict) -> list[dict]:
    return [m for m in sock["sent"] if isinstance(m, dict)]


# 확인: 멈춤 0.5 s 뒤 다시 시작하면 2 s가 지나도 새 소켓은 열려 있고(클라이언트가 닫지 않음) 상태는 '듣는
# 중'이며, 이전 소켓은 320 ms 패킷 뒤 'flush'를 받고 그 응답(final·stats)이 오면 바로 닫힌다. 이전 연결의
# 확정 자막은 새 연결이 듣는 동안에도 자막 상자에 붙는다. 이전 AudioContext·마이크 트랙은 정리되고 새 것은
# 살아 있다. 왜: 연결 객체가 전역 하나씩이던 때는 멈춤 뒤 1.5 s 닫기 타이머가 '그 시점의 전역 ws'(새 연결)를
# 닫아 '연결 끊김'이 됐고, 이전 연결은 닫히지 않아 서버 세션으로 남았다.
@needs_node
def test_restart_half_a_second_after_stop_keeps_the_new_connection_alive() -> None:
    out = run_page(
        [
            start(),
            open_socket(0),
            packet(5360),
            stop(),
            advance(500),
            start(),
            open_socket(1),
            snap("restarted"),
            message(final("가나다"), 0),
            message(stats(), 0),
            snap("old_final"),
            advance(2000),
        ]
    )
    restarted, old_final, later = out["snapshots"]["restarted"], out["snapshots"]["old_final"], out["final"]
    assert restarted["status"] == "듣는 중" and restarted["sockets"][1]["readyState"] == 1
    assert restarted["sockets"][0]["sent"] == [{"bytes": 10720}, "flush"], out["log"]
    assert "가나다" in old_final["committed"] and old_final["seg"] == "1"
    assert old_final["sockets"][0]["closedByClient"], out["log"]  # stats가 왔으니 1.5 s를 기다리지 않는다
    assert later["status"] == "듣는 중", out["log"]
    assert later["sockets"][1]["readyState"] == 1 and not later["sockets"][1]["closedByClient"]
    assert later["start_disabled"] and not later["stop_disabled"]
    assert [c["state"] for c in later["contexts"]] == ["closed", "running"]
    assert later["tracks"] == ["ended", "live"]
    assert later["unhandled"] == []


# 확인: 멈춤 0.1 s 뒤(150 ms 정리 타이머 전) 다시 시작하면, 새 연결이 그 타이머 전에 열리든 뒤에 열리든
# flush는 이전 소켓에만 가고 이전 소켓만 1.5 s 뒤 닫히며, 새 AudioContext는 닫히지 않고 새 소켓에는 flush가
# 가지 않는다. 왜: 전에는 타이머가 실행 시점의 전역 node·ctx·ws를 봐서 새 연결이 먼저 열리면 새 AudioContext를
# 닫고 새 소켓에 flush를 보냈고, 새 연결이 아직 연결 중이면 flush가 어디에도 가지 않아 마지막 확정 자막을
# 잃었다.
@needs_node
@pytest.mark.parametrize("open_before_timer", [True, False])
def test_restart_right_after_stop_flushes_only_the_old_connection(open_before_timer: bool) -> None:
    steps = [start(), open_socket(0), stop(), advance(100), start()]
    if open_before_timer:
        steps += [open_socket(1), advance(100)]  # t=200: 150 ms 타이머가 열린 새 연결을 본다
    else:
        steps += [advance(100), open_socket(1)]  # t=200: 타이머가 연결 중인 새 소켓을 본다
    steps += [snap("t200"), advance(1800)]
    out = run_page(steps)
    t200, end = out["snapshots"]["t200"], out["final"]
    assert texts(t200["sockets"][0]) == ["flush"], out["log"]
    assert t200["sockets"][1]["sent"] == [] and t200["sockets"][1]["readyState"] == 1
    assert [c["state"] for c in t200["contexts"]] == ["closed", "running"]
    assert t200["status"] == "듣는 중"
    assert end["sockets"][0]["closedByClient"] and end["sockets"][0]["readyState"] == 3, out["log"]
    assert end["sockets"][1]["readyState"] == 1 and not end["sockets"][1]["closedByClient"]
    assert texts(end["sockets"][1]) == []
    assert end["contexts"][1]["state"] == "running" and end["status"] == "듣는 중"
    assert end["unhandled"] == []


# 확인: 시작·멈춤을 두 번 하면 마이크 트랙 두 개가 모두 멈추고(브라우저의 마이크 사용 표시가 꺼진다), 멈출
# 때마다 worklet에 drain을 보내 320 ms 미만으로 남은 음성을 먼저 내보내며, 노드는 끊기고 AudioContext는
# 닫히고 소켓은 flush 뒤 닫힌다. 왜: 전에는 getUserMedia의 stream이 지역 변수라 track.stop()을 부를 수 없어
# 멈춘 뒤에도 트랙이 live로 남았다(소스 노드가 다음 시작까지 그 스트림을 붙들었다).
@needs_node
def test_stop_ends_every_microphone_track_and_drains_the_worklet() -> None:
    steps: list[dict] = []
    for _ in range(2):
        steps += [start(), open_socket(), packet(5360), stop(), advance(2000)]
    end = run_page(steps)["final"]
    assert end["tracks"] == ["ended", "ended"]
    worklets = [n for n in end["nodes"] if n["kind"].startswith("worklet:")]
    assert [n["drains"] for n in worklets] == [1, 1] and not any(n["connected"] for n in end["nodes"])
    assert [c["state"] for c in end["contexts"]] == ["closed", "closed"]
    for sock in end["sockets"]:
        assert texts(sock) == ["flush"] and sock["closedByClient"] and sock["readyState"] == 3
    assert end["status"] == "멈춤" and not end["start_disabled"] and end["stop_disabled"]
    assert end["unhandled"] == []


# 확인: 연결은 열렸지만 마이크를 못 켜는 세 경우(권한 거부, navigator.mediaDevices 없음 = 비보안 출처,
# createMediaStreamSource의 NotSupportedError)에 상태 줄이 '시작 실패: <사유>'가 되고, 시작 버튼이 돌아오며
# 멈춤 버튼은 꺼지고, 소켓은 클라이언트가 닫고, 얻은 트랙·AudioContext는 정리되며, 처리되지 않은 거부가
# 없다. 왜: 전에는 onopen의 async 본문에 try/catch가 없어 거부가 처리되지 않은 채 두 버튼이 모두 꺼진 상태로
# 멈췄다.
@needs_node
@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ({"denyMic": True}, "권한"),
        ({"noMediaDevices": True}, "출처"),
        ({"mssThrows": True}, "NotSupportedError"),
    ],
)
def test_start_failures_restore_the_start_button_with_a_reason(options: dict, fragment: str) -> None:
    end = run_page([start(), open_socket(0), advance(100)], **options)["final"]
    assert end["status"].startswith("시작 실패") and fragment in end["status"], end["status"]
    assert not end["start_disabled"] and end["stop_disabled"]
    assert end["sockets"][0]["closedByClient"] and end["sockets"][0]["readyState"] == 3
    assert all(t == "ended" for t in end["tracks"])
    assert all(c["state"] == "closed" for c in end["contexts"])
    assert end["unhandled"] == []


# 확인: 소켓이 열리기 전에 실패하면(서버가 꺼졌거나 Host·Origin 거부로 핸드셰이크가 실패해 코드 1006) 상태
# 줄이 '연결 실패'와 코드를 보이고 시작 버튼이 돌아온다. 왜: 전에는 onclose가 상태 글자만 바꾸고 버튼을 켜지
# 않아 새로 고치기 전에는 다시 시작할 수 없었다.
@needs_node
def test_connection_failure_before_open_restores_the_start_button() -> None:
    out = run_page([start(), snap("connecting"), fail_socket(0, 1006)])
    assert out["snapshots"]["connecting"]["status"] == "연결 중"
    end = out["final"]
    assert "연결 실패" in end["status"] and "1006" in end["status"], end["status"]
    assert not end["start_disabled"] and end["stop_disabled"]
    assert end["unhandled"] == []


# 확인: AudioContext가 16 kHz가 아닌 표본화율로 열리면 패킷을 하나도 보내지 않고(서버는 표본화율 정보 없이
# int16을 16 kHz로 해석한다) 마이크·AudioContext·소켓을 정리하며, 상태 줄에 실제 표본화율과 16 kHz를 적는다.
# 왜: 전에는 경고를 쓰고도 그대로 보냈고, 곧이어 '듣는 중'이 경고를 덮어 경고조차 보이지 않았다.
@needs_node
def test_unsupported_sample_rate_sends_nothing_and_releases_the_microphone() -> None:
    end = run_page([start(), open_socket(0), packet(5360), advance(500)], sampleRate=48000)["final"]
    assert binaries(end["sockets"][0]) == [] and end["sockets"][0]["closedByClient"]
    assert end["tracks"] == ["ended"]
    assert end["contexts"] == [{"sampleRate": 48000, "state": "closed"}]
    assert "48000" in end["status"] and "16 kHz" in end["status"], end["status"]
    assert not end["start_disabled"] and end["stop_disabled"]
    assert end["unhandled"] == []


# 확인: endpoint 이벤트(2차 패스 결과가 빈 세그먼트, 또는 토큰 없이 닫힌 세그먼트)가 잠정 자막을 비우고 확정
# 텍스트는 건드리지 않는다. final은 확정 텍스트에 줄을 더하고 세그먼트 수를 올리며 잠정 자막을 비운다. 멈춘
# 뒤 다시 시작했을 때 이전 연결의 늦은 partial(멈춤 직전 drain한 음성의 결과)은 새 연결의 잠정 자막을 덮지
# 않지만, 이전 연결의 final은 확정 텍스트에 붙고 stats가 오면 이전 소켓이 닫힌다. 왜: 전에는 endpoint를 무시해
# 확정되지 않은 부분 자막이 다음 partial이 올 때까지(멈춘 뒤에는 계속) 남았고, 부분 자막 처리가 연결을 가리지
# 않았다.
@needs_node
def test_endpoint_clears_the_provisional_caption_and_late_partials_do_not_override() -> None:
    out = run_page(
        [
            start(),
            open_socket(0),
            message(partial("가나"), 0),
            snap("partial"),
            message(endpoint(0), 0),
            snap("endpoint"),
            message(partial("다", 1), 0),
            message(final("다라", 1), 0),
            snap("final"),
            stop(),
            advance(100),
            start(),
            open_socket(1),
            message(partial("마", 0), 1),
            advance(100),  # t=200: 이전 연결의 150 ms 타이머가 flush를 보냈다
            message(partial("가가", 2), 0),
            message(final("가가가", 2, "eos"), 0),
            message(stats(), 0),
        ]
    )
    s = out["snapshots"]
    assert s["partial"]["partial"] == "가나" and s["partial"]["committed"] == ""
    assert s["endpoint"]["partial"] == "" and s["endpoint"]["committed"] == ""
    assert s["final"]["partial"] == "" and s["final"]["committed"] == "다라\n" and s["final"]["seg"] == "1"
    end = out["final"]
    assert end["partial"] == "마", out["log"]
    assert end["committed"] == "다라\n가가가\n" and end["seg"] == "2"
    assert texts(end["sockets"][0]) == ["flush"] and end["sockets"][0]["closedByClient"], out["log"]
    assert end["sockets"][1]["readyState"] == 1 and end["status"] == "듣는 중"
    assert end["unhandled"] == []


# 확인: 서버가 보낸 오류 줄({"type": "error", "code": 1013, ...})의 코드·사유가 그 뒤의 종료 때 상태 줄에
# 보이고, 시작 버튼이 돌아오며 마이크·AudioContext가 정리된다. 권한 창이 떠 있는 동안(getUserMedia가 끝나기
# 전) 연결이 끊겨도 권한 창이 끝난 뒤 얻은 트랙을 바로 멈추고 AudioContext를 만들지 않는다. 왜: 전에는 오류
# 줄을 무시해 '연결 끊김'만 보였고, 끊긴 뒤에도 onopen 본문이 이어져 '듣는 중'이 됐다.
@needs_node
@pytest.mark.parametrize("mic_delay_ms", [0, 500])
def test_server_error_line_is_shown_when_the_connection_closes(mic_delay_ms: int) -> None:
    error = {"type": "error", "code": 1013, "reason": "8 sessions are already open; try again later"}
    steps = [start(), open_socket(0), message(error, 0), server_close(0, 1013, "sessions"), advance(600)]
    end = run_page(steps, micDelayMs=mic_delay_ms)["final"]
    status = end["status"]
    assert "연결 끊김" in status and "1013" in status and "sessions" in status, status
    assert not end["start_disabled"] and end["stop_disabled"]
    assert end["tracks"] == ["ended"]
    assert all(c["state"] == "closed" for c in end["contexts"])
    if mic_delay_ms:
        assert end["contexts"] == []  # 끊긴 뒤에 얻은 마이크는 바로 놓고 오디오 그래프를 만들지 않는다
    assert end["unhandled"] == []


# 확인: 멈춤 → 150 ms 뒤 flush → stats가 오면 즉시 소켓을 닫고(1.5 s 타이머를 기다리지 않음) 표에 청크당
# 계산 시간을 쓴다. 그 뒤 시간이 더 지나도 예외가 없다(닫기 타이머가 지워졌다).
@needs_node
def test_stats_closes_the_flushed_socket_without_waiting() -> None:
    out = run_page(
        [
            start(),
            open_socket(0),
            stop(),
            advance(150),
            snap("flushed"),
            message(stats(3), 0),
            snap("stats"),
            advance(3000),
        ]
    )
    flushed, after = out["snapshots"]["flushed"], out["snapshots"]["stats"]
    assert texts(flushed["sockets"][0]) == ["flush"] and flushed["sockets"][0]["readyState"] == 1
    assert after["sockets"][0]["closedByClient"] and after["sockets"][0]["readyState"] == 3, out["log"]
    assert after["stats"] == "3 chunks, p50 1.0 ms, p90 2.0 ms"
    assert out["final"]["unhandled"] == []


# 확인: /info 줄이 2차 패스(final_pass)와 step 경로(step_path)를 보이고, /info를 못 받아도 페이지는 그대로
# 쓸 수 있다(처리되지 않은 거부 없음). 왜: 서버 기본값이 끝점 뒤 2차 패스·GPU graph 경로로 바뀌었는데 표시가
# 6단계 설정(청크·좌측·장치·rule2)만 보여 줬다.
@needs_node
def test_info_line_shows_the_second_pass_and_step_path() -> None:
    info = {
        "chunk_ms": 320,
        "left_frames": 128,
        "device": "cuda",
        "endpoint": {"rule2_s": 0.8},
        "final_pass": "full",
        "step_path": "graph",
    }
    end = run_page([], info=info)["final"]
    assert "2차 패스 full" in end["info"] and "graph" in end["info"] and "320 ms" in end["info"], end["info"]
    failed = run_page([], infoFails=True)["final"]
    assert failed["info"] == "-" and failed["unhandled"] == []


# 확인(Node 없이): 잠정 자막 span에 class가 있고 그 클래스의 CSS 규칙이 있으며(인라인 style 속성 금지), 안내
# 문구가 2차 패스(끝점 뒤 전체 문맥으로 다시 디코딩한 확정 자막이 부분 자막을 대체)를 설명한다.
def test_page_marks_provisional_captions_and_explains_the_second_pass() -> None:
    html = WEB.read_text(encoding="utf-8")
    assert re.search(r'<span id="partial" class="provisional"></span>', html)
    assert re.search(r"\.provisional\s*\{[^}]*font-style:\s*italic", html)
    guide = re.search(r'<p class="meta">(.*?)</p>', html, re.S)
    assert guide and "2차 패스" in guide.group(1) and "덧붙" in guide.group(1) and "확정" in guide.group(1)


# 확인(Node 없이): 서버의 CSP 해시 계산(server.page_csp)이 찾는 속성 없는 <script>·<style> 블록이 하나씩만
# 있고, CSP가 막는 인라인 style 속성이 없으며, 하네스가 돌리는 블록이 바로 그 스크립트다.
def test_page_inline_blocks_stay_hashable_for_the_csp() -> None:
    from stt_scratch.service import server

    html = WEB.read_text(encoding="utf-8")
    assert len(re.findall(r"<script\b", html)) == 1 and len(re.findall(r"<script>", html)) == 1
    assert len(re.findall(r"<style\b", html)) == 1 and len(re.findall(r"<style>", html)) == 1
    assert not re.search(r"\sstyle=", html)
    assert server.page_csp(html).count("'sha256-") == 2
