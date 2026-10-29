# ZAP Scanning Report

ZAP by [Checkmarx](https://checkmarx.com/).


## Summary of Alerts

| Risk Level | Number of Alerts |
| --- | --- |
| High | 0 |
| Medium | 0 |
| Low | 0 |
| Informational | 2 |




## Insights

| Level | Reason | Site | Description | Statistic |
| --- | --- | --- | --- | --- |
| Info | Informational | http://host.docker.internal:8098 | Percentage of responses with status code 2xx | 50 % |
| Info | Informational | http://host.docker.internal:8098 | Percentage of responses with status code 4xx | 50 % |
| Info | Informational | http://host.docker.internal:8098 | Percentage of endpoints with content type application/json | 66 % |
| Info | Informational | http://host.docker.internal:8098 | Percentage of endpoints with content type text/html | 33 % |
| Info | Informational | http://host.docker.internal:8098 | Percentage of endpoints with method GET | 100 % |
| Info | Informational | http://host.docker.internal:8098 | Count of total endpoints | 3    |
| Info | Informational | http://host.docker.internal:8098 | Percentage of slow responses | 50 % |







## Alerts

| Name | Risk Level | Number of Instances |
| --- | --- | --- |
| Modern Web Application | Informational | 1 |
| Non-Storable Content | Informational | 3 |




## Alert Detail



### [ Modern Web Application ](https://www.zaproxy.org/docs/alerts/10109/)



##### Informational (Medium)

### Description

The application appears to be a modern web application. If you need to explore it automatically then the Client Spider may well be more effective than the standard one.

* URL: http://host.docker.internal:8098/
  * Node Name: `http://host.docker.internal:8098/`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `<script>
// 패킷 규격: 320 ms = 16 kHz × 0.32 = 5,120표본(서버의 인코더 청크 8프레임과 같은 길이). 첫 패킷만 240표본을
// 더 보내야 서버 log-mel이 패킷마다 정확히 32프레임(청크 하나)을 만든다(docs/experiments.md 규칙 변경 9).
const CHUNK_MS = 320, RATE = 16000, CHUNK_SAMPLES = RATE * CHUNK_MS / 1000;
const FIRST_SAMPLES = CHUNK_SAMPLES + 240; // 첫 패킷만 프레임 창 하나만큼(240표본)을 더 싣는다
const $ = id => document.getElementById(id);
// 화면 상태(페이지 수명 동안 누적): 보낸 패킷 수, 확정 세그먼트 수, 확정 텍스트, 잠정 자막을 마지막으로 쓴 연결.
let sentChunks = 0, segments = 0, committed = "", partialOwner = null;
// 지금 연결(시작 ~ 멈춤·실패·끊김 사이) 하나. 연결마다의 자원(WebSocket, AudioContext, 소스·worklet 노드, 마이크
// 스트림, 송신 시각, 닫기 타이머)은 그 연결 객체 s에 묶어 두고, 모든 콜백·타이머가 전역이 아니라 자기 s를 쓴다.
// 왜: 전역 하나씩이던 때는 멈춤 뒤 1.5 s 닫기 타이머가 '그 시점의 전역 ws'(다시 시작한 새 연결)를 닫았고, 이전
// 연결은 flush·close되지 않아 서버 세션으로 남았으며, 멈춰도 마이크 트랙을 끄지 못했다(stream이 지역 변수).
let cur = null;
// 자막 상자 맨 앞의 텍스트 노드가 확정 텍스트 자리다(final 처리에서 이 노드의 값만 바꾼다).
const committedNode = document.createTextNode("");
$("final").insertBefore(committedNode, $("partial"));

function setStatus(text) { $("status").textContent = text; }

// 페이지를 열 때 GET /info로 서버 설정(청크 길이, 좌측 문맥, 장치, rule2 묵음 시간, 2차 패스, step 경로)을 받아
// 표에 보여 준다. 실패해도 페이지는 그대로 쓸 수 있다.
async function loadInfo() {
  try {
    const r = await fetch("/info"); const j = await r.json();
    $("info").textContent = `청크 ${j.chunk_ms} ms, 좌측 ${j.left_frames}프레임, ${j.device}, 끝점 rule2 ${j.endpoint.rule2_s}s, `
      + `2차 패스 ${j.final_pass}, step ${j.step_path}`;
  } catch (e) {}
}
loadInfo();

// AudioWorklet 프로세서 코드(문자열). 오디오 스레드에서 렌더 단위(보통 128표본)로 들어오는 마이크 표본을 모아
// 첫 패킷은 5,360표본, 이후 5,120표본이 차면 메인 스레드로 넘긴다. 시간이 아니라 표본 수로 세므로 패킷 길이가
// 서버의 규약과 정확히 맞는다. "drain" 메시지를 받으면 남은 표본(320 ms 미만)도 내보낸다. 별도 파일 없이 Blob
// URL로 불러오므로 CSP의 script-src에 blob:이 열려 있다. 연결마다 새 노드를 만들므로 첫 패킷 규칙도 연결마다
// 다시 적용된다.
const workletCode = `
class PcmSender extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; this.limit = ${FIRST_SAMPLES}; this.port.onmessage = () => this.emit(); }
  emit() {
    if (!this.n) return;
    const out = new Float32Array(this.n); let o = 0;
    for (const b of this.buf) { out.set(b, o); o += b.length; }
    this.buf = []; this.n = 0; this.limit = ${CHUNK_SAMPLES}; this.port.postMessage(out, [out.buffer]);
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    this.buf.push(Float32Array.from(ch)); this.n += ch.length;
    if (this.n >= this.limit) this.emit();  // "drain" messages call emit() for the remainder
    return true;
  }
}
registerProcessor("pcm-sender", PcmSender);`;

// Web Audio의 float(−1~1)을 서버가 기대하는 int16 PCM으로 바꾼다. 범위를 넘는 값은 잘라 내고, 음수는 32768,
// 양수는 32767을 곱해 int16 범위를 꽉 채운다.
function floatToInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); out[i] = s < 0 ? s * 32768 : s * 32767; }
  return out;
}

// 연결 하나의 자원 묶음. opened: 서버가 연결을 받았는지(onopen이 왔는지), flushed: 멈춤 뒤 "flush"를 보냈는지(그
// 뒤의 패킷은 버린다), failed: 소켓 error 이벤트가 있었는지, error: 서버가 닫기 전에 보낸 오류 줄({"type":
// "error"})의 코드·사유, lastSend: 마지막 패킷 송신 시각(부분 자막 지연 계산, 아직 보낸 것이 없으면 null),
// closeTimer: flush 뒤 stats가 오지 않을 때를 위한 1.5 s 닫기 타이머.
function newConnection() {
  return { ws: null, ctx: null, source: null, node: null, stream: null, opened: false, flushed: false, failed: false, error: "", lastSend: null, closeTimer: 0 };
}

// 오디오 쪽 자원 해제(멱등): worklet·소스 노드를 끊고, 마이크 트랙을 모두 멈추고(브라우저의 마이크 사용 표시가
// 꺼진다), AudioContext를 닫는다. 멈춤·시작 실패·연결 끊김이 모두 이 함수를 거친다. 이미 닫힌 AudioContext를
// 다시 닫으면 거부(InvalidStateError)되므로 상태를 보고 닫고, 거부는 무시한다.
function releaseAudio(s) {
  if (s.node) { s.node.port.onmessage = null; s.node.disconnect(); s.node = null; }
  if (s.source) { s.source.disconnect(); s.source = null; }
  if (s.stream) { for (const t of s.stream.getTracks()) t.stop(); s.stream = null; }
  if (s.ctx) { const ctx = s.ctx; s.ctx = null; if (ctx.state !== "closed") ctx.close().catch(() => {}); }
}

// 소켓 닫기(멱등): 대기 중인 닫기 타이머를 지우고, 아직 연결 중이거나 열려 있으면(readyState 0·1) 닫는다.
function closeSocket(s) {
  if (s.closeTimer) { clearTimeout(s.closeTimer); s.closeTimer = 0; }
  if (s.ws && s.ws.readyState < 2) s.ws.close();
}

// 실패 원인을 상태 줄에 쓸 문구로. getUserMedia의 DOMException 이름을 먼저 보고, 그 밖에는 이름과 메시지를 그대로
// 둔다(예: 표본화율이 다른 입력을 거부하는 브라우저의 NotSupportedError).
function reasonOf(e) {
  const name = e && e.name ? String(e.name) : "", msg = e && e.message ? String(e.message) : String(e);
  if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") return "마이크 권한이 거부되었습니다";
  if (name === "NotFoundError" || name === "DevicesNotFoundError") return "마이크 장치를 찾을 수 없습니다";
  if (name === "NotReadableError") return "마이크를 열 수 없습니다(다른 프로그램이 쓰고 있을 수 있음)";
  return name && name !== "Error" ? `${name}: ${msg}` : msg;
}

// 시작 실패: 이 연결이 지금 연결이면 사유를 상태 줄에 쓰고 시작 버튼을 되살린 뒤, 연결의 자원을 모두 정리한다.
// 지금 연결이 아니면(그 사이 끊겨 handleClose가 먼저 처리함) 화면은 두고 정리만 한다.
function fail(s, reason) {
  if (cur === s) { cur = null; setStatus(`시작 실패: ${reason}`); $("start").disabled = false; $("stop").disabled = true; }
  releaseAudio(s); closeSocket(s);
}

// 연결이 열린 뒤: 마이크 → AudioWorklet → 패킷 송신 그래프를 만든다. 실패하면(권한 거부, 비보안 출처라 마이크
// API가 없음, AudioContext가 16 kHz로 열리지 않음, createMediaStreamSource 예외) fail()로 간다. await 뒤마다
// 이 연결이 아직 지금 연결인지 보는 이유: 권한 창이 떠 있는 동안 서버가 연결을 닫을 수 있고(세션 상한 1013 등),
// 그러면 얻은 마이크를 바로 놓고 그래프를 만들지 않아야 한다.
async function startAudio(s) {
  s.opened = true;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("이 출처에서는 마이크를 쓸 수 없습니다(브라우저는 https 또는 localhost에서만 마이크를 엽니다)");
    }
    // 모노 마이크를 연다(에코 제거 켬, 잡음 억제 끔).
    s.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false } });
    if (cur !== s) { releaseAudio(s); return; }
    // AudioContext를 16 kHz로 열어 브라우저가 마이크 음성을 모델 입력 표본화율로 다시 샘플링하게 한다. 다른 값으로
    // 열렸으면 보내지 않는다: 서버는 표본화율 정보 없이 int16을 16 kHz로 해석하므로 그대로 보내면 인식이 틀어진다.
    s.ctx = new AudioContext({ sampleRate: RATE });
    if (s.ctx.sampleRate !== RATE) {
      throw new Error(`브라우저가 AudioContext를 ${s.ctx.sampleRate} Hz로 열어 16 kHz PCM을 보낼 수 없습니다`);
    }
    const url = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
    try { await s.ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    if (cur !== s) { releaseAudio(s); return; }
    s.source = s.ctx.createMediaStreamSource(s.stream);
    s.node = new AudioWorkletNode(s.ctx, "pcm-sender");
    // worklet이 넘긴 패킷을 int16으로 바꿔 바이너리 프레임으로 보내고, 송신 시각을 남겨 부분 자막 지연 계산에
    // 쓴다. 연결이 열려 있지 않거나 flush를 이미 보냈으면 버린다(flush 뒤의 패킷은 서버에서 새 스트림이 된다).
    s.node.port.onmessage = ev => {
      if (s.flushed || s.ws.readyState !== 1) return;
      s.ws.send(floatToInt16(ev.data).buffer);
      sentChunks += 1; s.lastSend = performance.now(); $("sent").textContent = sentChunks;
    };
    // 마이크 → worklet → 출력 장치로 그래프를 잇는다. worklet은 출력에 아무것도 쓰지 않으므로 소리는 나지 않는다.
    s.source.connect(s.node); s.node.connect(s.ctx.destination);
    setStatus("듣는 중"); $("stop").disabled = false;
  } catch (e) {
    fail(s, reasonOf(e));
  }
}

// 서버 이벤트 처리. partial: 지금 연결의 것만 잠정 자막으로 보이고 지연을 표시한다(멈춘 연결의 늦은 부분 자막이
// 새 연결의 자막을 덮지 않게). final: 확정 텍스트에 줄을 바꿔 덧붙이고 세그먼트 수를 올린다. endpoint(빈
// 세그먼트를 닫은 끝점. 2차 패스를 켰으면 2차 패스 결과가 비었다는 뜻): 보여 줄 텍스트는 없지만, 그 연결이 쓴
// 잠정 자막은 확정되지 않은 것이므로 지운다. final도 같은 연결의 잠정 자막을 지운다. stats: flush 뒤 세션 전체의
// 청크당 계산 시간. 멈춘 연결(flushed)이면 멈춤의 마지막 응답이므로 1.5 s를 기다리지 않고 바로 닫는다. error:
// 서버가 이 줄 뒤 같은 코드로 닫으므로 사유를 기억해 두고 onclose에서 보인다.
function handleMessage(s, ev) {
  let m;
  try { m = JSON.parse(ev.data); } catch (e) { return; }
  if (m.type === "partial") {
    if (cur !== s) return;
    $("partial").textContent = m.text; partialOwner = s;
    if (s.lastSend !== null) $("lat").textContent = `${Math.round(performance.now() - s.lastSend)} ms (계산 ${m.compute_ms} ms)`;
  } else if (m.type === "final" || m.type === "endpoint") {
    if (partialOwner === s) { $("partial").textContent = ""; partialOwner = null; }
    if (m.type === "final") {
      committed += (committed ? "\n" : "") + m.text; committedNode.nodeValue = committed + "\n";
      segments += 1; $("seg").textContent = segments;
    }
  } else if (m.type === "stats") {
    if (m.chunks) $("stats").textContent = `${m.chunks} chunks, p50 ${m.compute_ms_p50.toFixed(1)} ms, p90 ${m.compute_ms_p90.toFixed(1)} ms`;
    if (s.flushed) closeSocket(s);
  } else if (m.type === "error") {
    s.error = `${m.code} ${m.reason}`;
  }
}

// 연결 종료. 멈춘(또는 실패 처리된) 연결의 늦은 종료는 화면을 건드리지 않는다(지금 연결의 '듣는 중'을 '연결
// 끊김'으로 덮지 않게). 지금 연결이 끊긴 것이면 마이크·AudioContext를 놓고, 열린 적 없는 연결은 '연결 실패'(서버가
// 꺼졌거나 Host·Origin 검사로 거부, 코드 1006), 열렸던 연결은 '연결 끊김'과 사유(서버 오류 줄, 종료 코드, 소켓
// 오류)를 쓰고, 두 경우 모두 시작 버튼을 되살린다.
function handleClose(s, ev) {
  if (s.closeTimer) { clearTimeout(s.closeTimer); s.closeTimer = 0; }
  if (cur !== s) return;
  cur = null;
  releaseAudio(s);
  const code = ev && ev.code ? ev.code : 0;
  if (s.opened) {
    const detail = s.error || (code && code !== 1000 && code !== 1005 ? `코드 ${code}` : "") || (s.failed ? "소켓 오류" : "");
    setStatus(detail ? `연결 끊김: ${detail}` : "연결 끊김");
  } else {
    setStatus(`연결 실패: 서버가 꺼졌거나 연결을 거부했습니다${code ? ` (코드 ${code})` : ""}`);
  }
  $("start").disabled = false; $("stop").disabled = true;
}

// 시작: 새 연결 객체를 만들어 지금 연결로 두고, 같은 호스트의 /ws로 WebSocket을 연다(서버는 Host 허용 목록 밖이나
// Origin이 다른 사이트면 연결을 거부한다). 연결이 열리면 startAudio가 마이크를 켠다. 멈춘 이전 연결이 아직
// flush 응답을 기다리고 있어도 새 연결은 그것과 독립이다(이전 연결의 final·stats는 그대로 받아 자막에 붙인다).
$("start").onclick = () => {
  if (cur) return;
  $("start").disabled = true;
  const s = newConnection();
  cur = s;
  setStatus("연결 중");
  s.ws = new WebSocket(`ws://${location.host}/ws`);
  s.ws.binaryType = "arraybuffer";
  s.ws.onopen = () => startAudio(s);
  s.ws.onmessage = ev => handleMessage(s, ev);
  s.ws.onerror = () => { s.failed = true; };  // 세부 정보가 없는 이벤트. 뒤따르는 onclose가 상태 줄을 쓴다
  s.ws.onclose = ev => handleClose(s, ev);
};

// 멈춤: 지금 연결을 떼어 내고(이 시점부터 다시 시작할 수 있고, 아래 타이머·콜백은 떼어 낸 s만 본다) worklet에
// 남은 320 ms 미만 음성을 먼저 보내게 한 뒤(drain), 150 ms 뒤 오디오를 정리하고 마이크 트랙을 멈추고 "flush"를
// 보내 서버가 마지막 부분 청크를 처리해 final과 stats를 돌려주게 한다. stats가 오면 handleMessage가 바로 닫고,
// 오지 않으면 1.5 s 뒤 닫는다. 소켓이 이미 열려 있지 않으면 바로 닫는다.
$("stop").onclick = () => {
  const s = cur;
  if (!s) return;
  cur = null;
  $("stop").disabled = true; $("start").disabled = false;
  setStatus("멈춤");
  if (s.node) s.node.port.postMessage("drain");
  setTimeout(() => {
    releaseAudio(s);
    if (s.ws && s.ws.readyState === 1) {
      s.ws.send("flush"); s.flushed = true;
      s.closeTimer = setTimeout(() => closeSocket(s), 1500);
    } else {
      closeSocket(s);
    }
  }, 150);
};
</script>`
  * Other Info: `No links have been found while there are scripts, which is an indication that this is a modern web application.`


Instances: 1

### Solution

This is an informational alert and so no changes are required.

### Reference




#### Source ID: 3

### [ Non-Storable Content ](https://www.zaproxy.org/docs/alerts/10049/)



##### Informational (Medium)

### Description

The response contents are not storable by caching components such as proxy servers. If the response does not contain sensitive, personal or user-specific information, it may benefit from being stored and cached, to improve performance.

* URL: http://host.docker.internal:8098/
  * Node Name: `http://host.docker.internal:8098/`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `no-store`
  * Other Info: ``
* URL: http://host.docker.internal:8098/robots.txt
  * Node Name: `http://host.docker.internal:8098/robots.txt`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `no-store`
  * Other Info: ``
* URL: http://host.docker.internal:8098/sitemap.xml
  * Node Name: `http://host.docker.internal:8098/sitemap.xml`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `no-store`
  * Other Info: ``


Instances: 3

### Solution

The content may be marked as storable by ensuring that the following conditions are satisfied:
The request method must be understood by the cache and defined as being cacheable ("GET", "HEAD", and "POST" are currently defined as cacheable)
The response status code must be understood by the cache (one of the 1XX, 2XX, 3XX, 4XX, or 5XX response classes are generally understood)
The "no-store" cache directive must not appear in the request or response header fields
For caching by "shared" caches such as "proxy" caches, the "private" response directive must not appear in the response
For caching by "shared" caches such as "proxy" caches, the "Authorization" header field must not appear in the request, unless the response explicitly allows it (using one of the "must-revalidate", "public", or "s-maxage" Cache-Control response directives)
In addition to the conditions above, at least one of the following conditions must also be satisfied by the response:
It must contain an "Expires" header field
It must contain a "max-age" response directive
For "shared" caches such as "proxy" caches, it must contain a "s-maxage" response directive
It must contain a "Cache Control Extension" that allows it to be cached
It must have a status code that is defined as cacheable by default (200, 203, 204, 206, 300, 301, 404, 405, 410, 414, 501).

### Reference


* [ https://datatracker.ietf.org/doc/html/rfc7234 ](https://datatracker.ietf.org/doc/html/rfc7234)
* [ https://datatracker.ietf.org/doc/html/rfc7231 ](https://datatracker.ietf.org/doc/html/rfc7231)
* [ https://www.w3.org/Protocols/rfc2616/rfc2616-sec13.html ](https://www.w3.org/Protocols/rfc2616/rfc2616-sec13.html)


#### CWE Id: [ 524 ](https://cwe.mitre.org/data/definitions/524.html)


#### WASC Id: 13

#### Source ID: 3


