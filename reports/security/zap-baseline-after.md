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
| Low | Warning |  | ZAP errors logged - see the zap.log file for details | 1    |
| Info | Informational | http://host.docker.internal:8765 | Percentage of responses with status code 2xx | 50 % |
| Info | Informational | http://host.docker.internal:8765 | Percentage of responses with status code 4xx | 50 % |
| Info | Informational | http://host.docker.internal:8765 | Percentage of endpoints with content type application/json | 66 % |
| Info | Informational | http://host.docker.internal:8765 | Percentage of endpoints with content type text/html | 33 % |
| Info | Informational | http://host.docker.internal:8765 | Percentage of endpoints with method GET | 100 % |
| Info | Informational | http://host.docker.internal:8765 | Count of total endpoints | 3    |







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

* URL: http://host.docker.internal:8765/
  * Node Name: `http://host.docker.internal:8765/`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `<script>
const CHUNK_MS = 320, RATE = 16000, CHUNK_SAMPLES = RATE * CHUNK_MS / 1000;
const FIRST_SAMPLES = CHUNK_SAMPLES + 240; // the first packet carries the 240 extra samples of one frame window
let ctx, source, node, ws, sentChunks = 0, sendTimes = [], committed = "", segment = 0;
const $ = id => document.getElementById(id);

async function loadInfo() {
  try { const r = await fetch("/info"); const j = await r.json();
    $("info").textContent = `청크 ${j.chunk_ms} ms, 좌측 ${j.left_frames}프레임, ${j.device}, 끝점 rule2 ${j.endpoint.rule2_s}s`; } catch (e) {}
}
loadInfo();

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

function floatToInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); out[i] = s < 0 ? s * 32768 : s * 32767; }
  return out;
}

$("start").onclick = async () => {
  $("start").disabled = true;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false } });
    ctx = new AudioContext({ sampleRate: RATE });
    if (ctx.sampleRate !== RATE) { $("status").textContent = `브라우저가 ${ctx.sampleRate} Hz로 열었습니다(16 kHz 필요)`; }
    const blob = new Blob([workletCode], { type: "application/javascript" });
    await ctx.audioWorklet.addModule(URL.createObjectURL(blob));
    source = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pcm-sender");
    node.port.onmessage = ev => {
      if (ws.readyState !== 1) return;
      ws.send(floatToInt16(ev.data).buffer);
      sentChunks += 1; sendTimes.push(performance.now()); $("sent").textContent = sentChunks;
    };
    source.connect(node); node.connect(ctx.destination);
    $("status").textContent = "듣는 중"; $("stop").disabled = false;
  };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === "partial") {
      $("partial").textContent = m.text;
      if (sendTimes.length) $("lat").textContent = `${Math.round(performance.now() - sendTimes[sendTimes.length - 1])} ms (계산 ${m.compute_ms} ms)`;
    } else if (m.type === "final") {
      committed += (committed ? "\n" : "") + m.text; segment = m.segment + 1; $("seg").textContent = segment;
      $("final").childNodes[0].nodeValue = committed + "\n"; $("partial").textContent = "";
    } else if (m.type === "stats" && m.chunks) {
      $("stats").textContent = `${m.chunks} chunks, p50 ${m.compute_ms_p50.toFixed(1)} ms, p90 ${m.compute_ms_p90.toFixed(1)} ms`;
    }
  };
  ws.onclose = () => { $("status").textContent = "연결 끊김"; };
};

$("stop").onclick = () => {
  $("stop").disabled = true;
  if (node) node.port.postMessage("drain");  // send the buffered remainder (< 320 ms) before flushing
  setTimeout(() => {
    if (node) { node.disconnect(); source.disconnect(); }
    if (ctx) ctx.close();
    if (ws && ws.readyState === 1) { ws.send("flush"); setTimeout(() => ws.close(), 1500); }
  }, 150);
  $("status").textContent = "멈춤"; $("start").disabled = false;
};
// the final box keeps committed text as its first text node
$("final").insertBefore(document.createTextNode(""), $("partial"));
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

* URL: http://host.docker.internal:8765/
  * Node Name: `http://host.docker.internal:8765/`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `no-store`
  * Other Info: ``
* URL: http://host.docker.internal:8765/robots.txt
  * Node Name: `http://host.docker.internal:8765/robots.txt`
  * Method: `GET`
  * Parameter: ``
  * Attack: ``
  * Evidence: `no-store`
  * Other Info: ``
* URL: http://host.docker.internal:8765/sitemap.xml
  * Node Name: `http://host.docker.internal:8765/sitemap.xml`
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


