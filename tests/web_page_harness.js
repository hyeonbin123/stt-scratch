// [파일 역할] web/index.html의 인라인 <script>를 브라우저 없이 Node.js vm 안에서 실행해 보는 시험 하네스.
// tests/test_web_page.py가 시나리오(단계 목록)를 JSON으로 표준 입력에 넘기면, 가짜 DOM·WebSocket·AudioContext·
// AudioWorkletNode·getUserMedia·가상 타이머 위에서 페이지 스크립트를 돌리고 관찰값(버튼 상태, 상태 줄, 자막,
// 소켓마다 보낸 메시지와 상태, AudioContext 상태, 마이크 트랙 상태, 처리되지 않은 예외)을 JSON으로 표준 출력에
// 쓴다. 실행: node tests/web_page_harness.js < 요청.json (저장소 어디서 불러도 페이지 경로는 이 파일 기준이다).
// [왜] 페이지 스크립트의 결함(멈춤 뒤 닫기 타이머가 새 연결을 닫음, 시작 실패에 정리 경로가 없음, 멈춰도 마이크
// 트랙을 끄지 않음)은 서버 테스트(tests/test_service.py)로는 잡히지 않고, 실제 브라우저 자동화는 CI에 무겁다. 가짜
// 환경은 페이지가 쓰는 API 표면만 흉내 내므로 실제 브라우저(권한 창, 오디오 스레드, 닫기 핸드셰이크 시간)와 다를
// 수 있다. 그 틈은 README 서비스 절의 수동 점검 항목이 메운다.
// [시간] setTimeout·clearTimeout·performance.now()는 가상 시계(ms)다. "advance" 단계가 시계를 미는 동안 그 사이에
// 만기가 되는 타이머를 시각·등록 순서대로 실행한다. Promise(마이크 권한, addModule, 소켓 종료 통지 등)는 실제
// 마이크로태스크로 돌고, 단계마다 setImmediate 두 번으로 비운다. 그래서 단계 사이의 상태는 결정적이다.
// [입력] {"options": {...}, "steps": [{"op": "...", ...}, ...]}
//   options.sampleRate     AudioContext가 실제로 여는 표본화율(기본은 요청값 그대로, 16 kHz가 아닌 환경을 흉내 낼 때)
//   options.denyMic        getUserMedia가 NotAllowedError로 거부(권한 거부)
//   options.noMediaDevices navigator.mediaDevices가 없음(비보안 출처에서 브라우저가 마이크 API를 숨기는 경우)
//   options.mssThrows      createMediaStreamSource가 NotSupportedError를 던짐(표본화율이 다른 입력을 거부하는 브라우저)
//   options.micDelayMs     getUserMedia가 가상 시계로 이만큼 뒤에 끝남(권한 창이 떠 있는 시간)
//   options.info           GET /info 응답 객체(기본은 서버 기본값 모양), options.infoFails면 fetch가 실패
//   steps: {"op": "start"} / {"op": "stop"} 버튼 클릭(비활성 버튼은 브라우저처럼 무시하고 로그만 남김)
//          {"op": "open", "socket": i} 서버가 i번째 소켓을 받음(onopen), {"op": "fail", "socket": i, "code": c}
//          연결 실패(onerror 뒤 onclose), {"op": "server_close", "socket": i, "code": c, "reason": r} 서버가 닫음,
//          {"op": "message", "socket": i, "data": {...}} 서버 메시지(JSON), {"op": "packet", "samples": n} worklet이
//          n표본 패킷을 메인 스레드로 넘김, {"op": "advance", "ms": t} 가상 시계 전진, {"op": "snapshot",
//          "label": s} 지금 관찰값을 이름 붙여 저장. socket 생략·음수는 마지막 소켓(-1)부터 센다.
// [출력] {"final": <관찰값>, "snapshots": {label: <관찰값>}, "log": [...]}. 관찰값: now, start_disabled,
//   stop_disabled, status, info, sent, lat, seg, stats, partial, committed(확정 텍스트), sockets[{url, readyState,
//   sent(문자열 또는 {bytes}), closedByClient, closeCode}], contexts[{sampleRate, state}], tracks["live"|"ended"],
//   nodes[{kind, connected, drains}], unhandled[처리되지 않은 예외·거부 문자열].
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

// 페이지 경로는 이 파일 기준(tests/ 옆 web/)이라 작업 디렉터리와 무관하다.
const PAGE = path.join(__dirname, "..", "web", "index.html");

// GET /info 응답의 기본 모양(server.build_app의 info 사전과 같은 키). 테스트가 options.info로 바꿀 수 있다.
const DEFAULT_INFO = {
  checkpoint: "m.pt",
  chunk_frames: 8,
  chunk_ms: 320,
  left_frames: 128,
  device: "cpu",
  precision: "fp32",
  endpoint: { blank_threshold: 0.8, rule1_s: 2.4, rule2_s: 0.8, rule3_s: 20.0 },
  reset_cache_on_endpoint: false,
  step_path: "fast",
  final_pass: "full",
  vocab: 2000,
  warm_up_s: 1.0,
};

// 페이지의 인라인 <script> 블록을 꺼낸다. 서버의 CSP(server.page_csp)와 같은 규칙(속성 없는 <script> 태그)으로
// 찾고, 블록이 정확히 하나여야 한다(페이지 구조가 바뀌면 여기서 멈춰 알린다).
function pageScript() {
  const html = fs.readFileSync(PAGE, "utf8");
  const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (blocks.length !== 1) {
    throw new Error(`web/index.html: expected exactly one <script> block, found ${blocks.length}`);
  }
  return blocks[0];
}

// 브라우저 DOMException 모양(name 속성이 있는 Error)의 오류를 만든다. 페이지는 e.name으로 사유를 고른다.
function domError(name, message) {
  const e = new Error(message);
  e.name = name;
  return e;
}

function describeError(e) {
  if (e && e.stack) return String(e.stack).split("\n").slice(0, 2).join(" ");
  return String(e);
}

// 가짜 환경 하나를 만들고 페이지 스크립트를 그 안에서 실행한다. 돌려주는 객체의 run(steps)가 단계를 차례로
// 실행하고 결과를 돌려준다. 환경은 프로세스 하나에 하나만 만든다(unhandledRejection 수집이 프로세스 전역).
function makeEnvironment(options) {
  let now = 0;
  let seq = 0;
  const timers = new Map();
  const log = [];
  const sockets = [];
  const contexts = [];
  const tracks = [];
  const nodes = [];
  const unhandled = [];
  process.on("unhandledRejection", reason => unhandled.push(`unhandledRejection: ${describeError(reason)}`));

  // --- 가상 타이머 -------------------------------------------------------------------------------------------
  function schedule(fn, ms) {
    const id = ++seq;
    timers.set(id, { id, at: now + (Number(ms) || 0), fn });
    return id;
  }
  function cancel(id) {
    timers.delete(id);
  }
  // 마이크로태스크를 비운다(setImmediate 두 번: 첫 번째가 끝난 뒤 큐에 든 것까지).
  const settle = () => new Promise(resolve => setImmediate(() => setImmediate(resolve)));
  async function advance(ms) {
    const end = now + ms;
    for (;;) {
      const due = [...timers.values()].filter(t => t.at <= end).sort((a, b) => a.at - b.at || a.id - b.id);
      if (!due.length) break;
      const t = due[0];
      timers.delete(t.id);
      now = Math.max(now, t.at);
      try {
        t.fn();
      } catch (e) {
        unhandled.push(`timer: ${describeError(e)}`);
      }
      await settle();
    }
    now = end;
    await settle();
  }

  // --- 가짜 DOM: 페이지가 쓰는 요소만(id로 찾고 textContent·disabled·onclick·childNodes·insertBefore) ---------
  const el = {};
  for (const id of ["start", "stop", "status", "info", "sent", "lat", "seg", "stats", "partial"]) {
    // textContent는 실제 DOM처럼 대입값을 문자열로 바꾼다(숫자를 넣어도 "1"; null·undefined는 "").
    let text = "";
    el[id] = {
      id,
      disabled: false,
      onclick: null,
      nodeValue: null,
      get textContent() {
        return text;
      },
      set textContent(value) {
        text = value == null ? "" : String(value);
      },
    };
  }
  // 페이지 HTML의 초기 상태와 같게 둔다: 멈춤 버튼 비활성, 상태 줄 '대기', 표의 칸은 '-' 또는 0.
  el.stop.disabled = true;
  el.status.textContent = "대기";
  el.info.textContent = "-";
  el.sent.textContent = "0";
  el.lat.textContent = "-";
  el.seg.textContent = "0";
  el.stats.textContent = "-";
  el.final = {
    id: "final",
    childNodes: [el.partial],
    insertBefore(node, ref) {
      const i = this.childNodes.indexOf(ref);
      if (i < 0) this.childNodes.push(node);
      else this.childNodes.splice(i, 0, node);
      return node;
    },
  };
  const document = {
    getElementById: id => el[id] || null,
    createTextNode: value => ({ nodeType: 3, nodeValue: String(value) }),
  };

  // --- 가짜 WebSocket: readyState 0(연결 중)·1(열림)·2(닫는 중)·3(닫힘), 보낸 것 기록, 종료는 비동기 -----------
  class FakeWebSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.binaryType = "blob";
      this.sent = [];
      this.closedByClient = false;
      this.closeCode = null;
      this.onopen = null;
      this.onmessage = null;
      this.onerror = null;
      this.onclose = null;
      this.index = sockets.length;
      sockets.push(this);
      log.push(`t=${now} WebSocket #${this.index} created ${url}`);
    }
    send(data) {
      // 브라우저처럼 연결 중이면 예외, 닫는 중·닫힘이면 조용히 버린다.
      if (this.readyState === 0) throw domError("InvalidStateError", "Still in CONNECTING state.");
      if (this.readyState > 1) {
        log.push(`t=${now} WebSocket #${this.index} send ignored (readyState ${this.readyState})`);
        return;
      }
      const item = typeof data === "string" ? data : { bytes: data.byteLength };
      this.sent.push(item);
      log.push(`t=${now} WebSocket #${this.index} send ${typeof item === "string" ? item : item.bytes + " bytes"}`);
    }
    close(code, reason) {
      if (this.readyState >= 2) return;
      const wasConnecting = this.readyState === 0;
      this.readyState = 2;
      this.closedByClient = true;
      log.push(`t=${now} client closes WebSocket #${this.index}`);
      // 닫기 핸드셰이크가 끝난 뒤(비동기) close 이벤트가 온다. 코드 없이 닫으면 브라우저는 1005를 보인다.
      Promise.resolve().then(() => {
        this.readyState = 3;
        this.closeCode = code || 1005;
        if (this.onclose) this.onclose({ code: this.closeCode, reason: reason || "", wasClean: !wasConnecting });
      });
    }
    // 아래 _ 메서드는 서버·네트워크 쪽 사건을 시험이 일으키는 용도다.
    _open() {
      if (this.readyState !== 0) throw new Error(`WebSocket #${this.index} is not connecting (readyState ${this.readyState})`);
      this.readyState = 1;
      log.push(`t=${now} WebSocket #${this.index} open`);
      return this.onopen ? this.onopen({}) : undefined;
    }
    _fail(code) {
      if (this.readyState >= 2) return undefined;
      this.readyState = 3;
      this.closeCode = code;
      log.push(`t=${now} WebSocket #${this.index} failed (${code})`);
      if (this.onerror) this.onerror({});
      return this.onclose ? this.onclose({ code, reason: "", wasClean: false }) : undefined;
    }
    _serverClose(code, reason) {
      if (this.readyState >= 2) return undefined;
      this.readyState = 3;
      this.closeCode = code;
      log.push(`t=${now} server closes WebSocket #${this.index} (${code})`);
      return this.onclose ? this.onclose({ code, reason, wasClean: true }) : undefined;
    }
    _message(data) {
      if (this.readyState !== 1) {
        log.push(`t=${now} WebSocket #${this.index} message dropped (readyState ${this.readyState})`);
        return undefined;
      }
      const text = typeof data === "string" ? data : JSON.stringify(data);
      return this.onmessage ? this.onmessage({ data: text }) : undefined;
    }
  }

  // --- 가짜 Web Audio: 노드 연결 상태, worklet 포트, AudioContext 표본화율·상태 -----------------------------
  class FakeAudioNode {
    constructor(kind) {
      this.kind = kind;
      this.connected = false;
      this.index = nodes.length;
      nodes.push(this);
    }
    connect(target) {
      this.connected = true;
      return target;
    }
    disconnect() {
      this.connected = false;
    }
  }
  class FakeAudioWorkletNode extends FakeAudioNode {
    constructor(ctx, name) {
      super(`worklet:${name}`);
      this.ctx = ctx;
      this.drains = 0;
      const node = this;
      this.port = {
        onmessage: null,
        postMessage(message) {
          if (message === "drain") node.drains += 1;
          log.push(`t=${now} worklet #${node.index} <- ${String(message)}`);
        },
      };
    }
  }
  class FakeAudioContext {
    constructor(opts) {
      const requested = opts && opts.sampleRate;
      this.sampleRate = options.sampleRate != null ? options.sampleRate : requested || 48000;
      this.state = "running";
      this.destination = { kind: "destination" };
      this.index = contexts.length;
      contexts.push(this);
      log.push(`t=${now} AudioContext #${this.index} ${this.sampleRate} Hz`);
      const ctx = this;
      this.audioWorklet = {
        addModule() {
          if (ctx.state === "closed") return Promise.reject(domError("InvalidStateError", "AudioContext is closed."));
          return Promise.resolve();
        },
      };
    }
    createMediaStreamSource(stream) {
      if (options.mssThrows) {
        throw domError(
          "NotSupportedError",
          "Connecting AudioNodes from AudioContexts with different sample-rate is currently not supported."
        );
      }
      const node = new FakeAudioNode("source");
      node.stream = stream;
      return node;
    }
    close() {
      if (this.state === "closed") return Promise.reject(domError("InvalidStateError", "Cannot close a closed AudioContext."));
      this.state = "closed";
      log.push(`t=${now} AudioContext #${this.index} closed`);
      return Promise.resolve();
    }
    resume() {
      return Promise.resolve();
    }
  }

  // --- 가짜 마이크: 트랙 하나짜리 MediaStream, 거부·지연 옵션 ----------------------------------------------
  function openMicrophone() {
    if (options.denyMic) throw domError("NotAllowedError", "Permission denied");
    const track = {
      kind: "audio",
      readyState: "live",
      index: tracks.length,
      stop() {
        this.readyState = "ended";
        log.push(`t=${now} track #${this.index} stopped`);
      },
    };
    tracks.push(track);
    return { getTracks: () => [track], getAudioTracks: () => [track] };
  }
  const mediaDevices = {
    getUserMedia() {
      log.push(`t=${now} getUserMedia`);
      return new Promise((resolve, reject) => {
        const finish = () => {
          try {
            resolve(openMicrophone());
          } catch (e) {
            reject(e);
          }
        };
        if (options.micDelayMs) schedule(finish, options.micDelayMs);
        else finish();
      });
    },
  };
  const navigator = options.noMediaDevices ? { userAgent: "fake" } : { userAgent: "fake", mediaDevices };

  // --- 샌드박스: 페이지가 쓰는 전역만 넣는다. Promise·JSON·Math·형식화 배열은 vm 문맥 고유의 것을 쓴다. -------
  const sandbox = {
    document,
    navigator,
    location: { host: "127.0.0.1:8765", protocol: "http:" },
    performance: { now: () => now },
    fetch: () =>
      options.infoFails
        ? Promise.reject(new TypeError("Failed to fetch"))
        : Promise.resolve({ ok: true, json: () => Promise.resolve(options.info || DEFAULT_INFO) }),
    WebSocket: FakeWebSocket,
    AudioContext: FakeAudioContext,
    AudioWorkletNode: FakeAudioWorkletNode,
    Blob: class Blob {
      constructor(parts, opts) {
        this.parts = parts;
        this.type = opts && opts.type;
      }
    },
    URL: { createObjectURL: () => `blob:fake/${++seq}`, revokeObjectURL() {} },
    setTimeout: schedule,
    clearTimeout: cancel,
    // 페이지가 console을 쓰더라도 표준 출력(JSON 결과)을 더럽히지 않게 표준 오류로 보낸다.
    console: {
      log: (...a) => process.stderr.write(a.join(" ") + "\n"),
      warn: (...a) => process.stderr.write(a.join(" ") + "\n"),
      error: (...a) => process.stderr.write(a.join(" ") + "\n"),
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(pageScript(), sandbox, { filename: "web/index.html#script" });

  function pick(index) {
    if (!sockets.length) throw new Error("no WebSocket has been created yet");
    const i = index == null ? sockets.length - 1 : index < 0 ? sockets.length + index : index;
    if (!sockets[i]) throw new Error(`no WebSocket #${index} (have ${sockets.length})`);
    return sockets[i];
  }

  // worklet이 n표본 패킷을 메인 스레드로 넘긴 상황. 가장 최근 worklet 노드의 port.onmessage를 부른다. 노드가
  // 없거나(페이지가 그래프를 만들기 전에 실패) 핸들러가 지워졌으면(정리 뒤) 로그만 남긴다.
  function emitPacket(samples, nodeIndex) {
    const worklets = nodes.filter(n => n.kind.startsWith("worklet:"));
    const node = nodeIndex == null ? worklets[worklets.length - 1] : worklets[nodeIndex];
    if (!node) {
      log.push(`t=${now} packet of ${samples} samples dropped: no worklet node`);
      return undefined;
    }
    if (!node.port.onmessage) {
      log.push(`t=${now} packet of ${samples} samples dropped: worklet #${node.index} has no onmessage`);
      return undefined;
    }
    return node.port.onmessage({ data: new Float32Array(samples) });
  }

  // 페이지 핸들러를 브라우저처럼 부른다: 동기 예외와, async 핸들러가 돌려준 Promise의 거부를 '처리되지 않은 예외'로
  // 기록한다. 돌려준 Promise를 기다리지는 않는다(기다리면 가상 타이머로 끝나는 권한 창을 영원히 기다리게 된다).
  async function call(fn) {
    try {
      const result = fn();
      if (result && typeof result.then === "function") {
        result.then(undefined, e => unhandled.push(describeError(e)));
      }
    } catch (e) {
      unhandled.push(describeError(e));
    }
    await settle();
  }

  function clickable(button, name) {
    if (!button.onclick) throw new Error(`${name} button has no onclick handler`);
    if (button.disabled) {
      log.push(`t=${now} ${name} click ignored (disabled)`);
      return () => undefined;
    }
    return () => button.onclick({});
  }

  function snapshot() {
    return {
      now,
      start_disabled: el.start.disabled,
      stop_disabled: el.stop.disabled,
      status: el.status.textContent,
      info: el.info.textContent,
      sent: el.sent.textContent,
      lat: el.lat.textContent,
      seg: el.seg.textContent,
      stats: el.stats.textContent,
      partial: el.partial.textContent,
      committed: el.final.childNodes.filter(n => n !== el.partial).map(n => (n.nodeValue == null ? "" : n.nodeValue)).join(""),
      sockets: sockets.map(w => ({
        url: w.url,
        readyState: w.readyState,
        sent: w.sent,
        closedByClient: w.closedByClient,
        closeCode: w.closeCode,
      })),
      contexts: contexts.map(c => ({ sampleRate: c.sampleRate, state: c.state })),
      tracks: tracks.map(t => t.readyState),
      nodes: nodes.map(n => ({ kind: n.kind, connected: n.connected, drains: n.drains == null ? null : n.drains })),
      unhandled: unhandled.slice(),
    };
  }

  async function run(steps) {
    const snapshots = {};
    await settle(); // 페이지가 로드 때 부르는 loadInfo()의 fetch를 끝낸다
    for (const step of steps) {
      switch (step.op) {
        case "start":
          await call(clickable(el.start, "start"));
          break;
        case "stop":
          await call(clickable(el.stop, "stop"));
          break;
        case "open":
          await call(() => pick(step.socket)._open());
          break;
        case "fail":
          await call(() => pick(step.socket)._fail(step.code || 1006));
          break;
        case "server_close":
          await call(() => pick(step.socket)._serverClose(step.code || 1000, step.reason || ""));
          break;
        case "message":
          await call(() => pick(step.socket)._message(step.data));
          break;
        case "packet":
          await call(() => emitPacket(step.samples || 5360, step.node));
          break;
        case "advance":
          await advance(Number(step.ms) || 0);
          break;
        case "snapshot":
          snapshots[step.label || String(Object.keys(snapshots).length)] = snapshot();
          break;
        default:
          throw new Error(`unknown step op: ${JSON.stringify(step.op)}`);
      }
    }
    return { final: snapshot(), snapshots, log };
  }

  return { run };
}

let written = false;

async function main() {
  let input = "";
  try {
    input = fs.readFileSync(0, "utf8");
  } catch (e) {
    throw new Error(`usage: node ${path.basename(__filename)} < request.json (${e.message})`);
  }
  const request = input.trim() ? JSON.parse(input) : {};
  const env = makeEnvironment(request.options || {});
  const result = await env.run(request.steps || []);
  process.stdout.write(JSON.stringify(result));
  written = true;
}

// 단계가 끝나기 전에 이벤트 루프가 비면(기다리는 Promise가 영원히 끝나지 않음) Node는 조용히 0으로 끝난다.
// 그 경우 결과 없이 성공한 것처럼 보이지 않게 종료 코드를 1로 바꾸고 알린다.
process.on("exit", code => {
  if (!written && code === 0) {
    process.stderr.write("web_page_harness: the event loop drained before the steps finished (no result)\n");
    process.exitCode = 1;
  }
});

main().catch(e => {
  process.stderr.write(describeError(e) + "\n");
  process.exit(1);
});
