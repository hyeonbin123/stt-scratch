"""실행 중인 WebSocket 서버에 발화를 실시간 속도로 보내고 발화마다 자막·지연을 기록하는 측정용 클라이언트.

6단계에서 종단 지연(마지막 오디오 바이트 송신 → 최종 자막 수신)을 재고, 서비스 경로가 offline 스트리밍
평가와 같은 텍스트를 내는지 확인하는 데 썼다. 서버 기본값이 끝점 뒤 2차 패스(--final-pass full)가 된 뒤에는
확정 텍스트가 2차 패스 결과라 simulated 가설과 다르므로, 서버가 닫힘 이벤트에 함께 보내는 stream_text(2차 패스
전 320 ms 스트리밍 텍스트)로 스트리밍 경로를 확인하고, 확정 텍스트는 같은 설정의 오프라인 재생과 대조한다
(docs/experiments.md 규칙 변경 27).

사용법: uv run python scripts/stream_client.py --manifest dev_h --limit 50 --url ws://127.0.0.1:8765/ws
발화마다 받은 자막·닫힘 이벤트(final·endpoint, stream_text·final_pass_ms 포함)·이벤트 시각·서버 계산 통계를
reports/stage6/client-<manifest>.jsonl(--name으로 이름을 바꿈)에 쓴다. --replay-check를 주면 송신이 모두 끝난
뒤 같은 PC에서 서버와 같은 모델·설정의 세션에 같은 패킷을 재생해 기대 기록을 <이름>-replay.jsonl에 쓴다.
`--summarize`(서버 불필요)는 그 파일들을 읽어 <이름>-summary.json을 쓴다: 모든 발화를 모은 부분 자막 지연, 첫
부분 자막, 마지막 오디오 → 최종 자막, 그 가운데 이 클라이언트가 마지막 패킷 뒤 실시간으로 기다린 몫(실제
마이크 클라이언트는 "flush"를 바로 보낸다), 그 대기를 뺀 flush 왕복, 서버 계산 시간, 2차 패스 시간, 그리고
서버 설정(-meta.json의 server_info.final_pass)에 맞춘 텍스트 비교와 유효성 조건.
"""

# [왜 필요한가] 6단계 등록 측정 가운데 종단 지연(로컬)과 서비스 경로 검증을 맡는다(docs/experiments.md 규칙
# 변경 21): 부분 자막 도착 − 직전 패킷 송신(p50 62.4 ms), 첫 부분 자막, 마지막 오디오 → 최종 자막, flush 왕복,
# 그리고 서버가 낸 텍스트가 오프라인 simulated 스트리밍 가설과 같은지(50/50 일치). 마이크 대신 저장된 PCM을
# 같은 패킷 규약으로 보내 재현 가능한 측정을 한다. 서버 기본값이 바뀐 뒤(2차 패스, 규칙 변경 26)의 종단 측정은
# 규칙 변경 27 (A)로 등록했다: 텍스트 비교를 서버 설정에 맞추지 않으면 2차 패스 서버의 확정 텍스트는
# 구조적으로 simulated 가설과 거의 모두 달라 서비스 경로 버그와 구별되지 않는다.
# [설계 이유] 결과 파일마다 서버 /info(체크포인트 SHA-256 포함)와 실행 조건을 -meta.json으로 남겨 어떤 서버를
# 쟀는지 출처를 기록하고, flush 송신 시각을 직접 기록해 flush 왕복을 명목값이 아니라 실측으로 낸다(Codex 6단계
# 검토 반영, docs/ai-verification.md). 등록 측정 파일을 덮어쓰지 않도록 --name으로 이름을 바꿔 쓴다(README).
# 닫힘 이벤트 원본을 발화마다 남겨 텍스트가 어긋나면 어느 세그먼트에서 달라졌는지 볼 수 있게 한다.
# [연결] 서버는 stt_scratch.service.server(프로토콜은 그 파일 머리말). 비교용 simulated 가설은 4단계
# 보고서(scripts/stage2_report.py가 쓴 reports/stage4/hyps/…)에서 읽는다. 오프라인 재생은 서버와 같은
# StreamSession·warm_up·event_message를 그대로 쓴다.

from __future__ import annotations

# 표준 라이브러리: asyncio(보내기와 받기를 동시에 하는 비동기 클라이언트), urllib.request(서버 /info 조회),
# hashlib(재생 전 체크포인트·어휘 SHA-256 대조).
import argparse
import asyncio
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

# numpy: PCM 저장소에서 int16 표본을 꺼내고 백분위를 계산한다. websockets: 서버와의 WebSocket 연결.
import numpy as np
import websockets

# 프로젝트 내부 모듈: dev 매니페스트와 PCM 저장소, 첫 패킷 보정 표본 수(240). 모델·세션 모듈은 재생과
# 요약에서만 필요해 그 함수 안에서 불러온다(실시간 송신만 할 때 torch를 올리지 않는다).
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.features import FIRST_PACKET_EXTRA

# 패킷 규약(규칙 변경 9): 첫 패킷 5,360 표본, 이후 5,120 표본(16 kHz에서 320 ms). 첫 패킷은 240 표본을 더
# 보내야 mel 32프레임이 채워진다. 서버·브라우저와 같은 규약이라 패킷마다 인코더 청크 하나가 나온다.
ROOT = Path(__file__).resolve().parents[1]
CHUNK_SAMPLES = 5120
FIRST_PACKET_SAMPLES = CHUNK_SAMPLES + FIRST_PACKET_EXTRA
# 서버가 세그먼트를 닫을 때 보내는 이벤트 종류. endpoint는 확정 텍스트 없이 닫힌 세그먼트다(2차 패스를 켜면
# 2차 패스 결과가 빈 세그먼트도 endpoint로 닫히고, 그때도 stream_text에는 스트리밍 텍스트가 있을 수 있다).
CLOSE_TYPES = ("final", "endpoint")


# 발화 하나를 보내는 패킷들의 (시작, 끝) 표본 위치: 첫 패킷 5,360, 이후 5,120씩, 마지막은 남은 만큼. 실시간
# 송신(stream_one)과 오프라인 재생(replay_expected)이 같은 경계로 자르게 한곳에 둔다.
def packet_bounds(length: int):
    start, size = 0, FIRST_PACKET_SAMPLES
    while start < length:
        yield start, min(start + size, length)
        start += size
        size = CHUNK_SAMPLES


# 서버 이벤트(JSON을 읽은 dict) 목록에서 발화 하나의 확정 기록을 만든다. 실시간 송신과 오프라인 재생이 같은
# 함수를 써서 두 결과를 같은 정의로 비교한다.
# 입력: events(도착 순서; partial·stats가 섞여 있어도 됨), final_pass(서버 /info의 값, 모르면 None).
# 출력:
#   text            final 이벤트 text를 공백으로 이은 확정 텍스트(6단계와 같은 정의; 2차 패스를 켜면 2차
#                   패스 결과)
#   stream_text     final_pass가 "full"이고 모든 닫힘 이벤트에 stream_text가 있을 때만: 비지 않은
#                   stream_text를 공백으로 이은 1-pass 스트리밍 텍스트(닫힘 이벤트가 없으면 "";
#                   scripts/followup.py 세션 측정의 1-pass 정의와 같음). 그 밖에는 None: 2차 패스를 끈 서버는
#                   확정 텍스트가 곧 스트리밍 텍스트이고, stream_text를 보내기 전 서버(2차 패스를 켰어도)는
#                   스트리밍 텍스트를 알 수 없다.
#   n_finals·rules  final 수와 그 끝점 규칙(6단계와 같음)
#   n_endpoints·endpoint_rules  endpoint 수와 그 끝점 규칙
#   closes          닫힘 이벤트 원본(type·segment·text·rule·frames·compute_ms·final_pass_ms, 있으면
#                   stream_text와 도착 시각 t). 2차 패스 시간 통계와 세그먼트 단위 대조에 쓴다.
def close_record(events: list[dict], final_pass: str | None) -> dict:
    closes = [e for e in events if e["type"] in CLOSE_TYPES]
    finals = [e for e in closes if e["type"] == "final"]
    endpoints = [e for e in closes if e["type"] == "endpoint"]
    stream_text = None
    if final_pass == "full" and all("stream_text" in e for e in closes):
        stream_text = " ".join(e["stream_text"] for e in closes if e["stream_text"].strip())
    return {
        "text": " ".join(e["text"] for e in finals),
        "stream_text": stream_text,
        "n_finals": len(finals),
        "rules": [e["rule"] for e in finals],
        "n_endpoints": len(endpoints),
        "endpoint_rules": [e["rule"] for e in endpoints],
        "closes": closes,
    }


# 발화 하나를 새 WebSocket 연결로 보내고 결과를 모은다.
# 입력: 서버 URL, int16 PCM, realtime(True면 패킷 길이만큼 기다리며 실시간 속도로 보냄), final_pass(서버
# /info의 2차 패스 설정; stream_text 이음을 만들지 정한다). 출력: close_record의 확정 기록(확정 텍스트,
# stream_text 이음, 닫힘 이벤트 원본) + 부분 자막 수, 첫 부분 자막 시각, 부분 자막 지연 목록, 마지막 오디오 →
# 최종과 flush → 최종 시간, 서버가 보낸 청크당 계산 시간 통계.
# 시각은 모두 연결 직전을 0으로 한 이 프로세스의 perf_counter 기준이다.
async def stream_one(url: str, pcm: np.ndarray, realtime: bool, final_pass: str | None = None) -> dict:
    events = []
    t_start = time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:

        async def reader() -> None:
            # 보내는 동안에도 서버 메시지를 받아야 하므로, 받기는 별도 태스크로 돌리며 도착 시각과 함께
            # 쌓는다.
            async for msg in ws:
                events.append({"t": time.perf_counter() - t_start, **json.loads(msg)})

        task = asyncio.create_task(reader())
        sent_t = []
        # 패킷을 little-endian int16 바이트로 보내고 송신 시각을 기록한다. 실시간 모드에서는 패킷 길이(320
        # ms)만큼 쉬어 마이크처럼 보낸다.
        for start, end in packet_bounds(len(pcm)):
            piece = pcm[start:end]
            await ws.send(piece.astype("<i2").tobytes())
            sent_t.append(time.perf_counter() - t_start)
            if realtime:
                await asyncio.sleep(len(piece) / 16000)
        # 모든 음성을 보낸 뒤 "flush"를 보낸다. 서버는 마지막 부분 청크를 처리해 남은 이벤트를 보내고 마지막에
        # stats를 보내므로, stats가 올 때까지(최대 5 s) 기다린 뒤 받기 태스크를 끝낸다. flush 송신 시각을
        # 기록해 flush 왕복을 실측한다.
        t_flush = time.perf_counter() - t_start
        await ws.send("flush")
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and not any(e["type"] == "stats" for e in events):
            await asyncio.sleep(0.02)
        task.cancel()
    record = close_record(events, final_pass)
    finals = [e for e in record["closes"] if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    stats = next((e for e in events if e["type"] == "stats"), {})
    first_partial = partials[0]["t"] if partials else None
    # 자막 지연: 부분 자막마다 도착 시각 − 그 전에 보낸 마지막 패킷의 송신 시각(네트워크 + 대기 + 계산). 청크
    # 자체의 길이는 알고리즘 지연이라 따로 보고한다.
    lags = []
    for e in partials:
        before = [s for s in sent_t if s <= e["t"]]
        if before:
            lags.append((e["t"] - before[-1]) * 1000)
    # 마지막 오디오 → 최종 자막의 기준은 마지막 패킷 송신 시각이다(규칙 변경 24). 이 값에는 실시간 모드에서
    # 마지막 패킷 뒤 그 길이만큼 기다린 시간이 들어가므로, flush 송신 시각 기준 값도 함께 남긴다. 2차 패스를
    # 켜면 final 도착이 그 계산 시간만큼 늦어지므로 이 값들에 2차 패스 시간이 들어간다.
    last_final = finals[-1]["t"] if finals else None
    return {
        **record,
        "n_partials": len(partials),
        "first_partial_s": first_partial,
        "partial_lag_ms": lags,
        "final_after_last_audio_s": (last_final - sent_t[-1]) if last_final and sent_t else None,
        "final_after_flush_s": (last_final - t_flush) if last_final else None,
        "paced": realtime,
        "chunks": stats.get("chunks"),
        "compute_ms_p50": stats.get("compute_ms_p50"),  # 서버가 보낸 인코더 청크당 계산 시간(H6.2)
        "compute_ms_p90": stats.get("compute_ms_p90"),
        "compute_ms_max": stats.get("compute_ms_max"),
    }


# 매니페스트 앞 --limit 발화를 차례로 보내 결과를 JSONL로 쓴다. 시작 전에 서버 /info를 받아 -meta.json에 실행
# 조건과 함께 남긴다. 발화마다 줄을 바로 써서(flush) 중간에 멈춰도 끝난 발화는 남는다. 출력: 재생 확인에 쓸
# (발화 목록, PCM 저장소, 서버 /info, 결과 경로, meta).
async def main_async(args: argparse.Namespace) -> tuple:
    utts = read_manifest(ROOT / "manifests" / f"{args.manifest}.jsonl")[: args.limit or None]
    store = PcmStore(ROOT / "data" / "pcm", args.manifest)
    out = ROOT / "reports" / "stage6" / f"{args.name or 'client-' + args.manifest}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # /info 주소는 --url(ws://…/ws)에서 스킴만 http(s)로 바꾸고 마지막 경로를 /info로 바꿔 만든다. 그 결과가
    # http·https가 아니면 거부한다. --url 값이 file:// 같은 다른 스킴으로 urlopen에 들어가지 않게 하려는
    # 것으로, 공개 전 bandit 점검(B310) 지적을 고친 것이다(docs/security.md).
    info_url = args.url.replace("wss://", "https://").replace("ws://", "http://").rsplit("/", 1)[0] + "/info"
    if not info_url.startswith(("http://", "https://")):
        raise SystemExit(f"--url must be ws:// or wss://, got {args.url}")
    with urllib.request.urlopen(info_url, timeout=10) as r:  # noqa: S310  # nosec B310: scheme checked above
        server_info = json.loads(r.read())
    meta = {"server_info": server_info, "manifest": args.manifest, "limit": args.limit,
            "paced": not args.fast, "started": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    meta_path = out.with_name(out.stem + "-meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    final_pass = server_info.get("final_pass")
    with open(out, "w", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            offset, length = store.index[u.id]
            pcm = np.asarray(store.bin[offset : offset + length])
            result = await stream_one(args.url, pcm, not args.fast, final_pass)
            result.update({"id": u.id, "ref": u.text, "duration_s": u.duration_s})
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(utts)}", flush=True)
    print("wrote", out)
    return utts, store, server_info, out, meta


# --replay-check: 서버와 같은 설정의 StreamSession을 이 프로세스에서 만들어 같은 패킷을 그대로 넣고(push_pcm16
# → flush, 서버 처리기와 같은 호출 순서) 발화마다 기대 확정 기록을 만든다. 왜: 2차 패스 서버의 확정 텍스트는
# simulated 가설과 비교할 수 없으므로, 같은 코드·같은 설정·같은 입력의 오프라인 실행과 비교해 서버 경로(패킷
# 처리·int16 변환·직렬화)가 텍스트를 바꾸지 않는지 본다(docs/experiments.md 규칙 변경 27 (A)).
# 설정은 /info(server_info)에서 읽는다: 체크포인트(SHA-256이 다르면 멈춤), 어휘(vocab_sha256이 있으면 대조),
# 청크·좌측, 장치, 정밀도, 끝점, 캐시 정책, 실제 step 경로(캡처 실패로 fast가 됐으면 fast), 2차 패스, CPU면
# torch 스레드 수. 서버처럼 먼저 warm_up을 한 번 돌리고(graph 경로면 CUDA Graph 캡처 포함) 발화마다 새 세션을
# 만든다(서버는 연결마다 새 세션). 이벤트는 서버와 같은 직렬화(server.event_message)를 거쳐 close_record로
# 모은다. 입력: server_info, 발화 목록, PCM 저장소, root(체크포인트 상대 경로와 manifests/vocab.json의 기준),
# checkpoint(서버 체크포인트가 저장소 밖이라 /info에 파일 이름만 있을 때 줄 경로). 출력: (id → 기대 기록, 재생
# 조건). 주의: 서버와 같은 GPU를 쓰므로 실시간 송신이 모두 끝난 뒤에 부른다(지연 측정에 섞이지 않게). 이전
# 서버의 /info에 step_path·final_pass가 없으면 그때의 동작(eager, 2차 패스 없음)으로 재생한다.
def replay_expected(
    server_info: dict, utts, store: PcmStore, root: Path = ROOT, checkpoint: str | None = None
) -> tuple[dict[str, dict], dict]:
    import torch

    from stt_scratch.evaluate import check_vocab, load_model
    from stt_scratch.model import streaming
    from stt_scratch.model.features import LogMel
    from stt_scratch.model.mask import ChunkSpec
    from stt_scratch.service.endpoint import EndpointConfig
    from stt_scratch.service.server import event_message
    from stt_scratch.service.session import StreamSession, warm_up
    from stt_scratch.text import Vocab

    path = Path(checkpoint or server_info["checkpoint"])
    path = path if path.is_absolute() else root / path
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != server_info["checkpoint_sha256"]:
        raise SystemExit(
            f"replay checkpoint {path} has SHA-256 {sha[:12]}..., not the server's checkpoint_sha256"
        )
    vocab_path = root / "manifests" / "vocab.json"
    vocab_sha = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    if server_info.get("vocab_sha256") not in (None, vocab_sha):
        raise SystemExit(f"{vocab_path} differs from the server's vocabulary (vocab_sha256)")
    device = torch.device(server_info["device"])
    if device.type == "cpu" and server_info.get("cpu_threads"):
        torch.set_num_threads(int(server_info["cpu_threads"]))
    model, _ = load_model(path, device)
    vocab = Vocab.load(vocab_path)
    check_vocab(model, vocab)
    spec = ChunkSpec(server_info["chunk_frames"], server_info["left_frames"])
    logmel = LogMel().to(device)
    endpoint = EndpointConfig(**server_info["endpoint"])
    final_pass = server_info.get("final_pass", "none")
    options = {
        "amp": server_info["precision"] == "fp16-autocast",
        "step_path": server_info.get("step_path", "eager"),
        "final_pass": final_pass,
    }

    def new_session() -> StreamSession:
        return StreamSession(
            model, vocab, spec, logmel, endpoint, server_info["reset_cache_on_endpoint"], **options
        )

    warm_s = warm_up(new_session())
    rows: dict[str, dict] = {}
    for u in utts:
        offset, length = store.index[u.id]
        pcm = np.asarray(store.bin[offset : offset + length])
        session = new_session()
        events = []
        for start, end in packet_bounds(len(pcm)):
            events.extend(session.push_pcm16(pcm[start:end].astype("<i2").tobytes()))
        events.extend(session.flush())
        rows[u.id] = {"id": u.id, **close_record([event_message(e) for e in events], final_pass)}
    provenance = {
        "checkpoint": path.name,
        "checkpoint_sha256": sha,
        "vocab_sha256": vocab_sha,
        "device": str(device),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        **options,
        "warm_up_s": round(warm_s, 2),
        "graph": streaming.graph_status(model, spec) if options["step_path"] == "graph" else None,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return rows, provenance


# 실시간 모드에서 클라이언트가 마지막 패킷 뒤 기다린 시간을 계산하는 데 쓴다. 실제 마이크 클라이언트는 이
# 대기 없이 바로 flush를 보내므로 요약에서 따로 보여 준다.
def last_packet_samples(length: int) -> int:
    """stream_one이 보낸 마지막 패킷의 표본 수(첫 패킷 5,360, 이후 5,120씩)."""
    if length <= FIRST_PACKET_SAMPLES:
        return length
    rest = (length - FIRST_PACKET_SAMPLES) % CHUNK_SAMPLES
    return rest or CHUNK_SAMPLES


# 기록이 잰 서버의 2차 패스 설정("none"/"full")과 그 근거를 돌려준다. 판별할 수 없으면 (None, 이유).
# - -meta.json이 없거나 server_info가 없으면 판별 불가(6단계 첫 실행 기록 client-dev_h처럼 meta 이전 기록
#   포함).
# - server_info에 final_pass 키가 없으면 "none": 이 키는 --final-pass 옵션과 같은 커밋(b1bbdf5, 규칙 변경
#   26)에서 생겼으므로, 키가 없는 /info는 2차 패스가 없던(1-pass 확정 자막) 서버의 것이다(6단계 기록).
def final_pass_mode(meta: dict | None) -> tuple[str | None, str]:
    if not meta or not isinstance(meta.get("server_info"), dict):
        return None, "no -meta.json with server_info, so the server's final-pass setting is unknown"
    info = meta["server_info"]
    if "final_pass" not in info:
        return "none", "server_info has no final_pass key: the server predates the final-pass option (1-pass)"
    if info["final_pass"] in ("none", "full"):
        return info["final_pass"], "server_info.final_pass"
    return None, f"unknown server_info.final_pass {info['final_pass']!r}"


# 저장된 클라이언트 기록을 요약한다(서버 불필요).
# - partial_lag_ms: 모든 발화의 부분 자막 지연을 한데 모은 p50/p90/max.
# - final_after_last_audio_s와 client_wait_after_last_packet_s: 마지막 오디오 → 최종과 그 안에 든 클라이언트
#   대기.
# - flush_round_trip_s: flush 송신 시각이 기록된 실행이면 실측값, 예전 기록이면 '마지막 오디오 → 최종 − 마지막
#   패킷 길이'의 명목값(어느 쪽인지 basis에 적음).
# - 서버 청크 계산 p50/p90의 발화 간 중앙값과 최대, 끝점 규칙별 최종 자막 수, 한글 전용 CER_ns.
# - simulated 가설 파일이 있으면 텍스트 비교. 무엇과 비교할지는 서버 설정(final_pass_mode)으로 정한다:
#   none = 확정 텍스트가 곧 스트리밍 텍스트라 6단계처럼 same_text_as_simulated(정확히 같은 발화 수)를 세고,
#   full = 확정 텍스트는 2차 패스 결과라 same_text_as_simulated는 null로 두고 stream_text 이음을 비교한다.
#   same_stream_text_as_simulated는 공백 포함(exact)·공백 무시(ignoring_spaces) 두 수다(rule2 0.8 s에서는
#   세그먼트가 늘어 이음 자리의 공백만 다를 수 있다). 판별할 수 없으면 null과 경고(warnings, 표준 오류에도
#   씀).
# - <이름>-replay.jsonl(--replay-check)이 모든 발화를 덮으면 확정 텍스트가 재생과 같은 발화 수
#   (same_final_text_as_replay)와, 2차 패스 서버면 stream_text가 재생과 같은 수(same_stream_text_as_replay).
# - validity: 규칙 변경 27 (A)의 유효성 조건 = stream_text가 simulated와 공백 무시로 모두 같고 확정 텍스트가
#   재생과 모두 같음(met). 한쪽이라도 판별할 수 없으면 null.
# - 2차 패스 서버면 닫힘 이벤트마다의 2차 패스 계산 ms(final_pass_ms)와 stream_text의 한글 전용 CER_ns,
#   endpoint 규칙별 수(이 값을 기록한 클라이언트일 때만).
# 입력: manifest, simulated(root 기준 경로), name(기본 client-<manifest>), root(매니페스트·PCM·가설의 기준),
# out_dir(기록과 요약 파일 폴더, 기본 root/reports/stage6). 출력: 요약 dict(같은 내용을 <이름>-summary.json에
# 씀).
def summarize(
    manifest: str, simulated: str, name: str = "", root: Path = ROOT, out_dir: Path | None = None
) -> dict:
    # 채점은 평가 모듈과 같은 함수를 쓴다(측정할 때는 필요 없어서 함수 안에서 가져온다).
    from stt_scratch.evaluate import score

    name = name or f"client-{manifest}"
    out_dir = out_dir or root / "reports" / "stage6"
    rows = [json.loads(x) for x in (out_dir / f"{name}.jsonl").open(encoding="utf-8")]
    store = PcmStore(root / "data" / "pcm", manifest)
    utts = {u.id: u for u in read_manifest(root / "manifests" / f"{manifest}.jsonl")}
    sim = {}
    sim_path = root / simulated
    if sim_path.exists():
        sim = {r["id"]: r["hyp"] for r in map(json.loads, sim_path.open(encoding="utf-8"))}
    meta_path = out_dir / f"{name}-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    mode, mode_basis = final_pass_mode(meta)
    warnings = [] if mode else [f"text comparisons are null: {mode_basis}"]
    replay_path = out_dir / f"{name}-replay.jsonl"
    replay = {}
    if replay_path.exists():
        replay = {r["id"]: r for r in map(json.loads, replay_path.open(encoding="utf-8"))}

    # None을 뺀 값들의 p50·p90·최대·개수. 값이 없으면 n 0만 돌려준다.
    def pct(values) -> dict:
        a = np.asarray([v for v in values if v is not None], dtype=float)
        if not a.size:
            return {"n": 0}
        return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "max": float(a.max()),
                "n": int(a.size)}  # fmt: skip

    waits = [last_packet_samples(store.index[r["id"]][1]) / 16000 for r in rows]
    finals = [r["final_after_last_audio_s"] for r in rows]
    # 모든 행에 flush 기준 시각이 있으면(수정 뒤 클라이언트로 잰 기록) flush 왕복을 실측값으로 계산한다.
    measured_flush = all("final_after_flush_s" in r for r in rows)
    picked = [utts[r["id"]] for r in rows]
    scored = score(picked, {r["id"]: r["text"] for r in rows})
    # 발화마다의 스트리밍(1-pass) 텍스트: 2차 패스를 끈 서버는 확정 텍스트, 켠 서버는 stream_text 이음. 켠
    # 서버인데 stream_text가 없는 행(필드를 보내기 전 서버나 이전 클라이언트의 기록)이 있으면 판별할 수 없다.
    streams = None
    if mode == "none":
        streams = {r["id"]: r["text"] for r in rows}
    elif mode == "full":
        streams = {r["id"]: r.get("stream_text") for r in rows}
        if any(v is None for v in streams.values()):
            warnings.append(
                "stream comparisons are null: some rows have no stream_text from the final-pass server"
            )
            streams = None
    same_stream = None
    if sim and streams is not None:
        same_stream = {
            "exact": sum(sim.get(i) == s for i, s in streams.items()),
            "ignoring_spaces": sum(
                i in sim and sim[i].replace(" ", "") == s.replace(" ", "") for i, s in streams.items()
            ),
        }
    same_replay = same_stream_replay = None
    if replay:
        missing = sum(r["id"] not in replay for r in rows)
        if missing:
            warnings.append(
                f"replay comparison is null: {replay_path.name} lacks {missing} of the utterances"
            )
        else:
            same_replay = sum(replay[r["id"]]["text"] == r["text"] for r in rows)
            if mode == "full" and streams is not None:
                same_stream_replay = sum(replay[i].get("stream_text") == s for i, s in streams.items())
    stream_ok = None if same_stream is None else same_stream["ignoring_spaces"] == len(rows)
    replay_ok = None if same_replay is None else same_replay == len(rows)
    out = {
        "meta": meta,
        "utterances": len(rows),
        "partial_lag_ms": pct(x for r in rows for x in r["partial_lag_ms"]),
        "first_partial_s": pct(r["first_partial_s"] for r in rows),
        "final_after_last_audio_s": pct(finals),
        "client_wait_after_last_packet_s": pct(waits),
        "flush_round_trip_s": pct(r["final_after_flush_s"] for r in rows)
        if measured_flush
        else pct(f - w for f, w in zip(finals, waits, strict=True) if f is not None),
        "flush_round_trip_basis": "measured: final arrival - flush send time"
        if measured_flush
        else "nominal: last audio -> final minus the last packet duration (paced runs only)",
        "server_compute_ms_p50_median": float(np.median([r["compute_ms_p50"] for r in rows])),
        "server_compute_ms_p90_median": float(np.median([r["compute_ms_p90"] for r in rows])),
        "server_compute_ms_max": float(max(r["compute_ms_max"] for r in rows)),
        "final_rules": {
            k: sum(r["rules"].count(k) for r in rows) for k in ("rule1", "rule2", "rule3", "eos")
        },
        "cer_ns_hangul_only": scored["hangul_only"]["cer_ns"]["rate"],
        "simulated_hyps": simulated if sim else None,
        "same_text_as_simulated": sum(sim.get(r["id"]) == r["text"] for r in rows)
        if sim and mode == "none"
        else None,
    }
    if sim:
        out["simulated_cer_ns_hangul_only_same_utts"] = score(picked, {u.id: sim[u.id] for u in picked})[
            "hangul_only"
        ]["cer_ns"]["rate"]
    out.update(
        {
            "final_pass": mode,
            "final_pass_basis": mode_basis,
            "same_stream_text_as_simulated": same_stream,
            "replay_file": replay_path.name if replay else None,
            "same_final_text_as_replay": same_replay,
            "same_stream_text_as_replay": same_stream_replay,
            "validity": {
                "stream_text_matches_simulated_ignoring_spaces": stream_ok,
                "final_text_matches_replay": replay_ok,
                "met": None if stream_ok is None or replay_ok is None else stream_ok and replay_ok,
            },
            "final_pass_ms": pct(e.get("final_pass_ms") for r in rows for e in r.get("closes", []))
            if mode == "full"
            else None,
            "cer_ns_hangul_only_stream_text": score(picked, streams)["hangul_only"]["cer_ns"]["rate"]
            if mode == "full" and streams is not None
            else None,
            "endpoint_rules": {
                k: sum(r["endpoint_rules"].count(k) for r in rows) for k in ("rule1", "rule2", "rule3", "eos")
            }
            if all("endpoint_rules" in r for r in rows)
            else None,
            "warnings": warnings,
        }
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    path = out_dir / f"{name}-summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# 인자: --manifest(기본 dev_h), --limit(앞 N발화, 0 = 전체), --url(서버 WebSocket 주소), --fast(실시간 대기
# 없이 보냄), --summarize(기록 요약만), --name(결과 이름), --simulated(비교할 simulated 가설 파일; 기본값은
# 최종 모델의 320 ms dev_h 가설), --replay-check(송신 뒤 오프라인 재생으로 기대 기록),
# --replay-checkpoint(재생에 쓸 체크포인트 경로; 기본은 /info의 경로).
def main() -> None:
    p = argparse.ArgumentParser(
        description="실행 중인 스트리밍 서버에 저장된 dev 발화를 실시간 속도로 보내 자막·지연을 "
        "기록하고(기본), --summarize로 그 기록을 요약한다(docs/experiments.md 규칙 변경 21·24·27)"
    )
    p.add_argument("--manifest", default="dev_h")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    p.add_argument("--fast", action="store_true", help="실시간 속도로 기다리지 않고 바로 보낸다")
    p.add_argument("--summarize", action="store_true", help="저장된 클라이언트 기록만 요약한다(서버 불필요)")
    p.add_argument("--name", default="", help="reports/stage6/ 아래 결과 이름(기본 client-<manifest>)")
    p.add_argument(
        "--simulated",
        default="reports/stage4/hyps/trimtail-avg10-dev_h-chunk8_320ms.jsonl",
        help="텍스트 비교에 쓸, 서비스 중인 모델의 simulated 스트리밍 가설 파일",
    )
    p.add_argument(
        "--replay-check",
        action="store_true",
        help="모든 발화를 보낸 뒤 이 PC에서 서버 /info와 같은 모델·설정의 세션에 같은 패킷을 재생해 "
        "기대 기록을 <이름>-replay.jsonl에 쓴다(요약이 확정 텍스트를 이것과 비교한다). 서버와 같은 장치에 "
        "모델을 한 번 더 올리므로 실시간 송신이 끝난 뒤에 돈다. 체크포인트 SHA-256이 /info와 다르면 멈춘다",
    )
    p.add_argument(
        "--replay-checkpoint",
        default=None,
        help="재생에 쓸 체크포인트 경로(기본: /info의 checkpoint를 저장소 기준으로 찾음; 서버 체크포인트가 "
        "저장소 밖이면 /info에는 파일 이름만 있으므로 경로를 준다)",
    )
    args = p.parse_args()
    if args.summarize:
        summarize(args.manifest, args.simulated, args.name)
        return
    utts, store, server_info, out, meta = asyncio.run(main_async(args))
    if args.replay_check:
        rows, provenance = replay_expected(server_info, utts, store, ROOT, args.replay_checkpoint)
        replay_path = out.with_name(out.stem + "-replay.jsonl")
        with open(replay_path, "w", encoding="utf-8") as f:
            for row in rows.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 재생 조건을 같은 -meta.json에 더해 다시 쓴다(송신 전에 쓴 server_info 등은 그대로).
        meta["replay"] = provenance
        meta_path = out.with_name(out.stem + "-meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", replay_path)


if __name__ == "__main__":
    main()
