"""실행 중인 WebSocket 서버에 발화를 실시간 속도로 보내고 발화마다 자막·지연을 기록하는 측정용 클라이언트.

6단계에서 종단 지연(마지막 오디오 바이트 송신 → 최종 자막 수신)을 재고, 서비스 경로가 offline 스트리밍
평가와 같은 텍스트를 내는지 확인하는 데 썼다.

사용법: uv run python scripts/stream_client.py --manifest dev_h --limit 50 --url ws://127.0.0.1:8765/ws
발화마다 받은 자막·이벤트 시각·서버 계산 통계를 reports/stage6/client-<manifest>.jsonl에 쓴다.
`--summarize`(서버 불필요)는 그 파일을 읽어 reports/stage6/client-<manifest>-summary.json을 쓴다: 모든 발화를
모은 부분 자막 지연, 첫 부분 자막, 마지막 오디오 → 최종 자막, 그 가운데 이 클라이언트가 마지막 패킷 뒤
실시간으로 기다린 몫(실제 마이크 클라이언트는 "flush"를 바로 보낸다), 그 대기를 뺀 flush 왕복, 서버 계산
시간, 서비스 텍스트와 simulated 스트리밍 가설의 비교.
"""

# [왜 필요한가] 6단계 등록 측정 가운데 종단 지연(로컬)과 서비스 경로 검증을 맡는다(docs/experiments.md 규칙
# 변경 21): 부분 자막 도착 − 직전 패킷 송신(p50 62.4 ms), 첫 부분 자막, 마지막 오디오 → 최종 자막, flush 왕복,
# 그리고 서버가 낸 텍스트가 오프라인 simulated 스트리밍 가설과 같은지(50/50 일치). 마이크 대신 저장된 PCM을
# 같은 패킷 규약으로 보내 재현 가능한 측정을 한다.
# [설계 이유] 결과 파일마다 서버 /info(체크포인트 SHA-256 포함)와 실행 조건을 -meta.json으로 남겨 어떤 서버를
# 쟀는지 출처를 기록하고, flush 송신 시각을 직접 기록해 flush 왕복을 명목값이 아니라 실측으로 낸다(Codex 6단계
# 검토 반영, docs/ai-verification.md). 등록 측정 파일을 덮어쓰지 않도록 --name으로 이름을 바꿔 쓴다(README).
# [연결] 서버는 stt_scratch.service.server(프로토콜은 그 파일 머리말). 비교용 simulated 가설은 4단계
# 보고서(scripts/stage2_report.py가 쓴 reports/stage4/hyps/…)에서 읽는다.

from __future__ import annotations

# 표준 라이브러리: asyncio(보내기와 받기를 동시에 하는 비동기 클라이언트), urllib.request(서버 /info 조회).
import argparse
import asyncio
import json
import time
import urllib.request
from pathlib import Path

# numpy: PCM 저장소에서 int16 표본을 꺼내고 백분위를 계산한다. websockets: 서버와의 WebSocket 연결.
import numpy as np
import websockets

# 프로젝트 내부 모듈: dev 매니페스트와 PCM 저장소, 첫 패킷 보정 표본 수(240).
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.features import FIRST_PACKET_EXTRA

# 패킷 규약(규칙 변경 9): 첫 패킷 5,360 표본, 이후 5,120 표본(16 kHz에서 320 ms). 첫 패킷은 240 표본을 더
# 보내야 mel 32프레임이 채워진다. 서버·브라우저와 같은 규약이라 패킷마다 인코더 청크 하나가 나온다.
ROOT = Path(__file__).resolve().parents[1]
CHUNK_SAMPLES = 5120
FIRST_PACKET_SAMPLES = CHUNK_SAMPLES + FIRST_PACKET_EXTRA


# 발화 하나를 새 WebSocket 연결로 보내고 결과를 모은다.
# 입력: 서버 URL, int16 PCM, realtime(True면 패킷 길이만큼 기다리며 실시간 속도로 보냄). 출력: 최종 텍스트,
# 부분·최종 이벤트 수, 끝점 규칙, 첫 부분 자막 시각, 부분 자막 지연 목록, 마지막 오디오 → 최종과 flush → 최종
# 시간, 서버가 보낸 청크당 계산 시간 통계.
# 시각은 모두 연결 직전을 0으로 한 이 프로세스의 perf_counter 기준이다.
async def stream_one(url: str, pcm: np.ndarray, realtime: bool) -> dict:
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
        i, packet = 0, FIRST_PACKET_SAMPLES
        while i < len(pcm):
            piece = pcm[i : i + packet]
            i += packet
            packet = CHUNK_SAMPLES
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
    finals = [e for e in events if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    stats = next((e for e in events if e["type"] == "stats"), {})
    # 최종 텍스트 = 세그먼트별 최종 자막을 공백으로 이은 것(평가 경로의 세션 측정과 같은 방식).
    text = " ".join(e["text"] for e in finals)
    first_partial = partials[0]["t"] if partials else None
    # 자막 지연: 부분 자막마다 도착 시각 − 그 전에 보낸 마지막 패킷의 송신 시각(네트워크 + 대기 + 계산). 청크
    # 자체의 길이는 알고리즘 지연이라 따로 보고한다.
    lags = []
    for e in partials:
        before = [s for s in sent_t if s <= e["t"]]
        if before:
            lags.append((e["t"] - before[-1]) * 1000)
    # 마지막 오디오 → 최종 자막의 기준은 마지막 패킷 송신 시각이다(규칙 변경 24). 이 값에는 실시간 모드에서
    # 마지막 패킷 뒤 그 길이만큼 기다린 시간이 들어가므로, flush 송신 시각 기준 값도 함께 남긴다.
    last_final = finals[-1]["t"] if finals else None
    return {
        "text": text,
        "n_partials": len(partials),
        "n_finals": len(finals),
        "rules": [e["rule"] for e in finals],
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
# 조건과 함께 남긴다. 발화마다 줄을 바로 써서(flush) 중간에 멈춰도 끝난 발화는 남는다.
async def main_async(args: argparse.Namespace) -> None:
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
    out.with_name(out.stem + "-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")  # fmt: skip
    with open(out, "w", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            offset, length = store.index[u.id]
            pcm = np.asarray(store.bin[offset : offset + length])
            result = await stream_one(args.url, pcm, not args.fast)
            result.update({"id": u.id, "ref": u.text, "duration_s": u.duration_s})
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(utts)}", flush=True)
    print("wrote", out)


# 실시간 모드에서 클라이언트가 마지막 패킷 뒤 기다린 시간을 계산하는 데 쓴다. 실제 마이크 클라이언트는 이
# 대기 없이 바로 flush를 보내므로 요약에서 따로 보여 준다.
def last_packet_samples(length: int) -> int:
    """stream_one이 보낸 마지막 패킷의 표본 수(첫 패킷 5,360, 이후 5,120씩)."""
    if length <= FIRST_PACKET_SAMPLES:
        return length
    rest = (length - FIRST_PACKET_SAMPLES) % CHUNK_SAMPLES
    return rest or CHUNK_SAMPLES


# 저장된 클라이언트 기록을 요약한다(서버 불필요).
# - partial_lag_ms: 모든 발화의 부분 자막 지연을 한데 모은 p50/p90/max.
# - final_after_last_audio_s와 client_wait_after_last_packet_s: 마지막 오디오 → 최종과 그 안에 든 클라이언트
#   대기.
# - flush_round_trip_s: flush 송신 시각이 기록된 실행이면 실측값, 예전 기록이면 '마지막 오디오 → 최종 − 마지막
#   패킷 길이'의 명목값(어느 쪽인지 basis에 적음).
# - 서버 청크 계산 p50/p90의 발화 간 중앙값과 최대, 끝점 규칙별 최종 자막 수, 한글 전용 CER_ns.
# - simulated 가설 파일이 있으면 텍스트가 정확히 같은 발화 수와 같은 발화들의 simulated CER.
def summarize(manifest: str, simulated: str, name: str = "") -> dict:
    # 채점은 평가 모듈과 같은 함수를 쓴다(측정할 때는 필요 없어서 함수 안에서 가져온다).
    from stt_scratch.evaluate import score

    name = name or f"client-{manifest}"
    rows = [json.loads(x) for x in (ROOT / "reports" / "stage6" / f"{name}.jsonl").open(encoding="utf-8")]
    store = PcmStore(ROOT / "data" / "pcm", manifest)
    utts = {u.id: u for u in read_manifest(ROOT / "manifests" / f"{manifest}.jsonl")}
    sim = {}
    sim_path = ROOT / simulated
    if sim_path.exists():
        sim = {r["id"]: r["hyp"] for r in map(json.loads, sim_path.open(encoding="utf-8"))}

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
    meta_path = ROOT / "reports" / "stage6" / f"{name}-meta.json"
    out = {
        "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None,
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
        "same_text_as_simulated": sum(sim.get(r["id"]) == r["text"] for r in rows) if sim else None,
    }
    if sim:
        out["simulated_cer_ns_hangul_only_same_utts"] = score(picked, {u.id: sim[u.id] for u in picked})[
            "hangul_only"
        ]["cer_ns"]["rate"]
    path = ROOT / "reports" / "stage6" / f"{name}-summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# 인자: --manifest(기본 dev_h), --limit(앞 N발화, 0 = 전체), --url(서버 WebSocket 주소), --fast(실시간 대기
# 없이 보냄), --summarize(기록 요약만), --name(결과 이름), --simulated(비교할 simulated 가설 파일; 기본값은
# 최종 모델의 320 ms dev_h 가설).
def main() -> None:
    p = argparse.ArgumentParser()
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
    args = p.parse_args()
    if args.summarize:
        summarize(args.manifest, args.simulated, args.name)
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
