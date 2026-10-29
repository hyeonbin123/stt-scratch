"""6단계 스트리밍 서비스 측정 도구(docs/experiments.md 6단계, 규칙 변경 21).

하위 명령(각각 reports/stage6/<name>.json을 쓴다):
  grid     청크 {4, 8, 16} × 좌측 {64, 128}의 simulated CER, 알고리즘 지연(평균/최대), GPU(fp16)와
           CPU 1스레드(fp32)의 청크당 계산 시간 p50/p90과 RTF. H6.2
  h61      서비스 구성에서 dev_H·dev_mix 전수의 실제 캐시 스트리밍 CER vs simulated CER. H6.1
  session  dev_mix에서 StreamSession 실행(발화마다 뒤에 -60 dBFS 잡음 3 s를 붙임): 끝점 끈 기준,
           캐시 초기화한 rule2 {0.8, 1.2} s, 그다음 고른 rule2에서 캐시 유지. 조기 끝점에 따른 CER
           증가, EP50/EP90(끝점 시각 − MFA 마지막 단어 끝), 부분 자막 안정성 UPWR/UPSR(H6.3),
           세션 CER과 측정 CER 비교
  drift    dev_H 음성을 이어 붙인 10분을 세션 하나로 흘려 분당 계산 시간과 GPU 메모리 변화를 잰다.
           --step-path·--final-pass·--warm-up을 주면 서버 기본 구성(graph step, 2차 패스, 예열)으로도
           잰다(규칙 변경 27)

사용법: uv run python scripts/stage6.py grid --checkpoint outputs/stage4/control/avg10.pt
"""

# [왜 필요한가] 측정 항목과 판정 규칙은 6단계 측정 전에 등록했다(docs/experiments.md 6단계, 규칙 변경 21·24).
# 서비스가 쓰는 것과 같은 코드(StreamingEncoder, StreamSession)를 그대로 돌려서, 측정값이 곧 서비스 동작을
# 대표하게 한다. 결과: H6.1 충족(dev_H +0.002포인트), H6.2 미충족(GPU p90 63.3 ms > 50, CPU 1스레드 46.5 ms는
# 충족), H6.3 충족, rule2 1.2 s, 끝점 뒤 캐시 유지.
# [연결] 측정 모델은 4단계 채택 모델(outputs/stage4/trimtail/avg10.pt; docstring 사용법 예시의 control 경로는
# 판정 전 작성된 예). README의 6단계 명령에서 하위 명령별로 실행한다. 결과 파일마다 체크포인트
# SHA-256·구성·부분집합 여부(limit)를 남기고, 7단계(scripts/stage7.py)는 h61.json의 이 값들을 확인한 뒤에만
# 시작한다.

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# numpy: 백분위·지연 통계. torch: 모델 실행, GPU 동기화·메모리 통계, CPU 스레드 수 제한.
import numpy as np
import torch

# 프로젝트 내부 모듈: 평가 경로(offline·실제 캐시 스트리밍 디코딩, 채점, autocast), 캐시 스트리밍 인코더와 그
# CUDA Graph 캡처 상태, 서비스와 같은 세션·예열·끝점 설정, 첫 패킷 보정 표본 수.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, check_vocab, decode_offline, decode_streaming, load_model, score
from stt_scratch.model.features import FIRST_PACKET_EXTRA, LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder, graph_status
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession, warm_up
from stt_scratch.text import Vocab

# SR: 표본화율. PACKET: 320 ms 패킷 표본 수(5,120). NOISE_DBFS: 세션 측정에서 발화 뒤에 붙이는 가우시안 잡음
# 크기(-60 dBFS). 발화 뒤에 묵음 구간을 만들어 끝점이 닫힐 수 있게 한다(규칙 변경 21).
ROOT = Path(__file__).resolve().parents[1]
SR = 16000
PACKET = 5120
NOISE_DBFS = -60.0


# 채점 결과에서 주지표(한글 전용 CER_ns)를 꺼낸다.
def rate(scored: dict) -> float:
    return scored["hangul_only"]["cer_ns"]["rate"]


# 계산 시간은 넣지 않는다(규칙 변경 21). 청크 안 위치에 따라 0~청크 길이만큼 기다리므로(평균 절반) 평균 =
# 청크/2 + 15 ms, 최대 = 청크 + 15 ms다.
def algorithmic_latency_ms(chunk: int) -> dict:
    """알고리즘 지연: 프레임은 자기 청크가 끝나야 나오고, mel 창(25 ms 창, 10 ms 간격)이 15 ms를 더한다."""
    chunk_ms = chunk * 40
    return {"mean_ms": chunk_ms / 2 + 15, "max_ms": chunk_ms + 15}


# seed가 고정된 가우시안 잡음(진폭 = 10^(dBFS/20)). 발화마다 다른 seed를 주되 매번 같은 잡음이 나와 재현된다.
def noise(seconds: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g) * (10 ** (NOISE_DBFS / 20))


# 파형을 서비스 패킷 규약대로 자른다: 첫 패킷 5,120 + 240 표본, 이후 5,120 표본(규칙 변경 9). 그래야 패킷마다
# mel 32프레임 = 인코더 청크 1개가 나온다. (시작 위치, 조각)을 내준다.
def packets(wav: torch.Tensor):
    pos, size = 0, PACKET + FIRST_PACKET_EXTRA
    while pos < wav.numel():
        yield pos, wav[pos : pos + size]
        pos += size
        size = PACKET


# 발화마다 log-mel을 미리 계산하고 청크씩 step만 재므로 특징 계산은 시간에 들어가지 않는다.
# GPU에서는 step 전후 모두 동기화한다. 앞서 쌓인 GPU 작업이 이 step 시간에 섞이지 않게 하려는 것이다(Codex
# 4단계 검토, 규칙 변경 24).
@torch.no_grad()
def time_steps(model, utts, store, spec, device, amp: bool) -> dict:
    """발화 전체에 걸친 청크당 인코더 step 시간(ms, 동기화)과 RTF(계산 시간 합 ÷ 음성 길이)를 잰다."""
    logmel = LogMel().to(device)
    enc = StreamingEncoder(model, spec)
    times, audio_s = [], 0.0
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        state = enc.new_state()
        step = 4 * spec.chunk_size
        for i in range(0, mel.shape[1], step):  # 마지막 부분 청크도 포함한다(RTF는 발화 전체 기준)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device, amp):
                enc.step(mel[0, i : i + step], state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        audio_s += u.duration_s
    a = np.asarray(times)
    return {
        "chunks": len(times),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "max_ms": float(a.max()),
        "rtf": float(a.sum() / 1000 / audio_s),
    }


# grid: 청크 {4, 8, 16}(160·320·640 ms) × 좌측 {64, 128} 각각에 대해 (1) dev_mix·dev_h 전수 simulated CER, (2)
# 알고리즘 지연, (3) GPU fp16 청크 계산 시간(dev_h 앞 30발화), (4) CPU fp32 1스레드 계산 시간(앞 10발화)을
# 잰다. 서비스 구성(청크 8, 좌측 128)의 p90으로 H6.2(GPU ≤ 50 ms, CPU 1스레드 ≤ 250 ms)를 판정한다.
def cmd_grid(args, model, vocab, device) -> dict:
    logmel = LogMel().to(device)
    out = {"configs": []}
    sets = {
        n: (
            read_manifest(ROOT / f"manifests/{n}.jsonl")[: args.limit or None],
            PcmStore(ROOT / "data/pcm", n),
        )
        for n in ("dev_mix", "dev_h")
    }
    timing_utts = sets["dev_h"][0][: args.timing_utts]
    # CPU 측정용으로 같은 체크포인트를 CPU에 한 벌 더 올린다. CPU 경로는 autocast 없이 fp32다.
    cpu_model, _ = load_model(ROOT / args.checkpoint, torch.device("cpu"))
    for chunk in (4, 8, 16):
        for left in (64, 128):
            spec = ChunkSpec(chunk, left)
            row = {
                "chunk_frames": chunk,
                "chunk_ms": chunk * 40,
                "left": left,
                **algorithmic_latency_ms(chunk),
            }
            for name, (utts, store) in sets.items():
                row[f"{name}_cer_ns"] = rate(
                    score(utts, decode_offline(model, vocab, utts, store, spec, device, logmel))
                )
            row["gpu"] = time_steps(model, timing_utts, sets["dev_h"][1], spec, device, amp=True)
            # CPU 1스레드 조건(등록 정의). 이 설정은 프로세스 전체에 남는다.
            torch.set_num_threads(1)
            row["cpu1"] = time_steps(
                cpu_model,
                timing_utts[: args.cpu_utts],
                sets["dev_h"][1],
                spec,
                torch.device("cpu"),
                amp=False,
            )
            print(json.dumps(row), flush=True)
            out["configs"].append(row)
    # H6.2 판정은 서비스 구성(320 ms, 좌측 128) 한 행으로 한다.
    service = next(r for r in out["configs"] if r["chunk_frames"] == 8 and r["left"] == 128)
    out["h62"] = {
        "gpu_p90_ms": service["gpu"]["p90_ms"],
        "cpu1_p90_ms": service["cpu1"]["p90_ms"],
        "met": service["gpu"]["p90_ms"] <= 50 and service["cpu1"]["p90_ms"] <= 250,
    }
    return out


# h61: 서비스 구성에서 dev_h·dev_mix 전수를 simulated(청크 마스크 offline)와 실제 캐시 스트리밍(패킷 단위
# StreamingFbank + StreamingEncoder)으로 각각 디코딩해 CER 차이를 낸다. H6.1: 실제 − simulated ≤
# 0.1포인트(dev_H 기준, 방향 있는 차이). 위반이면 캐시 버그로 보고 진행하지 않기로 등록했다. 두 경로 모두 같은
# fp16 autocast라 차이는 캐시 경로에서만 나온다(규칙 변경 8).
def cmd_h61(args, model, vocab, device) -> dict:
    logmel = LogMel().to(device)
    spec = ChunkSpec(args.chunk, args.left)
    out = {"spec": f"chunk{args.chunk} left{args.left}"}
    for name in ("dev_h", "dev_mix"):
        utts = read_manifest(ROOT / f"manifests/{name}.jsonl")[: args.limit or None]
        store = PcmStore(ROOT / "data/pcm", name)
        sim = rate(score(utts, decode_offline(model, vocab, utts, store, spec, device, logmel)))
        real_hyps = decode_streaming(model, vocab, utts, store, spec, device, logmel)
        real = rate(score(utts, real_hyps))
        out[name] = {"simulated": sim, "real_streaming": real, "diff_pp": (real - sim) * 100}
        print(name, out[name], flush=True)
    out["h61_met"] = out["dev_h"]["diff_pp"] <= 0.1
    return out


# EP(끝점 지연)의 기준이다. 4단계 지연 스크립트의 MFA CSV 읽기 함수를 그대로 쓴다. 마지막 조각의 끝이 곧
# 마지막 어절의 끝이므로 어절 묶기는 필요 없다.
def last_word_end(name: str) -> dict[str, float]:
    """발화별 MFA 마지막 단어 끝 시각(초). 정렬이 있는 한글 전용 발화만 들어간다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from stage4_latency import load_alignment

    ids = json.loads((ROOT / f"work/mfa/corpus/{name}.ids.json").read_text(encoding="utf-8"))
    ends = {}
    for stem, uid in ids.items():
        path = ROOT / f"work/mfa/aligned/{name}/{stem}.csv"
        if path.exists():
            pieces = load_alignment(path)
            if pieces:
                ends[uid] = pieces[-1][1]
    return ends


# 세션은 서비스와 같은 StreamSession이고, 발화 뒤에 잡음 tail_s초를 붙여 패킷 단위로 흘린다(끝점이 닫힐 수
# 있게). 반환: 가설(세그먼트 최종 자막을 공백으로 이은 것), 첫 끝점 시각, (세그먼트의 부분 자막 목록, 최종
# 자막) 목록.
# reset: 끝점 뒤 인코더 캐시를 초기화할지(True) 유지할지(False).
def run_sessions(
    model, vocab, spec, utts, store, endpoint: EndpointConfig, reset: bool, device, tail_s: float
):
    """발화마다 StreamSession을 하나씩(뒤에 잡음을 붙여) 돌려 가설, 끝점 시각, 부분 자막을 돌려준다."""
    logmel = LogMel().to(device)
    hyps, first_endpoint, partial_runs = {}, {}, []
    for k, u in enumerate(utts):
        wav = torch.cat([store.waveform(u.id), noise(tail_s, seed=k)]).to(device)
        session = StreamSession(model, vocab, spec, logmel, endpoint, reset)
        finals, partials_by_segment = [], {}
        for pos, piece in packets(wav):
            for ev in session.push_samples(piece):
                if ev.kind == "partial":
                    partials_by_segment.setdefault(ev.segment, []).append(ev.text)
                # 첫 끝점 시각 = 그 끝점이 난 패킷의 끝 시각(오디오 기준). 빈 세그먼트를 닫는 endpoint
                # 이벤트도 첫 끝점에 포함한다(규칙 변경 24).
                elif ev.kind in ("final", "endpoint"):
                    if ev.kind == "final":
                        finals.append(ev)
                    first_endpoint.setdefault(u.id, (pos + piece.numel()) / SR)
        # 스트림 끝: 남은 부분 청크를 처리하고 열린 세그먼트를 닫는다.
        for ev in session.flush():
            finals.append(ev)
        hyps[u.id] = " ".join(f.text for f in finals)
        for f in finals:
            partial_runs.append((partials_by_segment.get(f.segment, []), f.text))
    return hyps, first_endpoint, partial_runs


# 정의는 규칙 변경 21에 등록했다. strict 판에서 append-only greedy의 미완 단어는 아직 덜 나온 상태일 뿐이라
# 0이 아니어도 '수정'은 아니다(실측 strict UPWR 0.85, UPSR 0.99는 진행 중인 단어가 있다는 뜻).
def stability(partial_runs) -> dict:
    """부분 자막 안정성(H6.3) 지표를 계산한다.

    UPWR: 부분 자막의 완결 단어(뒤에 공백이 온 단어. 마지막 미완 단어는 제외) 가운데 최종 자막의 같은 위치
    단어와 다른 것의 수 ÷ 최종 단어 수. UPSR: 그런 단어가 하나라도 있는 세그먼트 비율. 'strict' 판은 마지막
    미완 단어도 아직 최종 단어와 다르면 센다(append-only 디코더는 그 단어를 이어 늘리기만 한다).
    """
    unstable = strict = final_words = bad_segments = bad_strict = 0
    for partials, final in partial_runs:
        fw = final.split()
        final_words += len(fw)
        seg_bad = seg_strict = False
        for p in partials:
            words = p.split()
            complete = words if p.endswith(" ") else words[:-1]
            for i, w in enumerate(complete):
                if i >= len(fw) or fw[i] != w:
                    unstable += 1
                    seg_bad = True
            if words and not p.endswith(" "):
                i = len(words) - 1
                if i >= len(fw) or fw[i] != words[-1]:
                    strict += 1
                    seg_strict = True
        bad_segments += seg_bad
        bad_strict += seg_bad or seg_strict
    return {
        "upwr": unstable / max(1, final_words),
        "upsr": bad_segments / max(1, len(partial_runs)),
        "upwr_strict_trailing": (unstable + strict) / max(1, final_words),
        "upsr_strict_trailing": bad_strict / max(1, len(partial_runs)),
        "segments": len(partial_runs),
    }


# session: dev_mix 전수(발화 + 잡음 3 s)로 서비스 세션을 돌려 끝점 규칙과 캐시 정책을 등록 규칙대로 한 번
# 고르고 그 성능을 잰다(규칙 변경 21). 순서: 끝점 끈 기준 → rule2 0.8·1.2 s(캐시 초기화) → 고른 rule2에서 캐시
# 유지 → 정책 선택 → H6.3 → 서비스 경로 CER 점검.
def cmd_session(args, model, vocab, device) -> dict:
    spec = ChunkSpec(args.chunk, args.left)
    utts = read_manifest(ROOT / "manifests/dev_mix.jsonl")[: args.limit or None]
    store = PcmStore(ROOT / "data/pcm", "dev_mix")
    ends = last_word_end("dev_mix")
    # 끝점을 사실상 끈 설정(세 규칙의 시간을 아주 크게). 이 기준 세션 CER과 비교해 조기 끝점이 CER을 얼마나
    # 올리는지 본다.
    off = EndpointConfig(rule1_s=1e9, rule2_s=1e9, rule3_s=1e9)
    out = {
        "spec": f"chunk{args.chunk} left{args.left}",
        "tail_noise_s": args.tail_s,
        "noise_dbfs": NOISE_DBFS,
    }
    base_hyps, _, base_partials = run_sessions(
        model, vocab, spec, utts, store, off, True, device, args.tail_s
    )
    base = rate(score(utts, base_hyps))
    out["baseline_no_endpoint"] = {"cer_ns": base, **stability(base_partials)}
    print("baseline", out["baseline_no_endpoint"], flush=True)
    runs = {}
    for rule2 in (0.8, 1.2):
        hyps, ep, partials = run_sessions(
            model, vocab, spec, utts, store, EndpointConfig(rule2_s=rule2), True, device, args.tail_s
        )
        runs[rule2] = (hyps, ep, partials)
        cer = rate(score(utts, hyps))
        # EP = 첫 끝점 시각 − MFA 마지막 단어 끝(정렬 있는 발화만). EP50/EP90은 그 중앙값·90분위(ms).
        delays = [ep[u] - ends[u] for u in ep if u in ends]
        out[f"rule2_{rule2}_reset"] = {
            "cer_ns": cer,
            "cer_increase_pp": (cer - base) * 100,
            "ep50_ms": float(np.percentile(delays, 50) * 1000) if delays else None,
            "ep90_ms": float(np.percentile(delays, 90) * 1000) if delays else None,
            "ep_n": len(delays),
            **stability(partials),
        }
        print(rule2, out[f"rule2_{rule2}_reset"], flush=True)
    # rule2 선택: 기준 대비 CER 증가가 0.2포인트 이하인 것 중 짧은 쪽, 둘 다 넘으면 1.2 s(등록된 대체 규칙).
    # 실측은 둘 다 넘어 1.2 s가 됐다.
    ok = [r for r in (0.8, 1.2) if out[f"rule2_{r}_reset"]["cer_increase_pp"] <= 0.2]
    chosen = min(ok) if ok else 1.2
    out["rule2_chosen"] = chosen
    # 고른 rule2에서 끝점 뒤 캐시를 유지하는 실행. 초기화보다 CER이 낮으면 keep, 같거나 높으면 reset으로
    # 고정한다. 정책을 keep으로 고르면 EP·안정성도 keep 실행의 값을 보고한다(chosen; 규칙 변경 24).
    hyps, keep_ep, keep_partials = run_sessions(
        model, vocab, spec, utts, store, EndpointConfig(rule2_s=chosen), False, device, args.tail_s
    )
    keep = rate(score(utts, hyps))
    reset = out[f"rule2_{chosen}_reset"]["cer_ns"]
    keep_delays = [keep_ep[u] - ends[u] for u in keep_ep if u in ends]
    out["keep_cache"] = {
        "cer_ns": keep,
        "reset_cer_ns": reset,
        "ep50_ms": float(np.percentile(keep_delays, 50) * 1000) if keep_delays else None,
        "ep90_ms": float(np.percentile(keep_delays, 90) * 1000) if keep_delays else None,
        "ep_n": len(keep_delays),
        **stability(keep_partials),
    }
    out["cache_policy"] = "keep" if keep < reset else "reset"
    chosen_metrics = out["keep_cache"] if out["cache_policy"] == "keep" else out[f"rule2_{chosen}_reset"]
    out["chosen"] = {"rule2_s": chosen, "cache_policy": out["cache_policy"], **chosen_metrics}
    # H6.3: 선택 구성과 기준 모두 UPWR = 0(부분 자막이 한 번 낸 완결 단어를 고치지 않음).
    out["h63_met"] = chosen_metrics["upwr"] == 0 and out["baseline_no_endpoint"]["upwr"] == 0
    # 서비스 경로 CER 점검: 서비스 경로(끝점 끈 세션)의 CER이 h61에서 잰 dev_mix 실제 스트리밍 CER과
    # ±0.1포인트 안이어야 한다(등록 규칙). 부분집합(--limit) 실행에서는 비교하지 않는다.
    h61 = ROOT / "reports" / "stage6" / "h61.json"
    if h61.exists() and not args.limit:
        real = json.loads(h61.read_text(encoding="utf-8"))["dev_mix"]["real_streaming"]
        out["service_vs_measured"] = {
            "session_cer_ns": base,
            "real_streaming_cer_ns": real,
            "diff_pp": (base - real) * 100,
            "met": abs(base - real) * 100 <= 0.1,
        }
    return out


# drift: dev_H 발화를 0.5 s 잡음 간격으로 이어 --minutes(기본 10)분 길이 오디오를 만들고 한 세션으로 흘려,
# 분당 청크 계산 p50/p90, 약 30초마다 GPU 할당 메모리, 피크 메모리를 잰다. 긴 세션에서 계산 시간이나 메모리가
# 늘어나는지 확인하려는 측정이다(RoPE 절대 위치처럼 세션 길이에 따라 커지는 값이 있으므로;
# docs/survey/README.md §6의 드리프트 항목).
# 실시간 속도가 아니라 가능한 한 빠르게 처리한 결과다(규칙 변경 24의 표기). 0.5 s 간격에서는 rule2가 닫지 않아
# 세그먼트 대부분이 20 s 상한(rule3)으로 닫혔다.
# 서버 기본값이 바뀐 뒤(규칙 변경 26: graph step, 끝점 뒤 2차 패스, rule2 0.8 s, 캐시 유지)의 드리프트를
# 재려고 세션 옵션을 인자로 연다(규칙 변경 27 (B)): --step-path·--final-pass를 세션에 그대로 넘기고,
# --warm-up이면 서버 시작과 같은 예열(session.warm_up: 부분 청크 길이·2차 패스 길이 묶음·graph면 CUDA Graph
# 캡처)을 다른 세션으로 먼저 한 번 돌린 뒤 측정 세션을 만든다. 왜 예열인가: 하지 않으면 새 텐서 모양의 1회성
# 비용(GPU에서 새 2차 패스 길이를 처음 계산할 때 약 0.47 s, 그래프 캡처)이 측정 첫 분과 2차 패스 시간에 섞여
# 서버가 실제로 겪지 않는 값이 된다. 피크 메모리는 예열 뒤부터 잰다(측정 스트림 동안의 피크).
# 2차 패스를 켜면 닫힌 세그먼트마다의 2차 패스 계산 시간 p50/p90/max를 보고하고, graph 경로면 CUDA Graph 캡처
# 상태(model/streaming.graph_status)를 남긴다. 닫힌 세그먼트는 규칙별로도 센다(closes_by_rule: final·endpoint
# 모두; segments_closed는 6단계와 같이 final만). 기본값(eager, 2차 패스 없음, 예열 없음)은 6단계 측정과 같은
# 동작이다.
def cmd_drift(args, model, vocab, device) -> dict:
    spec = ChunkSpec(args.chunk, args.left)
    utts = read_manifest(ROOT / "manifests/dev_h.jsonl")
    store = PcmStore(ROOT / "data/pcm", "dev_h")
    pieces, total = [], 0.0
    for k, u in enumerate(utts):
        pieces += [store.waveform(u.id), noise(0.5, seed=10_000 + k)]
        total += u.duration_s + 0.5
        if total >= args.minutes * 60:
            break
    wav = torch.cat(pieces).to(device)
    endpoint = EndpointConfig(rule2_s=args.rule2)
    logmel = LogMel().to(device)

    def new_session() -> StreamSession:
        return StreamSession(
            model,
            vocab,
            spec,
            logmel,
            endpoint,
            args.policy == "reset",
            step_path=args.step_path,
            final_pass=args.final_pass,
        )

    warm_s = warm_up(new_session()) if args.warm_up else None
    session = new_session()
    per_minute, mem = {}, []
    finals = 0
    rules: Counter = Counter()
    pass_ms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # 패킷마다 이번에 새로 생긴 청크 계산 시간만 그 패킷의 분(minute)에 모으고, 30초 경계를 지나는 패킷에서
    # 메모리를 기록한다. 닫힘 이벤트(final·endpoint)마다 규칙과(2차 패스를 켰으면) 2차 패스 시간을 모은다.
    for pos, piece in packets(wav):
        n_before = len(session.stats.compute_ms)
        for ev in session.push_samples(piece):
            if ev.kind == "partial":
                continue
            finals += ev.kind == "final"
            rules[ev.rule] += 1
            if args.final_pass == "full":
                pass_ms.append(ev.final_pass_ms)
        minute = int(pos / SR // 60)
        per_minute.setdefault(minute, []).extend(session.stats.compute_ms[n_before:])
        if device.type == "cuda" and pos % (30 * SR) < PACKET:
            mem.append({"t_s": pos / SR, "allocated_mb": torch.cuda.memory_allocated() / 2**20})
    rows = {
        m: {"p50_ms": float(np.percentile(v, 50)), "p90_ms": float(np.percentile(v, 90)), "chunks": len(v)}
        for m, v in sorted(per_minute.items())
        if v
    }
    final_pass_ms = None
    if args.final_pass == "full":
        final_pass_ms = {"n": len(pass_ms), "p50": None, "p90": None, "max": None}
        if pass_ms:
            final_pass_ms.update(
                p50=float(np.percentile(pass_ms, 50)),
                p90=float(np.percentile(pass_ms, 90)),
                max=float(max(pass_ms)),
            )
    return {
        "audio_s": wav.numel() / SR,
        "segments_closed": finals,
        "per_minute": rows,
        "memory": mem,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "policy": args.policy,
        "rule2_s": args.rule2,
        "step_path": args.step_path,
        "final_pass": args.final_pass,
        "warm_up_s": None if warm_s is None else round(warm_s, 2),
        "closes_by_rule": dict(rules),
        "final_pass_ms": final_pass_ms,
        "graph": graph_status(model, spec) if args.step_path == "graph" else None,
    }


# 인자: command, --checkpoint, --name(결과 파일 이름, 기본 명령 이름), --chunk·--left(서비스 구성, 8·128),
# --timing-utts·--cpu-utts(grid 계산 시간 발화 수), --limit(앞 N발화만; 0 = 전수), --tail-s(세션 잡음 길이),
# --minutes·--rule2·--policy·--step-path·--final-pass·--warm-up(drift 설정), --device. 테스트가 기본값을 볼 수
# 있게 파서를 따로 만든다(drift의 새 인자 기본값 = 6단계 측정 동작).
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="6단계 스트리밍 서비스 측정 도구(docs/experiments.md 6단계·규칙 변경 27)"
    )
    p.add_argument("command", choices=["grid", "h61", "session", "drift"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--timing-utts", type=int, default=30)
    p.add_argument("--cpu-utts", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tail-s", type=float, default=3.0)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--rule2", type=float, default=1.2)
    p.add_argument("--policy", default="reset", choices=["reset", "keep"])
    p.add_argument(
        "--step-path",
        choices=["eager", "fast", "graph"],
        default="eager",
        help="drift: 세션의 인코더 step 경로(기본 eager = 6단계 측정 경로; 서버 GPU 기본값은 graph)",
    )
    p.add_argument(
        "--final-pass",
        choices=["none", "full"],
        default="none",
        help="drift: full이면 끝점마다 세그먼트를 전체 문맥으로 다시 디코딩하고(서버 기본값) 그 계산 시간을 "
        "보고한다(기본 none = 6단계 측정의 1-pass)",
    )
    p.add_argument(
        "--warm-up",
        action="store_true",
        help="drift: 측정 전에 서버 시작과 같은 예열(부분 청크 길이·2차 패스 길이 묶음·graph면 CUDA Graph "
        "캡처)을 다른 세션으로 한 번 돌린다. 1회성 비용이 측정에 섞이지 않게 한다(기본: 하지 않음 = 6단계 "
        "동작)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


# 하위 명령을 실행하고 결과에 체크포인트 경로·SHA-256·장치·구성·limit·소요 시간을 붙여 저장한다. 같은
# 체크포인트·구성·전수일 때만 결과를 다시 쓰거나 7단계 사전 검사에서 통과로 보게 하려는 출처 기록이다(규칙
# 변경 24·25). 결과 파일이 이미 있어도 덮어쓰므로 등록 결과와 다른 측정은 --name으로 이름을 바꾼다.
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    check_vocab(model, vocab)  # 어휘 토큰 수 ≠ 체크포인트 출력 크기면 측정 전에 멈춘다(잘못된 수치 기록 방지)
    t0 = time.perf_counter()
    result = {"grid": cmd_grid, "h61": cmd_h61, "session": cmd_session, "drift": cmd_drift}[args.command](
        args, model, vocab, device
    )
    import hashlib

    result.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(),
            "device": str(device),
            "spec": f"chunk{args.chunk} left{args.left}",
            "limit": args.limit,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    )
    out = ROOT / "reports" / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or args.command
    (out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote reports/stage6/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
