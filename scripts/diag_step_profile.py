"""탐색용 진단(6단계 판정에 쓰지 않음): GPU 스트리밍 인코더 step 하나의 시간이 어디에 쓰이는지 본다.

서비스 구성(청크 8, 좌측 128)으로 dev_H 몇 발화를 흘리며 step마다 ATen 연산 수와 CUDA 커널 수, CPU 쪽
벽시간(프로파일러를 켠 상태), GPU 커널 시간 합, GPU에서 fp16 autocast와 일반 fp32의 step 시간을 보고한다.
다른 GPU 작업이 없을 때만 돌린다.

    uv run python scripts/diag_step_profile.py
"""

# [왜 필요한가] 6단계 H6.2(320 ms 청크 계산 p90 ≤ 50 ms, GPU)가 63.3 ms로 미충족이었고, GPU 시간이 청크 크기와
# 무관하게 약 62 ms로 일정했다. 연산량이 아니라 다른 병목이 있는지 확인하려는 것이다.
# [결과] step 하나에 CUDA 커널 약 1,990개, GPU 커널 시간 합 5.5 ms. 프로파일러를 켠 벽시간(103 ms)은 보통
# 때(61 ms)보다 길어 정확한 분해는 아니지만, GPU는 대부분 쉬고 CPU 쪽 연산 디스패치·커널 실행 호출이
# 병목이라는 진단에 부합한다. fp32는 p50 51 ms로 autocast(61 ms)보다 빨랐다(형 변환 연산이 줄어서). 기록은
# reports/stage6/diag-step_profile.json, 해석은 docs/experiments.md 6단계 결과 H6.2 항목.
# [주의] CUDA 전용이며, 다른 GPU 작업이 없을 때만 돌린다(docstring).

from __future__ import annotations

import json
import time
from pathlib import Path

# numpy: step 시간 백분위 계산. torch.profiler: CPU·CUDA 활동을 함께 기록해 연산 수·커널 수·커널 시간을 센다.
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

# 프로젝트 내부 모듈: 서비스와 같은 캐시 스트리밍 인코더(StreamingEncoder), 평가와 같은 autocast
# 설정·체크포인트 로더, log-mel, 청크 설정, dev_H 데이터.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.model.streaming import StreamingEncoder

# 측정 대상은 최종 모델(4단계 TrimTail 상위 10개 평균)이다.
ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs/stage4/trimtail/avg10.pt"


# 발화마다 log-mel을 한 번에 계산해 인코더 청크 단위(mel 4 × chunk_size = 32프레임)로 자른 목록을 차례로
# 내준다.
# 특징 계산을 미리 끝내 두므로 아래 시간 측정에는 인코더 step만 들어간다. 마지막 조각은 32프레임보다 짧을 수
# 있다(부분 청크).
def chunks_of(model, store, utts, device, spec):
    logmel = LogMel().to(device)
    for u in utts:
        wav = store.waveform(u.id).to(device)
        mel, _ = logmel(wav[None], torch.tensor([wav.numel()], device=device))
        step = 4 * spec.chunk_size
        yield [mel[0, i : i + step] for i in range(0, mel.shape[1], step)]


# 주어진 정밀도(amp=True면 fp16 autocast, False면 fp32)로 발화들을 청크 단위로 흘려 step 시간 p50/p90을 잰다.
# 발화마다 새 스트림 상태로 시작한다. step 앞뒤로 GPU를 동기화해 그 step의 GPU 작업까지 벽시계에 넣는다(6단계
# 측정과 같은 방식, 규칙 변경 24). 앞 5 step은 초기화 비용이 섞이므로 버린다.
def timed(model, store, utts, device, spec, amp):
    enc = StreamingEncoder(model, spec)
    times = []
    for chunk_list in chunks_of(model, store, utts, device, spec):
        state = enc.new_state()
        for c in chunk_list:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device, amp):
                enc.step(c, state)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times[5:])  # 예열에 해당하는 앞 5 step은 버린다
    return {"p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)), "n": len(a)}


# fp16/fp32 step 시간을 비교하고, 정상 상태 20 step을 프로파일해 결과를 저장·출력한다. 서비스 구성(청크 8 =
# 320 ms, 좌측 128)과 dev_H 앞 10발화를 쓴다.
def main() -> None:
    device = torch.device("cuda")
    model, _ = load_model(CKPT, device)
    spec = ChunkSpec(8, 128)
    utts = read_manifest(ROOT / "manifests/dev_h.jsonl")[:10]
    store = PcmStore(ROOT / "data/pcm", "dev_h")
    # 같은 발화로 fp16 autocast와 fp32를 차례로 잰다. autocast는 연산마다 형 변환을 끼우므로, 연산 수가
    # 병목이면 fp32가 오히려 빠를 수 있다는 것을 확인하려는 비교다.
    out = {"fp16_autocast": timed(model, store, utts, device, spec, True)}
    out["fp32"] = timed(model, store, utts, device, spec, False)
    # 정상 상태 step 20개를 프로파일한다. 좌측 문맥 캐시(128프레임 = 8프레임 청크 16개)는 16청크 뒤에 가득
    # 차므로, 프로파일용 스트림에 첫 발화의 앞 16청크를 먼저 흘려 캐시를 채운 뒤 정상 상태의 step만 잰다.
    enc = StreamingEncoder(model, spec)
    chunk_list = next(
        chunks_of(model, store, read_manifest(ROOT / "manifests/dev_h.jsonl")[:1], device, spec)
    )
    state = enc.new_state()
    with autocast(device, True):
        for c in chunk_list[:16]:
            enc.step(c, state)
    # 16번째 청크부터 20 step을 고른다. 발화가 짧아도 20 step이 나오도록 청크 목록을 세 번 이어 붙인다. 이때
    # 발화의 마지막 짧은 조각이 중간에 끼일 수 있지만, 판정에 쓰지 않는 시간 분해용 진단이다.
    steady = (chunk_list * 3)[16:36]
    torch.cuda.synchronize()
    # CPU와 CUDA 활동을 함께 기록한다. 프로파일러 자체의 부담 때문에 이 구간의 벽시간은 보통 때보다
    # 길다(결과에서 103 ms vs 61 ms).
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for c in steady:
            with autocast(device, True):
                enc.step(c, state)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000 / len(steady)
    # n_aten: 다른 연산 안에서 불린 것을 뺀 최상위 aten 연산 호출 수(20 step 합계, 참고값). kernels: GPU에서
    # 실제로 실행된 커널 이벤트 목록. gpu_ms: 커널 시간 합(마이크로초 → ms)을 step 수로 나눈 값으로,
    # 벽시간과의 차이가 GPU가 쉬는 시간이다.
    events = prof.key_averages()
    n_aten = sum(e.count for e in events if e.key.startswith("aten::") and e.cpu_parent is None)
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_ms = sum(e.device_time for e in kernels) / 1000 / len(steady)
    out["profile"] = {
        "steps": len(steady),
        "wall_ms_per_step_profiled": wall,
        "cuda_kernels_per_step": len(kernels) / len(steady),
        "gpu_kernel_ms_per_step": gpu_ms,
        # CPU 시간이 가장 큰 연산 15개와 step당 호출 수·CPU 시간. 어느 연산의 디스패치가 시간을 쓰는지 보려는
        # 것이다.
        "top_aten_ops": [
            {
                "op": e.key,
                "calls_per_step": e.count / len(steady),
                "cpu_ms_per_step": e.cpu_time_total / 1000 / len(steady),
            }
            for e in sorted(events, key=lambda e: -e.cpu_time_total)[:15]
        ],
    }
    out["aten_top_level_calls_hint"] = n_aten
    path = ROOT / "reports/stage6/diag-step_profile.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1))
    print(json.dumps({k: v for k, v in out["profile"].items() if k != "top_aten_ops"}, indent=1))
    for r in out["profile"]["top_aten_ops"]:
        print(r)


if __name__ == "__main__":
    main()
