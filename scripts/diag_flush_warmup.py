"""탐색용 진단(6단계 판정에 쓰지 않음): 서버 시작 직후 flush가 느린 것이 일회성 비용인지 확인한다.

서버를 띄운 직후 몇몇 발화에서 flush(마지막 부분 청크 처리)가 약 0.5 s씩 걸린 현상이 '새로운 부분 청크
길이를 처음 처리할 때 드는 일회성 비용'인지 본다. 한 프로세스에서 dev_H 앞 50발화를 종단 클라이언트가 보낸
순서 그대로 세션에 흘리며 session.flush()(와 그 안의 인코더 step) 시간을 잰 뒤, 같은 50발화를 두 번째
패스로 한 번 더 돌린다. 첫 번째 패스에서 처음 나온 길이만 느리고 두 번째 패스에서는 모두 빠르면 일회성
비용이다.

    uv run python scripts/diag_flush_warmup.py
"""

# [결과와 조치] 마지막 부분 청크의 인코더 프레임 수가 처음 나올 때마다 한 번 약 0.47 s, 다시 오면 약 62
# ms였다. 그래서 서버 시작 때 가능한 부분 청크 길이(mel 1~31프레임)를 한 번씩 flush하는
# 예열(service/session.py warm_up)을 넣었다(docs/experiments.md "측정 뒤 서비스 수정",
# docs/ai-verification.md). 결과 파일은 reports/stage6/diag-flush_warmup.json.
# [주의] 이 스크립트는 warm_up을 부르지 않는다(예열 없는 상태를 재현하려는 것). 함수 없이 위에서 아래로
# 실행되는 일회성 진단이고 CUDA가 필요하다.

import json
import statistics as st
import time
from pathlib import Path

# torch: GPU 동기화와 시간 측정.
import torch

# 프로젝트 내부 모듈: 서버와 똑같은 스트리밍 세션(StreamSession)과 끝점 설정, 체크포인트 로더, log-mel, 청크
# 설정, dev_H 매니페스트·PCM 저장소. 서버 코드 경로를 그대로 써야 서버에서 본 현상을 같은 프로세스 안에서
# 재현할 수 있다.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import load_model
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

# 측정 조건: 최종 모델(4단계 TrimTail 평균), GPU, dev_H 앞 50발화(종단 클라이언트 측정과 같은 발화·순서),
# 6단계에서 정한 끝점 rule2 1.2 s.
ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda")
model, _ = load_model(ROOT / "outputs/stage4/trimtail/avg10.pt", device)
vocab = Vocab.load(ROOT / "manifests/vocab.json")
logmel = LogMel().to(device)
utts = read_manifest(ROOT / "manifests/dev_h.jsonl")[:50]
store = PcmStore(ROOT / "data/pcm", "dev_h")
endpoint = EndpointConfig(rule2_s=1.2)
rows = []
# 두 번 돈다. rep 0은 '처음 보는 길이'가 섞인 패스, rep 1은 모든 길이를 이미 한 번 본 뒤의 기준 패스다.
for rep in range(2):
    for k, u in enumerate(utts):
        # PCM 저장소의 int16 표본을 그대로 바이트로 꺼낸다. 클라이언트가 WebSocket으로 보내는 little-endian
        # int16과 같은 형식이다.
        offset, length = store.index[u.id]
        pcm = bytes(memoryview(store.bin[offset : offset + length]).cast("B"))
        # 발화마다 새 세션: 청크 8(320 ms)·좌측 128, 마지막 인자 False = 끝점 뒤 캐시 유지(6단계에서 고른
        # 서비스 구성).
        s = StreamSession(model, vocab, ChunkSpec(8, 128), logmel, endpoint, False)
        # 패킷 규약(docs/experiments.md 규칙 변경 9): 첫 패킷 5,360 샘플, 이후 5,120 샘플(320 ms). 바이트로는
        # 2배다. 이렇게 보내야 패킷마다 mel 32프레임 = 인코더 청크 1개가 나온다.
        i, size = 0, 5360 * 2
        while i < len(pcm):
            s.push_pcm16(pcm[i : i + size])
            i += size
            size = 5120 * 2
        # flush 직전 버퍼에 남은 mel 프레임 수 = 마지막 부분 청크 길이(0~31). 이 길이가 flush에서 인코더 입력
        # 모양을 정한다.
        pending = int(s.pending.shape[0])
        # 앞서 쌓인 GPU 작업을 끝내고 재야 flush 시간에 다른 작업이 섞이지 않는다.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev = s.flush()
        torch.cuda.synchronize()
        rows.append(
            {
                "pass": rep,
                "k": k,
                "pending_mel": pending,
                "flush_ms": (time.perf_counter() - t0) * 1000,
                # flush가 낸 최종 이벤트에 담긴 인코더 step 시간. 텍스트가 비어 이벤트가 없으면 None이다.
                "step_ms": ev[0].compute_ms if ev and hasattr(ev[0], "compute_ms") else None,
            }
        )
# 두 패스를 통틀어 각 행의 부분 청크 길이(mel 프레임 수)가 처음 나온 것인지 표시한다. rep 1의 행은 rep 0에서
# 이미 본 길이라 모두 False가 된다.
seen = set()
for r in rows:
    r["new_length"] = r["pending_mel"] not in seen
    seen.add(r["pending_mel"])
# 발화별 결과를 저장하고 앞 15행을 확인용으로 찍는다.
out = ROOT / "reports/stage6/diag-flush_warmup.json"
out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
for r in rows[:15]:
    print(r)

# 패스별 요약: 전체 중앙값·최대, 처음 나온 길이와 이미 본 길이 각각의 개수·중앙값. 처음 나온 길이만 느리면
# 일회성 비용이라는 해석이 맞다.
for rep in (0, 1):
    a = [r["flush_ms"] for r in rows if r["pass"] == rep]
    new = [r["flush_ms"] for r in rows if r["pass"] == rep and r["new_length"]]
    old = [r["flush_ms"] for r in rows if r["pass"] == rep and not r["new_length"]]
    print(
        "pass",
        rep,
        "median",
        round(st.median(a), 1),
        "max",
        round(max(a), 1),
        "new-length n",
        len(new),
        "median",
        round(st.median(new), 1) if new else None,
        "seen-length n",
        len(old),
        "median",
        round(st.median(old), 1) if old else None,
    )
