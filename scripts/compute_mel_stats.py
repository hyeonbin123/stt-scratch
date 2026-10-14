"""학습 매니페스트 전체의 log-mel 전역 평균·표준편차를 구해 manifests/mel_stats.json에 쓴다.

학습 매니페스트의 모든 발화에서 80차 log-mel의 채널별 전역 평균·표준편차를 구한다. 속도 교란은 걸지
않으므로 원래 음성 기준의 통계다. 0단계 데이터 준비의 마지막 명령이다(README "데이터 준비").

사용법: uv run python scripts/compute_mel_stats.py [--manifest train]
"""

# [왜 필요한가] 모델은 입력 log-mel을 이 통계로 정규화한다(ChunkConformerCTC.normalize_mel). 발화마다 자기
# 평균을 빼는 방식은 발화 전체(미래 프레임)를 봐야 해서 스트리밍과 맞지 않으므로, 학습 세트 전체에서 한 번
# 구한 고정값을 쓴다. 이 값은 모델 설정(ModelConfig.mel_mean·mel_std)에 들어가 체크포인트와 함께 저장되므로
# 학습·평가·서비스가 같은 정규화를 쓴다.
# [흐름] 발화마다 log-mel 계산 → 유효 프레임만 float64로 합과 제곱합 누적 → 평균과 표준편차 계산 → JSON 저장.
# [연결] 입력은 scripts/prepare_data.py가 만든 매니페스트와 PCM 저장소. 출력은 train.py(모델 생성)와
# scripts/bench_stage0.py가 읽는다.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# torch: log-mel 계산과 누적(GPU가 있으면 GPU에서).
import torch

# 프로젝트 내부 모듈: 매니페스트 읽기와 int16 PCM 저장소(data.py), 학습·추론과 같은 프레임 규칙(25 ms 창, 10
# ms 간격, center=False)의 LogMel(features.py). 같은 LogMel을 써야 통계가 실제 모델 입력과 맞는다.
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.model.features import LogMel

# 저장소 루트. manifests/와 data/pcm/을 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# 통계를 계산해 저장한다. --manifest(기본 train)로 다른 매니페스트를 지정할 수 있지만 출력 파일 이름은 항상
# manifests/mel_stats.json이라 train이 아닌 값을 주면 학습용 통계를 덮어쓴다.
# 속도 교란은 걸지 않는다(원래 음성 기준 통계, docstring 참조).
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="train")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utts = read_manifest(ROOT / "manifests" / f"{args.manifest}.jsonl")
    store = PcmStore(ROOT / "data" / "pcm", args.manifest)
    logmel = LogMel().to(device)
    # 채널(80)별 합과 제곱합. 학습 세트 전체(약 52 h, 초당 100프레임)의 프레임을 모두 더하므로 float64로
    # 누적해 반올림 오차와 분산 계산의 상쇄 오차를 줄인다.
    total = torch.zeros(80, dtype=torch.float64, device=device)
    total_sq = torch.zeros(80, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for i, u in enumerate(utts):
            # 발화 하나씩 처리한다(배치 패딩이 없으므로 패딩 프레임이 통계에 섞일 일이 없다).
            wav = store.waveform(u.id).to(device)
            mel, frames = logmel(wav[None], torch.tensor([wav.numel()], device=device))
            # 유효 프레임만 잘라 float64로 바꾼 뒤 채널별 합·제곱합과 프레임 수를 더한다.
            m = mel[0, : int(frames[0])].double()
            total += m.sum(0)
            total_sq += (m * m).sum(0)
            count += m.shape[0]
            # 오래 걸리는 작업이라 5,000발화마다 진행 상황을 찍는다.
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(utts)} utterances, {count} frames", flush=True)
    # 평균과 표준편차: Var = E[x²] − (E[x])². 아주 작은 음수·0이 나와도 sqrt와 나눗셈이 안전하도록 1e-8로
    # 하한을 둔다.
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(1e-8).sqrt()
    # 어느 매니페스트에서 몇 발화·몇 프레임으로 구했는지도 함께 남긴다.
    out = {
        "manifest": args.manifest,
        "n_utts": len(utts),
        "n_frames": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (ROOT / "manifests" / "mel_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("mean range", float(mean.min()), float(mean.max()), "std range", float(std.min()), float(std.max()))


if __name__ == "__main__":
    main()
