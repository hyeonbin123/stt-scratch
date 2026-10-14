"""학습 런의 epoch 체크포인트 가운데 dev 성적이 좋은 k개를 골라 가중치를 평균한 모델 파일을 만든다.

학습 런 폴더(outputs/<단계>/<런>)에 매 epoch 저장된 체크포인트 가운데 dev 세트(기본 dev_mix)의 full context
CER_ns가 가장 낮은 k개(기본 10개, docs/experiments.md 등록값)를 골라 가중치를 산술 평균하고, 그 결과를
모델 파일(보통 avg10.pt)로 저장한다.

사용법: uv run python scripts/average_checkpoints.py --run outputs/stage2/runA --k 10 --out <path.pt>
"""

# [왜 필요한가] 사전 등록 규칙이 "체크포인트는 매 epoch 저장, 평가 모델은 dev_mix 상위 10개
# 평균"이다(docs/experiments.md "선택 규칙"). 2~4단계의 보고·비교와 6·7단계 측정에 쓴 모델이 모두 이 평균
# 모델이다(최종 모델 outputs/stage4/trimtail/avg10.pt 포함).
# [흐름] epochs.jsonl에서 epoch별 dev CER을 읽는다 → 낮은 순으로 k개를 고른다 → epoch-NNN.pt를 차례로 불러
# float64로 더한다 → 개수로 나눠 float32로 되돌려 저장한다.
# [연결] 입력 파일(epochs.jsonl, epoch-NNN.pt)은 src/stt_scratch/train.py가 쓴다. README의 2단계 명령에서 직접
# 실행하고, scripts/stage3.py와 scripts/stage4.py가 하위 프로세스로 부른다. 출력은
# stt_scratch.evaluate.load_model(평가)과 train.py --init(4단계 이어 학습 시작점)이 "model"·"config" 키로
# 읽는다.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# torch: 체크포인트(.pt) 읽기·쓰기와 텐서 덧셈·나눗셈에 쓴다.
import torch

# 저장소 루트. --run과 --out을 저장소 기준 상대 경로로 받으려고 이 파일 위치(scripts/의 부모)에서 계산한다.
# 어느 폴더에서 실행해도 같은 경로를 가리킨다.
ROOT = Path(__file__).resolve().parents[1]


# 인자를 읽어 상위 k개 평균 체크포인트를 저장한다.
# 인자: --run(런 폴더), --k(평균할 개수, 등록값 10), --dev(순위를 매길 dev 세트 이름, 기본 dev_mix),
# --out(저장 경로).
# 출력: {"model": 평균 가중치, "config": 모델 구조 설정, "averaged": 고른 epoch 목록, "dev": 순위 기준 세트}.
# 어떤 epoch를 평균했는지 파일 안에 남겨 보고서가 그 목록을 그대로 옮길 수 있다.
# 왜 평균인가: 여러 epoch 가중치를 평균하는 것은 마지막 몇 epoch의 흔들림을 줄이려고 음성 인식 레시피에서 흔히
# 쓰는 방법이다. 이 프로젝트에서는 "어떤 모델로 평가하는가"를 측정 전에 고정하는 규칙으로 썼다.
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--dev", default="dev_mix")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run = ROOT / args.run
    # epochs.jsonl은 한 줄에 epoch 하나의 요약(JSON)이다. 빈 줄은 건너뛴다.
    epochs = [
        json.loads(line) for line in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    # (dev CER, epoch) 짝을 만든다. 발산 판정이나 --max-steps로 멈춘 epoch의 요약에는 train.py가 "dev"를 넣지
    # 않으므로 여기서 자연히 빠진다. 튜플로 정렬하므로 CER이 같으면 앞 epoch가 먼저 온다.
    scored = [
        (e["dev"][args.dev]["full"], e["epoch"]) for e in epochs if "dev" in e and e["dev"].get(args.dev)
    ]
    # CER이 낮은 순으로 k개. 기록된 epoch가 k개보다 적으면 있는 만큼만 쓴다.
    chosen = sorted(scored)[: args.k]
    # epoch 번호는 train.py가 0부터 센 값이고 파일 이름은 세 자리(epoch-089.pt 등)다. 보고서의 "1부터 센
    # epoch로는 각 +1"이라는 문구가 이 때문이다(docs/experiments.md 2단계 결과).
    paths = [run / f"epoch-{ep:03d}.pt" for _, ep in chosen]
    avg = None
    config = None
    for path in paths:
        # map_location="cpu": GPU를 쓰지 않고 CPU 메모리에서 평균한다. weights_only=True: 피클 안의 임의
        # 객체를 실행하지 않고 텐서·기본 자료형만 읽는 안전한 로드 방식이다.
        state = torch.load(path, map_location="cpu", weights_only=True)
        # 첫 체크포인트로 누적 버퍼를 만든다. 부동소수 텐서는 float64로 바꿔 더해야 float32를 10번 더할 때의
        # 반올림 오차가 쌓이지 않는다. 부동소수가 아닌 텐서(정수 버퍼 등)는 평균할 수 없으므로 첫 체크포인트
        # 값을 그대로 둔다. 모델 구조 설정(config)은 같은 런 안에서 모두 같으므로 첫 것을 쓴다.
        if avg is None:
            avg = {
                k: v.clone().double() if v.is_floating_point() else v.clone()
                for k, v in state["model"].items()
            }
            config = state["config"]
        else:
            # 두 번째 체크포인트부터는 부동소수 텐서만 누적 버퍼에 더한다.
            for k, v in state["model"].items():
                if v.is_floating_point():
                    avg[k] += v.double()
    # dev 기록이 있는 epoch가 하나도 없으면 평균할 것이 없으므로 여기서 멈춘다.
    assert avg is not None, "no checkpoints found"
    # 합을 개수로 나눠 평균을 만들고 학습·추론에서 쓰는 float32로 되돌린다.
    for k, v in avg.items():
        if v.is_floating_point():
            avg[k] = (v / len(paths)).float()
    out = ROOT / args.out
    # 옵티마이저·GradScaler 상태는 넣지 않는다. 평균 모델은 평가와 이어 학습의 시작 가중치로만 쓰이고, 이어
    # 학습(--init)은 옵티마이저·스케줄을 새로 시작하기 때문이다(docs/experiments.md 규칙 변경 16).
    torch.save({"model": avg, "config": config, "averaged": [ep for _, ep in chosen], "dev": args.dev}, out)
    print("averaged epochs", [ep for _, ep in chosen], "->", out)


if __name__ == "__main__":
    main()
