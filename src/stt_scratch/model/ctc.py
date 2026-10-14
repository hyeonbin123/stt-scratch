"""fp32 log-확률로 계산하는 CTC 손실들(docs/experiments.md "fp16 규칙").

학습 손실 전부를 둔다: CTC 손실(최종·중간 헤드), 둘을 합치는 total_loss, 3단계 (d)의 CR-CTC 일관성 항,
4단계의 Peak-First 정규화 항.
"""

# [fp32 규칙] 모델은 fp16 autocast로 돌지만 손실 입력은 항상 log_softmax(dtype=float32)로 만든다. ctc_loss는
# CUDA autocast의 fp32 자동 승격 목록에 없어 fp16 텐서를 넘기면 오류가 나고(이 PC에서 재현), CTC는 수치
# 정밀도에 민감하다는 조사 결과에 따른 규칙이다(docs/experiments.md "fp16 규칙",
# docs/survey/03-objectives-decoding-units.md §5). 단위 테스트가 이 dtype을 고정한다(tests/test_model.py
# test_ctc_input_is_fp32_under_autocast).
# [연결] train.py가 total_loss·cr_consistency·peak_first_regularizer를, evaluate.ctc_loss_manifest가 dev CTC
# 손실(발산 판정)을 위해 ctc_loss를, streaming.py가 log_probs_fp32를 쓴다.

from __future__ import annotations

import torch
from torch import nn


# 손실뿐 아니라 스트리밍 출력(streaming.StreamingEncoder.step)과 서비스 끝점 판정의 blank 확률도 이 값을
# 쓴다.
def log_probs_fp32(logits: torch.Tensor) -> torch.Tensor:
    """autocast 여부와 무관하게 float32로 log_softmax를 계산한다. ctc_loss에 넘기는 텐서는 이것뿐이다."""
    return nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)


# 입력: 로짓 (B, T, V), 발화별 출력 프레임 수, 이어 붙인 1차원 정답 토큰, 발화별 정답 길이. 출력: 발화별
# 손실의 합(스칼라). 평균은 호출자가 B로 나눠 낸다.
# 아래 docstring의 두 안전장치는 1단계 첫 시도의 세 런이 모두 100~120 step에서 멈춘 원인을 추적해 찾은 CUDA
# CTC backward 문제의 수정이다. 같은 배치를 fp32로 돌려도 NaN이 나서 fp16 문제가 아니었고, 발화별 기준
# 기울기와 비교해 원인을 확인했다(docs/ai-verification.md "1단계 첫 시도 발산 → CTC 커널 버그"). 회귀 테스트는
# tests/test_model.py test_ctc_loss_matches_per_utterance_reference.
def ctc_loss(
    logits: torch.Tensor, out_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    """배치 안 발화별 CTC 손실의 합(blank = 0, zero_infinity).

    PyTorch 기본 CUDA CTC backward(torch 2.11+cu128, RTX 2080 Ti)에 대한 안전장치 두 가지가 있으며, 둘 다
    발화별 기준 기울기와 비교해 찾았다.
    1. log_probs는 연속 메모리의 (T, B, V) 텐서여야 한다. 전치한 뷰를 넘기면 유효 프레임에서도 기준과 최대
       55까지 다른 기울기가 (비결정적으로) 나왔고 때로 NaN이 났다.
    2. 발화별 out_length 이상의 프레임은 torch.where로 바꿔 넣는다. 커널이 그 위치의 기울기를 초기화하지
       않고 남기는데, torch.where의 backward가 그쪽으로 오는 값을 버린다.
    정답은 패딩한 (B, S) int32 CPU 텐서로, 길이는 int32 CPU 텐서로 넘긴다. 이 배치는 기준과 1e-7까지
    맞았고, 1차원 정답 + CUDA int32 길이 조합은 1e-3 차이가 났다.
    """
    lp = log_probs_fp32(logits)  # (B, T, V)
    # 안전장치 2: 발화별 출력 길이 뒤의 패딩 프레임 값을 torch.where로 0으로 바꾼다. CUDA 커널은 그 위치의
    # 기울기를 초기화하지 않고 남기는데, torch.where의 backward는 선택되지 않은 쪽(원래 로짓)으로 기울기를
    # 보내지 않으므로 쓰레기 값이 모델로 역전파되지 않는다.
    valid = torch.arange(lp.shape[1], device=lp.device)[None, :] < out_lengths[:, None]
    lp = torch.where(valid[:, :, None], lp, torch.zeros((), dtype=lp.dtype, device=lp.device))
    # 안전장치 1: ctc_loss가 요구하는 (T, B, V) 배치를 전치 '뷰'가 아니라 연속 메모리 텐서로 만들어 넘긴다.
    # 비연속 뷰를 넘기면 유효 프레임에서도 기준과 최대 55 차이 나는 비결정적 기울기와 NaN이 나왔다. assert로
    # fp32·연속 조건을 실행 중에도 확인한다.
    lp = lp.transpose(0, 1).contiguous()  # (T, B, V)
    assert lp.dtype == torch.float32 and lp.is_contiguous()
    # 정답은 (B, S) 패딩 int32 CPU 텐서로, 길이는 int32 CPU 텐서로 넘긴다. 1차원 정답 + CUDA int32 길이 조합은
    # 기준과 1e-3, 이 패딩 정답 방식은 약 1e-7 차이였다(docstring). 그래서 이어 붙인 정답을 발화별 행으로 다시
    # 펼친다.
    tl_cpu = target_lengths.to(torch.int32).cpu()
    flat = targets.to(torch.int32).cpu()
    padded_targets = torch.zeros(lp.shape[1], int(tl_cpu.max()), dtype=torch.int32)  # (B, S) 모양
    off = 0
    for b, n in enumerate(tl_cpu.tolist()):
        padded_targets[b, :n] = flat[off : off + n]
        off += n
    # reduction="sum": 발화별 손실을 더해 total_loss에서 발화 수로 나눈다(토큰 수로 나누지 않으므로 긴 발화의
    # 손실이 더 크게 반영된다). zero_infinity=True: 출력 프레임이 정답보다 모자라 손실이 무한대가 되는 경우 그
    # 손실과 기울기를 0으로 처리하는 안전장치다(학습 데이터는 전처리에서 CTC 길이 조건으로 이미 걸렀다).
    return nn.functional.ctc_loss(
        lp,
        padded_targets,
        out_lengths.to(torch.int32).cpu(),
        tl_cpu,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )


# 3단계 (d)에서 썼다. train.py가 같은 배치를 두 번 이어 붙여(2B) 서로 다른 SpecAugment 마스크로 한 번에
# forward하면, 앞 절반과 뒤 절반의 프레임별 분포가 서로 비슷해지도록 양방향 KL을 건다. 각 방향에서 '목표'
# 쪽은 detach(sg)해 그쪽으로는 기울기를 보내지 않는다.
# 결과: 2e-3·1e-3에서 정렬이 시작되지 않아 등록 규칙대로 5e-4로 재시작했고, dev_mix 개선 없음·dev_H 8.65포인트
# 악화로 채택하지 않았다(docs/experiments.md 3단계 결과 (d)).
def cr_consistency(logits: torch.Tensor, out_lengths: torch.Tensor) -> torch.Tensor:
    """CR-CTC 일관성 항(arXiv 2410.05101). logits (2B, T, V)에는 증강한 두 뷰가 뷰 1부터 들어 있다.

    0.5 * [KL(sg(p1) || p2) + KL(sg(p2) || p1)]을 유효 프레임과 클래스에 대해 더하고 발화 수로 평균한다
    (ctc_loss / B와 같은 정규화). fp32로 계산한다.
    """
    lp = log_probs_fp32(logits)
    b = lp.shape[0] // 2
    lp1, lp2 = lp[:b], lp[b:]
    valid = torch.arange(lp.shape[1], device=lp.device)[None, :] < out_lengths[:b, None]
    w = valid[:, :, None].to(lp.dtype)
    # KL(p‖q) = Σ p·(log p − log q). log-확률 차이를 직접 쓰므로 0 확률에서도 log 0 계산이 없다. 유효 프레임
    # 가중치 w로 패딩을 빼고, 발화 수 b로 나눠 ctc_loss / B와 같은 척도로 맞춘다.
    t1, t2 = lp1.detach(), lp2.detach()
    kl12 = (t1.exp() * (t1 - lp2) * w).sum()
    kl21 = (t2.exp() * (t2 - lp1) * w).sum()
    return 0.5 * (kl12 + kl21) / b


# 4단계에서 썼다. train.py가 --peak-first-lambda λ를 곱해 손실에 더한다(λ = 1.62e-3,
# scripts/calibrate_peak_first.py로 정규화 항이 최종 헤드 CTC 손실의 약 10 %가 되게 고정).
# 결과: FTD P50을 50 ms 줄였지만 dev_mix CER이 상대 3.16 % 늘어 조건(≤ 3 %)을 넘어 채택하지
# 않았다(docs/experiments.md 4단계 결과).
def peak_first_regularizer(logits: torch.Tensor, out_lengths: torch.Tensor) -> torch.Tensor:
    """Peak-First CTC 정규화 항(arXiv 2211.03284): 유효한 t < T-1에 대한 KL(sg(p_{t+1}) || p_t)의 합.

    모든 프레임의 분포를 다음 프레임의 분포(detach) 쪽으로 당기므로 CTC 피크가 더 이른 프레임으로 옮겨진다.
    프레임과 클래스에 대해 더하고 발화 수로 평균한다(ctc_loss / B와 같은 정규화). fp32로 계산한다.
    """
    lp = log_probs_fp32(logits)
    nxt = lp[:, 1:].detach()
    cur = lp[:, :-1]
    t = torch.arange(lp.shape[1] - 1, device=lp.device)[None, :]
    # t와 t+1이 모두 유효한 프레임 쌍만 센다(발화 끝 프레임은 목표로만 쓰이고 기울기를 받지 않는다).
    w = (t + 1 < out_lengths[:, None]).to(lp.dtype)[:, :, None]
    return (nxt.exp() * (nxt - cur) * w).sum() / lp.shape[0]


# Run A는 w = 0.5(3·6·9층), 최종 모델은 w = 0.3(6층 하나)이다. 각 손실은 발화 수로 나눠 '발화당 평균
# 손실'이 된다. 로그용으로 각 항의 float 값을 parts에 담아 돌려준다(train.jsonl에 기록).
# 호출 위치: train.py 학습 루프, tests/test_model.py.
def total_loss(
    final_logits: torch.Tensor,
    inter_logits: list[torch.Tensor],
    out_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    sc_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """최종·중간 헤드 손실을 합친다: (1 - w) * 최종 + w * mean(중간), 발화 수로 정규화한다."""
    b = final_logits.shape[0]
    final = ctc_loss(final_logits, out_lengths, targets, target_lengths) / b
    parts = {"ctc_final": float(final.detach())}
    # 중간 헤드가 없거나 가중치가 0이면 최종 CTC만 쓴다.
    if inter_logits and sc_weight > 0:
        inter = torch.stack(
            [ctc_loss(lg, out_lengths, targets, target_lengths) / b for lg in inter_logits]
        ).mean()
        parts["ctc_inter"] = float(inter.detach())
        loss = (1.0 - sc_weight) * final + sc_weight * inter
    else:
        loss = final
    parts["loss"] = float(loss.detach())
    return loss, parts
