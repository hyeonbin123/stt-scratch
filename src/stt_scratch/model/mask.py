"""청크 경계 어텐션 마스크(U2식)와 동적 청크 샘플링.

청크 단위 스트리밍을 학습에서 흉내 내는 마스크와, 배치마다 청크 설정을 뽑는 샘플러를 둔다.
프레임 i는 chunk(j) <= chunk(i)이고 chunk(j) >= chunk(i) - left_chunks이며 j가 유효(패딩 아님) 프레임일
때만 프레임 j를 본다. 모든 층이 같은 경계를 쓰므로 층이 깊어져도 look-ahead가 쌓이지 않는다. 한 프레임은
많아야 자기 청크의 끝까지만 본다.
"""

# [왜 필요한가] 스트리밍 추론에서는 현재 청크까지의 음성만 있다. 학습 때 같은 경계의 마스크를 걸면 모델이
# 미래를 보지 않는 법을 배우고, 전체 발화 + 청크 마스크 forward의 결과가 캐시 스트리밍 결과와 수학적으로
# 같아진다(tests/test_model.py의 미래 누출·스트리밍 == 오프라인 테스트).
# [동적 청크] 배치마다 전체 문맥과 여러 청크 크기를 섞어 학습하면 모델 하나로 오프라인과 여러 지연 설정을 모두
# 쓸 수 있다(U2, docs/survey/README.md "결정" 스트리밍 학습). 2단계 H2.1에서 이 통합 모델(Run A)이 전체 문맥
# 전용 통제군(Run B)보다 dev_mix에서 나빠지지 않았다(22.05 vs 22.90, 단 두 런의 LR이 다름; docs/experiments.md
# 2단계 결과).
# [연결] train.py가 sample_chunk_spec으로 배치마다 ChunkSpec을 뽑고, conformer.ChunkConformerCTC.forward가
# attention_mask를 만든다. ChunkSpec은 평가·스트리밍·서비스 전체에서 '청크 크기·좌측 문맥' 설정을 나르는 값
# 객체로 쓰인다.

from __future__ import annotations

# random: train.py가 시드를 준 random.Random을 넘겨 배치별 청크 설정을 재현 가능하게 뽑는다.
import random
from dataclasses import dataclass

import torch


# 청크 설정 값 객체. chunk_size는 인코더 프레임(25 fps, 1프레임 40 ms) 단위이며 0 이하는 전체 문맥,
# left_context는 프레임 단위이며 음수는 무제한이다. 예: 서비스 구성 ChunkSpec(8, 128) = 320 ms 청크, 좌측 5.12
# s. frozen이라 여러 곳에서 공유해도 바뀌지 않는다.
@dataclass(frozen=True)
class ChunkSpec:
    chunk_size: int  # 25 fps 프레임 단위. 0 이하면 전체 문맥
    left_context: int  # 프레임 단위. 음수면 무제한

    @property
    def full(self) -> bool:
        return self.chunk_size <= 0

    # 좌측 문맥을 청크 개수로 바꾼 값(예: 128 ÷ 8 = 16청크). 마스크는 청크 단위로 자르므로, 한 프레임이 실제로
    # 보는 과거는 '이전 16청크 전체 + 자기 청크 안의 앞부분'이다. -1은 무제한.
    @property
    def left_chunks(self) -> int:
        if self.full or self.left_context < 0:
            return -1
        return max(0, self.left_context // self.chunk_size)


# 배치 하나의 청크 설정을 뽑는다: full_context_prob(기본 0.5) 확률로 전체 문맥, 아니면 청크 크기와 좌측 문맥을
# 각각 균등하게 고른다. 발화 단위가 아니라 배치 단위로 뽑는 것은 조사 검증 C13의 결론(배치 단위 샘플링만
# 작동)을 따른 것이다.
# 호출 위치: train.main의 학습 루프(--offline-only면 부르지 않고 항상 전체 문맥).
def sample_chunk_spec(
    full_context_prob: float, chunk_sizes: tuple[int, ...], left_contexts: tuple[int, ...], rng: random.Random
) -> ChunkSpec:
    if rng.random() < full_context_prob:
        return ChunkSpec(0, -1)
    return ChunkSpec(rng.choice(chunk_sizes), rng.choice(left_contexts))


# 쿼리 프레임 i가 키 프레임 j를 볼 수 있는 조건은 (1) j의 청크 ≤ i의 청크(미래 청크 금지), (2) j의 청크 ≥ i의
# 청크 − left_chunks(좌측 문맥 제한), (3) j가 패딩이 아님.
# 자기 청크 안에서는 뒤쪽 프레임도 볼 수 있다. 스트리밍에서는 청크가 다 모인 뒤 한 번에 계산하므로 이 정보는
# 이미 도착해 있다.
def attention_mask(
    lengths: torch.Tensor,
    t: int,
    spec: ChunkSpec,
    device: torch.device,
    query_offset: int = 0,
    t_q: int | None = None,
) -> torch.Tensor:
    """SDPA에 넘기는 bool 마스크 (B, 1, Tq, Tk). 어텐션을 허용하는 칸이 True다.

    lengths: 발화별 유효 키 프레임 수 (B,). 키는 절대 프레임 [0, t), 쿼리는 절대 프레임
    [query_offset, query_offset + t_q)를 덮는다(기본은 키 전체). query_offset·t_q는 키가 '캐시 + 현재
    청크'인 스트리밍 경우까지 쓸 수 있게 일반화한 것이지만, 현재 스트리밍 블록(forward_streaming)은 같은
    규칙의 마스크를 직접 만들고 이 함수는 오프라인 forward에서만 부른다.
    """
    t_q = t if t_q is None else t_q
    key_idx = torch.arange(t, device=device)
    query_idx = torch.arange(t_q, device=device) + query_offset
    # 키 쪽 패딩 마스크: 발화별 유효 길이 뒤의 프레임은 어떤 쿼리도 보지 못한다.
    valid = key_idx[None, :] < lengths[:, None]  # (B, Tk)
    if spec.full:
        allowed = torch.ones(t_q, t, dtype=torch.bool, device=device)
    else:
        # 프레임 번호를 청크 번호로 바꿔 청크 단위로 비교한다. 모든 층이 같은 경계를 쓰므로 층을 쌓아도
        # look-ahead가 누적되지 않는다(어느 층에서도 자기 청크 끝까지만 본다).
        cq = query_idx // spec.chunk_size
        ck = key_idx // spec.chunk_size
        allowed = ck[None, :] <= cq[:, None]
        if spec.left_chunks >= 0:
            allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
    mask = allowed[None, :, :] & valid[:, None, :]
    # 패딩 쿼리 행은 모든 키가 막혀 softmax가 NaN이 될 수 있고, 그 NaN이 BACON conv의 청크 내 오른쪽 탭을 통해
    # 유효 프레임으로 번질 수 있었다(설계 검토에서 발견, docs/ai-verification.md). 그래서 대각선을 열어 자기
    # 자신을 보게 해 softmax에 모두 False인 행이 들어가지 않게 한다. 이 행들의 출력은 길이 마스크로 버려진다.
    if query_offset == 0 and t_q == t:
        eye = torch.eye(t, dtype=torch.bool, device=device)
        mask = mask | eye[None]
    # (B, 1, Tq, Tk): 헤드 축은 1로 두어 SDPA가 모든 헤드에 브로드캐스트한다.
    return mask[:, None, :, :]
