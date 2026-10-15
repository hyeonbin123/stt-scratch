"""회전 위치(RoPE)와 bool 청크·패딩 마스크를 쓰는 멀티헤드 자기 어텐션(SDPA).

Conformer·E-Branchformer 블록이 공유한다. 위치 정보는 RoPE(회전 위치 인코딩), 계산은 PyTorch
scaled_dot_product_attention(SDPA)에 bool 마스크를 넘겨서 한다.
"""

# [왜 RoPE + SDPA인가] 조사에서 순수 CTC 조건의 RoPE가 상대 위치(RelPos)와 동급으로 보고됐고(2501.06051),
# RelPos의 rel-shift 같은 커스텀 코드가 없어 캐시 스트리밍과 오프라인의 등가성 테스트가
# 쉽다(docs/survey/README.md "결정" 위치). 클래스 이름은 RelPosSelfAttention이지만 실제 위치 방식은 RoPE다.
# [QK-norm] 1단계 진단에서 LR 7e-4 이상일 때 어텐션 로짓 표준편차가 60~111까지 커지며 학습이 불안정해졌고,
# q·k에 헤드별 LayerNorm을 넣자 1~3으로 묶였다. 그래서 규칙 변경 13부터 모든 모델에 켠다(docs/experiments.md
# 1단계 진단, 규칙 변경 13).
# [연결] conformer.ConformerBlock, ebranchformer.EBranchformerBlock이 오프라인(mask = mask.attention_mask)과
# 스트리밍(kv = 캐시 + 청크) 두 방식으로 부른다.

from __future__ import annotations

import torch
from torch import nn


# RoPE 회전각의 cos·sin 표: 차원 쌍마다 주파수 1/θ^(i/half)를 두고, 위치(절대 프레임 번호) × 주파수를 각도로
# 쓴다. [angles, angles]로 두 번 이어 붙이는 것은 apply_rope가 앞 절반과 뒤 절반을 짝지어 회전하기 때문이다.
# 각도를 fp32로 계산하는 이유: 스트리밍 위치는 세션이 길수록 커지므로(10분 = 15,000프레임) fp16으로 위치 ×
# 주파수를 계산하면 각도 오차가 커질 수 있다. 결과만 q·k의 dtype으로 바꾼다.
def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple:
    """positions: (T,) 절대 프레임 번호. cos, sin (T, head_dim)을 fp32로 계산한 뒤 dtype으로 바꿔 돌려준다."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half))
    angles = positions.float()[:, None] * inv_freq[None, :]  # (T, half)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# 회전 적용: (x1, x2) → (x1·cos − x2·sin, x2·cos + x1·sin). q와 k를 각자 절대 위치로 돌리면 내적 q_i·k_j가
# 위치 차이 i − j에만 의존한다(상대 위치 효과). 스트리밍도 오프라인과 같은 절대 프레임 번호를 넘기므로 같은
# 회전이 걸린다.
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D), cos/sin: (T, D). 짝 (x[:half], x[half:])을 회전한다."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# 멀티헤드 자기 어텐션(RoPE, 선택적 QK-norm, SDPA). q·k·v는 Linear 하나(qkv)로 투영한 뒤 잘라 쓴다.
# 스트리밍에서는 쿼리 = 현재 청크, 키·값 = 캐시(LayerNorm 뒤의 과거 입력) + 현재 청크를 다시 투영한
# 것이다('입력 캐시 + 재투영', streaming.py 머리말). 이 방식은 오프라인 청크 마스크 forward와 수학적으로 같아
# 1e-5 단위 테스트가 가능하다는 이유로 택했고, 대가로 청크마다 좌측 문맥의 K/V를 다시
# 계산한다(docs/survey/README.md §5).
class RelPosSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, theta: float, qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.theta = theta
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        # QK-LayerNorm(헤드별, 프레임별, RoPE 전): |q|, |k|가 약 sqrt(head_dim)이 되어 로짓이 qkv 가중치와
        # 함께 커지지 않고 대략 ±sqrt(head_dim) 안에 묶인다. 프레임마다 독립적으로 정규화하므로 미래 프레임
        # 정보가 섞이지 않아 스트리밍 정확성이 그대로다(tests/test_model.py
        # test_streaming_equals_offline_qk_norm). LayerNorm의 affine 파라미터가
        # 커지면 경계가 보장되지 않으므로 2단계 보고에서 그 크기를 감시했다(docs/experiments.md 2단계 결과
        # '안정성').
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        kv: torch.Tensor | None = None,
        kv_positions: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """어텐션 출력 (B, Tq, D)를 돌려준다.

        x: (B, Tq, D) 쿼리(kv가 None이면 키·값도 x에서 만든다). mask: bool (B 또는 1, 1, Tq, Tk), True =
        주목한다. kv: 스트리밍용 키·값 원천 (B, Tk, D)(캐시 + 청크). positions·kv_positions는 절대 프레임
        번호다. rope: 빠른 스트리밍 경로가 step마다 한 번 계산해 모든 블록에 넘기는 fp32 표
        (cos_q, sin_q, cos_k, sin_k)이며, 주면 positions·kv_positions로 표를 다시 만들지 않는다.
        """
        # 쿼리는 x에서, 키·값은 src(오프라인은 x 자체, 스트리밍은 캐시 + 청크)에서 만든다. 같은 qkv Linear를
        # x와 src에 각각 적용하므로 오프라인 경로에서는 x에 대한 투영이 두 번 계산된다(결과는 같고 연산만
        # 중복).
        b, tq, d = x.shape
        q = self.qkv(x)[..., :d]
        src = x if kv is None else kv
        kvp = self.qkv(src)[..., d:]
        k, v = kvp.chunk(2, dim=-1)
        tk = src.shape[1]
        # (B, T, D) → (B, H, T, D/H): SDPA가 기대하는 헤드 축 배치로 바꾼다.
        q = q.view(b, tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        # QK-norm은 RoPE보다 먼저 건다(규칙 변경 13의 정의). 회전은 벡터 크기를 바꾸지 않으므로 정규화로 묶은
        # 크기가 그대로 유지된다.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 오프라인은 쿼리와 키의 위치가 같아 표를 한 번만 만들고, 스트리밍은 키 위치(캐시 시작 ~ 청크 끝)가
        # 쿼리 위치(청크)와 달라 따로 만든다.
        if rope is not None:
            # 빠른 스트리밍 경로(streaming.StreamingEncoder path="fast"/"graph"): 표는 rope_cos_sin을 fp32로
            # 부른 결과라, 여기서 q·k dtype으로 바꾸면 아래 기본 경로가 블록마다 만드는 표와 비트 단위로 같다
            # (rope_cos_sin도 fp32로 계산한 뒤 마지막에 dtype을 바꾼다). 블록 12개 × 표 2개를 step마다 다시
            # 계산하던 커널 수를 줄이려는 것이다(docs/experiments.md 6단계 결과 H6.2 진단).
            cos_q, sin_q, cos_k, sin_k = rope
            cos_q, sin_q = cos_q.to(q.dtype), sin_q.to(q.dtype)
            cos_k, sin_k = cos_k.to(k.dtype), sin_k.to(k.dtype)
        else:
            cos_q, sin_q = rope_cos_sin(positions, self.head_dim, self.theta, q.dtype)
            if kv_positions is None:
                cos_k, sin_k = cos_q, sin_q
            else:
                cos_k, sin_k = rope_cos_sin(kv_positions, self.head_dim, self.theta, k.dtype)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        # SDPA: bool 마스크는 True인 칸만 softmax에 들어간다. RTX 2080 Ti(sm_75)에서는 flash 커널을 쓸 수 없어
        # memory-efficient 커널이 선택된다(docs/experiments.md 0단계 결과). 어텐션 dropout은 학습 때만 건다.
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, tq, d)
        return self.out(out)
