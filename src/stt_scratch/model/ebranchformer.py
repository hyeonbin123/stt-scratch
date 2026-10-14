"""청크 인과 E-Branchformer 블록(Kim et al., SLT 2022, arXiv 2210.00077). 3단계 후보 (b)에서만 썼다.

Macaron FFN(1/2) → [MHSA 가지 | cgMLP 가지] → 병합(이어 붙이기 → depthwise conv → 더하기 → linear) →
FFN(1/2) → LayerNorm. 두 depthwise conv(cgMLP 게이트와 병합)는 Conformer conv 모듈과 같은 분할 depthwise
conv(절반 인과, 절반은 청크 끝까지)를 쓰므로, 청크 마스크, 미래 누출 게이트 테스트, 정확한 캐시
스트리밍이 그대로 적용된다(tests/test_model.py test_ebranchformer_*). ConformerBlock과 같은
인터페이스(forward, forward_streaming, init_conv_cache)라 config.block = "ebranchformer"만으로 모델·학습·
스트리밍 코드를 그대로 쓸 수 있다.
"""

# [왜 비교했나] 조사에서 E-Branchformer는 100 h 규모 통제 비교에서 Conformer와 같거나 낫고 소규모 CTC 학습에서
# 실패가 적다는 보고가 있어 2순위 인코더였다. 다만 차이가 시드 분산 수준이라 기본 구조로 삼지 않고 동일 예산
# 비교군으로만 뒀다(docs/survey/README.md "결정" 인코더, docs/survey/01-encoders.md).
# [결과] 파라미터를 Run A의 +2.5 %(21.32M)로 맞춰 100 epoch 학습했으나 dev_mix 22.49 % vs Run A 22.05 %로 교체
# 조건(1.86포인트·상대 8 % 이상 개선)을 못 넘어 Conformer를 유지했다. dev_H는 0.86포인트 나았다. epoch 시간은
# Run A의 약 1.9배였다(docs/experiments.md 3단계 결과 (b)).

from __future__ import annotations

import torch
from torch import nn

from stt_scratch.model.attention import RelPosSelfAttention
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conv import SplitDepthwiseConv1d
from stt_scratch.model.mask import ChunkSpec


# cgMLP 게이트와 병합 conv가 같이 쓰는 공용 도우미다. 다음 캐시는 (이전 캐시 + 입력)의 마지막 k−1프레임이다.
# conv.ConvModule.forward의 캐시 처리와 같은 방식이다.
def _cached_depthwise(
    conv: SplitDepthwiseConv1d,
    y: torch.Tensor,
    chunk_size: int,
    left_cache: torch.Tensor | None,
    frame_offset: int,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y: (B, C, T). 패딩 프레임을 0으로 만들고 분할 conv를 돌려 (출력, 새 k-1프레임 캐시)를 돌려준다."""
    if valid is not None:
        y = y * valid[:, None, :].to(y.dtype)
    k1 = conv.kernel - 1
    new_cache = torch.cat([left_cache, y] if left_cache is not None else [y], dim=2)[:, :, -k1:]
    if new_cache.shape[2] < k1:
        new_cache = nn.functional.pad(new_cache, (k1 - new_cache.shape[2], 0))
    return conv(y, chunk_size, left_cache, frame_offset), new_cache


# cgMLP 가지(국소 문맥 담당): 투영해 넓힌 뒤 반으로 나눠 한쪽(g)만 depthwise conv로 시간 문맥을 섞고, 다른
# 쪽(r)과 곱해 게이트한다. 어텐션 가지가 전역 문맥을 맡는 것과 나란히 돈다.
# 커널은 cgmlp_kernel(기본 31프레임 = 1.24 s)이며, 3단계 (b)는 --cgmlp-units 896으로 폭을 줄여 파라미터 수를
# Run A에 맞췄다(규칙 변경 17).
class ConvolutionalGatingMLP(nn.Module):
    """cgMLP: LN → Linear(d, U) → GELU → (r, g) 분할. g: LN → depthwise conv. r * g → Linear(U/2, d)."""

    def __init__(self, d_model: int, units: int, kernel: int, dropout: float, bacon: bool) -> None:
        super().__init__()
        assert units % 2 == 0
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, units)
        self.act = nn.GELU()
        self.gate_norm = nn.LayerNorm(units // 2)
        self.conv = SplitDepthwiseConv1d(units // 2, kernel, bacon)
        self.proj2 = nn.Linear(units // 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        left_cache: torch.Tensor | None = None,
        frame_offset: int = 0,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # LayerNorm → Linear(d → U) → GELU → (r, g) 분할 → g에 LayerNorm·depthwise conv → r × g → Linear(U/2 →
        # d) → dropout. conv 입출력은 (B, C, T) 배치라 전치한다.
        y = self.act(self.proj1(self.norm(x)))
        r, g = y.chunk(2, dim=-1)
        g = self.gate_norm(g).transpose(1, 2)  # (B, U/2, T)
        g, new_cache = _cached_depthwise(self.conv, g, chunk_size, left_cache, frame_offset, valid)
        return self.dropout(self.proj2(r * g.transpose(1, 2))), new_cache


# E-Branchformer 블록: FFN 절반 → [어텐션 가지 | cgMLP 가지] 병렬 → 병합(이어 붙이기 → depthwise conv → 더하기
# → Linear) → FFN 절반 → LayerNorm. 어텐션·FFN·RoPE·QK-norm은 Conformer 블록과 같은 부품이다.
class EBranchformerBlock(nn.Module):
    """ConformerBlock과 같은 인터페이스(forward, forward_streaming, init_conv_cache)를 가진 블록."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # 속성 이름(ffn1, attn_norm, attn)을 ConformerBlock과 같게 둬서 보고 스크립트가 두 블록을 똑같이
        # 들여다볼 수 있게 한다.
        from stt_scratch.model.conformer import FeedForward

        # 병합 conv는 두 가지를 이어 붙인 2d 채널에 걸리고(merge_kernel 기본 31), merge_proj가 2d → d로
        # 되돌린다.
        d = cfg.d_model
        self.ffn1 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = RelPosSelfAttention(d, cfg.n_heads, cfg.dropout, cfg.rope_theta, cfg.qk_norm)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.cgmlp = ConvolutionalGatingMLP(d, cfg.cgmlp_units, cfg.cgmlp_kernel, cfg.dropout, cfg.bacon)
        self.merge_conv = SplitDepthwiseConv1d(2 * d, cfg.merge_kernel, cfg.bacon)
        self.merge_proj = nn.Linear(2 * d, d)
        self.merge_dropout = nn.Dropout(cfg.dropout)
        self.ffn2 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.final_norm = nn.LayerNorm(d)

    # 스트리밍 첫 청크용 0 캐시 두 개(cgMLP conv, 병합 conv). StreamState.conv_caches에는 블록마다 이 튜플이
    # 그대로 들어간다.
    def init_conv_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            x.new_zeros(1, self.cgmlp.conv.weight.shape[0], self.cgmlp.conv.kernel - 1),
            x.new_zeros(1, self.merge_conv.weight.shape[0], self.merge_conv.kernel - 1),
        )

    # 어텐션 출력을 받아 cgMLP 가지·병합·FFN 절반·최종 LayerNorm까지 계산하는 공통 뒷부분.
    # 오프라인(caches=None, valid 있음)과 스트리밍(이전 캐시, valid 없음)이 함께 쓰며 새 캐시 두 개를
    # 돌려준다.
    def _merge(
        self,
        x: torch.Tensor,
        att: torch.Tensor,
        chunk_size: int,
        caches: tuple[torch.Tensor, torch.Tensor] | None,
        frame_offset: int,
        valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cg_cache, merge_cache = caches if caches is not None else (None, None)
        local, new_cg = self.cgmlp(x, chunk_size, cg_cache, frame_offset, valid)
        cat = torch.cat([self.attn_dropout(att), local], dim=-1)  # (B, T, 2d)
        # 병합: 두 가지를 이어 붙인 것에 depthwise conv를 걸어 이웃 프레임 정보를 섞고, 원래 이어 붙인 값에
        # 더한 뒤(cat + fused) Linear로 d 차원으로 줄여 잔차에 더한다.
        fused, new_merge = _cached_depthwise(
            self.merge_conv, cat.transpose(1, 2), chunk_size, merge_cache, frame_offset, valid
        )
        x = x + self.merge_dropout(self.merge_proj(cat + fused.transpose(1, 2)))
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x), (new_cg, new_merge)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        # 오프라인 경로: 마스크는 모델이 만든 청크·패딩 마스크, conv 캐시는 0(과거 없음), frame_offset 0.
        att = self.attn(self.attn_norm(x), mask, positions)
        out, _ = self._merge(x, att, chunk_size, None, 0, valid)
        return out

    def forward_streaming(
        self,
        x: torch.Tensor,
        attn_cache: torch.Tensor,
        conv_cache: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        spec: ChunkSpec,
        frame_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # 스트리밍 경로: 어텐션 부분은 ConformerBlock.forward_streaming과 같다(LayerNorm 뒤 입력을 캐시, 절대
        # 위치, 같은 청크 규칙의 마스크). 이어서 _merge에 이전 conv 캐시 두 개를 넘긴다.
        x = x + 0.5 * self.ffn1(x)
        q_in = self.attn_norm(x)
        kv = torch.cat([attn_cache, q_in], dim=1)
        t_c, n_cached = x.shape[1], attn_cache.shape[1]
        kv_positions = torch.arange(frame_offset - n_cached, frame_offset + t_c, device=x.device)
        cq = positions // spec.chunk_size
        ck = kv_positions // spec.chunk_size
        allowed = ck[None, :] <= cq[:, None]
        if spec.left_chunks >= 0:
            allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
        att = self.attn(q_in, allowed[None, None], positions, kv=kv, kv_positions=kv_positions)
        out, caches = self._merge(x, att, spec.chunk_size, conv_cache, frame_offset, None)
        return out, kv, caches
