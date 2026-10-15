"""자기조건(중간) CTC 헤드를 둔 청크 인과 Conformer 인코더.

이 프로젝트의 모델 본체 ChunkConformerCTC와 그 블록(ConformerBlock)이다. 입력 log-mel → 정규화(+학습 중
SpecAugment) → 인과 서브샘플링(25 fps) → 블록 12개 → CTC 출력(최종 헤드 + 중간 헤드).
"""

# [설계] 청크 인과 Conformer-CTC 약 21M(docs/survey/README.md "결정"). 모든 블록이 같은 청크 경계 마스크를
# 쓰고, 시간축 연산은 어텐션과 depthwise conv 두 개뿐이라 스트리밍용 캐시가 블록당 두 종류(어텐션 입력, conv
# 입력)로 끝난다.
# [두 가지 forward] forward는 학습·오프라인 평가용(전체 발화 + 청크 마스크), forward_streaming은 청크 하나씩
# 캐시와 함께 계산하는 실제 스트리밍용(streaming.StreamingEncoder가 부름)이다. 같은 청크 설정이면 두 결과가
# fp32에서 1e-5 안으로 같다(tests/test_model.py).
# [연결] train.py(학습), evaluate.py(full·simulated 디코딩, dev 손실), streaming.py(실제 스트리밍), scripts/의
# 단계별 보고 스크립트가 이 모델을 쓴다.

from __future__ import annotations

import torch
from torch import nn

# 블록 부품(어텐션, conv 모듈), 설정, SpecAugment, 청크 마스크, 인과 서브샘플링을 모두 같은 패키지의 모듈에서
# 가져온다(외부 ASR 라이브러리 없이 순수 PyTorch로 구현).
from stt_scratch.model.attention import RelPosSelfAttention
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conv import ConvModule
from stt_scratch.model.features import spec_augment
from stt_scratch.model.mask import ChunkSpec, attention_mask
from stt_scratch.model.subsampling import CausalConv2dSubsampling


# pre-LN 피드포워드: LayerNorm → Linear(d → ffn_dim) → SiLU → dropout → Linear(ffn_dim → d) → dropout.
# 블록에서 출력에 0.5를 곱해 앞뒤 두 번 쓴다(Macaron 구조). 프레임마다 독립 계산이라 스트리밍에 캐시가 필요
# 없다. ebranchformer.py도 이 클래스를 그대로 쓴다.
class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Conformer 블록 하나. 각 하위 모듈이 입력을 LayerNorm한 뒤 계산하고 잔차로 더하는 pre-LN 구조이며, 블록 끝에
# LayerNorm을 한 번 더 건다. pre-LN은 깊은 층에서도 잔차 경로가 정규화되지 않아 학습이 안정적인 쪽으로 알려져
# 있다(docs/survey/04-small-data-korean.md: Conformer 계열은 LayerNorm·pre-norm).
class ConformerBlock(nn.Module):
    """Macaron FFN(1/2) → MHSA → Conv → FFN(1/2) → LayerNorm. 모두 pre-norm 잔차 연결이다."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ffn1 = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = RelPosSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_theta, cfg.qk_norm)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout, cfg.bacon)
        self.ffn2 = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # FFN 절반 → 어텐션(청크 마스크·RoPE) → conv 모듈(분할 depthwise, 패딩 프레임 0) → FFN 절반 → 최종
        # LayerNorm. 오프라인 경로라 conv 캐시 출력은 버린다.
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), mask, positions))
        y, _ = self.conv(x, chunk_size, valid=valid)
        x = x + y
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)

    # 오프라인 conv가 과거를 0으로 채우는 것과 같은 조건이다. E-Branchformer는 같은 이름의 메서드가 캐시
    # 두 개(cgMLP, 병합 conv)를 돌려주므로 StreamingEncoder는 블록 종류를 몰라도 된다.
    def init_conv_cache(self, x: torch.Tensor) -> torch.Tensor:
        """스트리밍 첫 청크용 conv 모듈 0 캐시 (1, d, k-1)."""
        return x.new_zeros(1, x.shape[-1], self.conv.cache_frames)

    # 어텐션 키·값은 '캐시된 과거 입력 + 현재 청크'를 다시 투영해서
    # 만든다(입력 캐시 + 재투영). 캐시를 몇 프레임 남길지(좌측 문맥)는 호출자 StreamingEncoder가 정한다.
    def forward_streaming(
        self,
        x: torch.Tensor,
        attn_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        positions: torch.Tensor,
        spec: ChunkSpec,
        frame_offset: int,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        fast_conv: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """청크 하나에 대한 블록 계산(실제 스트리밍).

        x: (1, Tc, D) 현재 청크. 첫 프레임의 절대 프레임 번호가 `frame_offset`이다.
        attn_cache: (1, L, D) 청크 앞 L프레임의 LayerNorm을 거친 어텐션 입력.
        conv_cache: (1, D, k-1) 청크 앞 k-1프레임의 conv 입력.
        mask·rope: 빠른 경로가 step마다 한 번 만든 청크 마스크와 RoPE fp32 표. 둘 다 주거나 둘 다 None이어야
        한다(None이면 아래에서 블록마다 만든다 — 기본 경로).
        fast_conv: 입력이 청크 하나 안에 들어갈 때 BACON 오른쪽 탭의 가리기를 건너뛴다(conv.py 참고, 결과는
        비트 단위로 같다).
        반환: 블록 출력, 키·값 원천 전체(캐시 + 청크; 잘라 내는 일은 호출자가 한다), 새 conv 캐시.
        """
        x = x + 0.5 * self.ffn1(x)
        # 캐시에는 LayerNorm까지 거친 어텐션 입력을 저장한다. 그래서 다음 청크에서 캐시 부분은 qkv 투영만 다시
        # 하면 되고, 오프라인 forward가 같은 프레임에 대해 계산하는 키·값과 같은 값이 나온다.
        q_in = self.attn_norm(x)
        kv = torch.cat([attn_cache, q_in], dim=1)
        t_c, n_cached = x.shape[1], attn_cache.shape[1]
        kv_positions = None
        if mask is None:
            # 키 위치는 절대 프레임 번호(캐시 시작 ~ 청크 끝)로 매겨 RoPE와 청크 경계 계산이 오프라인과
            # 같아지게 한다.
            kv_positions = torch.arange(frame_offset - n_cached, frame_offset + t_c, device=x.device)
            # mask.attention_mask와 같은 규칙(미래 청크 금지, 좌측 left_chunks청크까지)의 마스크를 쿼리 = 현재
            # 청크, 키 = 캐시 + 청크에 대해 직접 만든다. 스트리밍에는 배치 패딩이 없으므로 패딩 조건은 없다.
            cq = positions // spec.chunk_size
            ck = kv_positions // spec.chunk_size
            allowed = ck[None, :] <= cq[:, None]
            if spec.left_chunks >= 0:
                allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
            mask = allowed[None, None]  # (1, 1, Tc, L + Tc)
        attn_out = self.attn(q_in, mask, positions, kv=kv, kv_positions=kv_positions, rope=rope)
        x = x + self.attn_dropout(attn_out)
        # conv 모듈은 이전 청크의 마지막 k−1프레임 캐시를 받아 이어서 계산하고, 다음 청크용 캐시를 돌려준다.
        # frame_offset으로 BACON 오른쪽 탭의 청크 끝 위치가 오프라인과 맞춰진다.
        y, new_conv_cache = self.conv(x, spec.chunk_size, conv_cache, frame_offset, fast=fast_conv)
        x = x + y
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x), kv, new_conv_cache


# 모델 전체. 출력은 (최종 로짓, 중간 로짓 목록, 출력 길이)이며 손실은 ctc.total_loss가 합친다. 중간 헤드는
# sc_layers에 적힌 블록 뒤에 붙고, sc_feedback이면 그 softmax를 다음 블록 입력에 더하는 자기조건 CTC, 아니면
# 손실만 더하는 InterCTC다(최종 모델은 InterCTC 6층).
class ChunkConformerCTC(nn.Module):
    """Log-mel → 정규화(+학습 중 SpecAugment) → 서브샘플링 → 블록 → CTC 헤드."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # 전역 정규화 통계를 버퍼로 등록한다. 버퍼는 state_dict에 저장되고 model.to(device)로 함께 옮겨지지만
        # 학습되지는 않는다. 통계가 없으면 평균 0·표준편차 1(정규화 없음).
        mean = torch.tensor(cfg.mel_mean or [0.0] * cfg.n_mels)
        std = torch.tensor(cfg.mel_std or [1.0] * cfg.n_mels)
        self.register_buffer("mel_mean", mean.float())
        self.register_buffer("mel_std", std.float())
        self.subsampling = CausalConv2dSubsampling(
            cfg.n_mels, cfg.subsampling_channels, cfg.d_model, cfg.causal_subsampling
        )
        self.in_dropout = nn.Dropout(cfg.dropout)
        # 블록 종류 선택. E-Branchformer는 3단계 (b)에서만 쓰였다. ebranchformer.py가 이 파일의 FeedForward를
        # 가져다 쓰므로, 순환 import를 피하려고 필요할 때 함수 안에서 import한다.
        if cfg.block == "conformer":
            block_cls = ConformerBlock
        elif cfg.block == "ebranchformer":
            from stt_scratch.model.ebranchformer import EBranchformerBlock

            block_cls = EBranchformerBlock
        else:
            raise ValueError(f"unknown block {cfg.block!r}")
        self.blocks = nn.ModuleList(block_cls(cfg) for _ in range(cfg.n_layers))
        # out_proj 하나를 최종 헤드와 모든 중간 헤드가 공유한다. sc_proj는 자기조건 CTC의 피드백 Linear(어휘
        # 확률 → d_model)이며, 중간 헤드가 없거나 InterCTC(피드백 끔)면 만들지 않는다.
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.sc_layers = set(cfg.sc_layers)
        self.sc_proj = nn.Linear(cfg.vocab_size, cfg.d_model) if self.sc_layers and cfg.sc_feedback else None

    # 원시 log-mel을 학습 세트 통계로 표준화한다. 오프라인 forward와 StreamingEncoder.step이 모두 이 메서드를
    # 쓴다.
    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mel_mean) / self.mel_std

    def forward(
        self, mel: torch.Tensor, frames: torch.Tensor, spec: ChunkSpec, augment: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """mel: (B, T, 80) 원시 log-mel, frames: 유효 프레임 수.

        반환: 최종 로짓 (B, T', V), 중간 로짓 목록, 출력 길이 (B,).
        """
        x = self.normalize_mel(mel)
        # SpecAugment는 정규화 뒤, 서브샘플링 앞에서 건다. augment 인자와 train 모드가 모두 참일 때만이라,
        # 같은 모델로 dev 평가(eval 모드)를 할 때는 자동으로 꺼진다. train.py는 SpecAugment 커리큘럼(epoch
        # 3부터, 규칙 변경 13)에 따라 augment를 넘긴다.
        if augment and self.training:
            x = spec_augment(
                x,
                frames,
                self.cfg.freq_masks,
                self.cfg.freq_mask_width,
                self.cfg.time_masks,
                self.cfg.time_mask_ratio,
            )
        x, out_lengths = self.subsampling(x, frames)
        x = self.in_dropout(x)
        t = x.shape[1]
        # 모든 블록이 같은 마스크·위치·유효 프레임을 공유한다. 위치는 0부터의 절대 프레임 번호, valid는 conv
        # 모듈이 패딩 프레임을 0으로 만들 때 쓴다.
        mask = attention_mask(out_lengths, t, spec, x.device)
        positions = torch.arange(t, device=x.device)
        valid = positions[None, :] < out_lengths[:, None]
        inter: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks, start=1):
            x = block(x, mask, positions, spec.chunk_size, valid)
            # 중간 헤드: 지정된 블록 뒤에서 공유 out_proj로 로짓을 내 손실용으로 모은다(마지막 블록은 최종
            # 헤드가 맡으므로 제외). 자기조건 CTC면 softmax(fp32로 계산 후 원래 dtype)를 sc_proj로 d_model에
            # 투영해 다음 블록 입력에 더한다. 프레임 단위 연산이라 스트리밍에서도 같은 순서로 하면 결과가
            # 같다(streaming.StreamingEncoder.step).
            # 2단계부터 중간 헤드(특히 6·9층)가 처음 보는 Zeroth 문장 dev_H에서 최종 헤드보다 나은 현상이
            # 관측됐다. 학습 문장 암기와 일관된 과적합으로 해석했고, 등록 밖이라 디코딩 헤드는 바꾸지
            # 않았다(docs/experiments.md 2단계 결과 H2.3).
            if i in self.sc_layers and i < len(self.blocks):
                logits = self.out_proj(x)
                inter.append(logits)
                if self.sc_proj is not None:
                    probs = torch.softmax(logits.float(), dim=-1).to(x.dtype)
                    x = x + self.sc_proj(probs)
        return self.out_proj(x), inter, out_lengths
