"""ChunkConformerCTC의 캐시 기반 스트리밍 추론.

학습된 모델을 '청크 하나씩' 돌린다. 청크 mel을 넣으면 그 청크의 인코더 프레임별 fp32 log-prob을
돌려주고 캐시를 갱신한다. 스트림 상태에는 블록마다 마지막 L(좌측 문맥)프레임의 어텐션 입력(LayerNorm 뒤)과
마지막 k-1프레임의 conv 입력, 그리고 서브샘플링 캐시와 지금까지 처리한 프레임 수가 들어 있다. 각 청크는
한 번만 forward하고, 캐시된 프레임의 키·값은 다시 투영한다('입력 캐시 + 재투영'). 그래서 결과가 같은 청크
마스크의 오프라인 forward와 같다(tests/test_model.py가 확인).
"""

# [상태] 캐시 크기가 좌측 문맥으로 고정되므로 세션이 길어져도 청크당 계산·메모리가 늘지 않는다(10분 드리프트
# 측정, docs/experiments.md 6단계 결과).
# [정확성] 같은 청크 설정의 오프라인 forward(전체 발화 + 청크 마스크)와 fp32에서 1e-5 안으로 같고(0단계
# 게이트), 학습된 모델 fp16에서도 실제 스트리밍 CER이 simulated와 0.03포인트 안이었다(H6.1).
# [연결] evaluate.decode_streaming(실제 스트리밍 평가·봉인 test), service.session.StreamSession(서비스),
# scripts/stage6.py 등. autocast(fp16)는 호출자가 건다.

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.mask import ChunkSpec


# 스트림 하나의 상태(발화·세션마다 새로 만든다). frames_seen은 지금까지 처리한 인코더 프레임 수이자 다음 청크
# 첫 프레임의 절대 번호로, RoPE 위치와 청크 경계 계산에 쓰인다. conv_caches는 Conformer 블록이면 텐서,
# E-Branchformer 블록이면 텐서 두 개짜리 튜플이다.
@dataclass
class StreamState:
    spec: ChunkSpec
    frames_seen: int = 0  # 지금까지 처리한 인코더 프레임(25 fps) 수
    sub_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    attn_caches: list[torch.Tensor] = field(default_factory=list)
    conv_caches: list = field(default_factory=list)  # 블록마다 텐서 하나 또는 텐서 튜플


class StreamingEncoder:
    # 입력 검사: 전체 문맥이나 좌측 문맥 0·무제한은 거부한다(무제한(-1)이면 캐시를 고정 길이로 자를 수 없고,
    # 0은 독립 검토 반영으로 거부한다; docs/ai-verification.md). 좌측 문맥이 청크 크기의 배수여야 캐시
    # L프레임이 정확히 '이전 L/청크 개의 청크 전체'가 되어 오프라인 마스크의 left_chunks와 맞는다.
    # 주의: 생성자에서 model.eval()을 호출하므로 넘긴 모델이 eval 모드로 바뀐다(dropout 끔).
    def __init__(self, model: ChunkConformerCTC, spec: ChunkSpec) -> None:
        if spec.full or spec.left_context <= 0:
            raise ValueError("streaming needs a finite chunk size and a positive left context")
        if spec.left_context % spec.chunk_size:
            raise ValueError("left context must be a multiple of the chunk size")
        self.model = model.eval()
        self.spec = spec
        self.device = next(model.parameters()).device

    def new_state(self) -> StreamState:
        return StreamState(spec=self.spec)

    @torch.no_grad()
    # 청크 하나 처리: 정규화 → 인과 서브샘플링(캐시) → 블록 12개(각각 캐시와 함께) → 공유 out_proj → fp32
    # log-softmax. 스트림 마지막의 짧은 부분 청크도 받을 수 있다(인과 프런트엔드라 길이와 무관하게 오프라인과
    # 같다).
    # 호출 위치: evaluate.decode_streaming, service.session.StreamSession._step(서비스가 청크당 계산 시간을
    # 재는 대상이 이 호출이다).
    def step(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        """mel: 청크 하나의 원시 log-mel 프레임 (T_mel, 80), T_mel = 4 * chunk_size.

        새 인코더 프레임의 fp32 log-prob (chunk_size, V)을 돌려주고 state를 제자리에서 갱신한다.
        """
        m = self.model
        cfg = m.cfg
        x = m.normalize_mel(mel.to(self.device).unsqueeze(0))
        x, state.sub_cache = m.subsampling.forward_streaming(x, state.sub_cache)
        t_c = x.shape[1]
        offset = state.frames_seen
        positions = torch.arange(offset, offset + t_c, device=self.device)
        # 첫 청크: 어텐션 캐시는 길이 0(과거 없음), conv 캐시는 0으로 채운 k−1프레임. 오프라인 forward의 '과거
        # = 0 패딩' 조건과 같다.
        if not state.attn_caches:
            d = cfg.d_model
            state.attn_caches = [x.new_zeros(1, 0, d) for _ in m.blocks]
            state.conv_caches = [block.init_conv_cache(x) for block in m.blocks]
        for i, block in enumerate(m.blocks, start=1):
            # 블록은 '캐시 + 현재 청크' 전체를 키·값 원천으로 돌려주고, 여기서 마지막 좌측 문맥 L프레임만
            # 남긴다. 처음 구현은 캐시 길이가 0일 때 새 캐시를 0길이로 잘라 첫 청크 뒤에도 캐시가 비는 버그가
            # 있어, '블록은 전부 돌려주고 호출자가 자르는' 구조로 바꿨다(docs/ai-verification.md).
            x, kv, new_conv = block.forward_streaming(
                x, state.attn_caches[i - 1], state.conv_caches[i - 1], positions, self.spec, offset
            )
            state.attn_caches[i - 1] = kv[:, -self.spec.left_context :]
            state.conv_caches[i - 1] = new_conv
            # 자기조건 CTC 모델이면 오프라인 forward와 같은 위치에서 같은 피드백을 더한다. 최종 모델(InterCTC,
            # 피드백 없음)은 sc_proj가 None이라 중간 헤드 계산 자체를 건너뛴다(스트리밍에는 중간 헤드 출력이
            # 필요 없다).
            if i in m.sc_layers and i < len(m.blocks) and m.sc_proj is not None:
                probs = torch.softmax(m.out_proj(x).float(), dim=-1).to(x.dtype)
                x = x + m.sc_proj(probs)
        # 다음 청크의 위치를 위해 처리한 프레임 수를 늘리고, 이 청크 프레임들의 fp32 log-prob (t_c, V)를
        # 돌려준다(배치 축 제거).
        state.frames_seen += t_c
        return log_probs_fp32(m.out_proj(x))[0]
