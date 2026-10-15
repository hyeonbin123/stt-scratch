"""ChunkConformerCTC의 캐시 기반 스트리밍 추론.

학습된 모델을 '청크 하나씩' 돌린다. 청크 mel을 넣으면 그 청크의 인코더 프레임별 fp32 log-prob을
돌려주고 캐시를 갱신한다. 스트림 상태에는 블록마다 마지막 L(좌측 문맥)프레임의 어텐션 입력(LayerNorm 뒤)과
마지막 k-1프레임의 conv 입력, 그리고 서브샘플링 캐시와 지금까지 처리한 프레임 수가 들어 있다. 각 청크는
한 번만 forward하고, 캐시된 프레임의 키·값은 다시 투영한다('입력 캐시 + 재투영'). 그래서 결과가 같은 청크
마스크의 오프라인 forward와 같다(tests/test_model.py가 확인).

계산 경로는 세 가지다(`path`).
  eager  6·7단계 측정과 봉인 test에 쓴 원래 경로. 블록마다 RoPE 표·마스크를 만들고, autocast가 step마다
         가중치를 fp16으로 바꾼다.
  fast   같은 값을 내도록 연산 수만 줄인 경로: RoPE 표·마스크를 step마다 한 번 만들어 모든 블록에 넘기고,
         BACON 오른쪽 탭 가리기(청크 하나 안에서는 0을 0으로 바꾸는 일)를 건너뛰며, CUDA fp16 autocast
         아래에서는 Linear·conv 가중치를 미리 fp16으로 바꾼 복사본을 쓴다(autocast가 step마다 하던 변환과
         같은 값). 모두 결과가 비트 단위로 같도록 고른 변경이다.
  graph  fast와 같은 계산을, 캐시가 꽉 찬 정상 상태 step에서는 고정 버퍼 위에서 돌린다. CUDA에서는 이 step을
         torch.cuda.CUDAGraph로 한 번 캡처해 재생하고(커널 실행 호출 수천 번 → 그래프 재생 한 번), CPU에서는
         같은 고정 버퍼 코드를 그대로 실행한다(테스트용). 첫 L/청크 개 청크와 flush의 부분 청크는 fast로 돈다.
fast·graph는 출력 동일성 관문(docs/experiments.md "7단계 뒤 후속 측정" P3, 규칙 변경 26)을 통과해야
서버 기본값이 된다.
"""

# [상태] 캐시 크기가 좌측 문맥으로 고정되므로 세션이 길어져도 청크당 계산·메모리가 늘지 않는다(10분 드리프트
# 측정, docs/experiments.md 6단계 결과).
# [정확성] 같은 청크 설정의 오프라인 forward(전체 발화 + 청크 마스크)와 fp32에서 1e-5 안으로 같고(0단계
# 게이트), 학습된 모델 fp16에서도 실제 스트리밍 CER이 simulated와 0.03포인트 안이었다(H6.1).
# [왜 빠른 경로인가] 6단계에서 GPU step p90이 63 ms로 목표 50 ms를 넘었고(H6.2 미충족), 진단에서 step
# 하나에 CUDA 커널 약 1,990개·autocast 가중치 변환 448회가 나오고 GPU 커널 시간 합은 5.5 ms뿐이었다. 즉
# 연산량이 아니라 연산 수(커널 실행 호출)가 병목이라 연산 수를 줄이고, CUDA Graph로 실행 호출을 묶는다.
# Windows GPU에서는 torch.compile(inductor)이 공식 지원되지 않아(Triton 필요) 수동 CUDA Graph를 쓴다.
# [exit_layer] 앞 N개 블록만 돌리고 공유 out_proj로 출력하는 조기 종료(후속 측정 P4, 보고만). InterCTC 중간
# 헤드가 같은 out_proj를 쓰므로 N이 중간 헤드 층이면 그 헤드의 스트리밍 출력이다.
# [연결] evaluate.decode_streaming(실제 스트리밍 평가·봉인 test), service.session.StreamSession(서비스),
# scripts/stage6.py·scripts/followup.py 등. autocast(fp16)는 호출자가 건다.

from __future__ import annotations

import copy
import weakref
from dataclasses import dataclass, field

import torch
from torch import nn

from stt_scratch.model.attention import rope_cos_sin
from stt_scratch.model.conformer import ChunkConformerCTC, ConformerBlock
from stt_scratch.model.conv import SplitDepthwiseConv1d
from stt_scratch.model.ctc import log_probs_fp32
from stt_scratch.model.mask import ChunkSpec

PATHS = ("eager", "fast", "graph")


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


# autocast 가중치 복사본 저장소: 원래 모델 → {dtype: 복사본}. 약한 키라 원래 모델이 사라지면 복사본도
# 사라진다(복사본은 원래 모델을 참조하지 않는다). 세션마다 복사하지 않고 모델 하나에 한 번만 만든다.
_WEIGHT_COPIES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def autocast_weight_copy(model: ChunkConformerCTC, dtype: torch.dtype) -> ChunkConformerCTC:
    """autocast가 step마다 하던 가중치 변환을 한 번 미리 해 둔 추론 전용 복사본을 돌려준다.

    Linear·Conv1d·Conv2d·분할 depthwise conv의 가중치·편향만 dtype으로 바꾸고, LayerNorm 파라미터와 mel 정규화
    버퍼는 fp32로 둔다. autocast는 이 연산들의 fp32 가중치를 매번 같은 반올림(`.to(dtype)`)으로 바꾸고
    LayerNorm은 fp32로 돌리므로, 같은 autocast 아래에서 이 복사본의 출력은 원래 모델과 비트 단위로 같다
    (이미 dtype인 텐서는 autocast가 변환하지 않는다). autocast 밖(CPU fp32)에서는 쓰면 안 된다.
    주의: 분할 depthwise conv는 `weight.to(x.dtype)`로 입력 dtype에 맞추는데, Conformer conv 모듈에서 그
    입력은 항상 Linear(pointwise1) 출력이라 autocast dtype이다. 그래서 Conformer 블록만 지원한다.
    """
    per_model = _WEIGHT_COPIES.setdefault(model, {})
    if dtype not in per_model:
        if any(not isinstance(b, ConformerBlock) for b in model.blocks):
            raise ValueError("the precast weight copy supports Conformer blocks only")
        twin = copy.deepcopy(model).eval()
        for module in twin.modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d | SplitDepthwiseConv1d):
                for param in module.parameters(recurse=False):
                    param.data = param.data.to(dtype)
        for param in twin.parameters():
            param.requires_grad_(False)
        per_model[dtype] = twin
    return per_model[dtype]


# 지금 이 장치에 걸린 autocast 설정(꺼져 있으면 None). 빠른 경로가 가중치 복사본을 쓸지 정하는 데 쓴다.
def _active_autocast(device: torch.device) -> torch.dtype | None:
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return None


def step_inputs(
    positions: torch.Tensor, kv_positions: torch.Tensor, spec: ChunkSpec, head_dim: int, theta: float
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """step 하나의 청크 마스크 (1, 1, Tc, Tk)와 RoPE fp32 표 (cos_q, sin_q, cos_k, sin_k)를 한 번에 만든다.

    ConformerBlock.forward_streaming과 RelPosSelfAttention.forward가 블록마다 만드는 것과 같은 식이라 결과가
    비트 단위로 같다. 모든 블록이 같은 위치·청크 설정을 쓰므로 블록마다 다시 만들 필요가 없다.
    """
    cq = positions // spec.chunk_size
    ck = kv_positions // spec.chunk_size
    allowed = ck[None, :] <= cq[:, None]
    if spec.left_chunks >= 0:
        allowed &= ck[None, :] >= (cq[:, None] - spec.left_chunks)
    cos_q, sin_q = rope_cos_sin(positions, head_dim, theta, torch.float32)
    cos_k, sin_k = rope_cos_sin(kv_positions, head_dim, theta, torch.float32)
    return allowed[None, None], (cos_q, sin_q, cos_k, sin_k)


class StreamingEncoder:
    # 입력 검사: 전체 문맥이나 좌측 문맥 0·무제한은 거부한다(무제한(-1)이면 캐시를 고정 길이로 자를 수 없고,
    # 0은 독립 검토 반영으로 거부한다; docs/ai-verification.md). 좌측 문맥이 청크 크기의 배수여야 캐시
    # L프레임이 정확히 '이전 L/청크 개의 청크 전체'가 되어 오프라인 마스크의 left_chunks와 맞는다.
    # exit_layer: None이면 모든 블록, 정수 N이면 앞 N개 블록 뒤에서 out_proj로 출력한다(1 ≤ N ≤ 블록 수).
    # path: "eager"(기본, 측정에 쓴 경로) / "fast" / "graph"(모듈 docstring). fast·graph는 Conformer 블록만.
    # 주의: 생성자에서 model.eval()을 호출하므로 넘긴 모델이 eval 모드로 바뀐다(dropout 끔).
    def __init__(
        self,
        model: ChunkConformerCTC,
        spec: ChunkSpec,
        exit_layer: int | None = None,
        path: str = "eager",
    ) -> None:
        if spec.full or spec.left_context <= 0:
            raise ValueError("streaming needs a finite chunk size and a positive left context")
        if spec.left_context % spec.chunk_size:
            raise ValueError("left context must be a multiple of the chunk size")
        if exit_layer is not None and not 1 <= exit_layer <= len(model.blocks):
            raise ValueError(f"exit_layer must be in 1..{len(model.blocks)}")
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        if path != "eager" and any(not isinstance(b, ConformerBlock) for b in model.blocks):
            raise ValueError("the fast and graph paths support Conformer blocks only")
        self.model = model.eval()
        self.spec = spec
        self.device = next(model.parameters()).device
        self.n_blocks = len(model.blocks) if exit_layer is None else exit_layer
        self.exit_layer = exit_layer
        self.path = path

    def new_state(self) -> StreamState:
        return StreamState(spec=self.spec)

    @torch.no_grad()
    # 청크 하나 처리: 정규화 → 인과 서브샘플링(캐시) → 블록(각각 캐시와 함께) → 공유 out_proj → fp32
    # log-softmax. 스트림 마지막의 짧은 부분 청크도 받을 수 있다(인과 프런트엔드라 길이와 무관하게 오프라인과
    # 같다).
    # 호출 위치: evaluate.decode_streaming, service.session.StreamSession._step(서비스가 청크당 계산 시간을
    # 재는 대상이 이 호출이다).
    def step(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        """mel: 청크 하나의 원시 log-mel 프레임 (T_mel, 80), T_mel = 4 * chunk_size.

        새 인코더 프레임의 fp32 log-prob (chunk_size, V)을 돌려주고 state를 제자리에서 갱신한다.
        """
        if self.path == "eager":
            return self._step_eager(mel, state)
        if self.path == "graph":
            stepper = _static_stepper(self)
            if stepper.eligible(mel, state):
                return stepper.step(mel, state)
        return self._step_fast(mel, state)

    # 원래 경로(6·7단계 측정·봉인 test에 쓴 계산). exit_layer가 None이면 전과 같은 코드를 그대로 지난다.
    def _step_eager(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        m = self.model
        cfg = m.cfg
        blocks = m.blocks[: self.n_blocks]
        x = m.normalize_mel(mel.to(self.device).unsqueeze(0))
        x, state.sub_cache = m.subsampling.forward_streaming(x, state.sub_cache)
        t_c = x.shape[1]
        offset = state.frames_seen
        positions = torch.arange(offset, offset + t_c, device=self.device)
        # 첫 청크: 어텐션 캐시는 길이 0(과거 없음), conv 캐시는 0으로 채운 k−1프레임. 오프라인 forward의 '과거
        # = 0 패딩' 조건과 같다.
        if not state.attn_caches:
            d = cfg.d_model
            state.attn_caches = [x.new_zeros(1, 0, d) for _ in blocks]
            state.conv_caches = [block.init_conv_cache(x) for block in blocks]
        for i, block in enumerate(blocks, start=1):
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
            # 필요 없다). 조기 종료 층(i == n_blocks)에서는 피드백 없이 바로 출력한다(오프라인 중간 헤드도
            # 피드백을 더하기 전의 로짓이다).
            if i in m.sc_layers and i < self.n_blocks and m.sc_proj is not None:
                probs = torch.softmax(m.out_proj(x).float(), dim=-1).to(x.dtype)
                x = x + m.sc_proj(probs)
        # 다음 청크의 위치를 위해 처리한 프레임 수를 늘리고, 이 청크 프레임들의 fp32 log-prob (t_c, V)를
        # 돌려준다(배치 축 제거).
        state.frames_seen += t_c
        return log_probs_fp32(m.out_proj(x))[0]

    # fast·graph 경로가 쓸 모델: autocast(서비스는 CUDA fp16) 아래면 가중치를 미리 그 dtype으로 바꾼 복사본,
    # 아니면 원래 모델.
    def weights_model(self) -> ChunkConformerCTC:
        dtype = _active_autocast(self.device)
        if dtype is None or dtype == torch.float32:
            return self.model
        return autocast_weight_copy(self.model, dtype)

    # 블록 계산 공통부(fast와 graph의 고정 버퍼 step이 함께 쓴다). 캐시 목록의 원소를 새 값으로 바꿔 넣는 대신
    # write 함수로 넘겨, fast는 목록에 대입하고 graph는 고정 버퍼에 복사한다.
    def run_blocks(self, m, x, attn_caches, conv_caches, positions, mask, rope, write):
        for i, block in enumerate(m.blocks[: self.n_blocks], start=1):
            # frame_offset 0: mask·rope를 넘기므로 블록은 위치를 다시 계산하지 않고, conv는 '입력이 청크 하나
            # 안'인지 보는 데만 쓴다. 스트리밍 step의 실제 offset은 항상 청크 배수라 0과 같은 판정이 나온다.
            x, kv, new_conv = block.forward_streaming(
                x, attn_caches[i - 1], conv_caches[i - 1], positions, self.spec, 0, mask, rope, True
            )
            write(i - 1, kv[:, -self.spec.left_context :], new_conv)
            if i in m.sc_layers and i < self.n_blocks and m.sc_proj is not None:
                probs = torch.softmax(m.out_proj(x).float(), dim=-1).to(x.dtype)
                x = x + m.sc_proj(probs)
        return x

    # 빠른 경로 step(모양이 정해지지 않은 첫 청크들·부분 청크도 받는다). eager와 같은 연산을 같은 순서로 하되
    # 마스크·RoPE 표를 한 번만 만들고, 탭 가리기를 건너뛰고, 가중치 복사본을 쓴다.
    def _step_fast(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        m = self.weights_model()
        cfg = m.cfg
        x = m.normalize_mel(mel.to(self.device).unsqueeze(0))
        x, state.sub_cache = m.subsampling.forward_streaming(x, state.sub_cache)
        t_c = x.shape[1]
        offset = state.frames_seen
        if not state.attn_caches:
            blocks = m.blocks[: self.n_blocks]
            state.attn_caches = [x.new_zeros(1, 0, cfg.d_model) for _ in blocks]
            state.conv_caches = [block.init_conv_cache(x) for block in blocks]
        n_cached = state.attn_caches[0].shape[1]  # 모든 블록의 캐시 길이는 같다
        positions = torch.arange(offset, offset + t_c, device=self.device)
        kv_positions = torch.arange(offset - n_cached, offset + t_c, device=self.device)
        mask, rope = step_inputs(positions, kv_positions, self.spec, cfg.head_dim, cfg.rope_theta)

        def write(i: int, attn: torch.Tensor, conv: torch.Tensor) -> None:
            state.attn_caches[i] = attn
            state.conv_caches[i] = conv

        x = self.run_blocks(m, x, state.attn_caches, state.conv_caches, positions, mask, rope, write)
        state.frames_seen += t_c
        return log_probs_fp32(m.out_proj(x))[0]


# 고정 버퍼 step 저장소: 원래 모델 → {(청크, 좌측, 블록 수): _StaticStep}. 세션마다 새 StreamingEncoder를
# 만들어도 같은 모델·설정이면 버퍼와 CUDA Graph 하나를 함께 쓴다(세션마다 캡처하면 연결마다 수백 ms가 든다).
# _StaticStep은 원래 모델을 약한 참조로만 가지므로 모델이 사라지면 항목도 사라진다.
_STEPPERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _static_stepper(enc: StreamingEncoder) -> _StaticStep:
    per_model = _STEPPERS.setdefault(enc.model, {})
    key = (enc.spec.chunk_size, enc.spec.left_context, enc.n_blocks)
    if key not in per_model:
        per_model[key] = _StaticStep(enc)
    return per_model[key]


class _StaticStep:
    """캐시가 꽉 찬 정상 상태 step을 고정 버퍼 위에서 돌린다(CUDA에서는 CUDA Graph로 캡처해 재생).

    정상 상태 = 좌측 문맥 L프레임 캐시가 모두 찼고(첫 L/청크 개 청크 뒤), mel이 온전한 청크 하나이며,
    처리한 프레임 수가 청크 배수인 step. 그러면 모든 텐서 모양이 매번 같아 그래프 하나로 재생할 수 있다.
    위치는 offset 텐서(1개 값) + 고정 arange로 만들어 그래프 안에서 계산한다.

    상태 묶기(bind): 그래프는 버퍼 주소를 기억하므로 스트림 상태(StreamState)의 캐시 목록이 이 버퍼 자체를
    가리키게 한다. 다른 상태가 오면 먼저 묶였던 상태의 캐시를 복제해 떼어 주고 새 상태를 버퍼로 복사한다.
    fast 경로(부분 청크 등)가 상태의 캐시를 새 텐서로 바꿔 넣으면 다음 step에서 다시 복사해 묶는다.
    """

    def __init__(self, enc: StreamingEncoder) -> None:
        self._model = weakref.ref(enc.model)
        self.spec = enc.spec
        self.n_blocks = enc.n_blocks
        self.device = enc.device
        self.signature = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_failed: str | None = None  # 캡처 실패 사유(실패하면 고정 버퍼 코드를 그대로 실행)
        self._bound: weakref.ref | None = None

    def _encoder(self) -> StreamingEncoder:
        return StreamingEncoder(self._model(), self.spec, self.n_blocks, path="fast")

    def eligible(self, mel: torch.Tensor, state: StreamState) -> bool:
        spec = self.spec
        return (
            state.sub_cache is not None
            and len(state.attn_caches) == self.n_blocks
            and mel.shape[0] == 4 * spec.chunk_size
            and state.frames_seen % spec.chunk_size == 0
            and all(c.shape[1] == spec.left_context for c in state.attn_caches)
        )

    # 버퍼 모양·dtype과 autocast 설정의 묶음. 바뀌면(예: autocast를 끄고 켬) 버퍼와 그래프를 새로 만든다.
    def _signature(self, state: StreamState) -> tuple:
        tensors = [*state.sub_cache, *state.attn_caches, *state.conv_caches]
        return (_active_autocast(self.device), tuple((t.shape, t.dtype) for t in tensors))

    def _allocate(self, state: StreamState, signature: tuple) -> None:
        self.release()
        spec = self.spec
        self.sub = [torch.empty_like(t) for t in state.sub_cache]
        self.attn = [torch.empty_like(t) for t in state.attn_caches]
        self.conv = [torch.empty_like(t) for t in state.conv_caches]
        self.mel = torch.zeros(4 * spec.chunk_size, state.sub_cache[0].shape[-1], device=self.device)
        self.offset = torch.zeros((), dtype=torch.long, device=self.device)
        self.ar_q = torch.arange(spec.chunk_size, device=self.device)
        self.ar_k = torch.arange(spec.left_context + spec.chunk_size, device=self.device)
        self.out: torch.Tensor | None = None
        self.graph = None
        self.signature = signature

    def _buffers(self) -> list[torch.Tensor]:
        return [*self.sub, *self.attn, *self.conv]

    def _is_bound(self, state: StreamState) -> bool:
        return (
            self._bound is not None
            and self._bound() is state
            and all(a is b for a, b in zip(state.sub_cache, self.sub, strict=True))
            and all(a is b for a, b in zip(state.attn_caches, self.attn, strict=True))
            and all(a is b for a, b in zip(state.conv_caches, self.conv, strict=True))
        )

    # 묶였던 상태의 캐시 가운데 아직 이 버퍼를 가리키는 것을 복제본으로 바꿔, 버퍼를 다른 상태에 넘겨도 그
    # 상태의 값이 바뀌지 않게 한다.
    def release(self) -> None:
        prev = self._bound() if self._bound is not None else None
        self._bound = None
        if prev is None or self.signature is None:
            return
        own = {id(b) for b in self._buffers()}
        if prev.sub_cache is not None:
            prev.sub_cache = tuple(t.clone() if id(t) in own else t for t in prev.sub_cache)
        prev.attn_caches = [t.clone() if id(t) in own else t for t in prev.attn_caches]
        prev.conv_caches = [t.clone() if id(t) in own else t for t in prev.conv_caches]

    def _bind(self, state: StreamState) -> None:
        if self._is_bound(state):
            return
        if self._bound is not None and self._bound() is not state:
            self.release()
        for buf, t in zip(self.sub, state.sub_cache, strict=True):
            if buf is not t:
                buf.copy_(t)
        for buf, t in zip(self.attn, state.attn_caches, strict=True):
            if buf is not t:
                buf.copy_(t)
        for buf, t in zip(self.conv, state.conv_caches, strict=True):
            if buf is not t:
                buf.copy_(t)
        state.sub_cache = tuple(self.sub)
        state.attn_caches = list(self.attn)
        state.conv_caches = list(self.conv)
        self._bound = weakref.ref(state)

    # 고정 버퍼만 읽고 쓰는 정상 상태 step 본체. 그래프 캡처 대상이므로 CPU로 값을 읽거나(.item 등) 버퍼 밖에
    # 결과를 남기는 일이 없어야 한다. 출력 log-prob은 self.out에 남는다.
    def _body(self) -> None:
        enc = self._encoder()
        m = enc.weights_model()
        cfg = m.cfg
        x = m.normalize_mel(self.mel.unsqueeze(0))
        x, (c0, c1) = m.subsampling.forward_streaming(x, (self.sub[0], self.sub[1]))
        positions = self.offset + self.ar_q
        kv_positions = self.offset - self.spec.left_context + self.ar_k
        mask, rope = step_inputs(positions, kv_positions, self.spec, cfg.head_dim, cfg.rope_theta)

        def write(i: int, attn: torch.Tensor, conv: torch.Tensor) -> None:
            self.attn[i].copy_(attn)
            self.conv[i].copy_(conv)

        x = enc.run_blocks(m, x, self.attn, self.conv, positions, mask, rope, write)
        self.sub[0].copy_(c0)
        self.sub[1].copy_(c1)
        out = log_probs_fp32(m.out_proj(x))[0]
        # 출력 버퍼는 그래프 밖에서 처음 실행할 때 만든다. 캡처 중에는 이 copy_만 기록된다.
        if self.out is None:
            self.out = torch.empty_like(out)
        self.out.copy_(out)

    # 버퍼를 만들 때의 autocast 설정으로 본체를 돈다. 캡처 중 autocast는 cache_enabled=False여야 한다
    # (torch.cuda.make_graphed_callables 문서). autocast가 꺼진 설정(CPU fp32 등)이면 그대로 돈다.
    def _autocast_body(self) -> None:
        dtype = self.signature[0]
        if dtype is None:
            self._body()
            return
        with torch.autocast(device_type=self.device.type, dtype=dtype, cache_enabled=False):
            self._body()

    # CUDA Graph 캡처: PyTorch 문서의 절차대로 옆 스트림에서 몇 번 미리 돌려(지연 초기화·할당 안정화) 캐시를
    # 원래 값으로 되돌린 뒤 캡처한다. autocast는 캡처 중 cache_enabled=False여야 한다(가중치 복사본을 쓰므로
    # 가중치 변환은 어차피 없다). 실패하면 사유를 남기고 고정 버퍼 코드를 그래프 없이 돈다.
    def _capture(self) -> None:
        saved = [b.clone() for b in self._buffers()]
        try:
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._autocast_body()
            torch.cuda.current_stream(self.device).wait_stream(side)
            for b, v in zip(self._buffers(), saved, strict=True):
                b.copy_(v)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._autocast_body()
            self.graph = graph
        except RuntimeError as err:  # 캡처를 지원하지 않는 연산 등
            self.graph = None
            self.graph_failed = str(err)[:500]
            for b, v in zip(self._buffers(), saved, strict=True):
                b.copy_(v)

    def step(self, mel: torch.Tensor, state: StreamState) -> torch.Tensor:
        signature = self._signature(state)
        if signature != self.signature:
            self._allocate(state, signature)
        self._bind(state)
        self.mel.copy_(mel)
        self.offset.fill_(state.frames_seen)
        use_graph = self.device.type == "cuda" and self.graph_failed is None
        if use_graph and self.graph is None:
            # 처음에는 그래프 없이 한 번 돌려 이 step의 결과를 얻고(출력 버퍼도 이때 생긴다), 다음 step부터 쓸
            # 그래프를 캡처한다. 캡처의 예비 실행이 출력·캐시 버퍼를 덮으므로 결과는 먼저 복제해 두고, 캐시는
            # 캡처가 이 step 뒤 값으로 되돌려 둔다.
            self._autocast_body()
            result = self.out.clone()
            self._capture()
        else:
            if use_graph:
                self.graph.replay()
            else:
                self._autocast_body()
            result = self.out.clone()
        state.frames_seen += self.spec.chunk_size
        return result
