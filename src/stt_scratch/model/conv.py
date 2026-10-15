"""BACON식 분할 depthwise conv를 쓰는 Conformer conv 모듈.

채널 절반은 완전 인과다(탭 i-k+1..i, 과거만). 나머지 절반은 오른쪽도 보지만 그 프레임이 속한 청크의
끝까지만 본다(탭 i-(k-1)/2 .. min(i+(k-1)/2, chunk_end)). 청크가 무제한이면 이 절반은 보통의 대칭
depthwise conv가 된다. 두 절반 모두 청크 경계를 넘어 과거를 보므로, 왼쪽 k-1프레임 캐시로 스트리밍하면
오프라인 결과가 정확히 재현된다. 오른쪽 탭은 청크 끝에서 막히므로 미래 청크 정보는 들어오지 않는다.
"""

# [왜 이렇게 나눴나] 청크 스트리밍에서는 청크가 다 모인 뒤 계산하므로 청크 안의 뒤쪽 프레임은 지연 없이 쓸 수
# 있다. 전 채널 인과 conv는 그 정보를 버리고, 전 채널 양방향(청크 경계까지)은 RNN-T 조건 보고에서 인과보다
# 나빴다. BACON(Interspeech 2026)의 '반반'은 둘을 섞은 것이다. 960 h·110M 조건의 근거라 65 h에서의 이득은
# 검증하지 않았고, bacon=False로 전 채널 인과로 되돌릴 수 있다(docs/survey/README.md §3, "결정" conv 모듈).
# [연결] ConvModule은 conformer.ConformerBlock이, SplitDepthwiseConv1d는 ebranchformer의 cgMLP·병합 conv도
# 함께 쓴다.

from __future__ import annotations

import torch
from torch import nn


# 현재 코드에서는 어디서도 부르지 않는다(SplitDepthwiseConv1d.forward가 같은 계산을 절대 프레임 번호로 직접
# 한다).
def chunk_end_index(t: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    """프레임 i마다 그 청크의 마지막 프레임 번호를 돌려준다(chunk_size <= 0, 즉 전체 문맥이면 t-1)."""
    idx = torch.arange(t, device=device)
    if chunk_size <= 0:
        return torch.full_like(idx, t - 1)
    return torch.clamp((idx // chunk_size + 1) * chunk_size - 1, max=t - 1)


# 채널별(depthwise) 1차원 conv, 커널 k(기본 15, 홀수). 앞 causal_channels개 채널은 탭 i−k+1 .. i(완전 인과),
# 나머지 채널은 탭 i−half .. i+half 중 청크 끝을 넘지 않는 것만 쓴다(half = k // 2).
# 가중치를 nn.Conv1d가 아니라 (C, k) Parameter로 직접 둔 이유: 인과 절반은 커널 전체를, 양방향 절반은 커널을
# '과거~현재' 부분과 오른쪽 탭 하나씩으로 나눠 쓰기 때문에 가중치를 잘라 쓸 수 있어야 한다. 초기화는 nn.Conv1d
# 기본값과 같은 kaiming_uniform(a=√5), 편향은 0이다.
# 주의: self.half 속성이 nn.Module.half() 메서드를 가리므로 이 모듈을 fp16으로 바꿀 때는 .to(torch.float16)을
# 쓴다(tests/test_model.py 참고).
class SplitDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel: int, bacon: bool) -> None:
        super().__init__()
        assert kernel % 2 == 1
        self.kernel = kernel
        self.half = kernel // 2
        self.bacon = bacon
        self.causal_channels = channels // 2 if bacon else channels
        self.weight = nn.Parameter(torch.empty(channels, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        left_cache: torch.Tensor | None = None,
        frame_offset: int = 0,
        fast: bool = False,
    ) -> torch.Tensor:
        """분할 depthwise conv를 적용한다.

        x: (B, C, T). left_cache: (B, C, k-1) 스트리밍용 과거 프레임(None이면 0). frame_offset: x[..., 0]의
        절대 프레임 번호(스트리밍에서도 청크 경계가 어긋나지 않게 한다). fast: 입력 전체가 청크 하나 안에
        있으면(스트리밍 step) 오른쪽 탭 가리기를 건너뛴다. 그때 청크 끝을 넘는 탭은 모두 오른쪽 0 패딩을
        읽으므로 가리기가 0을 0으로 바꾸는 일뿐이라 결과가 비트 단위로 같다(tests/test_model.py).

        두 절반 모두 보통의 grouped conv1d로 계산한다. 청크 경계 절반은 대칭 커널의 인과 부분(탭 i-half ..
        i)에 오른쪽 탭 i+1 .. i+half를 더한 것이며, 오른쪽 탭은 프레임의 청크 끝을 넘으면 더하기 전에 0으로
        가린다. 그래서 미래 값이 반올림 연산에 아예 들어가지 않아 fp16에서도 결과가 미래 불변이다(다 더한 뒤
        나중에 빼는 방식은 그렇지 않았다).
        """
        b, c, t = x.shape
        k, half, cc = self.kernel, self.half, self.causal_channels
        # 오프라인(캐시 없음)은 과거를 0으로 채운다. 스트리밍 첫 청크의 캐시도 0이므로(init_conv_cache) 두
        # 경로의 결과가 같다. 가중치는 입력 dtype(autocast 아래 fp16)으로 맞춘다.
        cache = x.new_zeros(b, c, k - 1) if left_cache is None else left_cache
        w = self.weight.to(x.dtype)
        # 인과 절반: 과거 k−1프레임을 앞에 붙이고 패딩 없는 grouped conv1d를 돌리면 출력 i가 정확히 입력 i−k+1
        # .. i를 본다.
        xc = torch.cat([cache, x], dim=2)  # (B, C, T + k - 1)
        out_c = nn.functional.conv1d(xc[:, :cc], w[:cc, None, :], groups=cc)
        if cc == c:
            return out_c + self.bias[None, :, None].to(x.dtype)
        # 양방향 절반: 과거 half프레임 + 현재 입력 + 오른쪽 0 half프레임을 이어 붙인다. 먼저 '과거~현재'
        # 탭(i−half .. i)을 xb[:, :, : t + half] 위에서 커널 앞 half+1개 가중치의 인과 conv로 한 번에
        # 계산하고, 오른쪽 탭 i+1 .. i+half는 아래 반복문에서 하나씩 더한다.
        xb = torch.cat([cache[:, cc:, -half:], x[:, cc:], x.new_zeros(b, c - cc, half)], dim=2)
        out_b = nn.functional.conv1d(xb[:, :, : t + half], w[cc:, None, : half + 1], groups=c - cc)
        # 청크 경계는 절대 프레임 번호로 계산한다(frame_offset = 이 입력 첫 프레임의 절대 번호). 그래야
        # 스트리밍에서 청크를 따로 넣어도 오프라인과 같은 경계가 잡힌다.
        # 빠른 경로: 입력 [frame_offset, frame_offset + t)가 청크 하나 안에 있으면 청크 끝 ≥ 입력 끝이다.
        # 그러면 청크 끝을 넘는 오른쪽 탭은 모두 xb 끝의 0 패딩을 읽으므로 가리기가 필요 없다. 실제
        # 스트리밍의 모든 step(온전한 청크와 flush의 부분 청크, frame_offset은 항상 청크 배수)이 여기에
        # 해당하고, 탭마다 비교·masked_fill 커널 3개(블록당 21개)가 빠진다. 오프라인 forward(입력이 여러
        # 청크)는 해당하지 않아 그대로 가린다.
        if fast and chunk_size > 0 and frame_offset % chunk_size + t <= chunk_size:
            end = None
        else:
            idx = torch.arange(t, device=x.device) + frame_offset
            end = (idx // chunk_size + 1) * chunk_size - 1 if chunk_size > 0 else None
        # 오른쪽 탭 j: 프레임 i+j가 i의 청크 끝을 넘으면 그 값을 먼저 0으로 가린 뒤 더한다. '모두 더한 다음
        # 넘는 탭을 빼는' 방식은 fp16 반올림에 미래 값이 섞여 미래 불변이 깨졌기 때문에 이렇게 바꿨다(독립
        # 검토 반영, docs/ai-verification.md). 덕분에 fp16에서도 경계 앞 출력이 비트 단위로
        # 같다(tests/test_model.py test_split_conv_future_invariant_fp16).
        # unfold로 (B, C, T, 15) 텐서를 만들던 첫 구현보다 이 'grouped conv1d + 오른쪽 탭 보정' 구현이 빨라서
        # 바꿨고, 출력은 독립 unfold 구현과 1e-5 안에서 같다(docs/experiments.md 0단계 결과,
        # test_split_conv_matches_reference).
        for j in range(1, half + 1):
            tap = xb[:, :, half + j : half + j + t]  # 프레임 i + j(x의 끝을 넘으면 0)
            if end is not None:
                tap = tap.masked_fill(((idx + j) > end)[None, None, :], 0.0)  # 청크 끝을 넘는 탭
            out_b = out_b + w[cc:, half + j][None, :, None] * tap
        return torch.cat([out_c, out_b], dim=1) + self.bias[None, :, None].to(x.dtype)


# Conformer conv 모듈: LayerNorm → pointwise Linear(d → 2d) → GLU → (패딩 프레임 0) → 분할 depthwise conv →
# LayerNorm → SiLU → pointwise Linear → dropout. 잔차 연결은 블록 쪽에서 한다.
# 원래 Conformer는 depthwise 뒤에 BatchNorm을 쓰지만 여기서는 LayerNorm(norm2)을 쓴다. 스트리밍·소규모
# 데이터에서 BN을 LN으로 바꾸는 것이 관행이고(docs/survey/01-encoders.md), LN은 프레임마다 계산되어 배치
# 구성이나 패딩, 청크 분할과 무관하게 같은 값을 낸다.
class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float, bacon: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = SplitDepthwiseConv1d(d_model, kernel, bacon)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 스트리밍 캐시 길이: depthwise 커널이 과거로 보는 최대 거리(k − 1 = 14프레임).
        self.cache_frames = kernel - 1

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int,
        left_cache: torch.Tensor | None = None,
        frame_offset: int = 0,
        valid: torch.Tensor | None = None,
        fast: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D), valid: (B, T) bool. 패딩 프레임은 depthwise conv 전에 0으로 만든다.

        fast는 빠른 스트리밍 경로용으로 depthwise conv에 그대로 넘긴다(SplitDepthwiseConv1d.forward 참고).

        반환: (출력, GLU 뒤 신호에서 잘라 낸 새 왼쪽 캐시 (B, D, k-1)).
        """
        # GLU는 2d 채널을 반으로 나눠 한쪽을 다른 쪽의 sigmoid로 곱한다(게이트). 이어서 (B, D, T)로 바꿔
        # conv1d 축 배치에 맞춘다.
        y = self.norm(x)
        y = nn.functional.glu(self.pointwise1(y), dim=-1).transpose(1, 2)  # (B, D, T)
        # 배치 패딩 프레임을 0으로 만든다. 그러지 않으면 패딩 위치의 값(어텐션 등에서 나온 것)이 양방향 절반의
        # 오른쪽 탭을 통해 유효 프레임으로 새어 들어올 수 있다(docs/ai-verification.md,
        # test_padding_does_not_change_valid_frames). 스트리밍에는 패딩이 없어 valid가 None이다.
        if valid is not None:
            y = y * valid[:, None, :].to(y.dtype)
        # 다음 청크용 캐시: (이전 캐시 + 이번 depthwise 입력)의 마지막 k−1프레임. 입력이 짧으면 왼쪽을 0으로
        # 채워 길이를 맞춘다. 오프라인 경로에서는 이 값을 버린다.
        new_cache = torch.cat([left_cache, y] if left_cache is not None else [y], dim=2)[
            :, :, -self.cache_frames :
        ]
        if new_cache.shape[2] < self.cache_frames:
            new_cache = nn.functional.pad(new_cache, (self.cache_frames - new_cache.shape[2], 0))
        y = self.depthwise(y, chunk_size, left_cache, frame_offset, fast=fast).transpose(1, 2)
        y = self.pointwise2(self.act(self.norm2(y)))
        return self.dropout(y), new_cache
