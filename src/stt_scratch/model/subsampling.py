"""인과 Conv2d 서브샘플링(시간축 4배). 모델 입력 단계다.

log-mel(100 fps)을 Conv2d 두 번(각각 시간 stride 2)으로 25 fps 인코더 프레임으로 줄이고 d_model 차원으로
투영한다. 출력 프레임 i는 입력 프레임 4i + 3 이하(실제로는 4i까지, 즉 자기 4프레임 묶음 안)만 보므로
전단부가 현재 청크 밖의 look-ahead를 더하지 않는다. 출력 길이 = ceil(ceil(T/2)/2).
"""

# [왜 인과형인가] 표준 Conformer 서브샘플링은 시간축을 양쪽으로 패딩해 20 ms 앞을 본다. 이 프로젝트는 청크
# 경계 뒤를 보지 않는 스트리밍이 목표라 시간축 패딩을 왼쪽에만 넣어 look-ahead를 0으로
# 했다(docs/survey/README.md "결정" 서브샘플링).
# [왜 4배인가] 학습 전사의 토큰률(음절 + 공백)이 p95 8.01, 최대 10.64 토큰/s라 8배(12.5 fps)에서는 CTC 길이
# 조건(출력 프레임 ≥ 토큰 수 + 반복 수 + 1)에 걸리는 발화가 생길 수 있다. 4배(25 fps)에서는 위반
# 0건이었다(docs/data.md 음절, docs/experiments.md 0단계 결과·H0.1).
# [연결] conformer.ChunkConformerCTC가 forward(학습·오프라인)를, streaming.StreamingEncoder.step이
# forward_streaming(캐시 스트리밍)을 부른다.

from __future__ import annotations

import torch
from torch import nn


# mel 프레임 수 T → 인코더 프레임 수 ceil(ceil(T/2)/2). 정수와 텐서를 모두 받는다(텐서는 배치의 발화별 길이).
# 음수 나눗셈 트릭 대신 올림을 명시한 식으로 쓴다(ruff B002 지적 뒤 수정, docs/ai-verification.md).
# tests/test_model.py가 실제 모듈 출력 길이와 대조한다.
def subsampled_length(t: torch.Tensor | int) -> torch.Tensor | int:
    if isinstance(t, int):
        return ((t + 1) // 2 + 1) // 2  # ceil(ceil(t/2)/2)
    return torch.div(torch.div(t + 1, 2, rounding_mode="floor") + 1, 2, rounding_mode="floor")


# 인과 Conv2d 2단 서브샘플링. 입력 (B, T, 80) log-mel을 채널 1장짜리 이미지로 보고 3×3 커널, stride 2를 두 번
# 적용한다. 주파수축도 80 → 40 → 20으로 줄어들어, 출력은 채널 256 × 주파수 20 = 5,120차원을 Linear로
# d_model(256)에 투영한다.
# 계산해 보면 출력 프레임 i는 mel 프레임 4i까지만 본다(모듈 docstring의 '4i + 3 이하'는 같은 4프레임 묶음
# 안이라는 상한). 그래서 청크 경계(mel 4 × chunk_size 단위) 뒤의 정보가 들어오지 않는다.
class CausalConv2dSubsampling(nn.Module):
    def __init__(self, n_mels: int, channels: int, d_model: int, causal: bool = True) -> None:
        super().__init__()
        self.causal = causal
        # 시간축 패딩은 _pad_time에서 직접 넣고(인과형이면 왼쪽만), 주파수축 패딩은 Conv2d가 대칭으로 넣는다.
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=(0, 1))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=(0, 1))
        self.act = nn.SiLU()
        # 주파수축 출력 크기: 대칭 패딩 1, 커널 3, stride 2를 두 번 거친 크기(80 → 40 → 20). 투영 Linear의
        # 입력 차원을 정한다.
        freq = (n_mels + 2 - 3) // 2 + 1
        freq = (freq + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * freq, d_model)
        # 스트리밍에서 각 conv가 청크 앞에서 빌려 와야 하는 과거 입력 프레임 수(커널 3 = 현재 + 과거 2).
        self.cache_frames = 2

    # 시간축 패딩만 따로 한다(Conv2d의 padding은 주파수축에만 1). 인과형은 왼쪽 2프레임을 0으로 채우며, 첫
    # 스트리밍 청크도 같은 0 패딩을 쓰므로 오프라인과 결과가 같다.
    def _pad_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F). 인과형: 왼쪽에 2. 비인과형: 양쪽에 1씩(표준 방식, 20 ms look-ahead).
        if self.causal:
            return nn.functional.pad(x, (0, 0, 2, 0))
        return nn.functional.pad(x, (0, 0, 1, 1))

    # 학습·오프라인 경로: 발화 전체를 한 번에 줄이고 발화별 출력 길이를 함께 돌려준다. 패딩 프레임에 대한
    # 출력은 뒤의 마스크가 무시한다.
    def forward(self, mel: torch.Tensor, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = mel.unsqueeze(1)  # (B, 1, T, F)
        x = self.act(self.conv1(self._pad_time(x)))
        x = self.act(self.conv2(self._pad_time(x)))
        b, c, t, f = x.shape
        # (B, C, T', F) → (B, T', C·F)로 펼쳐 프레임마다 채널·주파수 값을 하나의 벡터로 만든 뒤 d_model로
        # 투영한다.
        x = self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return x, subsampled_length(frames)

    # 캐시 스트리밍 경로. 청크 하나의 mel을 받아 그 청크의 인코더 프레임만 계산한다. 캐시는 (conv1 입력의
    # 마지막 2프레임, conv2 입력의 마지막 2프레임)이며, 오프라인에서 왼쪽 0 패딩이 하던 역할을 이전 청크의
    # 실제 값이 대신한다.
    # T가 4의 배수여야 하는 이유: stride 2가 두 번이라 청크 길이가 4의 배수일 때만 다음 청크의 stride 위상이
    # 오프라인과 맞는다. 스트림의 마지막 부분 청크는 뒤에 이어지는 청크가 없으므로 아무 길이여도 오프라인과
    # 같다(tests/test_model.py test_streaming_partial_last_chunk).
    def forward_streaming(
        self, mel: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """mel: (1, T, F), T는 4의 배수. cache = (마지막 mel 2프레임, conv1 출력의 마지막 2프레임)."""
        assert self.causal, "streaming needs the causal front-end"
        x = mel.unsqueeze(1)
        # 첫 청크는 오프라인과 같은 왼쪽 0 패딩, 이후 청크는 이전 청크의 마지막 2프레임을 앞에 붙인다. 붙인
        # 입력의 마지막 2프레임이 다음 청크의 캐시가 된다.
        if cache is None:
            x1_in = nn.functional.pad(x, (0, 0, 2, 0))
        else:
            x1_in = torch.cat([cache[0], x], dim=2)
        new_mel_cache = x1_in[:, :, -2:]
        x1 = self.act(self.conv1(x1_in))
        # conv2도 같은 방식으로 conv1 출력의 과거 2프레임을 캐시로 이어 붙인다.
        if cache is None:
            x2_in = nn.functional.pad(x1, (0, 0, 2, 0))
        else:
            x2_in = torch.cat([cache[1], x1], dim=2)
        new_x1_cache = x2_in[:, :, -2:]
        x2 = self.act(self.conv2(x2_in))
        b, c, t, f = x2.shape
        out = self.out(x2.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return out, (new_mel_cache, new_x1_cache)
