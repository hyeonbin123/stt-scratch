"""규칙형 CTC 끝점 검출(docs/experiments.md 6단계; sherpa/WeNet 방식).

스트리밍 서비스에서 발화 구간의 끝을 정한다. 모델의 blank 확률만 보고 '말이 끝났다'를 판정한다.
blank 사후확률이 `blank_threshold` 이상인 프레임을 '묵음'으로 본다. 끝점은 다음 중 하나가 맞으면 난다.
  rule 1: 꼬리 묵음 >= rule1_s(무엇이 디코딩됐는지와 무관),
  rule 2: 이 세그먼트에서 무엇이든 디코딩됐고 꼬리 묵음 >= rule2_s,
  rule 3: 세그먼트 길이가 rule3_s 이상.
프레임은 25 fps(40 ms)의 인코더 프레임이다.
"""

# [왜 규칙형인가] 별도 VAD 모델 없이 CTC 출력만으로 동작하고, sherpa·WeNet이 쓰는 방식이라 바로 적용할 수
# 있다(docs/survey/02-streaming.md). 기본값(blank ≥ 0.8을 묵음, rule1 2.4 s, rule3 20 s)은 사전 등록 값이고,
# rule2는 6단계에서 {0.8, 1.2} s 중 '조기 끝점으로 인한 CER 증가 ≤ 0.2포인트'를 만족하는 짧은 쪽을 고르기로
# 했는데 둘 다 넘어 등록된 대체 규칙대로 1.2 s가 됐다(docs/experiments.md 6단계 결과 '세션'). 이때 끝점 확정
# 지연 EP50/EP90은 1,295 / 1,465 ms였다. 캐시 유지 상태로 다시 잰 7단계 뒤 후속 측정(P2, 규칙 변경 26)에서는
# 0.8 s가 조건을 만족해(CER 증가 0.000포인트, EP50 905 ms) 서버 기본값이 0.8 s다. 이 설정 클래스의 기본값은
# 6단계 측정 도구가 쓰는 1.2 s 그대로 둔다.
# [연결] service.session.StreamSession이 청크마다 update를 부르고, 규칙 이름이 나오면 세그먼트를 닫는다.
# 서버는 --rule2로 rule2만 바꿀 수 있다.

from __future__ import annotations

from dataclasses import dataclass

import torch

# 인코더 프레임 하나의 길이(25 fps = 40 ms). 프레임 수 × FRAME_S로 초 단위 규칙과 비교한다.
FRAME_S = 0.04


# 끝점 규칙 설정. blank_threshold: blank 사후확률이 이 값 이상인 프레임을 묵음으로 센다. rule1_s: 아무것도
# 디코딩하지 않았어도 묵음이 이만큼 이어지면 닫음(긴 무음 구간 정리). rule2_s: 토큰이 하나라도 나온 뒤 묵음이
# 이만큼이면 닫음(정상적인 발화 끝). rule3_s: 세그먼트 길이 상한(묵음 없이 계속 말해도 20 s에서 강제로 닫음).
@dataclass
class EndpointConfig:
    blank_threshold: float = 0.8
    rule1_s: float = 2.4
    rule2_s: float = 1.2
    rule3_s: float = 20.0


# 세그먼트 하나의 끝점 상태: 끝에서부터 이어진 묵음 프레임 수, 세그먼트 전체 프레임 수, 토큰이 나왔는지 여부.
# 세그먼트가 닫히면 세션이 reset으로 초기화한다.
class Endpointer:
    def __init__(self, cfg: EndpointConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.trailing_silence = 0
        self.segment_frames = 0
        self.decoded_something = False

    # 판정은 청크 단위로 한 번만 하므로 실제 끝점 시각은 청크 경계(320 ms 단위)로 올림된다. 규칙 우선순위는
    # rule1 → rule2 → rule3.
    def update(self, log_probs: torch.Tensor, new_tokens: int) -> str | None:
        """청크 하나의 새 프레임들로 상태를 갱신하고, 맞은 규칙 이름(rule1/rule2/rule3) 또는 None을 돌려준다.

        log_probs: 새 프레임들의 fp32 (T, V). new_tokens: 이 프레임들에서 나온 토큰 수.
        """
        # fp32 log-prob에서 blank(id 0) 확률을 꺼내 프레임별 묵음 여부를 정한다. 묵음이 아닌 프레임이 하나라도
        # 오면 꼬리 묵음 카운터는 0으로 돌아간다.
        blank = log_probs[:, 0].exp()
        is_sil = (blank >= self.cfg.blank_threshold).tolist()
        if new_tokens > 0:
            self.decoded_something = True
        for s in is_sil:
            self.trailing_silence = self.trailing_silence + 1 if s else 0
            self.segment_frames += 1
        trailing_s = self.trailing_silence * FRAME_S
        if trailing_s >= self.cfg.rule1_s:
            return "rule1"
        if self.decoded_something and trailing_s >= self.cfg.rule2_s:
            return "rule2"
        if self.segment_frames * FRAME_S >= self.cfg.rule3_s:
            return "rule3"
        return None
