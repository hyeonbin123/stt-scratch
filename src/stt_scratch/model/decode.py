"""CTC 디코딩: 배치 greedy(greedy_batch), 청크마다 덧붙이기만 하는 증분 스트리밍(GreedyState), 확정 자막용
편향 디코딩 분기(biased_batch: 편향 목록이 비면 greedy_batch 그대로)."""

# [왜 greedy인가] 프레임별 argmax → 같은 토큰 반복 축약 → blank 제거만 하면 되고 상태가 '직전 프레임 토큰'
# 하나뿐이라, 캐시 스트리밍에서 이미 낸 토큰이 바뀌지 않는다(부분 자막이 덧붙기만 함). 조사에서 1차 결과는
# greedy와 prefix beam의 차이가 거의 없다고 봤고(WeNet 보고), LM을 쓰는 2차 디코딩(5단계)은
# 생략했다(docs/survey/README.md "결정" 디코딩, docs/experiments.md 5단계 판정).
# [연결] evaluate.py(full·simulated는 greedy_batch, 실제 스트리밍은 GreedyState), service.session(서비스),
# scripts/stage2_report.py. biased_batch는 서비스에 아직 연결하지 않았다: 확정 자막 핫워드 편향은 등록 측정
# (docs/experiments.md 규칙 변경 29) 뒤에 옵션으로만 붙이고, 지금은 scripts/bias_eval.py(지연 측정)만 쓴다.

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from stt_scratch.biasing import ContextTrie, prefix_beam_search


# 발화별 유효 프레임까지만 본다. 결과는 발화별 토큰 id 목록. argmax는 장치에서 한 번에 하고, 반복 축약은
# CPU로 옮긴 목록에서 발화별로 한다.
def greedy_batch(logits: torch.Tensor, out_lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """배치 greedy: 프레임별 argmax, 직전 프레임과 같은 토큰은 한 번만, blank는 버린다. logits: (B, T, V)."""
    best = logits.argmax(dim=-1).cpu()
    results: list[list[int]] = []
    for b in range(best.shape[0]):
        seq = best[b, : int(out_lengths[b])].tolist()
        out: list[int] = []
        prev = blank
        for tok in seq:
            if tok != blank and tok != prev:
                out.append(tok)
            prev = tok
        results.append(out)
    return results


# 확정 자막(끝점 뒤 2차 패스)의 디코딩 분기. 편향 목록이 없거나(None) 항목이 하나도 없으면(빈 목록, 모든
# 어절이 빠진 목록) greedy_batch를 그대로 부르므로 출력이 지금 서비스와 비트 단위로 같다(기록 수치가 바뀌지
# 않음; tests/test_biasing.py). 목록이 있으면 발화마다 유효 프레임만 잘라 fp32 log_softmax를 하고, 프레임별
# 상위 top_k (id, log 확률)와 blank log 확률만 CPU로 옮겨 biasing.prefix_beam_search로 디코딩한다. 자르고 나서
# 같은 연산을 하는 것은 scripts/final_pass_dump.frame_stats와 같은 계산이라, 덤프로 잰 오프라인
# 평가(scripts/bias_eval.py)가 이 함수의 출력과 같다(tests/test_biasing.py). 입력: logits (B, T, V)(fp16·fp32
# 모두; log 확률을 넘겨도 된다), 발화별 출력 길이, 편향 trie, 빔 크기, 프레임당 후보 수. 출력: 발화별 토큰 id
# 목록(greedy_batch와 같은 형식). 주의: 빔은 CPU의 순수 Python이라 세그먼트 길이에 비례해 느려진다(지연은
# scripts/bias_eval.py --latency로 잰다).
def biased_batch(
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    context: ContextTrie | None = None,
    *,
    beam: int = 8,
    top_k: int = 10,
    blank: int = 0,
) -> list[list[int]]:
    """확정 자막 디코딩: 편향 목록이 없거나 비면 greedy_batch, 있으면 top-k prefix beam + 음절 trie 보너스."""
    if context is None or context.entries == 0:
        return greedy_batch(logits, out_lengths, blank)
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    results: list[list[int]] = []
    for b, n in enumerate(out_lengths.tolist()):
        lp = torch.log_softmax(logits[b, : int(n)].float(), dim=-1)
        top = torch.topk(lp, min(top_k, lp.shape[-1]), dim=-1)
        results.append(
            prefix_beam_search(
                top.indices.cpu().numpy(),
                top.values.cpu().numpy(),
                lp[:, blank].cpu().numpy(),
                context,
                beam=beam,
                blank=blank,
            )
        )
    return results


# 검증: tests/test_model.py test_greedy_state_matches_batch_decoding. tokens는 지금까지 낸 전체 토큰,
# frame_of_token은 각 토큰이 나온 출력 프레임 번호(호출자가 넘긴 frame_offset + 청크 안 위치)다. 평가
# 경로(evaluate.decode_streaming)는 발화 시작부터 센 번호를 넘기고, 서비스 세션은 StreamState.frames_seen으로
# 만든 번호를 넘긴다. 그래서 서비스에서는 인코더 상태를 새로 만든 뒤(연결, flush, 캐시를 비우는 끝점) 처리한
# 프레임이 REBASE_AT(2^16프레임, 25 fps에서 약 44분)에 이르면 이 값이 연결 이후 절대 번호가 아니라 청크
# 배수만큼 당겨진 '재기준된 번호'가 되고, 재기준을 가로지르는 세그먼트에서는 값이 그만큼
# 줄어든다(model/streaming.py의 StreamState·rebase_positions 설명). 현재 이 값을 읽는 코드는 없다.
# 서비스에서는 세그먼트마다 새 상태로 시작한다. 그래서 끝점이 같은 토큰이 이어지는 구간에 걸리면 그 토큰이 한
# 번 더 나올 수 있다(docs/experiments.md 6단계 결과 '세션').
@dataclass
class GreedyState:
    """스트리밍 greedy 상태.

    직전 프레임의 토큰(prev)을 청크 사이에 넘겨, 청크 경계를 가로질러 이어지는 같은 토큰이 올바르게 한 번으로
    축약되게 한다.
    """

    prev: int = 0
    tokens: list[int] = field(default_factory=list)
    frame_of_token: list[int] = field(default_factory=list)  # 토큰마다 나온 출력 프레임(frame_offset 기준)

    # 로짓 대신 log-prob을 넘겨도 된다. 이전 토큰은 절대 고치지 않으므로 6단계에서 잰 부분 자막
    # 수정률(UPWR)이 구조상 0이다(H6.3). frame_offset은 이 청크 첫 프레임의 번호다(서비스 세션은
    # StreamState.frames_seen 기준이라 긴 세션에서는 재기준된 번호; 위 설명).
    def push(self, logits: torch.Tensor, frame_offset: int, blank: int = 0) -> list[int]:
        """logits: 새 프레임들의 (T, V). 새로 나온 토큰만 돌려주고 tokens에 덧붙인다(이전 토큰은 그대로)."""
        new: list[int] = []
        for i, tok in enumerate(logits.argmax(dim=-1).tolist()):
            if tok != blank and tok != self.prev:
                new.append(tok)
                self.frame_of_token.append(frame_offset + i)
            self.prev = tok
        self.tokens.extend(new)
        return new

    # 상태를 처음으로 되돌린다. prev를 blank(0)로 두어 다음 첫 프레임의 토큰이 새 토큰으로 인정되게 한다. 현재
    # 세션은 세그먼트마다 새 GreedyState를 만들기 때문에 이 메서드를 부르는 곳은 없다.
    def reset(self) -> None:
        self.prev = 0
        self.tokens = []
        self.frame_of_token = []
