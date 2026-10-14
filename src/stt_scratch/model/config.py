"""모델과 스트리밍 설정(docs/survey/README.md "결정"에서 정함).

모델 구조, 동적 청크 학습, SpecAugment의 하이퍼파라미터를 ModelConfig 하나에 모은다.
"""

# [설계 이유] 기본값은 docs/survey/README.md "결정" 표의 구조(Conformer 12층, d=256, 4헤드, FFN 1024,
# depthwise 커널 15, 약 21M)와 2단계 Run A 처방이다. 학습 스크립트는 asdict(cfg)를 체크포인트의 "config"에
# 통째로 저장하고, 평가·서비스는 ModelConfig(**state["config"])로 같은 구조를 다시
# 만든다(evaluate.load_model). 그래서 명령행 옵션으로 바꾼 값(--sc-layers, --block 등)도 체크포인트 하나로
# 재현된다.
# [주의] 최종 모델은 3단계 (a) 판정에 따라 sc_layers=(6,), sc_weight=0.3, sc_feedback=False(InterCTC 단독)로
# 학습됐으므로 아래 기본값과 다르다(docs/experiments.md 3단계 결과 (a)).
# [연결] train.build_model이 명령행 옵션과 manifests/mel_stats.json으로 채우고,
# conformer·ebranchformer·streaming 모듈이 읽는다.

from __future__ import annotations

from dataclasses import dataclass, field


# dataclass로 둔 이유: asdict로 JSON·체크포인트에 그대로 직렬화되고, 키워드 인자로 바로 복원된다. 필드를 새로
# 더해도 기본값이 있으면 옛 체크포인트의 config로도 생성할 수 있다.
@dataclass
class ModelConfig:
    # 출력 클래스 수 = blank·<unk>·공백 3개 + 학습 전사에 나온 한글 음절(실제 어휘 1,288개,
    # docs/experiments.md 0단계 결과). 어휘 파일에서 정해지므로 기본값이 없다.
    vocab_size: int
    # 입력 log-mel 차원(features.N_MELS와 같은 80).
    n_mels: int = 80
    # 인코더 크기: 12층, 폭 256, 헤드 4개(헤드당 64차원), FFN 1024. 약 65 h 데이터에는 20~40M 파라미터가
    # 적정하다는 조사 결론(docs/survey/README.md "결정")에 맞춘 약 21M 구성이다.
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 4
    ffn_dim: int = 1024
    # conv 모듈 depthwise 커널 길이(25 fps에서 15프레임 = 600 ms). 홀수여야 한다(SplitDepthwiseConv1d가 중앙
    # 탭을 기준으로 좌우를 나눈다).
    conv_kernel: int = 15
    dropout: float = 0.1
    # 서브샘플링 Conv2d 채널 수와 인과 여부. True면 시간축 패딩을 왼쪽에만 넣어 look-ahead가 0이다. False(양쪽
    # 패딩, 20 ms look-ahead)는 스트리밍 경로(forward_streaming)가 거부한다.
    subsampling_channels: int = 256
    causal_subsampling: bool = True
    # RoPE 회전 주파수의 밑(일반적인 기본값 10000).
    rope_theta: float = 10000.0
    # RoPE 전에 q와 k에 헤드별 LayerNorm을 건다(QK-norm). 이것이 없으면 LR 7e-4 이상에서 어텐션 로짓
    # 표준편차가 60~111까지 커졌으므로 로짓을 묶어 둔다(docs/experiments.md 규칙 변경 13). False = 처음 등록한
    # 구조.
    qk_norm: bool = True
    # 중간 CTC 헤드 설정. 기본값 (3, 6, 9)·λ 0.5·피드백 켬은 2단계 Run A의 자기조건 CTC 처방이다. 3단계
    # (a)에서 '자기조건 CTC가 dev_mix에서 상대 5 % 이상 좋다'는 H3a가 충족되지 않아 InterCTC(6층, w 0.3,
    # 피드백 끔)로 교체했다(docs/experiments.md 3단계 결과 (a)). 손실 합산식은 ctc.total_loss에 있다.
    # sc_layers: 출력에 중간 CTC 헤드를 붙이고(자기조건 CTC면) 그 softmax를 다음 블록에 되먹이는 블록
    # 번호(1부터). 빈 튜플 = 일반 CTC.
    sc_layers: tuple[int, ...] = (3, 6, 9)
    sc_weight: float = 0.5  # 전체 손실 = (1 - w) × 최종 + w × mean(중간)
    # False = 일반 InterCTC: 중간 손실은 그대로 두고, 다음 블록으로의 softmax 피드백만 끈다.
    sc_feedback: bool = True
    # BACON식 분할 conv: 960 h·110M 조건의 근거라 65 h에서의 이득은 검증되지 않았으므로, False 한 줄로 전 채널
    # 인과 conv로 되돌릴 수 있게 플래그로 뒀다(docs/survey/README.md "결정" conv 모듈).
    # conv 모듈: True = BACON 분할(채널 절반은 인과, 절반은 청크 경계 안 양방향), False = 모든 채널 인과.
    bacon: bool = True
    # 인코더 블록 종류: "conformer"(Run A) 또는 "ebranchformer"(3단계 후보 b; MHSA와 cgMLP 두 갈래를
    # depthwise conv로 병합). cgMLP와 병합 conv도 같은 청크 경계 분할 depthwise conv를 쓴다.
    # E-Branchformer는 3단계 (b) 비교용으로만 쓰였고, dev_mix가 Run A보다 개선되지 않아(22.49 vs 22.05)
    # Conformer를 유지했다(docs/experiments.md 3단계 결과 (b)). (b) 실행 때는 파라미터 수를 맞추려고
    # --ffn-dim 768 --cgmlp-units 896을 줬다(규칙 변경 17).
    block: str = "conformer"
    cgmlp_units: int = 1024
    cgmlp_kernel: int = 31
    merge_kernel: int = 31
    # U2식 동적 청크 학습: 배치마다 50 % 확률로 전체 문맥, 나머지는 청크 {4, 8, 16,
    # 32}프레임(160/320/640/1,280 ms)과 좌측 문맥 {64, 128, 무제한} 중 하나를 고른다. 배치 단위로만 뽑는 것은
    # 조사 검증 C13(배치 단위 샘플링만 작동)을 따른 것이고, 4프레임은 160 ms 평가 축을 채우려고 더했다. 우측
    # 문맥 혼합 학습은 넣지 않았다(C12, docs/survey/README.md "결정" 스트리밍 학습).
    # full_context_prob 확률로 배치가 전체 문맥을 보고, 아니면 청크 크기(25 fps 프레임)와 좌측 문맥
    # 크기(프레임)를 각각 균등하게 뽑는다.
    full_context_prob: float = 0.5
    chunk_sizes: tuple[int, ...] = (4, 8, 16, 32)
    left_contexts: tuple[int, ...] = (64, 128, -1)  # -1 = 무제한
    # SpecAugment 크기: 발화마다 주파수 마스크 2개(폭 0~27 bin), 시간 마스크 5개(각각 길이 0~유효 프레임의 5
    # %). 모델 안에서 log-mel에 걸며 학습 때만 쓴다. 처음부터 켜면 CTC 정렬이 시작되지 않아 학습에서는 epoch
    # 3부터 켠다(규칙 변경 13). 3단계 (d) CR-CTC는 시간 마스크를 12개로 늘려 썼다(규칙 변경 17).
    freq_masks: int = 2
    freq_mask_width: int = 27
    time_masks: int = 5
    time_mask_ratio: float = 0.05
    # 전역 정규화 통계(80차 평균·표준편차). 0단계에서 scripts/compute_mel_stats.py가 만든
    # manifests/mel_stats.json을 train.build_model이 넣는다. config에 들어 있으므로 체크포인트만 있으면
    # 평가·서비스가 같은 정규화를 쓴다. 비어 있으면 모델이 평균 0·표준편차 1(정규화 없음)로 둔다.
    mel_mean: list[float] = field(default_factory=list)
    mel_std: list[float] = field(default_factory=list)

    # 헤드당 차원(256 / 4 = 64). RoPE와 QK-norm이 이 크기 단위로 동작한다.
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
