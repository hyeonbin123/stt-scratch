# 01. ASR 인코더 구조 조사 (2022-01 ~ 2026-10)

조사일: 2026-10-10. 대상: Conformer 계열, Branchformer 계열, 다중 프레임률(Squeezeformer·Zipformer·Efficient/Fast Conformer), SSM/Mamba 계열, 선형 어텐션·기타, 2025~2026 신규. 모든 수치는 아래 5절의 페이지에서 직접 읽은 것이며, 출처를 arXiv id 또는 URL로 표기한다. 확인하지 못한 것은 (확인 필요)로 표시한다.

우리 조건: 공개 한국어 약 65 h(Zeroth 51.6 h + FLEURS ko + Common Voice ko), 사전학습 없음, RTX 2080 Ti 11 GB(fp16만, bf16 없음) 1장, 최종 목표는 스트리밍 STT 서비스.

## 1. 요약

- 1순위 **Zipformer**(ICLR 2024, 2310.11230): 같은 틀에서 Zipformer-S 23.3M이 2.42/5.73(LibriSpeech test-clean/other)로 Conformer-M 28.4M 2.96/7.11보다 좋고, 30초 입력 FLOPs가 40.8 G로 WER 2.5/6.0 이하 모델 중 가장 적다(Conformer-M 77.0 G 대비 47% 적음; 표 2 안에는 Squeezeformer-XS 18.2 G, 재현 Conformer-S 29.1 G 등 더 적은 모델도 있다). 혼합정밀(V100, 사실상 fp16)로 학습했고 Balancer만 혼합정밀 불안정 대책으로 서술됨(BiasNorm·Whitener·ScaledAdam의 동기는 별개). 공식 레시피에 청크 인과 스트리밍(320 ms 청크 3.06/7.81, 66M; simulated 값, 실제 청크 디코딩 3.06/7.79)과 `--use-fp16 1`이 들어 있다. 한국어 스트리밍 전례(KsponSpeech Zipformer2)도 있다.
- 2순위 **E-Branchformer**(SLT 2022, 2210.00077): 100 h 규모에서 가장 통제된 비교가 있고(LS-100 dev-clean/other: 38.5M 6.1/16.7 vs Conformer 39.0M 6.3/17.0; test 6.3/17.0 vs 6.6/17.2; AED+joint CTC, LM 없음), 소규모 데이터·CTC에서 Conformer보다 학습 실패가 적다(TEDLIUM2 CTC 33M 인코더, 피크 LR 2e-3, pure CTC: Conformer 10회 중 6회 실패, E-Branchformer 2회 실패 — 분모 미명시, 표 없는 일화적 보고). 표준 LayerNorm·Adam으로 구현이 단순하지만 FLOPs가 Zipformer보다 크고 스트리밍은 청크 마스킹을 직접 붙여야 한다.
- 3순위 **Fast Conformer(8x) + 캐시 인식 스트리밍**(ASRU 2023, 2305.05084; 2312.17279): 연산 2.9배 절감, 성숙한 스트리밍 레시피, Zeroth-Korean 50 h에서 FastConformer-Transducer 약 50M(hidden 400)으로 문자 단위 CER 1.51%라는 직접 전례가 있다(학습 환경 미기재). 다만 소형 모델 정확도는 Zipformer/E-Branchformer보다 근거가 약하고 8x 프레임률은 CTC·한국어 음절 단위와의 궤합을 검토해야 한다.
- **ConMamba/LC-BiMamba**는 저자원(80 h, 5 h)에서 Conformer와 동급 WER, 학습시간 40~45% 절감, 스트리밍 듀얼모드에서 Conformer보다 좋은 결과가 있어 유망하지만, Mamba CUDA 커널이 Linux 전제이고(sm_75 컴파일 대상에는 포함) CTC 단독에서는 test_other가 약하다는 보고가 있어 "실험군"으로만 둔다.
- 공통 설계 메모: 위치 인코딩은 상대위치(Transformer-XL식 또는 Zipformer의 compact rel-pos)가 기본, RoPE는 WER 동급에 학습시간 최대 21% 절감(단, 18층 256차원 순수 CTC에서는 RelPos가 미세하게 우세). 서브샘플링 4x→8x는 CTC greedy 기준 WER 0.3~0.7 절대 상승(HydraFormer), Transducer 기준 0.13(Multi-blank). 65 h에는 20~40M 파라미터가 적정 구간이라는 간접 근거가 많다.

## 2. 논문별 정리

### 2.1 요약표

| 모델 | 출처 | 핵심 아이디어 | 대표 결과(데이터, WER/CER, 파라미터) | 소규모·스트리밍 근거 | 우리 조건에서 장/단 |
|---|---|---|---|---|---|
| Conformer | Interspeech 2020, 2005.08100 | MHSA(상대위치) + 깊이별 conv + 마카론 FFN | LS960 S 10.3M 2.7/6.3, M 30.7M 2.3/5.0, L 118.8M 2.1/4.3 (LM 없음) | 모든 후속 연구의 기준선. 소규모 CTC에서 학습 실패 보고(2305.11073, TEDLIUM2 1개 설정·10회 시도의 일화적 보고) | 구현 단순, 참고자료 최다 / 소규모·fp16에서 안정성 열위 |
| Efficient Conformer | ASRU 2021, 2109.01163 | 3단계 점진적 다운샘플링(총 8x), 1단계 그룹 어텐션 | LS960 CTC 13.2M 3.57/8.99 vs Conformer 13.0M 4.07/10.25; 추론 29%·학습 36% 빠름 | **RTX 2080 Ti 4장으로 학습**. 스트리밍 결과 없음 | 우리 GPU에서 검증된 소형 구조 / 코드·커뮤니티 작음 |
| Squeezeformer | NeurIPS 2022, 2206.00888 | 시간축 U-Net(7블록 40 ms→80 ms→업샘플), post-LN+스케일, Swish 단일화 | LS960 CTC XS 9.0M 3.74/9.09, S 18.6M 3.08/7.47, SM 28.2M 2.79/6.89 (Conformer-CTC-M 27.4M 3.20/7.90) | 업샘플 없이 다운샘플만 하면 수렴 실패 보고. 스트리밍 없음 | 소형 CTC 정확도 우수 / TPU 레시피, 스트리밍 설계 부재 |
| Branchformer | ICML 2022, 2207.02971 | MHSA 가지 + cgMLP 가지 병렬, concat+linear 병합 | Aishell 45.4M 4.19/4.43 (Conformer 46.3M 4.24/4.62); SWBD300 43.7M 10.9 vs 11.1 | "짧은 발화·제한된 데이터에서 Conformer보다 학습이 안정적"(저자 서술), Speech Commands에서 Conformer 발산 | 안정성 / E-Branchformer에 대체됨 |
| E-Branchformer | SLT 2022, 2210.00077; 비교연구 Interspeech 2023, 2305.11073 | 병합을 concat→DwConv(k=31)→linear로 강화, FFN 추가 | LS960 B 41.1M 2.49/5.61, L 148.9M 2.14/4.55; **LS-100 dev-clean/other 38.5M 6.1/16.7 vs Conformer 39.0M 6.3/17.0** (test 6.3/17.0 vs 6.6/17.2; AED+joint CTC, LM 없음) | 15개 ASR 코퍼스에서 거의 모두 ≥ Conformer; TEDLIUM2 CTC 33M 학습 실패 2회 vs 6/10(안정성 실험은 표 외 서술, 해당 설정 WER 미보고, 정밀도 미기재); OWSM v3.1에서 10% 데이터로 가장 빨리 수렴 | 소규모 안정성 최상 / 블록당 연산 큼, 스트리밍 레시피 약함 |
| Zipformer | ICLR 2024, 2310.11230 | 6스택 U-Net(50/25/12.5/6.25/12.5/25 Hz), 블록 내 어텐션 가중치 공유, BiasNorm, SwooshR/L, ScaledAdam | LS960 pruned RNN-T S 23.3M 2.42/5.73, M 65.6M 2.21/4.79, L 148.4M 2.06/4.63; GFLOPs(30 s) 40.8/62.9/107.7; Aishell-1 S 30.2M 4.67 | icefall 스트리밍(66M, causal) 320 ms 3.06/7.81, 640 ms 2.81/7.15(simulated streaming, epoch 30 avg 8; 실제 청크 디코딩 3.06/7.79, 2.84/7.16); 한국어 Zipformer2 스트리밍 155.7M FP16(KsponSpeech CER 7.16 오프라인/7.8 청크32); Hakka 74 h에서 사용 | 소형 정확도·FLOPs 최상, 혼합정밀 안정화 장치(Balancer) / 구조·옵티마이저가 특수, k2 의존 |
| Fast Conformer | ASRU 2023, 2305.05084 | 8x 깊이별 분리 conv 서브샘플링(256ch, K=9), conv 커널 31→9, 제한 컨텍스트+글로벌 토큰 | LS test-other greedy: RNNT Conformer 115M 5.19 → FC 109M 4.99; CTC 121M 5.74 → 115M 5.64; MACs 2.9배↓, 처리량 2.8배↑ | 캐시 인식 스트리밍(2312.17279): 114M FP16, 오프라인 CTC/RNNT 5.7/5.0, 지연 240 ms 8.8/7.8, 1360 ms 7.1/6.3; Zeroth 50 h FastConformer-T ≈50M, 문자 단위, CER 1.51 (2404.04295; 학습 환경 미기재) | 연산 절감·스트리밍 성숙·한국어 전례 / 32 GPU 레시피, 소형 정확도 근거 부족 |
| ConMamba(Speech Slytherin) | 2407.09732 (ICASSP 2025) | Conformer의 MHSA를 양방향 Mamba로 교체, 위치 임베딩 불필요 | LS960 S 14.1M 4.0/9.5 vs Conformer 13.3M 4.1/10.0; CTC 31.6M 3.9/10.3 vs 28.8M 4.3/11.3; L은 2.8/6.7 vs 2.6/6.7 | 메모리 이점은 약 30 s, 속도 이점은 약 60 s 이상에서만 | 소형에서 소폭 우세 / 짧은 발화에서는 효율 이점 없음 |
| Mamba in Speech | 2405.12609 (v6 2025-04) | ExtBiMamba(방향별 별도 투영)로 MHSA 교체 | LS-100 ConExtBiMamba 41.6M 5.9/6.0 vs Conformer-Large 42.2M 6.2/6.4; 단독 Mamba는 40.8/40.0으로 붕괴 | LS-100 V100 epoch당 19.8분 vs 21.4분 | Mamba는 conv와 결합해야만 함 |
| Mamba(Miyazaki) | Interspeech 2024, 2406.16808 | 양방향 Mamba 인코더/디코더, 위치 인코딩 제거 | LS-100 E-Branchformer+Mamba 디코더 36.7M 6.1/16.5 vs Transformer 디코더 38.5M 6.4/17.0 | CTC 단독 Mamba는 test_other에서 Conformer/E-Branchformer 미달, 학습 불안정(AdamW, dropout 0.2 필요) | CTC-only 설계에는 경고 신호 |
| LC-BiMamba | Interspeech 2025 (Moriya, NTT) | 청크 단위 역방향 Mamba로 듀얼모드(오프/스트리밍) 단일 모델, 위치 임베딩·히스토리 캐시 불필요 | 인코더 ≈108M; TEDLIUM2 듀얼모드 오프/스트 6.9/7.9 RTF 0.40 vs Conformer 7.1/8.4 RTF 0.49; LS 3.0/7.1·3.7/8.9 vs 3.0/7.2·3.9/9.4 | 청크 15프레임(≈300 ms); 오프라인 최대 길이 739분 vs Conformer 5분(RTX 6000 Ada) | 스트리밍 SSM의 가장 강한 근거 / 대형(108M) 결과만 |
| ConMamba 저자원 | Interspeech 2025 (Zevallos, BSC) | 9개 언어 80 h/8~18 h/≤5 h에서 Conformer vs ConMamba 동일 설정 | 80 h 단문 평균 22.65 vs 23.14; 8~18 h 48.98 vs 48.74; ≤5 h 64.83 vs 65.09; 장문(>10 s) 80 h 28.15 vs 54.43 | 학습시간 40~45%↓, 장문 추론 메모리 50%↓, 속도 63~70%↑ (S: 144차원 12층) | 우리 규모 직접 근거 / 단문 WER 차이는 미미 |
| Samba-ASR | 2501.02832 | 인코더·디코더 모두 Mamba | 표2: GigaSpeech 9.12, LS 1.17/2.48, SPGI 1.84 (canary-1b 4.15 평균 대비 3.65) | 파라미터·코드·평가 프로토콜 미기재, GigaSpeech/SPGI가 학습·평가에 모두 등장 (확인 필요) | 재현 불가, 참고만 |
| SummaryMixing | Interspeech 2024, 2307.07421 | 발화 평균 요약 벡터로 MHSA 대체(선형 시간) | LS960 Conformer 2.3/5.4(137 GPU-h, 46 GB) → SM 2.1/5.1(98 GPU-h, 21 GB); CV Dutch 40 h 31.5 vs 32.6 | 스트리밍 확장 2409.07165(초록: 두 모드 모두 MHSA보다 우세, 수치 미확인) | 메모리 절반·소규모에서도 우세 / 생태계 작음 |
| REB-former | Interspeech 2025 (Song) | E-Branchformer:RWKV = 2:1 교차 적층 | LS-100 AED: Conformer 34.2M 6.6/17.2, E-Branchformer 38.5M 6.4/16.7, REB-former 42.1M 6.0/15.8 | RTX 4090 1장 학습 | LS-100 소형 비교표가 유용 |
| Deformable-conv 스트리밍 | ICASSP 2026, 2601.19960 | 스트리밍 Conformer-T에서 MHSA를 변형 conv로 대체/제거 | LS test-other 320 ms 청크: 10.36(기준 81.3M)/10.33(soft 67.6M)/10.39(hard 65.5M); 1280 ms: 8.91/9.34/9.62 | CPU RTF hard 44.9%↓; "스트리밍에서 어텐션은 국소 연산자" | 작은 청크에서는 어텐션 생략 가능성 |
| Granite TurboCTC | 2609.20104 (2026-09) | 470M CTC Conformer, 100→50 Hz 스태킹 + 첫 두 블록 내부 2x씩(총 8x, 12.5 Hz), 청크 128 어텐션, self-conditioning | OpenASR 평균 5.83; 동일 8x conv 프런트엔드(FastConformer식) 5.92 | 60k h 대형 | 블록 내부 점진 다운샘플이 conv 프런트엔드보다 소폭 우세 |

### 2.2 보충 메모

- **Conformer 제거 실험**(2005.08100 표3): 상대위치 제거 2.1/4.3→2.4/5.6, conv 제거→2.1/4.9, 마카론 제거→2.1/5.0. 상대위치·conv가 핵심.
- **Zipformer 제거 실험**(2310.11230, M 65.6M 2.21/4.79 기준): 시간 다운샘플링 제거 94.2M 2.23/5.09(파라미터 44%↑), LayerNorm 대체 2.29/4.97, Swish 2.27/5.37, Adam 2.38/5.51. 같은 틀로 재현한("reproduced by us", pruned transducer, 같은 툴킷) Conformer pruned RNN-T는 S 9.8M 3.75/9.24, M 28.4M 2.96/7.11, L 122.5M 2.46/5.55. 단, 재현 Conformer의 epoch·옵티마이저·GPU 예산은 논문에 없으므로 동일 예산 비교로 쓰지 말 것. V100 32 G 기준 epoch당 S 87분(2 GPU), M 69분·L 80분(4 GPU; GPU 수는 논문 표 7). 이 epoch 시간은 960 h × 속도 교란 3배(0.9/1.0/1.1, 표 7·부록 A.4.1 명시) 기준이다.
- **Zipformer 위치 인코딩**(icefall `zipformer.py`): `CompactRelPositionalEncoding`(상대 오프셋을 log 압축·atan 후 cos/sin, 마지막 채널 1.0 bias) + `RelPositionMultiheadAttentionWeights`(`linear_pos`로 학습 투영). 한 층에서 계산한 어텐션 가중치를 `nonlin_attention`, `self_attn1`, `self_attn2`가 재사용. 스트리밍은 `causal=True`, `chunk_size`, `left_context_frames`로 청크 마스킹(`c = t // chunk_size`)과 `ChunkCausalDepthwiseConv1d`.
- **Fast Conformer 긴 입력**: 제한 컨텍스트 128스텝(≈10 s)+글로벌 토큰; A100 단일 배치 최대 길이 Conformer 15분 → FC 25분 → FC+제한 컨텍스트 675분. 다중 지연 학습(2312.17279 표4, test-other, 학습 데이터 상이 가능성 확인 필요): CTC 40/240/520 ms 7.6/6.5/6.0.
- **스트리밍 Zipformer 우측 컨텍스트**(ACL 2025 Industry, 2506.14434): 학습 시 RC∈{0,64,128,256} 무작위, 청크 16/32/64. LS960: 기준 RC0 3.33/8.90 → RC256 2.81/7.36; 제안 모델 RC256 2.43/6.55 vs 비스트리밍 2.38/5.72. **LS-100 h**(부록 A, Figure 4 히트맵; 표 아님)에서 RC 없이 학습한 기준 모델의 RC 0 디코딩 9.61 → RC{0,64,128,256} 혼합 학습 모델의 RC 128 디코딩 5.83(test-other 20.36→12.79). 같은 디코딩 RC 128끼리는 7.86→5.83, 같은 RC 0끼리는 9.61→12.46으로 혼합 학습 모델이 오히려 나쁨(부록 모델의 크기·epoch 미기재). 8×V100, 30 epoch.
- **Zipformer를 80 h에서 처음부터 학습한 반례**(XLSR-Transducer, 2407.04439): AMI 80 h(원거리 회의) Zipformer 70M 비스트리밍 21.0, 320 ms 청크 28.5, 1280 ms 24.6 (XLSR 300M 사전학습 12.7). 저자원·잡음 데이터에서 처음부터 학습하면 사전학습 대비 크게 뒤진다는 경고.
- **Mamba 스트리밍 단방향**(ICASSP 2025, 2410.00070): AISHELL-1 42.5M, Mamba+UMA 6.59%(미리보기 0 ms)/5.55%(256 ms) vs 청크 Conformer+UMA 6.04%. 인코더는 인과 conv(k=4)+Mamba, 미리보기는 인코더 뒤 비인과 conv로 부여.
- **TC-BiMamba**(2602.11546, 2026-02): "핵심 결과에 치명적 실험 오류" 사유로 v2 철회. 인용 금지.
- **남아공 7개 언어 Mamba**(2607.01502, 2026-07): 언어당 50 h, Conformer 114M vs ConMamba 123M(18층 512, CTC 문자), 단문 평균 35.49 vs 32.91, 학습시간 34 h vs 18 h, **Conformer는 A100 40 GB에서 학습 불가(80 GB 필요)**.
- **MLMA**(2510.18684): 18층 256차원 ConMamba 31.6M CTC LS 4.05/10.50 vs Conformer 28.8M 4.27/11.29.
- **S4+conv 온라인 Conformer**(2309.08551): 좌측 컨텍스트만으로 LS 4.01/8.53. 추가로 ChunkFormer(ICASSP 2025, 2502.14673)는 80 GB에서 16 h 오디오 처리, 장문 WER 최대 7.7 절대 개선(초록).
- **Selective Lookahead**(SLT 2026, 2609.37611), **Chunkwise Aligners**(ICASSP 2026, 2605.11422), **All-in-One ASR**(ASRU 2025, 2512.11543)는 디코더/정렬 쪽 기여라 02-streaming에서 다루는 것이 맞다.
- **Mamba 커널 제약**(state-spaces/mamba, Dao-AILab/causal-conv1d `setup.py`): gencode는 `compute_75(sm_75)`, 80, 87(+CUDA≥11.8이면 90)이므로 RTX 2080 Ti(sm_75)는 컴파일 대상. 최소 CUDA 11.6. README 요구사항은 "Linux"이며 Windows 미언급(커뮤니티 가이드·WSL2 보고만 있음). 우리 작업 환경이 Windows라면 WSL2가 전제다 (확인 필요).

## 3. 설계 선택지

### 3.1 위치 인코딩
- 상대위치(Transformer-XL식 sinusoidal): Conformer·Branchformer·E-Branchformer·Efficient Conformer·Fast Conformer가 모두 사용. 제거 시 Conformer 2.1/4.3→2.4/5.6.
- RoPE(2501.06051, SpeechBrain 1.0): LS960 AED 12층 512 RelPos 109M 2.00/4.57 vs RoPE 106M 1.96/4.50, 학습시간 0.86x(효율 어텐션 병용 0.81x); **순수 CTC 18층 256**: RelPos 28.8M 4.03/11.06 vs RoPE 27.6M 4.07/11.07(0.81x, RelPos 미세 우세); VoxPopuli 550 h 트랜스듀서 스트리밍 1280/640/320 ms RelPos 10.02/10.36/11.13 vs RoPE 9.75/10.27/11.06; CV Dutch 111 h 33.27 vs 33.14. 5초 미만 짧은 클립에서는 속도 이점 없음. θ=10000 유지 권고. RelPos는 PyTorch SDPA/Flash 계열과 호환이 어렵다는 지적.
- Zipformer compact rel-pos(상기), Granite는 Shaw 상대위치. ConMamba/Mamba 계열은 위치 인코딩 불필요(Moriya; Miyazaki).
- 참고: Conformer AED 인코더가 시간축을 뒤집는 현상 보고(2410.00680) — AED 디코더와 결합 시 주의.

### 3.2 서브샘플링(프레임률)
- 4x(40 ms): Conformer·Branchformer·E-Branchformer·Squeezeformer 출력·Efficient Conformer 1단계. 6x·8x 단일 모델 비교(HydraFormer 2408.04325, Conformer 48.3M CTC greedy): LS test-clean 4.73/4.87/5.04, test-other 11.48/11.83/12.17, AISHELL 5.94/6.03/6.01; 어텐션 리스코어 3.92/3.91/4.01. "서브샘플링이 커지면 WER↑ RTF↓".
- 8x(80 ms): Fast Conformer(깊이별 분리 conv), Efficient Conformer(점진), Squeezeformer 중간부, Granite(블록 내부 2x×2). Multi-blank(2211.03541, conformer-rnnt-large ≈120M, test-other): 4x 5.43(243 s) / 8x 5.56(150 s) / 16x 6.38(106 s).
- Zipformer: 50→25→12.5→6.25 Hz U-Net, 최종 25 Hz(=4x) 출력. 다운샘플링 제거 시 파라미터 44%↑에 WER 손해 없음→효율 핵심이 다운샘플링.
- 한국어 CTC 주의: 8x(12.5 프레임/s)에서는 음절 토큰 비율이 프레임 수를 넘지 않아야 한다(CTC는 출력 길이 ≥ 라벨 길이). 65 h 데이터의 음절/초 통계를 먼저 측정해야 한다 (03·04 조사와 연결).

### 3.3 깊이·너비·파라미터(20M~120M)
- 동일 틀 소형 비교: Zipformer-S 23.3M 2.42/5.73 vs Conformer-M 28.4M 2.96/7.11(2310.11230); Squeezeformer-S 18.6M 3.08/7.47 vs Conformer-CTC-M 27.4M 3.20/7.90(2206.00888); Efficient Conformer 13.2M 3.57/8.99 vs Conformer 13.0M 4.07/10.25(2109.01163); ConMamba-S 14.1M 4.0/9.5 vs Conformer-S 13.3M 4.1/10.0(2407.09732).
- 100 h 규모 비교(모두 AED 약 35~42M, test-clean/other): Conformer 34.2M 6.6/17.2, E-Branchformer 38.5M 6.4/16.7, ConExtBiMamba 41.6M 6.0(test), REB-former 42.1M 6.0/15.8(2305.11073, 2405.12609, REB-former).
- Zeroth-Korean 50 h: FastConformer-T 인코더 hidden 640→400으로 축소, 약 50M(2404.04295; 인용된 NeMo 기본 설정의 인코더 d_model은 512이므로 '640'의 출처는 불명확하고 정확한 구조는 재구성 불가). Irish 290 h: Conformer 113M 16.7 vs E-Branchformer Small 45.8M 17.7, Large 130M 15.7(2405.01293). 65 h에서는 20~40M이 적정이고 100M급은 과적합·메모리 위험.
- Moriya는 LibriSpeech(큰 데이터)에서는 "깊은" 구성이, TEDLIUM2에서는 "넓은" 구성이 유리했다고 보고(깊이-데이터 크기 상호작용).

### 3.4 정규화·활성·옵티마이저
- Conformer: conv 모듈 BatchNorm. 스트리밍/소규모에서는 LayerNorm으로 교체하는 것이 관행(Stateful Conformer BN→LN; Moriya BN→LN, 커널 31→15; Branchformer는 Conformer 발산 원인으로 BN 의심).
- Squeezeformer: pre-LN을 학습형 스케일로 대체(post-LN only), 스케일 제거 시 dev-other 14.00%로 급락. Zipformer: BiasNorm(평균 빼기 제거), ScaledAdam(파라미터 RMS 기준 업데이트), Adam 사용 시 모든 모듈에 BiasNorm이 필요.
- 비교연구 2305.11073: 피크 LR 2e-3, pure CTC, 33M 인코더(TEDLIUM2)에서 Conformer 10회 중 6회 실패, E-Branchformer 2회 실패; LR을 낮추면 안정되지만 WER 손해; "joint CTC에서 초기 몇 epoch에 valid loss가 내려가지 않으면 LR↓ 또는 warmup↑".
- Granite: Muon(Polar Express) 5.83 vs AdamW 5.99 (대형 기준, 소형 적용은 확인 필요).

### 3.5 fp16에서 깨지는 것(논문·레시피 서술)
- Zipformer: Balancer는 활성이 "지나치게 크거나 작아지는" 현상을 막으며 혼합정밀에서 특히 불안정 요인이라고 서술(원문 "particularly when employing mixed precision training", 부록 A.3.1; 표 6 No Balancer 2.23/4.97 vs 2.21/4.91, 40 epoch — 발산 위험에 관한 정성적 서술이며 발산 통계는 없음); Whitener는 공분산 고유값 쏠림(발산 전조) 억제; 소스에서는 어텐션 점수에 `penalize_abs_values_gt(limit=25.0)` 패널티, 마스크 값 -1000(NaN 방지). icefall Zipformer 계열 레시피의 문서화된 학습 명령은 `--use-fp16 1`을 명시한다(train.py 기본값은 False; lstm/emformer/stateless5 등 일부 레시피는 `--use-fp16 0`). LS-100 pruned_transducer_stateless2(약 80M) 1 GPU max-duration 300: fp16 6.63/17.39(epoch 30 avg 10) vs fp32 6.64/17.19(epoch 39 avg 10); 같은 체크포인트 비교는 epoch 29 avg 8에서 fp32 6.71/17.77 vs fp16 6.81/17.84.
- Fast Conformer 캐시 스트리밍(2312.17279): FP16 혼합정밀 200 epoch 이상 학습. E-Branchformer: PyTorch 혼합정밀(8×A100). 한국어 Zipformer2: FP16.
- 어텐션 점수 overflow(WavLM 2110.13900, IV-D "Stabilization of Training"): 대형 모델에서 fp16 어텐션 점수가 상한을 넘어 NaN 손실로 학습이 불안정했고, 쿼리를 c·√d로 나눈 뒤 키별 최댓값을 빼고 다시 c를 곱하는 트릭(c=32, CogView 인용)으로 해결. 소형 모델에서도 fp32 softmax·dynamic loss scaling은 기본으로 둘 것.
- Mamba: 학습 불안정(Miyazaki: AdamW·dropout 0.2 필요). 커널 dtype은 fp16/bf16/fp32 지원이라 서술하지만 README 예시는 bf16 (fp16 전용 GPU에서의 안정성은 확인 필요).
- 메모리 참고: SummaryMixing 비교에서 Conformer 110M LS 학습이 46 GB VRAM(SM 21 GB); 남아공 연구 Conformer 114M CTC가 40 GB에서 OOM. 11 GB에서는 20~40M + fp16 + max-duration 축소 + (Zipformer 다운샘플 또는 8x)가 현실적이다.

## 4. 우리 조건에서의 순위 (Top 3)

### 1위: Zipformer (icefall 레시피, CTC 또는 pruned RNN-T, `causal=True` 청크 스트리밍)
근거: (a) 소형 파라미터·FLOPs 효율이 가장 좋다(S 23.3M 2.42/5.73, 40.8 GFLOPs; 같은 틀 Conformer-M보다 작고 정확). (b) 혼합정밀(V100, 사실상 fp16)로 학습했고 Balancer가 혼합정밀 불안정을 막는 장치로 서술되며(BiasNorm·Whitener·ScaledAdam의 동기는 별개), 공식 레시피의 문서화된 학습 명령이 모두 `--use-fp16 1`을 쓰므로(플래그 기본값은 False) bf16 없는 2080 Ti에 가장 잘 맞는다. (c) 스트리밍이 레시피에 내장(청크 마스킹+청크 인과 conv+상태 캐시)되어 있고, 320 ms 청크에서 3.06/7.81(simulated, epoch 30; 실제 청크 3.06/7.79) vs 오프라인 2.22/4.87(epoch 50 avg 25, 65.55M)로 손실 폭이 문서화되어 있으며, 우측 컨텍스트 혼합 학습으로 LS-100 스트리밍이 기준 모델 RC 0 디코딩 9.61 → RC 128 디코딩 5.83까지 개선됨이 보고됐다(부록 Figure 4; 같은 RC 0끼리는 9.61→12.46으로 오히려 나쁨). (d) 한국어 스트리밍 Zipformer2(KsponSpeech, FP16) 전례와 74 h Hakka 적용 사례가 있다. 단점·열린 질문: 구조와 옵티마이저가 특수해 직접 구현 시 재현 위험이 크다(icefall 코드 재사용 권장); k2 CUDA는 Windows pip 휠이 없고 소스 빌드도 torch 2.8+에서 실패 보고만 있어 WSL2 없는 Windows에서는 사용 불가로 간주(WSL2 또는 CTC만 PyTorch로 구현하는 우회 검토); 65 h 오프라인 소규모 비교표가 없다(AMI 80 h 반례 존재); S(23M)와 M(65M) 사이 어느 폭이 65 h에 맞는지는 실험으로 정해야 한다.

### 2위: E-Branchformer (ESPnet식 CTC/AED, 4x conv2d, rel-pos, 청크 마스킹 추가)
근거: (a) 100 h 규모에서 가장 통제된 다중 코퍼스 비교가 있고 거의 모든 세트에서 Conformer 이상(LS-100 dev 6.1/16.7 vs 6.3/17.0, test 6.3/17.0 vs 6.6/17.2; CHiME4 6.8/8.4/10.8/13.0 vs 7.8/9.5/12.5/14.8). (b) "모델이 크거나 데이터가 작을 때" Conformer보다 안정적이며 CTC 33M에서 학습 실패 2회 vs 6/10(2305.11073, 일화적 보고), OWSM 10% 데이터에서 가장 빠른 수렴. (c) LayerNorm·Adam·표준 모듈만 써서 구현·디버깅이 쉽고 PyTorch 혼합정밀(8×A100) 학습 사례가 있다(2210.00077; 2305.11073은 정밀도 미기재). 단점·열린 질문: cgMLP+MHSA 병렬로 블록당 연산이 커서 11 GB에서 배치가 작아진다(Zipformer-L 107.7 vs E-Branchformer-L 284.4 GFLOPs); 스트리밍 레시피가 공식적으로 약해 청크 마스킹·인과 conv를 직접 설계해야 한다; RoPE 교체 시 CTC-only에서 미세 손해 가능.

### 3위: Fast Conformer 8x + 캐시 인식 청크 스트리밍 (NeMo식, 또는 자체 구현)
근거: (a) 연산 2.9배 절감으로 11 GB에서 가장 여유롭고, Granite 2026도 8x·12.5 Hz 토큰률을 채택해 추세가 유지된다. (b) 스트리밍 설계(완전 인과 conv, BN→LN, 다중 지연 학습, 활성 캐시)가 가장 성숙하고 지연별 WER이 표로 공개돼 있다. (c) Zeroth-Korean 50 h에서 FastConformer-T 약 50M으로 CER 1.51%라는 우리 데이터 직접 전례가 있다(토큰 단위는 문자(음절)로 확인; 사전학습 가중치 언급이 없어 처음부터 학습으로 보이나 명시되지 않음; GPU·에폭·정밀도·디코딩 방식 미기재). 단점·열린 질문: 소형 모델에서 Zipformer/E-Branchformer 대비 정확도 근거가 약하고(Zipformer 표에서 Eff.Conformer/Squeezeformer 125M이 FC보다 WER 높음), 8x에서는 한국어 음절 CTC의 길이 제약과 0.1~0.7 WER 손해를 감수해야 하며, 공개 레시피가 32 GPU·대규모 데이터 전제다.

### 보류(실험군): ConMamba / LC-BiMamba
저자원 80 h 동급 WER·학습시간 40~45% 절감(Zevallos), 듀얼모드 스트리밍에서 Conformer 우세·위치 인코딩 불필요(Moriya), 장문 메모리 안정성은 매력적이다. 그러나 CTC 단독 Mamba의 test_other 열위와 학습 불안정 보고(Miyazaki), Linux 전제 CUDA 커널, 2026년 TC-BiMamba 철회 사례, 디코딩 도구 미성숙을 고려해 1차 구조로는 두지 않고, 시간이 남으면 1·2위 구조에서 MHSA만 BiMamba로 바꾼 소규모 비교(14M급, Speech Slytherin 설정)를 한다.

### 공통 열린 질문
1. 한국어 토큰(음절 vs 자모 vs BPE)과 프레임률(4x/8x)의 조합에서 CTC 길이 제약·WER 영향(03·04 조사 결과와 합칠 것).
2. 11 GB에서 Zipformer-S/M, E-Branchformer-B(41M)의 fp16 max-duration과 epoch 시간 실측.
3. 청크 크기(320 vs 640 ms)와 우측 컨텍스트 학습을 1차 모델에 넣을지.
4. Windows 환경에서 k2/icefall·mamba-ssm 설치 가능성(WSL2 필요 여부). k2 부분은 답이 나왔다: k2 CUDA는 Windows pip 휠 없음(CPU 휠만), 소스 빌드는 torch 2.8+에서 실패 보고만 있음, fast_rnnt는 C++/CUDA 확장 필수 → WSL2 없는 Windows에서는 사용 불가로 간주. mamba-ssm은 미확인.

## 5. 읽은 자료 목록

- Conformer: https://arxiv.org/abs/2005.08100 (ar5iv 전문)
- Efficient Conformer: https://arxiv.org/abs/2109.01163 (ar5iv 전문)
- Squeezeformer: https://arxiv.org/abs/2206.00888 (ar5iv 전문)
- Branchformer: https://arxiv.org/abs/2207.02971 (ar5iv 전문)
- E-Branchformer: https://arxiv.org/abs/2210.00077 (ar5iv 전문)
- E-Branchformer vs Conformer 비교: https://arxiv.org/abs/2305.11073 (ar5iv 전문)
- Zipformer: https://arxiv.org/abs/2310.11230 , https://arxiv.org/html/2310.11230v4
- Zipformer 소스: https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/zipformer/zipformer.py
- icefall RESULTS(스트리밍/오프라인/LS-100): https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/RESULTS.md
- icefall 스트리밍 문서: https://icefall.readthedocs.io/en/latest/recipes/Streaming-ASR/librispeech/zipformer_transducer.html
- 한국어 스트리밍 Zipformer2 모델 카드: https://huggingface.co/kangkyu/icefall-asr-ko-streaming-zipformer-174m
- 스트리밍/비스트리밍 통합 Zipformer: https://arxiv.org/abs/2506.14434 , https://arxiv.org/html/2506.14434v1
- Fast Conformer: https://arxiv.org/abs/2305.05084 , https://arxiv.org/html/2305.05084v6
- Stateful(캐시 인식) Conformer: https://arxiv.org/abs/2312.17279 , https://arxiv.org/html/2312.17279v1
- Speech Slytherin(ConMamba): https://arxiv.org/abs/2407.09732 , https://arxiv.org/html/2407.09732v1
- Mamba in Speech: https://arxiv.org/abs/2405.12609 , https://arxiv.org/html/2405.12609v6
- Mamba in speech applications(Miyazaki): https://arxiv.org/abs/2406.16808 , https://arxiv.org/html/2406.16808v1
- LC-BiMamba(Moriya, Interspeech 2025): https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.pdf
- Mamba 저자원(Zevallos, Interspeech 2025): https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.pdf
- Mamba 스트리밍+UMA: https://arxiv.org/abs/2410.00070 , https://arxiv.org/html/2410.00070v2
- Samba-ASR: https://arxiv.org/abs/2501.02832 , https://arxiv.org/html/2501.02832v1
- MLMA: https://arxiv.org/html/2510.18684
- 남아공 언어 Mamba: https://arxiv.org/html/2607.01502
- TC-BiMamba(철회): https://arxiv.org/abs/2602.11546
- S4 증강 온라인 Conformer: https://arxiv.org/abs/2309.08551
- SummaryMixing: https://arxiv.org/abs/2307.07421 , https://arxiv.org/html/2307.07421v3 ; 스트리밍판 https://arxiv.org/abs/2409.07165
- RoPE 벤치마크: https://arxiv.org/abs/2501.06051 , https://arxiv.org/html/2501.06051v2
- REB-former(Interspeech 2025): https://www.isca-archive.org/interspeech_2025/song25b_interspeech.pdf
- Deformable conv 스트리밍(ICASSP 2026): https://arxiv.org/abs/2601.19960 , https://arxiv.org/html/2601.19960
- Granite TurboCTC: https://arxiv.org/html/2609.20104
- HydraFormer(서브샘플링 비교): https://arxiv.org/html/2408.04325
- Multi-blank Transducer(프레임률 비교): https://arxiv.org/html/2211.03541
- ChunkFormer: https://arxiv.org/abs/2502.14673
- Conformer 시간 역전 현상: https://arxiv.org/abs/2410.00680
- WavLM fp16 안정화 절(IV-D): https://ar5iv.labs.arxiv.org/html/2110.13900
- OWSM v3.1(인코더 비교): https://arxiv.org/html/2401.16658v2
- XLSR-Transducer(Zipformer 80 h 반례): https://arxiv.org/html/2407.04439v1
- Zeroth-Korean FastConformer-T: https://arxiv.org/html/2404.04295
- Irish 저자원 E-Branchformer: https://arxiv.org/html/2405.01293v1
- Hakka 저자원 Zipformer: https://arxiv.org/html/2602.22522
- 기타 2026 신규(참고): https://arxiv.org/abs/2603.16922 , https://arxiv.org/abs/2609.37611 , https://arxiv.org/abs/2605.11422 , https://arxiv.org/abs/2512.11543 , https://arxiv.org/abs/2506.19761
- Mamba 커널 요구사항: https://github.com/state-spaces/mamba , https://raw.githubusercontent.com/state-spaces/mamba/main/setup.py , https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/main/setup.py

## 6. 검증 후 정정 (2026-10-11)

적대적 검증(README §2, `work/survey_workflow_results.json` verify)에서 지적된 문구·조건을 반영했다. 수치 자체는 모두 원문과 일치해 바꾸지 않았다.

- C1: §1 Zipformer "fp16 혼합정밀을 기본으로 설계(…)" → 혼합정밀(V100, 사실상 fp16) 학습 + Balancer만 혼합정밀 대책으로 서술; §2.1 표 장/단 열과 §4 1위 (b)도 같은 취지로 수정.
- C1: §1 "FLOPs 40.8 G로 가장 적다" → "WER 2.5/6.0 이하 모델 중 가장 적다(Conformer-M 대비 47%)"로 한정하고 표 2의 더 작은 모델(Squeezeformer-XS 18.2 G, Conformer-S 29.1 G) 병기.
- C1: §2.2 "같은 틀로 재현" 항목에 재현 Conformer의 epoch·옵티마이저·예산 미기재(동일 예산 비교 금지) 주의, GPU 수 출처를 표 7로, epoch 시간이 960 h × 속도 교란 3배 기준임을 추가.
- C1: §3.5 Balancer 서술에 원문 표현·표 6(No Balancer 2.23/4.97 vs 2.21/4.91)·발산 통계 없음을 추가.
- C2: §1·§2.1·§4 1위 (c)의 스트리밍 3.06/7.81, 2.81/7.15에 simulated 값임과 실제 청크 디코딩 3.06/7.79, 2.84/7.16을 병기; 오프라인 2.22/4.87은 epoch 50 avg 25임을 명시.
- C2: §3.5 "icefall 전 레시피가 `--use-fp16 1`" → Zipformer 계열 문서화 명령이 명시(train.py 기본값 False, 일부 레시피 0); LS-100 fp16/fp32 비교에 체크포인트(epoch 30 avg 10 vs epoch 39 avg 10)와 동일 체크포인트 비교(epoch 29 avg 8: fp32 6.71/17.77 vs fp16 6.81/17.84) 추가; §4 1위 (b) "기본값이 `--use-fp16 1`" → "문서화된 학습 명령이 모두 사용(기본값 False)".
- C3: §1·§2.1·§4 2위 (a)의 LS-100 6.1/16.7 vs 6.3/17.0을 dev 수치로 명시하고 test 6.3/17.0 vs 6.6/17.2(AED+joint CTC, LM 없음) 병기; §3.3 100 h 비교에 test 라벨.
- C3: "발산 6/10 vs 2/10", "발산 보고 다수" → 논문 용어 "실패(failed)", E-Branchformer 분모 미명시, TEDLIUM2 1개 설정·10회 시도의 일화적 보고(표 없음, WER·정밀도 미기재)로 §1·§2.1(Conformer·E-Branchformer 행)·§3.4·§4 2위 (b) 수정.
- C3: §4 2위 (c) fp16 혼합정밀 사례 출처를 2210.00077(8×A100)로 분리(2305.11073은 정밀도 미기재).
- C10: 2404.04295 관련 §1·§2.1·§3.3·§4 3위 (c) → 토큰 단위는 문자(음절)로 확인, 사전학습 언급 없음(처음부터 학습으로 보이나 명시 안 됨), GPU·에폭·정밀도·디코딩 미기재, "hidden 640"은 NeMo 기본 인코더 d_model 512와 맞지 않아 구조 재구성 불가.
- C11: 공통 열린 질문 4의 k2 부분에 답 기재(Windows CUDA 휠 없음, torch 2.8+ 소스 빌드 실패 보고, fast_rnnt C++ 확장 필수 → WSL2 없는 Windows 사용 불가); §4 1위 단점의 "확인 필요"도 같은 결론으로 교체.
- C12: §2.2 "LS-100 RC 추가로 9.61→5.83" → 부록 A Figure 4 히트맵, 기준 모델 RC 0 디코딩 9.61 → RC 혼합 학습 모델 RC 128 디코딩 5.83(test-other 20.36→12.79), 같은 RC 128끼리 7.86→5.83, 같은 RC 0끼리 9.61→12.46으로 오히려 나쁨; §4 1위 (c)도 동일하게 수정.
