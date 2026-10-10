# 02. 스트리밍 / 온라인 ASR 방법 조사 (2020-01 ~ 2026-10)

작성일 2026-10-10. 모든 수치는 본문에 적힌 arXiv id 또는 URL 페이지에서 직접 읽은 값이다. 직접 확인하지 못한 항목은 (확인 필요)로 표시했다.
대상 조건: 한국어 약 65 h(Zeroth 51.6 h + FLEURS ko + Common Voice ko), RTX 2080 Ti 11 GB(fp16 전용), 처음부터 학습, 최종 목표는 마이크 청크 입력의 실시간 부분 전사.

## 1. 요약

- 권장 방식: **청크 단위(block/chunk-wise) 인과(causal) 어텐션 인코더 + 인과 컨볼루션**을 **동적 청크 학습(dynamic chunk training)**으로 학습하고, 1차 출력은 **CTC(greedy 또는 prefix beam)**로 내보내며, 보조 손실로 **pruned transducer(또는 어텐션 디코더)**를 함께 두는 구성. 디코딩 캐시(과거 key/value·컨볼루션 상태)를 유지하는 "cache-aware" 추론으로 청크당 계산을 중복 없이 수행한다.
- 이유: (1) 65 h·11 GB 조건에서 가장 단순하고 검증된 경로다. U2는 AISHELL-1(초록 170 h, 학습셋 150 h), 47 M 파라미터(인코더 12층 + 어텐션 디코더 6층 포함)로 청크 16(640 ms)에서 CER 5.33 %, 전체 문맥 4.90 %를 보였다(arXiv 2012.05481). (2) CTC는 디코더 상태가 없어 부분 결과가 단조 추가(append-only)로 안정적이고 청크당 비용이 가장 낮다. (3) 보조 transducer를 붙이면 스트리밍 CTC WER가 4.56 → 4.00 %(test-clean), 12.21 → 10.42 %(test-other)로 개선된다(arXiv 2305.11539; 지연 패널티 없는 보조 transducer 기준이며 이때 CTC 헤드의 MSD는 273 → 342 ms로 늘어난다. 결합 손실은 0.2·L_ctc + 1.0·L_transducer(α=0.2). CTC λ 0.010 + transducer λ 0.0050으로 지연 패널티를 함께 주면 4.74/11.89 %, MSD 77 ms). (4) icefall의 Zipformer 레시피가 `--causal 1 --chunk-size 16,32,64,-1 --left-context-frames 64,128,256,-1`로 이를 그대로 지원하고 fp16 학습(`--use-fp16 1`)을 사용한다.
- 핵심 트레이드오프 1 — 청크 크기: icefall 스트리밍 Zipformer 66 M, LibriSpeech greedy: 320 ms 청크 3.06/7.81 → 640 ms 2.81/7.15(RESULTS.md; simulated streaming 값이며 실제 청크 디코딩은 3.06/7.79, 2.84/7.16). NeMo cache-aware FastConformer 114 M, test-other CTC: 0 ms 10.6 → 240 ms 8.8 → 1360 ms 7.1, 오프라인 5.7(arXiv 2312.17279).
- 핵심 트레이드오프 2 — 좌측 문맥: 같은 0.56 s 지연에서 과거 문맥을 1.12 s → 5.6 s로 늘리면 평균 WER 8.51 → 7.28 %(Nemotron 0.6B, arXiv 2604.14493). 좌측 문맥은 지연을 늘리지 않고 계산만 늘린다.
- 핵심 트레이드오프 3 — 방출 지연(emission delay): 스트리밍 CTC의 평균 시작 지연(MSD) 273 ms는 delay-penalty λ=0.010으로 108 ms까지 줄지만 WER 4.56 → 5.32 %(arXiv 2305.11539). Transducer는 λ=0.0030에서 MAD 373 → 102 ms, WER 3.4 → 3.70 %(arXiv 2211.00490).
- 핵심 트레이드오프 4 — 우측 문맥(look-ahead): 학습 때 우측 문맥을 섞으면(0/64/128/256 프레임) 디코딩 시 RC 64로 2.74/6.89, RC 0에서는 4.43/9.50(arXiv 2506.14434; LS960, 평가 청크 32·좌측 128. RC 없이 학습한 기준 모델은 RC 0에서 3.33/8.90이므로 혼합 학습 모델은 RC 0 디코딩에서 오히려 퇴보한다. 논문은 프레임=10 ms로 서술하나 icefall 레시피 단위는 20 ms다). 레이어별 look-ahead는 깊이만큼 누적되므로(17층×2프레임×40 ms=1360 ms, NeMo 문서) 청크 경계형(chunk-aware) 마스크를 써야 한다.
- 지연은 알고리즘 지연(청크+우측 문맥), 청크당 계산 시간, 토큰 방출 지연, 끝점(endpoint) 지연을 따로 측정해야 하며, 사용자 체감 지연(UPL)의 50 % 이상이 끝점 지연이었다(arXiv 2104.02207).

## 2. 방법별 정리

### 2.1 표

| 방법 | 학회/연도, arXiv | 기법 | 보고된 지연 vs 정확도 | 모델 크기 | 학습 시 필요 | 추론 복잡도 |
|---|---|---|---|---|---|---|
| U2 (WeNet) | arXiv 2020, 2012.05481 | 동적 청크 어텐션(배치마다 50 % 전체, 50 % U(1,25) 청크) + CTC 1차 + 어텐션 리스코어링 | AISHELL-1 CER(리스코어링) full 4.90 / 청크16 5.33 / 8 5.52 / 4 5.71 / 1 6.23; CTC prefix beam 5.49/6.08/6.41/6.64/7.58; 리스코어링 Δ 50–100 ms([10,15] 인용값) | 47 M(인코더 12층 Conformer d256 + 어텐션 디코더 6층 포함 전체, 표 3) | 동적 청크 마스크, 인과 conv(kernel 8), CTC+어텐션 손실 | 1차 CTC 청크당 1회; 2차는 발화 끝에 1회 |
| U2++ | arXiv 2021, 2106.05642 | U2 + 우→좌(R2L) 디코더 + SpecSub | AISHELL-1 Conformer full 4.63 / 청크16 5.05(CTC prefix 5.19/5.81); "청크16 지연 0–640 ms, 평균 320 ms" | U2와 거의 동일(3+3 디코더층) | R2L 가중치 0.3 | U2와 동일 |
| WeNet 런타임 | arXiv 2021, 2102.01547 | TorchScript, CTC prefix beam + 리스코어링, int8 | AISHELL-1 CER(리스코어링) full 5.52 / 16 6.05 / 8 6.28 / 4 6.62; x86 RTF fp32 0.095(16)/0.186(4), int8 0.081/0.134; Android fp32 0.251/0.505, int8 0.114/0.201; L1 모델지연 380/220/140 ms(청크 16/8/4), L2 리스코어링 ≈115 ms | 12층 Transformer | 동일 | 청크가 작을수록 RTF 증가 |
| Emformer | ICASSP 2021 제출, 2010.10759 | 블록 처리 + 메모리 뱅크 + 좌측 문맥 KV 캐시 + 우측 문맥 하드 복사(look-ahead 누출 방지) | LibriSpeech EIL 960 ms: 2.50/5.62; EIL 80 ms: 3.01/7.09(EM-36L+sMBR); EM-24L RTF 0.13(중지연) → 0.33(80 ms) | EM-24L ≈80 M, EM-36L 120 M | 32×V100 fp16, 180 epoch; 보조 손실 0.3 | 좌측 캐시로 AM-TRF 대비 학습 4.6배, RTF 0.16→0.13 |
| Cache-aware FastConformer | ICASSP 2024, 2312.17279 | 완전 인과 / regular look-ahead / chunk-aware look-ahead 3종, 활성값 캐시, 다중 look-ahead 학습, CTC+RNNT 하이브리드 | test-other CTC(chunk-aware) 0 ms 10.6, 40 ms 10.1, 240 ms 8.8, 520 ms 8.4, 680 ms 8.0, 1360 ms 7.1; RNNT 9.5/9.0/7.8/7.5/7.3/6.3; 오프라인 CTC 5.7, RNNT 5.0; buffered 기준 CTC 8.0(1.5 s), RNNT 11.3(2 s) | ≈114 M, 17층 | 배치마다 look-ahead 무작위 선택; 하이브리드 손실 | 과거 재계산 없음; buffered 방식은 중복 계산 |
| Zipformer(비스트리밍 논문) | ICLR 2024, 2310.11230 | U-Net형 다중 프레임률(50/25/12.5/6.25/12.5/25 Hz, 출력 25 Hz), ScaledAdam, BiasNorm | 스트리밍 수치 없음. Zipformer-S pruned transducer 23.3 M: 2.42/5.73(40.8 GFLOPs/30 s); Conformer-S 10.3 M: 2.7/6.3 | S 23.3 M / M 65.6 M / L 148.4 M | S: 2×V100 32 G, max-duration 1500 s, 50 epoch | FLOPs가 Conformer-L 대비 50 % 이상 절감 |
| icefall 스트리밍 Zipformer | RESULTS.md(k2-fsa/icefall) | `--causal 1`, 청크 16/32/64/-1(50 Hz 프레임), 좌측 64/128/256/-1 프레임, pruned stateless transducer | 66.11 M, LibriSpeech greedy 320 ms(청크16, 좌128) 3.06/7.81, modified beam 3.01/7.69, fast beam 3.04/7.68; 640 ms(청크32, 좌256) greedy 2.81/7.15; 앞 수치는 simulated(decode.py에 전체 발화 입력) 값이며 실제 청크 디코딩(streaming_decode.py)은 greedy 3.06/7.79·2.84/7.16; simulated vs 실제 차이는 greedy ≤0.03, beam 계열 최대 0.06(640 ms modified/fast beam test-other 7.05→7.11, 7.04→7.10) | 66 M | 4 GPU, `--use-fp16 1`, max-duration 1000, num-epochs 40(보고 WER은 epoch 30 avg 8); 청크 16,32,64,-1·좌측 64,128,256,-1은 train.py 기본값 | greedy/modified beam/fast beam 지원, 다중 스트림 배치 |
| 한국어 스트리밍 Zipformer(모델 카드) | HF kangkyu/icefall-asr-ko-streaming-zipformer-174m | Zipformer2 causal, 청크 16/32/64/-1 다중 청크 학습, 좌측 128, 음절 어휘 2,460 | KsponSpeech eval CER: 오프라인+KenLM 7.16; int8 청크64 7.53 / 32 7.815 / 16 8.255; 1스레드 int8 RTF 0.049/0.054/0.073(fp32 0.086/0.098/0.134) | 155.7 M | 6,500 h, fp16, pruned RNN-T, 30 epoch | sherpa-onnx는 청크 16/32/64만 지원 |
| Dual-mode ASR | ICLR 2021, 2010.06030 | 가중치 공유(마스크된 conv, 좌측만 softmax), 모드별 norm, in-place KD | Conformer 30.7 M: 스트리밍 4.6/9.9 → 듀얼 3.7/9.2; latency@50 140→10 ms, @90 280→90 ms; 전체 문맥 2.3/5.0 vs 듀얼 2.5/5.9 | 30.7 M | 두 모드 동시 forward(비용 2배) | 스트리밍 추론은 단일 모드 |
| Dynamic Chunk Convolution | ICASSP 2023, 2304.09325 | 인과 conv 대신 청크 내부 우측만 보는 conv, 전체 문맥 모델에서 초기화 | LibriSpeech 640 ms 청크·좌 1280 ms: 인과 conv 2.9/6.8 → DCConv 2.6/6.6 → 미세조정 2.5/6.6; 전체 문맥 모델을 그대로 스트리밍하면 3.6/9.4 | Conformer 12×512×8 | 청크 8–32 프레임 샘플링 | 인과 conv와 동일 |
| DCTX-Conformer | arXiv 2023, 2306.08175 | 문맥 임베딩 carry-over(좌측 문맥 0일 때 보상) | test-clean 320/640/1280 ms: 좌측 full 3.1/2.6/2.4, 좌측 0 5.4/3.6/2.8, DCTX 좌측 0 4.4/3.2/2.6; test-other 좌측 full 8.2/6.8/6.0; 평균 지연 816 → 820 ms | 115 M | 청크 320–1280 ms 60 %, 전체 40 %; 전체 문맥 30 epoch 후 DCT 30 epoch | 임베딩 1–16개 추가, 지연 영향 "무시 가능" |
| Transformer-Transducer | ICASSP 2020, 2002.02562 | 레이어별 좌/우 문맥 제한, 우측 문맥은 레이어 수만큼 누적 | 좌10/우0 4.2/11.3; 좌512 우2 3.0/7.7, 우6 2.8/6.9, 우10 2.7/6.6, full 2.4/5.6(30 ms 프레임, 18층) | 139 M | 마스크만 | 레이어별 look-ahead 누적이 단점 |
| MS 스트리밍 T-T/C-T | arXiv 2020, 2010.11395 | 청크 마스크(전 레이어 공유, 우측 수용 범위 고정) vs 레이어별 look-ahead(18층×5프레임=90프레임) | [0,720] ms look-ahead, 60프레임 히스토리: T-T 8.28, C-T 8.19; 전체 7.78/7.69; look-ahead 0: 8.88/8.80; 16코어 CPU 1스레드 RTF 0.40, int8 0.22(WER 8.28→8.50); 히스토리 전체→60프레임 RTF 3.44→2.38 | ≈80 M | 청크 마스크 | KV 캐시로 프레임당 Q/K/V 1회 |
| DCN self-attention | Interspeech 2021, 2107.01269 | 인과/비인과 두 스트림으로 look-ahead 누적 차단 | Transformer TA 480 ms: RSA 2.9/7.6, CSA(청크) 2.7/7.3, DCN 2.8/7.4; Conformer DCN 640/320 ms 2.6/6.7, +Tr-LM 2.5/6.3 | 12층 512 | 두 스트림(프레임 2배 ≈ 50 % 겹침 청크) | CSA와 유사 |
| Blockwise synchronous beam search(ESPnet) | arXiv 2020, 2006.14941 | 문맥 블록 인코더 {16,16,8}(이론 지연 0.64 s) + 블록 동기 빔 탐색, 반복 감지 | LibriSpeech(큰 LM, beam 10) 3.0/7.8, RTF 0.35, 응답 1.19 s vs 배치 2.9/6.7, 3.49 s; 반복 기준 제거 시 32.3/39.8 | 미기재 | 블록 인코더 | 어텐션 디코더를 블록마다 재실행 |
| CUSIDE / CUSIDE-T | arXiv 2022, 2203.16758 / ISCSLP 2024, 2407.10255 | GRU로 미래 400 ms 프레임을 시뮬레이션(L1 손실), 청크 400±200 ms, 좌측 800 ms | CUSIDE(CTC-CRF, 37 M) AISHELL-1 400+2 ms CER 5.47 vs U2++ 640 ms 5.81; CUSIDE-T(RNN-T 90 M) 400 ms: U2++ 6.11 / CUSIDE-T 6.02, LM 리스코어링 5.74 / 5.51; 시뮬레이션 비용 ≈2 ms/청크(GTX 1080) | 37 M / 90 M | 시뮬레이션 모듈(≈5 %) | 청크당 GRU 1회 |
| Unifying streaming/non-streaming Zipformer | ACL Industry 2025, 2506.14434 | 청크 {16,32,64} + 우측 문맥 {0,64,128,256} 프레임 무작위 학습, 평가 시 좌측 128·청크 32(논문은 프레임=10 ms로 서술, icefall 레시피 단위는 20 ms) | LibriSpeech RC0 4.43/9.50, RC64 2.74/6.89, RC128 2.46/6.70, RC256 2.43/6.55; RC 없이 학습한 기준 RC0 3.33/8.90; 비스트리밍 2.38/5.72; 서버 최종 청크 지연(동시 100) 1.41/1.44/1.47 s(RC 0/32/64) | Zipformer-M(LS)/L(사내), 파라미터 수·fp16·디코딩 방식 미기재 | 8×V100, 30 epoch | 우측 문맥만큼 지연 증가 |
| TSCA + DRC | Interspeech 2024, 2502.15158 | 시간 이동 문맥 어텐션 + 동적 우측 문맥 마스크 | LibriSpeech 상대 WER 10–13.9 % 감소(절대값 미기재) | 미기재 | 마스크 | 미기재 |
| FaT(Future-aware Transformer) | Interspeech 2024(yang24m) | 뒤 청크→앞 청크 look-ahead 창 자기증류 | AISHELL-1 dev/test 5.2/5.9 %(640 ms); 청크16 방출지연 FTD P50: 기준 50 ms, look-ahead 기준 0 ms, FaT −10 ms; LTD −50/−100/−100 ms | 미기재 | 자기증류 | look-ahead 창 |
| Delayed-KD | Interspeech 2025, 2505.22069 | 비스트리밍 교사 → 스트리밍 CTC 지연 KD, Temporal Alignment Buffer(0–160 ms) | AISHELL-1 청크1(40 ms): CER 스트리밍/리스코어링 6.11/5.42(TAB 80 ms), U2++ 청크1 7.76/5.98, U2++ 청크8(320 ms) 6.53/5.44; FTD P50/P90 140/180 ms, LTD 90/130 ms; WenetSpeech Test_Net 40 ms 11.37/12.75 vs U2++ 320 ms 11.54/12.23 | 미기재(U2++ 설정) | 비스트리밍 교사 모델 필요 | U2와 동일 |
| LC-BiMamba 듀얼 모드 | Interspeech 2025(moriya25) | 어텐션 없는 양방향 Mamba, 청크(Lc=15 → 평균 알고리즘 지연 300 ms), AdaALSD 디코딩 | TED-LIUM2 test: 단일 모드 Conformer(Lh15,Lc15) 8.6 % RTF 0.48; LC-BiConMamba 8.3 % RTF 0.40; 듀얼 DB1 오프라인 6.9 / 스트리밍 7.9 RTF 0.40; Lc 1→25: 9.6→7.8 | ≈108 M 인코더 | pruned transducer, 100 epoch | 선형 복잡도, 히스토리 재계산 없음 |
| Chunkwise Aligner | ICASSP 2026, 2605.11422 | 라벨을 청크 왼쪽 프레임에 정렬, 청크 종료 확률(EOC) 학습, 강제 정렬 필요 | LibriSpeech 오프라인 2.2/5.0 RTF 0.12 vs transducer 2.2/4.9 RTF 0.30; 스트리밍 3.2/7.9(320 ms 지연) vs transducer 3.1/7.6; 알고리즘 지연 600 ms | 110 M(Conformer-L 17층) | MFA 또는 CTC 정렬, 청크 15 | 디코딩 2.6배 빠름 |
| Unified RNNT + MCR-RNNT | Interspeech 2026(andrusenko26, NVIDIA) | 청크 제한 어텐션 + 우측 문맥 + 동적 청크 conv, 모드 일치 정규화(Triton) | Open ASR 평균 WER: 오프라인 6.63, 1.12 s 7.09, 0.56 s 7.47, 0.40 s 7.83, 0.32 s 8.24, 0.24 s 9.04, 0.16 s 10.51; 스트리밍 기준 모델 0.16 s 9.84 / 오프라인 7.75; 오프라인 모델을 0.56 s로 스트리밍하면 13.56 | 128 M FastConformer | 120 k h, 32×A100, 100 k step | 지연 = 청크 + 우측 문맥(예 0.56 = 0.16 + 0.40) |
| Selective Lookahead | SLT 2026, 2609.37611 | 토큰별로 look-ahead 대기/확정 결정, 안정 접두(stable-prefix), 대기 예산 | LibriSpeech test-clean(train-clean-360 학습, 청크 768 ms): 정적 R=0 8.5 % 중앙값 238 ms; R=1 6.7 % 860 ms; 동적 θ=0.70+안정접두 6.5 % 306 ms(p90 2800); +예산 B=1 6.8 % p90 964 ms; FastEmit 6.7 % 326 ms(p90 656) | 12층 Conformer d=256 | 트리거 학습 | 재디코딩 비용 |
| Nemotron Speech Streaming 0.6B / 온디바이스 | HF 모델 카드, arXiv 2604.14493(2026) | 24층 cache-aware FastConformer RNNT, att_context_size [70,0/1/6/13] | LibriSpeech test-other 1.12 s 4.84, 0.56 s 5.07, 0.16 s 5.57, 0.08 s 6.01; 히스토리 1.12 s→5.6 s(0.56 s 지연) 평균 WER 8.51→7.28; int4 0.67 GB, CPU RTFx 7.20(0.56 s), 2.87(0.16 s) | 600 M | 530 k h | 유효 지연 ≈ 지연 + 청크/RTFx |
| X2Streaming-ASR | arXiv 2026, 2609.08672 | 위치별 추가 문맥량 결정(hard-commit, GRPO형 보상) | 문자 확정 지연 평균 24–97 ms vs 기준 409–585 ms(AISHELL-1/3, 강제 정렬 기준) | 미기재 | 3단계 학습 | 단일 패스 |
| LLM 기반 스트리밍 | ICASSP 2026 2601.22779; 2603.11123; 2602.11298 | MoChA read/write 정책 + LLM; Uni-ASR; Voxtral Realtime(480 ms 지연에서 Whisper 수준) | 2601.22779: AISHELL-1 CER 5.1, 토큰 생성 지연 62.5 % 감소 | 대형 | 대규모 데이터 | 우리 조건에 부적합 |

### 2.2 지연 제어 손실(latency-aware loss)

| 방법 | 출처 | 대상 | 보고 수치 |
|---|---|---|---|
| FastEmit | ICASSP 2021, 2010.11148 | transducer | 비blank 기울기를 (1+λ)배. ContextNet-M LibriSpeech λ 0: 4.5/10.0, PR50/PR90 70/270 ms; λ 0.004: 3.5/8.4, −30/100 ms; λ 0.01: 3.5/8.6, −110/40 ms; λ 0.04: 4.4/10.0, −230/−90 ms. Voice Search Conformer-T: 5.6 % EP50/90 260/590 → FastEmit 5.8 % 290/660 ms(이 경우 EP 지연은 늘었다) |
| Delay-penalized transducer | ICASSP 2023 제출, 2211.00490 | transducer(k2) | 비blank log-prob에 λ(T/2−t) 가산. 12층 Conformer 640 ms 청크, MFA 기준 MAD/MED: λ 0 3.4/8.66, 373/485 ms; 0.0015 3.42/9.01, 213 ms; 0.0030 3.70/9.35, 102 ms; 0.0060 3.74/9.56, −3 ms; 0.0100 4.67/10.44, −93 ms. FastEmit과 "유사한 trade-off" |
| Delay-penalized CTC(FST) | Interspeech 2023, 2305.11539 | CTC(k2) | 77 M Conformer, 320 ms 청크, torchaudio 정렬, MSD/MED: 기준 4.56/12.21, 273/189 ms; λ 0.005 5.00/12.86, 179 ms; 0.010 5.32/13.82, 108/33 ms; 0.015 5.58/14.04, 42/−36 ms; 0.020 5.96/14.81, 3/−75 ms. 보조 transducer(λ 0.0050)+CTC λ 0.010: 4.74/11.89, MSD 77 ms |
| Peak-First CTC | ICASSP 2023, 2211.03284 | CTC | 인접 프레임 KL로 스파이크 좌이동. AISHELL-1 510 ms 미래 문맥: λ 0 CER 7.03 APL 384.95 ms; λ 3.0 6.84 / 283.22 ms; λ 5.0 7.22 / 206.44 ms; 비스트리밍 APL −202 ms(GMM 정렬) |
| Bayes Risk CTC | ICLR 2023, 2210.07499 | CTC | 조기 방출 위험 함수 e^{−λ(τ−τ')/T}. Aishell-2 온라인 최소 전체 지연 431 → 302 ms(−30 %); 오프라인 다운샘플링으로 RTF 0.42→0.22(−47 %). 온라인 CER 열은 추출이 깨져 있음(확인 필요) |
| Minimum latency training | Interspeech 2022, 2211.02333 | transducer | 격자 대각선별 기대 지연을 forward-backward로 계산. WSJ 인과 Conformer-T 220 → 27 ms(WER 0.7 % 악화); 정렬 제한 110 ms, FastEmit 67 ms |
| TrimTail | ICASSP 2023 제출, 2211.00522 | CTC/transducer | 스펙트로그램 꼬리 프레임 절단. Aishell-1·LibriSpeech 100–200 ms 지연 감소, 동등 이상 정확도 |
| Delayed-KD, FaT | 위 표 | CTC | 증류로 스파이크를 교사 쪽으로 당김(위 표 수치) |

### 2.3 메모

- 동적 청크 학습의 세부: U2는 배치마다 x~U(0,1), x>0.5면 전체 문맥, 아니면 청크 길이 U(1, min(25, l_max−1)); epoch 단위 스케줄은 "작동하지 않음"으로 보고(2012.05481). icefall은 청크 목록(`16,32,64,-1`)과 좌측 문맥 목록에서 무작위 선택(train.py). DCTX는 60 % 청크 / 40 % 전체(2306.08175).
- 우측 문맥은 "레이어별"로 주면 깊이만큼 누적된다(T-T: 2프레임×18층 ≈1 s, 2002.02562; NeMo: 17×2×10 ms×4=1360 ms). 청크 경계 마스크(U2/CSA/chunk-aware)는 전 레이어가 같은 경계를 보므로 누적이 없고 평균 look-ahead는 청크의 절반이다(2010.11395, 2312.17279).
- 좌측 문맥은 지연을 늘리지 않고 정확도를 올리며, NeMo 문서는 "약 6 s면 무제한과 비슷"이라 한다. 좌측 문맥을 0으로 두면 손실이 매우 크다(DCTX 표: 320 ms에서 3.1 → 5.4).
- Mamba(LC-BiMamba)는 어텐션 없이 RTF 0.40으로 Conformer 0.48보다 빨랐지만 ≈108 M 규모·NTT 내부 구현이고 2080 Ti에서의 커널 지원은 미확인(확인 필요).
- Zipformer 논문 자체에는 스트리밍 수치가 없으므로 icefall RESULTS.md와 한국어 모델 카드를 근거로 삼았다. 두 레시피의 청크 단위가 다르다: 구 레시피(`pruned_transducer_stateless7_streaming`)는 `--decode-chunk-len 32`를 320 ms(100 Hz 입력 프레임)로, 신 레시피(`zipformer`)는 `--chunk-size 16`을 320 ms(50 Hz 프레임)로 적는다.
- 2026년 신작(Chunkwise Aligner, MCR-RNNT, Selective Lookahead, X2Streaming, Nemotron)은 공통적으로 "지연 = 청크 + 우측 문맥"을 명시하고, 우측 문맥을 학습 때 섞어 하나의 모델로 여러 지연을 지원하는 쪽으로 수렴한다.

## 3. 디코딩

### 3.1 CTC 스트리밍 디코딩

- greedy: 프레임별 argmax 후 반복·blank 축약. 상태가 없어 캐시형 인코더(과거 재계산 없음)에서는 이미 낸 토큰이 바뀌지 않으므로 부분 결과가 append-only로 안정적이다. 청크형 인코더에서는 토큰 시각이 "청크 끝"으로 찍힌다(2505.22069). WeNet AISHELL-1에서 greedy 6.99 vs prefix beam 6.98(청크 16)로 1차 결과 차이는 거의 없고, 2차 리스코어링이 6.05로 끌어올린다(2102.01547).
- prefix beam search: 프레임마다 각 접두어를 어휘로 확장하고 blank/비blank 확률을 합친다. WeNet 구현은 프레임별 top-k 가지치기 뒤 확장하며, 발화 간 배치가 되지 않는다(검색 결과; 복잡도 대략 O(T·K·k)는 본 조사의 추정, 확인 필요). 음절 어휘(2,460개, 한국어 카드) 정도면 CPU에서도 부담이 작다(확인 필요).
- n-gram LM: WeNet 2.0은 TLG WFST(T∘min(det(L∘G)))로 CTC WFST 빔 탐색을 하고 blank 확률 0.98 초과 프레임을 건너뛴다. 이득은 AISHELL-1 4.63 → 4.40, LibriSpeech test-other 6.53 → 5.98(2203.15455). icefall은 `fast_beam_search_LG`(FSA 기반, GPU 실행 가능)를 제공한다. 신경 LM shallow fusion은 프레임 동기 탐색에서 호출이 잦아 RTF가 크게 늘며, delayed fusion으로 RTF 0.082 → 0.066을 보고(2501.09258, 검색 요약만 확인, 확인 필요).
- 2차 어텐션 리스코어링(U2 계열): 끝점에서 n-best를 한 번 다시 채점. 추가 지연 Δ 50–100 ms(2012.05481이 [10,15]를 인용한 추정치이며 자체 측정값은 아님; 2407.10255), WeNet 측정 L2 ≈115 ms. 최종 결과가 부분 결과와 달라질 수 있어 화면 깜빡임(flicker)을 유발한다.
- 블록 동기 빔 탐색(ESPnet 어텐션 디코더)은 블록마다 디코더를 재실행하고 신뢰도·반복 감지가 없으면 WER가 3.0/7.8 → 32.3/39.8로 붕괴한다(2006.14941). 처음부터 학습하는 소형 모델에는 과하다.

### 3.2 Transducer 스트리밍 디코딩

- greedy / modified beam search(프레임당 기호 1개, 배치 가능) / fast beam search(FSA, GPU). icefall 66 M에서 3.06 / 3.01 / 3.04(test-clean, simulated streaming; 실제 청크 디코딩 3.06 / 3.05 / 3.07)로 차이가 작다(RESULTS.md).
- 가설마다 예측망 상태를 들고 다녀야 하나, stateless decoder(임베딩 + kernel 2 conv)로 비용이 작다(2206.13236). 학습 메모리는 pruned RNN-T가 결정적이다: 손실 계산 벤치마크(batch 30·어휘 500·V100; 전체 학습 메모리가 아니라 손실 계산의 피크 메모리)에서 torchaudio 544 ms·18.48 GB vs pruned 64 ms·3.73 GB; WER은 pruned 2.56/6.27 vs 비pruned optimized_transducer 2.61/6.46(빔 4, LM 없음, 8×V100; 2206.13236 Table 1·3). 11 GB GPU에서 transducer를 쓰려면 사실상 pruned 손실이 필요하다.
- TDT(토큰+지속시간 예측)는 추론을 최대 2.82배 가속(2304.06795)하지만 스트리밍 적용은 별도 확인 필요. Chunkwise Aligner는 transducer 대비 RTF 0.30 → 0.12이나 강제 정렬이 필요하다(2605.11422).
- 방출 지연: 2305.11539는 transducer 헤드 자체의 지연을 보고하지 않는다. 지연 패널티 없는 보조 transducer를 붙이면 CTC 헤드의 MSD가 273 → 342 ms로 늘어나고(Table 2), transducer에 λ=0.0050을 주면 CTC MSD 151 ms, WER 4.19/10.95로 돌아온다. transducer 단독의 기준 지연은 다른 모델·지표에서 MAD 373 ms(2211.00490)이므로, transducer 주력 시 delay penalty가 거의 필수다.

### 3.3 부분 결과 안정성(partial-result stability)

- 지표: UPWR(불안정 단어 수 / 최종 단어 수), UPSR(수정된 구간 수 / 발화 수), "접두어가 이후 모든 가설에 유지될 때만 안정"(2006.01416). 불안정 원인의 47.6 %가 텍스트 정규화(구두점·대소문자·숫자), 52.4 %가 스트리밍 조기 방출이었다. 부분 결과 방출 간격(PEI) 50 → 200 ms로 늘리면 UPWR 0.14 → 0.04, UPSR 0.48 → 0.15, 평균 부분 지연 +75 ms.
- 정책: Whisper-Streaming의 LocalAgreement-n은 연속 n회 출력의 최장 공통 접두어만 확정(영어 ESIC 0.5 s 청크 WER 8.5 %·평균 지연 3.27 s, A40 GPU, 2307.14743); 부분 가설 선택(hold-n/local agreement)은 How2에서 지연 −83 %(0.8 s)에 WER +1 %(2005.11185); Selective Lookahead는 안정 접두 + 대기 예산으로 p90 2800 → 964 ms(WER 6.5 → 6.8)(2609.37611).
- CTC greedy + 캐시형 인코더 + 우측 문맥 0이면 구조적으로 수정이 없으므로 가장 안정적이고, 우측 문맥이나 빔/리스코어링을 넣는 만큼 안정화 정책이 필요하다.

### 3.4 청크당 추론 비용

- CPU: WeNet x86 fp32 RTF 청크16 0.095 → 청크4 0.186(int8 0.081 → 0.134), Android 0.251 → 0.505(int8 0.114 → 0.201). 청크가 작을수록 오디오 1초당 호출 수가 늘어 RTF가 커진다(2102.01547). MS T-T 16코어 1스레드 RTF 0.40(720 ms look-ahead), int8 0.22; 프레임 15개 묶음 배치로 RTF 0.19이나 지연 +450 ms(2010.11395). 한국어 Zipformer 155.7 M int8 1스레드 RTF 0.073/0.054/0.049(청크 16/32/64). Nemotron 600 M int4는 0.56 s 청크에서 RTFx 7.20(청크당 ≈78 ms, 본 조사 환산, 확인 필요).
- 작은 세그먼트는 비용이 크다: Emformer RTF 0.13(중지연) → 0.30–0.49(80 ms 저지연).
- 중복 계산 회피: NeMo chunk-aware look-ahead는 "look-ahead 활성값 대부분을 캐시해 중복이 거의 0", buffered streaming은 겹침 청크로 중복 계산(NeMo 문서). 캐시 상태는 어텐션 KV 캐시·conv 캐시·유효 길이 세 텐서(NeMo 스크립트). 입력 정규화는 전체 발화를 볼 수 없으므로 "스트리밍 모델은 입력 정규화 없이 학습하는 것을 권장"(NeMo 스크립트).
- GPU: 여러 스트림을 배치로 묶어 청크 단위 forward(icefall `--num-decode-streams`, NeMo batch_size>1). 단일 스트림 소형 모델이면 2080 Ti의 청크당 시간은 수 ms 수준으로 예상되나 측정 필요(확인 필요).

## 4. 지연 측정 방법

### 4.1 알고리즘 지연(algorithmic latency)

- Emformer EIL = look-ahead + 0.5 × 중앙 블록(예 320 + 0.5×1280 = 960 ms)(2010.10759). U2/U2++: 청크 16 ⇒ 0–640 ms, 평균 320 ms(2106.05642). WeNet L1 = (청크/2 × 4 + 6) × 10 ms(서브샘플 4, CNN look-ahead 6 프레임)(2102.01547). CUSIDE·MCR-RNNT: 지연 = 청크 + 우측 문맥(0.56 s = 0.16 + 0.40)(andrusenko26). NeMo: 평균 지연 = look-ahead의 절반(2312.17279). LC-BiMamba: Lc 15 ⇒ 평균 300 ms(moriya25).
- 보고 시 평균과 최대를 함께 적어야 비교가 된다(같은 "320 ms"가 평균인지 최대인지 논문마다 다름).

### 4.2 토큰 방출 지연(emission delay)

- 기준 정렬: MFA 강제 정렬(2211.00490, 2609.37611), torchaudio 강제 정렬(2305.11539), Kaldi GMM-HMM(2211.03284, 2505.22069), 또는 음성 종료 시각(FastEmit).
- 지표: PR latency(마지막 토큰 방출 − 음성 끝)·EP latency, 50/90 백분위(2010.11148); MAD/MED(단어 시작·끝 평균 차)(2211.00490); MSD/MED(2305.11539); APL(2211.03284); FTD/LTD(첫·마지막 토큰, P50/P90)(2505.22069); 단어별 커밋 시각 평균/중앙값/p90(2609.37611). 음수 값은 발음 종료 전에 방출했음을 뜻한다(FastEmit, FaT, Peak-First).
- CTC 스파이크 위치는 정렬 기준점으로 신뢰할 수 없다: Zeyer 등은 균등 초기화 FFNN이 "단어 전체를 가능한 한 짧게 압축"하는 피키한 국소 최적점으로 수렴함을 보였고, 라벨 prior를 더하면 피키함이 사라진다(2105.14849). 따라서 방출 지연은 별도 정렬기 기준으로 재야 한다.

### 4.3 끝점·사용자 체감 지연(UPL)

- UPL = 음성 끝 → 마이크 닫힘 = 끝점 지연 + 디코더 catch-up. 모델 크기·FLOPS·RTF는 UPL과 "항상 강하게 상관하지 않음"(2104.02207). 측정값(LSTM RNN-T 8×640, 69.9 MB, StaticEP 0.9 s): 첫 토큰 지연 0.62 s, UPL P50 1.60 s, EP 지연 1.00 s, catch-up 0.56 s; 정렬 제한 RNN-T는 UPL 1.37 s(WER 1.94 → 2.22); 신경 끝점기(NEP)로 EP 지연 0.19 s, UPL 0.94 s. 끝점 지연이 UPL의 50 % 이상.
- RNN-T 끝점기(</s> 토큰 + 조기/지연 패널티): EP50/EP90 기준 410/710 ms → 380/580 ms(E5, WER 7.5 → 6.9); 조기 패널티 0.1, 유예 3/5/7 프레임(180/300/420 ms), 기준 시각은 "마지막 비묵음 음소 다음 프레임"(2004.11544). E2E Segmenter: P50 250 ms 감소·WER 최대 8.5 % 상대 개선(2204.10749, 검색 요약만 확인, 확인 필요).
- 규칙형 끝점(우리 조건에 즉시 적용 가능): sherpa는 연속 blank 수 × 프레임 이동(서브샘플 4 → 40 ms)으로 꼬리 묵음을 재고, rule1(무조건 꼬리 묵음, 예 2.4 s)·rule2(무언가 디코딩 후 묵음, 예 1.2 s)·rule3(발화 길이, 예 20 s) 중 하나라도 켜지면 끝점(k2-fsa sherpa 문서). WeNet 기본값 rule1(false, 5000 ms), rule2(true, 1000 ms), rule3(20000 ms), blank 확률 0.8 이상을 묵음으로 간주(ctc_endpoint.h).

### 4.4 계산·종단 지연

- RTF/RTFx와 함께 "유효 지연 ≈ 알고리즘 지연 + 청크 길이/RTFx"(2604.14493); 서버-클라이언트에서는 마지막 청크 전송 → 최종 전사 수신 시간(final-chunk latency, 동시 100스트림 1.41 s)(2506.14434).

### 4.5 우리가 측정할 것

1. 알고리즘 지연: 청크 + 우측 문맥(평균·최대), 프레임률·서브샘플 명시.
2. 청크당 계산 시간: 목표 장치(2080 Ti, 그리고 배포 CPU)에서 p50/p90 ms와 RTF, 좌측 문맥 길이별.
3. 방출 지연: 한국어 강제 정렬 기준(MFA 한국어 모델 또는 자체 비스트리밍 CTC 정렬, 확인 필요) FTD/LTD P50/P90 또는 MSD/MED; 음수 허용.
4. 부분 결과 안정성: UPWR/UPSR, 부분 결과 방출 간격.
5. 끝점 지연과 UPL: 끝점 규칙별 EP50/EP90, 조기 끝점에 의한 삭제 오류.
6. 위 각 항목을 CER과 함께 하나의 표로(청크 160/320/640 ms × 좌측 문맥 × 우측 문맥 0/1 청크).

## 5. 우리 조건에서의 순위(top 3)

1. **청크형 인과 Zipformer-S(또는 Conformer-S) + CTC 주력 + 보조 pruned transducer, 동적 청크 학습, delay-penalized CTC, 규칙형 끝점.**
   근거: 65 h·단일 11 GB에서 가장 단순한 디코딩(greedy, append-only 부분 결과)과 가장 낮은 청크당 비용. 보조 transducer가 CTC WER를 4.56 → 4.00 / 12.21 → 10.42로 개선하고(2305.11539; 지연 패널티 없는 transducer 기준이며 이때 CTC MSD 273 → 342 ms, 결합 손실 0.2·L_ctc + 1.0·L_transducer. CTC λ 0.010 + transducer λ 0.0050이면 4.74/11.89, MSD 77 ms), icefall 레시피가 `--use-ctc`·`--use-transducer`·`--causal`·fp16을 모두 지원한다. 소형 기준점: Zipformer-S 23.3 M(2.42/5.73 비스트리밍), Conformer-S 10.3 M(2.7/6.3). AISHELL-1(초록 170 h, 학습셋 150 h)·47 M(디코더 6층 포함) U2가 청크 16에서 5.33 %를 냈으므로 65 h에서는 더 작은 모델(10–25 M)이 적절해 보인다(확인 필요). 메모리: Zipformer-S는 2×V100-32G·max-duration 1500 s로 학습됐으므로 11 GB에서는 max-duration 300–500 s 수준으로 낮춰야 할 것으로 추정(확인 필요).
   미결: 청크 320 vs 640 ms, 우측 문맥 0 vs 1청크 혼합 학습(2506.14434식) 여부, 음절 vs BPE 어휘, delay penalty λ(CTC 0.005–0.010 구간부터), 한국어 정렬기.
2. **pruned stateless transducer 단독(icefall 스트리밍 Zipformer 방식).**
   근거: 스트리밍 정확도가 가장 좋은 공개 레시피(66 M, 320 ms에서 3.06/7.81 simulated, 실제 청크 디코딩 3.06/7.79)와 한국어 선례(155.7 M, 6,500 h, 청크 16 CER 8.255)가 있고, pruned 손실로 11 GB에서도 학습 가능(3.73 GB @ batch 30). 단점: 예측망 상태를 가진 빔 탐색, 더 큰 방출 지연(MAD 373 ms)으로 delay penalty 필수, 65 h에 대한 소형 transducer 수치가 없음(확인 필요).
3. **U2/U2++식 2패스(CTC 1차 + 어텐션 리스코어링), WeNet 런타임.**
   근거: 생산 런타임·TLG LM·int8이 갖춰져 있고 AISHELL-1(초록 170 h, 학습셋 150 h)에서의 수치가 가장 풍부하다(47 M, 디코더 6층 포함, 청크 16 5.33 %, 리스코어링 Δ 50–115 ms). 단점: 리스코어링은 끝점 뒤에만 가능해 부분 결과와 최종 결과가 달라지고, 6층(U2++ 3+3) 디코더가 65 h에서 과적합할 위험(확인 필요), WeNet은 Zipformer형 다중 프레임률 인코더가 없다.

제외: Emformer 메모리 뱅크(80 M+, 구현 복잡), LLM 기반(데이터·연산), Chunkwise Aligner(강제 정렬 필요), Mamba(2080 Ti 커널 지원 미확인), 블록 동기 어텐션 디코딩(붕괴 위험·비용).

## 6. 읽은 자료 목록

- Zhang et al., Unified Streaming and Non-streaming Two-pass E2E Model (U2), arXiv 2012.05481 (ar5iv 전문)
- Wu et al., U2++, arXiv 2106.05642 (ar5iv 전문)
- Yao et al., WeNet toolkit, arXiv 2102.01547 (ar5iv 전문); Zhang et al., WeNet 2.0, arXiv 2203.15455 (ar5iv 전문)
- Shi et al., Emformer, arXiv 2010.10759 (ar5iv 전문)
- Noroozi et al., Stateful Conformer with Cache-based Inference, arXiv 2312.17279 (HTML v3); NeMo ASR models 문서(docs.nvidia.com/nemo-framework/.../asr/models.html); NeMo `speech_to_text_cache_aware_streaming_infer.py`
- Yao et al., Zipformer, arXiv 2310.11230 (HTML v4); icefall `egs/librispeech/ASR/RESULTS.md`, `zipformer/train.py`; icefall 문서 `Streaming-ASR/librispeech/zipformer_transducer`
- HF 모델 카드 kangkyu/icefall-asr-ko-streaming-zipformer-174m; nvidia/nemotron-speech-streaming-en-0.6b
- Yu et al., Dual-mode ASR, arXiv 2010.06030 (ar5iv 전문)
- Li et al., Dynamic Chunk Convolution, arXiv 2304.09325 (ar5iv 전문); Huybrechts et al., DCTX-Conformer, arXiv 2306.08175 (HTML v2)
- Zhang et al., Transformer Transducer, arXiv 2002.02562 (ar5iv 전문); Chen et al., Developing Real-time Streaming Transformer Transducer, arXiv 2010.11395 (ar5iv 전문)
- Moritz et al., Dual Causal/Non-Causal Self-Attention, arXiv 2107.01269 (ar5iv 전문)
- Tsunoo et al., Blockwise Synchronous Beam Search, arXiv 2006.14941 (ar5iv 전문)
- An et al., CUSIDE, arXiv 2203.16758 (ar5iv 전문); CUSIDE-T, arXiv 2407.10255 (HTML v2)
- Sharma et al., Unifying Streaming and Non-streaming Zipformer-based ASR, arXiv 2506.14434 (HTML v1)
- Le & Chau, Time-Shifted Contextual Attention / Dynamic Right Context, arXiv 2502.15158 (abs)
- Yang et al., Learning from Back Chunks (FaT), Interspeech 2024 (ISCA PDF, 로컬 텍스트 추출)
- Li et al., Delayed-KD, Interspeech 2025 / arXiv 2505.22069 (HTML v1)
- Moriya et al., Attention-Free Dual-Mode ASR with LC-BiMamba, Interspeech 2025 (ISCA PDF, 로컬 텍스트 추출)
- Zhou et al., Adapting Whisper for Streaming via Two-Pass Decoding, Interspeech 2025 (ISCA 초록)
- Teo et al., Chunkwise Aligners, ICASSP 2026 / arXiv 2605.11422 (HTML v1)
- Andrusenko et al., Reducing the Offline-Streaming Gap for Unified ASR Transducer (MCR-RNNT), Interspeech 2026 (ISCA PDF, 로컬 텍스트 추출)
- Jia et al., Selective Lookahead for Attention-Based Streaming ASR, SLT 2026 / arXiv 2609.37611 (HTML)
- Banfic et al., Pushing the Limits of On-Device Streaming ASR, arXiv 2604.14493 (HTML v2)
- Lin et al., X2Streaming-ASR, arXiv 2609.08672 (abs); Wan et al., Streaming with Decoder-Only LLMs, ICASSP 2026 / arXiv 2601.22779 (abs); Uni-ASR arXiv 2603.11123 (abs); Voxtral Realtime arXiv 2602.11298 (abs); Beyond Fresh Starts arXiv 2608.22101 (abs); Alignment-Path Distillation arXiv 2609.20121 (abs)
- Yu et al., FastEmit, arXiv 2010.11148 (ar5iv 전문)
- Kang et al., Delay-penalized transducer, arXiv 2211.00490 (ar5iv 전문); Yao et al., Delay-penalized CTC (FST), arXiv 2305.11539 (ar5iv 전문)
- Tian et al., Bayes Risk CTC, arXiv 2210.07499 (ar5iv 전문, 부록 일부 깨짐); Tian et al., Peak-First CTC, arXiv 2211.03284 (ar5iv 전문)
- Shinohara & Watanabe, Minimum Latency Training of Sequence Transducers, arXiv 2211.02333 (abs); Song et al., TrimTail, arXiv 2211.00522 (abs)
- Zeyer et al., Why does CTC result in peaky behavior?, arXiv 2105.14849 (ar5iv 전문)
- Li et al., Towards Fast and Accurate Streaming E2E ASR (RNN-T endpointer), arXiv 2004.11544 (ar5iv 전문); E2E Segmenter arXiv 2204.10749 (검색 요약만)
- Shangguan et al., Dissecting User-Perceived Latency, arXiv 2104.02207 (PDF 로컬 텍스트 추출); Shangguan et al., Analyzing the Quality and Stability of a Streaming On-Device Recognizer, arXiv 2006.01416 (ar5iv 전문)
- Liu et al., Partial Hypothesis Selection, arXiv 2005.11185 (abs); Macháček et al., Whisper-Streaming, arXiv 2307.14743 (HTML v2); Li et al., Encoder States Revision, arXiv 2207.02495 (abs)
- Kuang et al., Pruned RNN-T, arXiv 2206.13236 (ar5iv 전문); Xu et al., TDT, arXiv 2304.06795 (abs)
- k2-fsa sherpa 끝점 문서(k2-fsa.github.io/sherpa/ncnn/endpoint.html); WeNet `runtime/core/decoder/ctc_endpoint.h`
- 검색만으로 확인한 항목(확인 필요 표시): Delayed fusion arXiv 2501.09258, Flexi-Transducer 2104.02232, Nemotron 관련 StreamHear 2608.13717

## 7. 검증 후 정정 (2026-10-11)

적대적 검증(README §2, `work/survey_workflow_results.json`의 verify C2·C6·C12·C13)에 따라 아래를 고쳤다. 수치 자체는 모두 원문과 일치했고 조건·비교 대상·표현만 바로잡았다.

- C6: §3.2 "보조 transducer 헤드 342 ms"는 오독이어서 "지연 패널티 없는 transducer와 공동 학습 시 CTC 헤드의 MSD 273 → 342 ms, transducer 헤드 지연은 미보고"로 고치고 transducer λ=0.0050 조합(CTC MSD 151 ms, 4.19/10.95)을 추가.
- C6: §1 (3)·§5 1순위에 보조 transducer 이득의 조건(지연 패널티 없음, CTC MSD 273 → 342 ms), 결합 손실 0.2·L_ctc + 1.0·L_transducer, 지연 패널티 결합(CTC λ 0.010 + transducer λ 0.0050: 4.74/11.89, MSD 77 ms)을 병기.
- C6: §3.2 pruned RNN-T WER 쌍을 "pruned 2.56/6.27 vs 비pruned optimized_transducer 2.61/6.46(빔 4, LM 없음)"으로 순서·비교 대상을 바로잡고, 18.48 → 3.73 GB는 손실 계산 벤치마크의 피크 메모리임을 명시.
- C2: §2.1 icefall 행의 "simulated vs 실제 ≤0.03"을 "greedy ≤0.03, beam 계열 최대 0.06"으로, 학습 조건을 "`--use-fp16 1`, num-epochs 40(보고 WER은 epoch 30 avg 8), 청크·좌측 목록은 train.py 기본값"으로 수정.
- C2: §1, §3.2, §5 2순위의 3.06/7.81·2.81/7.15가 simulated 값임을 표시하고 실제 청크 디코딩 3.06/7.79·2.84/7.16(beam 3.05/3.07)을 병기.
- C12: §1 트레이드오프 4와 §2.1 2506.14434 행에 LS960·평가 청크 32·좌측 128, 기준 모델 RC 0 3.33/8.90 대비 혼합 학습 모델의 RC 0 퇴보, 프레임 단위 차이(논문 10 ms vs icefall 레시피 20 ms), "Zipformer-M(LS)/L(사내), 파라미터 수·fp16·디코딩 방식 미기재"를 추가.
- C13: U2 서술(§1, §2.1, §5)의 "170 h AISHELL-1"을 "AISHELL-1(초록 170 h, 학습셋 150 h)"로, "47 M"에 "인코더 12층 + 어텐션 디코더 6층 포함 전체"를 명시.
- C13: §2.1·§3.1의 리스코어링 Δ 50–100 ms가 U2 논문이 [10,15]를 인용한 값이며 자체 측정이 아님을 표시.
- C13 (d)의 U2 표 1(4.92/4.93/4.64) 비고와 C2의 오프라인 2.22/4.87 epoch 비고는 이 문서에 해당 수치가 없어 적용하지 않음(03·01 문서 대상).
