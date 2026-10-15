# 조사 종합 — 한국어 처음부터 학습 스트리밍 STT 구조 선정

작성일: 2026-10-11. 상태: **결정 완료**(맨 아래 "결정" 절, 2026-10-11). 이 문서는 조사 문서 4편, 검증 결과 13건(C1~C13), 누락 논문 조사 3묶음, 설계안 4건, 심사 3건을 한 곳에 모은 것이다. 처음 초안을 쓸 때는 조사 문서 본문을 고치지 않았고, 아래 §2의 정정은 그 뒤 각 조사 문서 끝의 "검증 후 정정" 절로 반영했다(커밋 be3f55f). 결정 뒤 측정으로 바뀐 구성(예: 자기조건 CTC → InterCTC 6층)은 "결정" 절 끝에 적었다.

프로젝트 조건: 공개 한국어 약 61~65 h(Zeroth-Korean 51.6 h, FLEURS ko 약 10 h, Common Voice ko 검증분 1.6~2.2 h), 사전학습 가중치 없음, RTX 2080 Ti 11 GB 1장(fp16만), Windows 11(WSL2 없음), 순수 PyTorch 2.11 + CUDA 12.8. 목표는 마이크 청크 → 중간 자막 스트리밍 서비스. 비교 기준선은 whisper-ko-ft의 Zeroth test CER(기본 whisper-large-v3-turbo 4.48 %, 파인튜닝 1.96 %). 측정은 사전 등록 방식(후보·판정 규칙 먼저 커밋, dev로 고르고 test는 한 번).

## 1. 조사 범위와 문서 목록

| 문서 | 범위 | 비고 |
|---|---|---|
| `01-encoders.md` | 인코더 구조 2022-01~2026-10 (Conformer·Branchformer·Zipformer·Fast Conformer·Mamba 등) | 1순위 Zipformer, 2순위 E-Branchformer로 정리 |
| `02-streaming.md` | 청크 인과 어텐션·캐시 추론·동적 청크·방출 지연·끝점·부분 결과 안정성 | §4.5에 측정 6항목 양식 |
| `03-objectives-decoding-units.md` | CTC/InterCTC/SC-CTC/CR-CTC, pruned RNN-T, 디코딩, 한국어 출력 단위, fp16 CTC | 1순위 CTC 계열, 2순위 pruned RNN-T |
| `04-small-data-korean.md` | ≤100 h 레시피·증강·한국어 데이터·라이선스·11 GB fp16 운영 | Zeroth/FLEURS/CV 데이터 계획 |

검증 13건은 조사 문서의 핵심 수치 주장을 원문(arXiv HTML, icefall RESULTS.md, PyTorch 문서, HF datasets-server, 로컬 실측)과 대조한 것이다. 판정: confirmed 7건(C1, C4, C5, C8, C9, C12, C13), partially_confirmed 6건(C2, C3, C6, C7, C10, C11). 수치 자체가 틀린 경우는 없었고, 조건 짝짓기·일반화·용어에서 어긋난 곳이 있었다.

## 2. 검증에서 정정된 것 (조사 문서 수정 목록)

아래는 조사 문서의 어느 문장을 어떻게 고쳐야 하는지의 목록이다. 이 목록을 만들 때는 문서를 고치지 않았고, 반영은 각 조사 문서 끝의 "검증 후 정정" 절에 했다.

**01-encoders.md**
- L9 "fp16 혼합정밀을 기본으로 설계(Balancer·Whitener·BiasNorm·ScaledAdam)" → "혼합정밀(V100, 사실상 fp16)로 학습했고 Balancer만 혼합정밀 불안정 대책으로 서술됨(BiasNorm·Whitener·ScaledAdam 동기는 별개)". 논문에 fp16이라는 단어는 없고 발산 통계도 없음 (C1).
- L9 "30초 입력 FLOPs 40.8 G가 가장 적다" → Table 2 안에 Squeezeformer-XS 18.2 G, Conformer-S 29.1 G 등이 더 적으므로 "WER 2.5/6.0 이하 모델 중 가장 적다" 또는 "Conformer-M 대비 47 % 적다"로 (C1).
- L9, L26, L91 및 02 L10, L84, L140의 스트리밍 "3.06/7.81", "2.81/7.15"는 simulated(전체 발화 입력, epoch 30 avg 8) 값. "실제 청크 디코딩 3.06/7.79, 2.84/7.16"을 1회 병기하고 오프라인 2.22/4.87은 epoch 50 avg 25임을 명시 (C2).
- L10, L25 "LS-100 38.5M 6.1/16.7 vs 39.0M 6.3/17.0"은 **dev** 수치. "(test 6.3/17.0 vs 6.6/17.2; AED+joint CTC, LM 없음)" 추가 (C3).
- L10, L21, L25, L78, L94 "발산 6/10 vs 2/10", "발산 보고 다수" → 논문 용어는 "failed", E-Branchformer 분모 미명시, TEDLIUM2 1개 설정·10회 시도의 일화적 보고(표 없음, WER·정밀도 미기재). "다수"는 과장 (C3). L94 (c)의 fp16 사례는 2305.11073이 아니라 2210.00077 출처로 분리.
- L42 및 04 L124 "같은 틀로 재현(4 GPU, icefall RESULTS 기준)" → 재현 Conformer의 epoch·옵티마이저·예산은 논문 미기재이므로 동일 예산 비교로 쓰지 말 것. GPU 수는 Table 7에 있음 (C1).
- L45 "LS-100에서 RC 추가로 9.61→5.83" → 부록 A Figure 4 히트맵이며 "기준 모델 RC 0 디코딩 9.61 → RC 혼합 학습 모델 RC 128 디코딩 5.83". 같은 RC 0끼리는 9.61→12.46으로 오히려 나쁨 (C12).
- L72, L97, L27 (2404.04295) → 토큰 단위는 문자(음절)로 확인, 사전학습 언급 없음(처음부터 학습으로 보이나 명시 안 됨), GPU·에폭·정밀도·디코딩 방식 미기재; "hidden 640→400"의 640은 NeMo 기본 인코더 d_model(512)과 맞지 않아 구조 재구성 불가 (C10).
- L82 "icefall 전 레시피가 `--use-fp16 1`" → Zipformer 계열 문서화 명령이 1을 명시(train.py 기본값 False; lstm/emformer/stateless5 일부는 0). LS-100 fp16 6.63/17.39(epoch 30)와 fp32 6.64/17.19(epoch 39)는 체크포인트가 다르며, 같은 epoch 29 avg 8 비교는 fp32 6.71/17.77 vs fp16 6.81/17.84 (C2).
- 공통 열린 질문 4(k2 설치) → 답: k2 CUDA는 Windows pip 휠 없음, torch 2.8+ 소스 빌드 실패 보고만 있음, fast_rnnt는 C++ 확장 필수 → WSL2 없는 Windows에서 사용 불가로 간주 (C11).

**02-streaming.md**
- L9(3), L85, L87, L137 → "보조 transducer 헤드 342 ms"는 오독. 342 ms는 지연 패널티 없는 transducer와 공동 학습했을 때 **CTC 헤드**의 MSD(273→342). transducer 헤드 지연은 미보고. WER 쌍 "2.56/6.27 vs 2.61/6.46"은 pruned가 앞이고 비교 대상은 optimized_transducer(torchaudio 아님) (C6). 결합 손실 0.2·L_ctc + 1.0·L_transducer, 지연 패널티 결합(CTC λ 0.010 + transducer λ 0.005: 4.74/11.89, MSD 77 ms) 추가 권장.
- L13, L38 (2506.14434) → "RC 없이 학습한 기준은 RC 0에서 3.33/8.90이므로 혼합 학습 모델은 RC 0 디코딩에서 퇴보", "논문은 프레임=10 ms로 서술하나 icefall 레시피 단위는 20 ms", 평가 청크 32·좌측 128, 파라미터·fp16·디코딩 방식 미기재 (C12).
- L28 "simulated vs 실제 청크 ≤0.03" → greedy ≤0.03, beam 계열 최대 0.06 (C2).
- L9, L22, L79, L137, L142 (U2) → "AISHELL-1(초록 170 h, 학습셋 150 h)", "47 M은 인코더 12층+디코더 6층 포함 전체", "Δ 50~100 ms는 [10,15] 인용값" (C13).

**03-objectives-decoding-units.md**
- L11, L61, L76 → KsponSpeech 음절 2306은 "CER·sWER 기준 모든 세트에서 같거나 좋음"(Eval-other CER은 10k와 9.0 동률, Eval-other WER은 5k·10k가 0.1 낮음; small Transformer+SpecAugment 조건). Zeroth 51.6 h는 CER은 음절 1.8 최저지만 WER은 음절-서브워드 6k 2.5 < 음절 2.6이고 논문 결론은 서브워드 6k 우위; WER 2.6은 morfessor 분절 기준(어절 WER 아님) (C7).
- L13 "ESPnet·icefall·NeMo 모두 `.float()`" → icefall은 k2 transducer 손실에만 `.float()`, CTC 경로는 autocast의 log_softmax fp32 승격에 의존 (C9).
- L23, L101 (CR-CTC) → "두 증강 뷰를 배치 축으로 이어 2× 배치 1회 forward(연산 2배)", 공개 icefall 레시피는 M 4×V100 fp16 50 ep, "스트리밍 적합성: CTC와 동일"은 추론(논문에 스트리밍 실험 없음) (C5).
- L26 "S(=4~5)" → 예시값, 실험값 미기재(icefall 기본 5); 18.48→3.73 GB는 손실 계산 벤치마크 피크(vocab 500, V100)이지 전체 학습 메모리가 아님 (C6).
- L29 (U2 표 1 4.92/4.93/4.64) → 비스트리밍 kernel-15 모델 수치이며 동적 청크 통합 모델(full 4.90)과 다른 모델 (C13).
- L64, L65 (NVIDIA 2편) → 1.22는 디코더 CV + 조이너 CVW(Table 5; 본문의 "V-VW"는 표상 1.27), 디코더만은 V 1.36; 2407.04368은 중국어-한국어 이중언어 모델(연결 어휘 6,380/2,441), Adam+cosine warmup 10k peak 1.5e-3 (C10).
- §5 → "각 타깃 길이 ≤256"은 ATen 코드상 "<256"; Windows cu128에서 cuDNN CTC 경로 실행 확인; torchaudio 2.11 rnnt_loss는 fp16 logits 허용·bf16 거부·비용은 fp16으로 반환(reduction='sum' overflow 주의), 실측 B=8·T=400·U=80·V=2,500 fp16 피크 3.62 GB (C11). §5-3 포럼 #148800은 "해결 확인 없음"으로.
- §5-7 및 04 §6.2 → KsponSpeech base-lr 0.045→0.035는 `zipformer` 레시피(74.8M, 4 GPU, max-duration 750) 조건; 스트리밍 레시피는 기본 lr 유지 (C9).

**04-small-data-korean.md**
- L9, L98, L152, L53~54, L66~70 (Zeroth) → 고유 문장 3,003(실측), test 457발화 중 456(99.8 %)의 전사가 train에 그대로 있음(화자는 비중첩), config 이름은 `default`. 열린 질문 L152는 "확인 완료"로 바꾸고 모든 Zeroth test CER에 "본 문장·새 화자(seen-text)" 라벨 (C8).
- L11 → "Conformer-12 CTC 12.4→InterCTC 10.8→+Stochastic depth 9.9; Transformer-18 CTC 14.9→InterCTC 12.7→SC-CTC 11.9"로 설정을 분리 (C4).
- L30, L91 → 음절 ≥ 서브워드(CER 기준, Eval-other 동률), Zeroth WER은 서브워드 6k 우위 명시 (C7).
- L43, L123, L139 → fp16 vs fp32 비교에 체크포인트(epoch/avg) 병기, 동일 조건 비교 fp32 6.71/17.77 vs fp16 6.81/17.84, max-duration 500 fp16은 6.89/17.75로 악화 (C2).
- L124 "(960 h, 속도 교란 포함 여부 미기재)" → "(960 h × 속도 교란 3배, Table 7·A.4.1 명시)" — 2080 Ti 처리량 환산은 ~2880 h/epoch 기준 (C1).
- L138 "Balancer ... especially with mixed precision" → Table 6 No Balancer 2.23/4.97 vs 2.21/4.91(40 epoch); 주장은 발산 위험에 관한 정성적 서술 (C1).
- §6.2 "is_bf16_supported()일 때만 bf16이며 2080 Ti에서는 fp16" → torch 2.11 기본 호출은 2080 Ti에서도 True(에뮬레이션 포함)를 반환하며 bf16 에뮬레이션은 fp32보다 느림(4096² matmul bf16 5.8 vs fp16 45.4 vs fp32 7.4 TFLOPS). `is_bf16_supported(including_emulation=False)`로 확인하고 autocast dtype을 명시해야 함 (C9).

## 3. 누락 논문 가운데 결정에 영향을 주는 것

누락 조사 3묶음(인코더·스트리밍 8편, 한국어 12편, 소규모 레시피 11편)에서 "권고를 바꾼다"고 표시된 것만 적는다. 나머지는 기존 순위를 뒷받침하거나 세부 참고다.

| 논문 | 영향 | 설계안 반영 |
|---|---|---|
| BACON (Xu 외, Interspeech 2026-1455) | 청크 스트리밍 conv를 "전 채널 인과"에서 "채널 절반 인과 + 절반 청크 경계 양방향"으로. all-bidir(DCConv형)는 RNN-T에서 인과보다 악화(8.78 vs 8.73). 960 h·110M 결과라 65 h 이득 크기는 미검증 | 4안 모두 arm 또는 기본으로 채택 |
| 오창한 외 2024 (말소리와 음성과학 16(3)) | 유일한 한국어 처음부터 학습 인코더 통제 비교: FLEURS-ko CER E-Branchformer 4.11 < Conformer 4.30 < Transformer 4.54(7,500 h·112M). 음절≈BPE at 7,500 h. AED 디코더는 25 s 이상 발화에서 붕괴(20 s 이하 분할 권고) | E-Branchformer 근거 보강, FLEURS-ko 기대치 재설정 |
| 2602.23171 Align-Consistency | 같은 Conformer에서 CR-CTC 이득이 LS-960 상대 13 %지만 LS-100은 4 %. forward 2배·배치 절반 비용 대비 65 h 기대 이득이 작음 | 4안 모두 CR-CTC를 기본에서 선택 arm으로 강등 |
| 2606.18466 MFA 2026 | MFA Korean 3.0(Seoul Corpus 경계 오차 14.78 ms)이 방출 지연 측정 기준 정렬기로 사용 가능. 02 §4.5의 "확인 필요" 해소 | 4안 모두 채택(Windows 설치 미확인) |
| Common Voice 22.0 CC0 미러 (fsicoli/common_voice_22_0) | CV17(검증 1.72 h·90명) 대신 CV22(2.17 h·154명, train 519/dev 474/test 472) 사용 가능. 27.0은 계정 필요 | 4안 모두 CV22 1순위·CV17 폴백 |
| 2412.00877 CBA, 2505.20606 Mixup/SpecMix | LS-100 처음부터 학습에서 적응형 SpecAugment·InterCTC 가중(13~14 % 상대), Mixup/SpecMix > SpecAugment(과소학습 조건) | 선택 arm으로만 |
| TTS 합성 보강 (2503.08954, 2410.16726 등) | ≤100 h에서 가장 큰 이득(31M Conformer-Transducer CV 50 h 46.3→37.1; 한국어 CER 8.48→6.31)이지만 사전학습 TTS 산출물 사용이 "처음부터 학습" 전제와 충돌 | 4안 모두 사용자 결정 보류 항목으로 분리 |

참고치 갱신(04 문서의 FLEURS 기준 "0.6B 사전학습 21.7 %"는 더 이상 유효하지 않음): FLEURS-ko CER Whisper-tiny 38M 11.9 / base 7.8 / small 4.5(HiKE 2509.24613), OWSM-CTC v4 6.74(2506.00338), Qwen3-ASR-0.6B 3.72·CV-ko 8.48(2601.21337), Raon-Speech 9B 1.81. Pruned CTC(2609.33645)는 음절 2.5k 어휘에서 이득이 작고 k2 의존이라 미채택. CHAT(2602.24245)는 k2 없는 transducer 메모리 절감 대안이나 NeMo·110M 규모만 있어 메모만.

## 4. 후보 구조 비교표

심사 3건(측정·재현성 / 실무 구현 / 스트리밍 제품)의 6개 항목(각 10점) 합계.

| | 1안 EBranch-KO-32M | 2안 KoStream-CTC | 3안 ChunkConformer-CTC-21M | 4안 KoStream-EB26 |
|---|---|---|---|---|
| 관점 | 정확도 우선 | 스트리밍 서비스 우선 | 구현 위험 최소화 | 소규모 데이터 효율 |
| 인코더 | E-Branchformer 12L d256, 약 32M(AED 포함) | Conformer 12L d256, 약 22M | Conformer 12L d256, 약 21M | E-Branchformer 12L d256, 약 26M |
| 위치 인코딩 | Transformer-XL RelPos | RelPos | RoPE + SDPA | RelPos |
| 서브샘플링 | 4배(Conv2d) | 4배 인과 Conv2d(look-ahead 0) | 4배 인과 Conv2d | 4배 인과 Conv2d |
| conv 모듈 | BACON 반반(cgMLP·병합 둘 다) | BACON 반반(k=15, LN) | BACON 반반(k=15, LN), 플래그 폴백 | 기본 인과, BACON은 4단계 arm |
| 스트리밍 | 청크 경계 마스크, 동적 청크(U2 50/50), 좌측 128, RC 0, KV 캐시 | 동일 + 끝점에서 캐시 초기화 | 동일, 입력 캐시·재투영(simulated==실제 수치 동일) | 동일, 청크 확률 {0.35,0.25,0.15,0.25}(추정) |
| 손실 | CTC 0.3 + SC-CTC(3·6·9) + AED CE 0.7 | CTC + InterCTC(6층, w 0.3) | SC-CTC(3·6·9, λ 0.5) | CTC + InterCTC/SC-CTC(3단계 비교) |
| 디코딩 | 1차 greedy, 끝점 prefix beam+AED 리스코어링, 오프라인 joint 빔, 5-gram arm | greedy 단일, LM은 7단계 arm | greedy, 끝점 뒤 prefix beam+4-gram 옵션 | greedy 부분 자막, 끝점 beam 8+5-gram |
| 옵티마이저 | Adam, warmup 8k, inverse-sqrt, peak {1e-3,2e-3} | AdamW, warmup 10k, cosine, peak 1e-3 | AdamW, warmup 4k, cosine, peak {1e-3,2e-3} | AdamW wd 1e-2, warmup 5k, cosine |
| 학습 길이 | 80 ep ≈160k step | 80 ep ≈160k step | 100 ep(14 h 상한 규칙) ≈66k step | 30 ep 프록시 + 80 ep 최종 |
| dev 1차 지표 | Zeroth 문장 홀드아웃(200문장) | Zeroth 텍스트 홀드아웃(10 %) | FLEURS val | (FLEURS val + CV dev)/2 |
| 자체 GPU 시간 추정 | 150~200 h(심사 재계산 250~300 h) | 160~400 h | 60~100 h | 150 h 이상 |
| 심사 1 (측정·재현성) | 37 | 44 | **49** | 40 |
| 심사 2 (실무 구현) | 35 | 44 | **47** | 38 |
| 심사 3 (스트리밍 제품) | 40 | 45 | **48** | 41 |
| **합계 (180점 만점)** | 112 | 133 | **144** | 119 |
| 순위(3심사 일치) | 4 | 2 | 1 | 3 |

공통점: 네 안 모두 한글 음절 출력 단위(+공백·<unk>, blank=0), CTC 계열 주손실, 청크 경계 마스크·RC 0·동적 청크, Zipformer·pruned RNN-T·k2·Mamba·8배 서브샘플링·우측 문맥 혼합 학습·epoch 단위 청크 커리큘럼 기각, Zeroth test를 seen-text 앵커로만 사용, fp16 autocast dtype 명시 + CTC 입력 fp32 log_softmax + GradScaler 운영 규칙, 속도 교란 3종 + SpecAugment, CV22 미러 1순위.

## 5. 심사가 가장 높게 본 구조와 근거

**심사 결과 권고안: 3안 ChunkConformer-CTC-21M**(세 심사 모두 1위, 합계 144/180). 2안 KoStream-CTC가 2위(133)로 모든 심사에서 일관되게 다음이었다.

근거(세 심사의 공통 서술):
- 구현 위험이 가장 낮다. 직접 구현해야 할 비표준 부품이 Conformer 블록 + RoPE(SDPA 바로 사용, rel-shift 없음) + 인과 Conv2d 서브샘플링 + BACON 반반 conv(unfold 1회) + SC-CTC(공유 Linear 2개)뿐이고, 외부 C++/CUDA 확장·AED·joint 빔·transducer·KenLM 의존이 없다. 비검증 부품(BACON·RoPE·인과 서브샘플링)마다 설정 플래그 폴백이 있고 최후 폴백으로 fp32 학습까지 정해 두었다.
- "입력 캐시·재투영" 스트리밍은 전체 발화 + 청크 마스크 forward와 수학적으로 동일해 fp32 atol 1e-5 단위 테스트가 가능하다(icefall조차 beam 계열에서 0.06 벌어진 문제, C2). 미래 누출·CTC dtype·20발화 과적합 테스트를 "통과 전 학습 금지" 게이트로 두었다.
- 사전 등록 양식이 가장 완결적이다: D1~D19 일정, 예산 규칙(epoch = floor(14 h/epoch 시간), 하한 60), arm 제거 순서(d→c→b→a), fp16 발산 판정의 수치 정의, Stage 7 봉인 test 1회·규칙 외 재측정 없음, 모든 수치에 체크포인트·디코딩 모드 병기.
- 4주 안에 서비스까지 끝날 가능성이 가장 높다(60~100 GPU-h, 100 epoch ≈ 8~14 h 추정 + 14 h 상한 규칙으로 흡수).
- 정확도는 SC-CTC 기본 + 끝점 뒤 4-gram 옵션으로 2안보다 상한이 높고 1·4안보다 낮다. 기대치(Zeroth 2~5 %, FLEURS 12~30 %, CV 15~40 %)는 보수적이다.

심사가 지적한 3안의 약점: (a) dev_primary가 FLEURS val 226발화 단독이라 0.2~0.3 절대 차이의 통계력 부족, (b) Stage 3 ablation 단일 시드·인코더 교체 임계(상대 8 %)가 시드 분산 추정 없이 설정됨, (c) FLEURS/CV 숫자·라틴 발화를 평가에서도 제외하면 test 세트가 바뀜, (d) 총 업데이트 수 약 66k step이 1·2안(160k)의 40 %, (e) 재투영 방식은 청크마다 좌측 128프레임 K/V를 다시 계산해 CPU 배포 비용 증가, (f) 평가 청크에 160 ms 누락, (g) torchaudio.models.Conformer 수치 대조는 FFN/MHA에만 적용 가능.

가져올 아이디어(세 심사의 graft 목록에서 겹치는 것):
1. 2안: 통제군 2시드로 시드 표준편차 s를 추정하고 채택 임계를 max(0.3 절대, 2s)로; 4중 조건(FLEURS val 악화 ≤0.2 AND 청크당 p90 +20 % 이내 AND 발산 횟수 ≤ 통제군). 3안 Stage 3 단일 시드 비교에 그대로 이식.
2. 4안: dev_mix = (FLEURS val + CV22 dev)/2를 1차 지표로, Zeroth 문장 홀드아웃을 2차 가드로(심사 3은 역할을 바꿔 홀드아웃 1차도 가능하다고 봄). 모든 판정에 발화 단위 부트스트랩 95 % CI.
3. 4안·1안: LM 코퍼스에서 모든 dev/test 문장 정확 일치 제거 후 건수 보고, Zeroth 홀드아웃은 LM α/β 튜닝에 쓰지 않음.
4. 1안: FLEURS/CV test는 "전체 발화(숫자·라틴은 오류로 계산)"와 "한글 전용 부분집합" 두 수치 보고; train+홀드아웃 refit 금지 사전 선언.
5. 2안: 기본 모델 전체 학습을 ablation보다 먼저 끝내 D8 전에 동작하는 스트리밍 모델 확보; 끝점에서 캐시 초기화 vs 유지를 dev에서 1회 비교; 서비스 경로 CER == 봉인 전 스트리밍 CER ±0.1; 10분 연속 입력 드리프트 측정; 평가 격자에 160 ms 추가; 끝점 규칙을 EP50/EP90 + 조기 끝점 삭제 CER ≤0.2로 1회 선택 후 고정; fbank fp16 사전 계산 샤드를 HDD 폴백으로.
6. 4안: 30 epoch 프록시 예산 + 최종만 긴 스케줄; "결합 ≥ 최고 단일 arm" 확인; LR 사다리(×0.5 → warmup ×2 → 2단 선형 warmup); 어텐션 점수 |>25| 비율 모니터; 스트리밍 중 LM 사용 게이트(LocalAgreement-2, UPWR ≤ greedy 2배, CPU 청크당 <160 ms); BACON 채택 조건에 MSD P50 증가 ≤20 ms; 부분 결과 방출 간격 200 ms.
7. 1안: Stage 0에서 max-duration {200,300,400,500} s OOM 스윕 + 15 % 여유 규칙, 메모리가 남아도 배치를 늘리지 않고 step 수 고정(C2); SC-CTC 중간 헤드 fp32 log_probs 메모리(B×T×V×4 B)를 미리 계산; 방출 지연 P50 400 ms 초과 시에만 TrimTail/Peak-First를 CER 손해 ≤0.1 조건으로 채택; A0(CTC+중간 CTC만) 폴백 arm.

## 6. 미해결 질문과 1단계(Stage 0~1) 측정으로의 해소

| 질문 | 왜 중요한가 | 해소 방법 (Stage 0~1에서) |
|---|---|---|
| whisper-ko-ft의 Zeroth test CER 정의(공백 포함/제외, 정규화) | 앵커 4.48/1.96과 정의가 다르면 헤드라인 비교 무효. 2안은 공백 제외, 1·3·4안은 공백 포함 | 사전 등록 커밋 전에 whisper-ko-ft 채점 코드를 그대로 가져와 `docs/experiments.md`에 정의를 고정 |
| 2080 Ti fp16 실제 처리량(epoch 시간, 데이터 대기 비율, SDPA 커널 종류) | 3안 5~8분/epoch vs 1·2안 15~25분/epoch(데이터량 비례로는 일관)가 모두 V100 대비 가정. 모든 arm 수·epoch 예산이 여기에 좌우 | Stage 0 마이크로벤치: max-duration {200,300,400,500} s 피크 메모리·step 시간, 데이터 대기 비율(>20 %면 fbank 사전 계산 폴백), sm_75에서 SDPA가 mem-efficient인지 math인지 기록 |
| 3안 총 66k step이 160k step 대비 CER에 미치는 영향 | SpecAugment 효과는 긴 스케줄 의존 | Stage 0 실측 epoch 시간이 5분 이하면 14 h 예산 안에서 epoch 150~160으로 올리는 규칙을 사전 등록 |
| 네 안이 임계값 근거로 쓴 조사·검증 밖 출처(BACON, Spiralformer 513 ms, 2602.23171, 2604.14493, 2607.05612, 2606.23306, 2412.00877, 2407.04439) | 사전 등록 가설의 임계값(80 ms, 0.3 절대, 3 %)을 결정 | C14 이후 검증 항목으로 원문 확인 후 커밋; 미확인 상태면 임계값을 "참고치"로 표기 |
| fp16·sm_75에서 Conformer pure CTC의 peak LR 2e-3 실패율(C3는 정밀도 미기재) | LR 선택·안정성 주장 모두 미검증 | Stage 1: 인코더 × LR {1e-3, 2e-3} × 3시드 짧은 런; 실패 정의(비유한 step 비율, grad-scale 임계, 초기 손실 하강 기준)를 하나로 통일하고 발산 횟수를 결과표로 보고 |
| dev 선택 지표의 분해능과 시드 분산 s | FLEURS val 226발화로 0.2~0.3 절대 차이를 가릴 수 있는지 불명 | Stage 0에서 각 dev의 총 음절 수(0.1 절대 = 몇 글자)를 보고, Stage 1~2 통제군 2시드로 s 추정 후 임계값 확정 |
| Zeroth 문장 홀드아웃 비율(1안 6.7 %, 2·3안 10 %, 4안 5 %)과 해시 규칙 | 학습량과 dev 분해능을 동시에 바꿈 | 비율·SHA-256 규칙·시드를 하나로 고정해 Stage 0에 커밋 |
| FLEURS/CV 숫자·라틴 문자 처리(제외 vs 규칙 변환) | 어휘 크기·test 세트·외부 수치 비교 가능성에 영향 | 하나의 규칙 + 제외율 보고 양식을 사전 등록; 변환기를 쓰면 규칙표 자체를 커밋; test는 두 수치(전체/한글 전용) 보고 |
| Zeroth LM 코퍼스(1.09억 문장)의 tar 포함 여부·라이선스·test 문장 포함률; 순수 Python n-gram의 메모리·시간; kenlm Windows 휠 | LM arm의 실행 가능성과 오염 통제 | Stage 0에서 tar 내용 확인, dev/test 문장 정확 일치 건수 집계, 문장 상한(2~5M)으로 n-gram 구축 시간 실측 |
| MFA Korean 3.0 Windows(conda) 설치 가능성 | 실패 시 방출 지연은 자체 CTC 정렬(피키 편향)로 상대 비교만 유효 | Stage 0에서 설치 시도; 실패 시 "상대 비교만 유효"를 사전 선언 |
| simulated == 실제 청크 추론 동일성 테스트가 fbank 경계(center=False, 240샘플 잔여)까지 포함하는지 | 캐시 버그를 놓치면 서비스 CER이 봉인 수치와 달라짐 | Stage 0 단위 테스트 항목에 fbank 프레임 동일성 명시; 10분 연속 입력 드리프트(RoPE 절대 카운터·RelPos 클램프 거동) 포함 |
| SC-CTC 중간 softmax(V≈2k × 3층)·재투영 방식의 CPU 청크당 비용 | CPU 배포 여부가 미정 | 배포 대상(GPU 전용 vs CPU 폴백)을 제품 요구로 먼저 정하고, Stage 6에서 InterCTC-only 모델과 나란히 p50/p90 측정 |
| CV22 미러에서 ko만 선별 다운로드 가능 여부·공식 JSON과 무결성 | 실패 시 CV17 폴백(1.6 h) | Stage 0에서 transcript/ko와 오디오 tar 구조 확인, 클립 수·시간을 공식 release JSON과 대조, 받기 전 사용자 통지 |
| GradScaler init_scale(기본 65536 vs icefall 1.0)과 warmup 상호작용 | 초기 skip 폭주 가능성 | Stage 1 로그에 scale 이력·skip 횟수를 반드시 남기고 2안처럼 1 epoch 비교 |
| 끝점 뒤 캐시 처리(초기화 vs 유지), 규칙형 끝점 1.2 s의 UX 허용 여부, 부분→최종 교체 상한 | 제품 기준 부재 | dev에서 1회 비교 후 고정; UPWR/UPSR·끝점 후 최종 변경률 상한을 제품 요구로 정의 |

## 7. 원문 위치 안내

설계안 4건과 심사 3건의 원문(JSON)은 `work/` 아래에 있다(릴레이 산출물; 파일명은 해당 라운드의 `work/relay/` 또는 `work/` 직하 JSON 참조). 검증 결과 C1~C13과 누락 논문 3묶음도 같은 위치의 JSON에 있다. 이 README는 그 내용을 요약한 것이며, 수치의 1차 출처는 각 JSON의 `evidence` 필드와 조사 문서 4편이다. `work/`는 커밋하지 않는 임시 작업 폴더라 이 원문들은 공개 저장소에 없다. 그래서 사전 등록 문서(`docs/experiments.md`)에는 이 README와 조사 문서만 인용한다.

## 심사 결과 권고안

**3안 ChunkConformer-CTC-21M을 기본 골격으로 채택**하고, §5의 가져올 아이디어 1~7(특히 2시드 s 추정 임계, dev_mix 1차 지표, LM 오염 통제, test 두 수치 보고, refit 금지, 기본 모델 우선 학습)을 사전 등록 문서에 병합한다. Stage 0(데이터·마이크로벤치·단위 테스트)과 Stage 1(LR·안정성 스윕)을 먼저 끝내 §6의 미해결 질문을 수치로 닫은 뒤 arm 수·epoch 예산을 확정한다. 2안 KoStream-CTC는 3안이 Stage 0~1에서 구현 문제(RoPE 캐시, 재투영 CPU 비용, SC-CTC 오버헤드)를 보이면 바꿔 탈 수 있는 대안으로 유지한다. 1안(AED 리스코어링)과 4안(E-Branchformer + 외부 5-gram)은 4주 예산 밖으로 판정되었으며, 그 안의 요소(AED 보조 손실, 외부 LM, E-Branchformer 통제군)는 시간이 남을 때의 선택 arm으로만 둔다. TTS 합성·YODAS 의사 라벨은 "처음부터 학습" 전제와의 양립 여부를 사용자가 결정할 때까지 보류한다.

## 결정 (2026-10-11, Claude. 사용자 지시 "조사 끝나면 네가 가장 좋다고 판단한 구조로 진행")

**채택: 3안 ChunkConformer-CTC-21M을 골격으로, 아래 수정을 더한 구조.** 이름은 코드에서 `chunk-conformer-ctc`로 쓴다.

| 항목 | 결정 | 왜 |
|---|---|---|
| 인코더 | Conformer 12층 d=256, 4헤드, FFN 1024, pre-LN, conv 모듈은 LayerNorm·GLU·SiLU, depthwise k=15, 약 21M | 65 h에 20~40M이 적정(01 §3.3), 순수 PyTorch로 가장 적은 비표준 부품, 스트리밍화할 시간축 연산이 블록당 2개뿐. Zipformer의 우위는 960 h·transducer 조건이고(C1) 재구현 위험이 크다. E-Branchformer와의 차이는 시드 분산 수준(C3)이라 3단계 통제군으로만 둔다 |
| conv 모듈 | BACON식 반반(128채널 완전 인과 + 128채널 청크 경계까지 양방향), 설정 플래그로 전 채널 인과로 폴백 | 960 h·110M 근거(Interspeech 2026)라 65 h 이득은 미검증. 폴백이 한 줄이라 위험이 작다 |
| 서브샘플링 | 인과 Conv2d 2단, 4배(25 fps), look-ahead 0 | 한국어 음절+공백 토큰률 p95 6.1자/s(실측)에서 8배(12.5 fps)는 CTC 길이 제약에 걸린다 |
| 위치 | RoPE + `scaled_dot_product_attention`(bool 마스크) + **QK-norm**(2026-10-11 1단계 진단 뒤 추가: q·k 헤드별 LayerNorm, RoPE 앞) | 순수 CTC에서 RelPos와 동급(2501.06051), rel-shift 커스텀 코드가 없어 캐시 추론 등가성 테스트가 쉽다. QK-norm은 LR 7e-4 이상에서 어텐션 로짓 표준편차가 60~111까지 커지던 것을 1~3으로 억제했다(`docs/experiments.md` 규칙 변경 13; LayerNorm의 affine 파라미터가 커지면 경계가 보장되지 않으므로 2단계 보고에서 감시) |
| 정규화 커리큘럼 | SpecAugment는 epoch 3부터(앞 3 epoch는 속도 교란·동적 청크·dropout만) | 처음부터 SpecAugment를 걸면 CTC 정렬이 5 epoch 넘게 시작되지 않았고, 끄면 epoch 2에 시작했다(1단계 진단, 규칙 변경 13) |
| 스트리밍 학습 | U2식 청크 경계 마스크 + 배치 단위 동적 청크: 50 % 전체 문맥, 50 % 청크∈{4,8,16,32}프레임(160/320/640/1,280 ms) 균등, 좌측 문맥∈{64,128,무제한}. 우측 문맥 혼합 학습은 넣지 않음 | C13(배치 단위 샘플링만 작동), C12(RC 혼합 학습은 RC0 디코딩 퇴보). 3안의 {8,16,32}에 4를 더한 것은 02 §4.5의 160 ms 평가 축을 채우기 위해서(심사 지적) |
| 손실 | 최종 CTC + 자기조건 CTC(3·6·9층, λ=0.5), 입력은 fp32 log_softmax. CR-CTC·transducer·AED는 쓰지 않음 | C4(81 h에서 거의 무비용 개선), C9·C11(fp32 입력 필수, k2 불가). InterCTC 단독과의 비교는 3단계 (a) |
| 출력 단위 | 학습 전사에 등장한 한글 음절 전부 + 공백 + `<unk>`, blank=0 | C7(음절이 CER 기준 최선), 실측 어휘 약 1.3k |
| 디코딩 | 스트리밍 greedy(append-only) + 규칙형 끝점. 끝점 뒤 prefix beam + 음절 4-gram은 5단계 선택 arm | 02 §3.3(부분 결과 안정성), WeNet 1차 greedy≈prefix beam |
| 옵티마이저 | AdamW(0.9, 0.98), weight decay 1e-6, 선형 warmup 4k → cosine, peak LR은 1단계에서 {1e-3, 2e-3}×3시드로 선택 | C3(2e-3 순수 CTC 실패 보고, 정밀도 미기재라 fp16에서 실측 필요) |
| 정밀도 | `autocast(dtype=float16)` 명시, 파라미터 fp32, GradScaler 기본값, 발산 판정·LR 하향 규칙 사전 등록 | C9(2080 Ti에서도 `is_bf16_supported()`가 True를 돌려주므로 자동 선택 금지) |
| 데이터 | Zeroth train(문장 단위 10 % 홀드아웃 제외) + FLEURS train(숫자·라틴 발화 제외) + **CV22** validated(dev·test 제외) ≈ 53 h | C8(Zeroth test는 본 문장 99.8 %), 누락 조사(CV22 미러 CC0·ko 파일 분리 확인·받음) |
| 선택 지표 | dev_mix = FLEURS validation(한글 전용) + CV22 dev(못 본 문장·못 본 화자)를 1차, Zeroth 문장 홀드아웃 dev를 2차 가드(절대 +0.3 넘게 나빠지면 채택 금지). 모든 판정에 발화 단위 부트스트랩 95 % 구간 | 심사 공통 지적(FLEURS val 226발화 단독은 분해능 부족), C8(화자 홀드아웃 dev는 문장 암기를 보상) |
| 주지표 | **공백 제외 음절 CER**(whisper-ko-ft와 같은 정규화·채점 코드), 공백 포함 CER 병기 | whisper-ko-ft `text_norm.py`를 열어 확인: 소문자화, 구두점→공백, 한국어는 공백·작은따옴표 제거 후 글자 단위. 앵커 4.48/1.96이 이 정의 |
| 봉인 test | Zeroth 공식 test(본 문장 표시), Zeroth 문장 홀드아웃 test(못 본 문장), FLEURS test, CV22 test — 7단계에서 각 1회. FLEURS·CV는 전체와 한글 전용 두 수치 | 1안 graft(두 수치 보고), C8 |
| 통계 | 1단계 3시드로 시드 표준편차 s를 추정하고 3단계 채택 임계를 max(0.3 절대, 2s)로 | 2안 graft(C3 함의) |
| 보류 | TTS 합성·YODAS 의사 라벨·Whisper 증류: "처음부터 학습" 원칙과 충돌하므로 사용자 결정 전까지 제외 | 누락 조사 묶음 3 |

3안에서 바꾼 것: 동적 청크 집합에 4프레임 추가, dev 1차 지표를 FLEURS val 단독에서 dev_mix로, Zeroth 문장 홀드아웃을 dev와 test로 반으로 나눔(못 본 문장 test 확보), 주지표를 공백 제외 CER로(앵커 정의 확인 결과), CV17→CV22, 시드 표준편차 기반 임계, FLEURS/CV test 두 수치 보고, LM 텍스트에서 dev/test 문장 제거·건수 보고. 2안의 서비스 검증 항목(서비스 경로 CER == 봉인 전 스트리밍 CER ±0.1, 10분 드리프트, 끝점 규칙 1회 선택 후 고정, 160 ms 평가)을 6단계에 넣는다.

기각한 큰 대안과 이유는 §4~5와 설계안 원문(`work/survey_workflow_results.json`)에 있다. 이 결정 뒤의 사전 등록 규칙은 `docs/experiments.md`, 단계 계획은 `docs/plan.md`.

**결정 뒤 측정으로 바뀐 것**(2026-10-15 덧붙임, 각 판정은 `docs/experiments.md`): 손실은 3단계 (a) 판정으로 자기조건 CTC(3·6·9층, λ 0.5)에서 InterCTC 6층(w 0.3, 피드백 없음)으로 바뀌었고, 4단계 판정으로 TrimTail 15 epoch 이어 학습을 더했다. warmup은 4k가 아니라 2,000 step(규칙 변경 1), QK-norm과 SpecAugment 커리큘럼은 규칙 변경 13으로 더했다. 5단계(prefix beam + 음절 4-gram)는 생략했다. 위 표는 결정 당시의 기록으로 그대로 둔다.
