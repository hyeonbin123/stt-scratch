# 04. 소규모 데이터 from-scratch 학습과 한국어 ASR 데이터·레시피·결과 조사

- 조사 기간: 2019-01 ~ 2026-10 문헌·데이터 카드. 조사일 2026-10-10/11.
- 조건: 한국어 STT를 사전학습 가중치 없이 약 65 h 공개 음성으로 학습, RTX 2080 Ti 1장(11 GB, fp16만), 최종 목표는 스트리밍 서비스.
- 표기: 모든 수치는 직접 열어 본 페이지(§8 목록)에서 가져왔다. 열지 못했거나 검색 스니펫에만 있던 값은 (확인 필요)로 표시했다. 본 문서가 계산한 값은 (계산)으로, 프로젝트 자체 재고 조사(`docs/data.md`, 2026-10-11)로 확인한 값은 (실측)으로 표시했다. 검증 후 정정 내역은 §9.

## 1. 요약

1. 65 h from-scratch로 기대할 CER(추정): 같은 도메인·낭독 음성(Zeroth 계열) 테스트는 CER 2~5 %, 도메인 밖 자발화는 CER 15 % 이상. 근거: Zeroth 51.6 h E2E 모델이 CER 1.4~2.5 %를 보이지만 훈련 고유 문장이 3,003개뿐이고(실측) test 457발화 중 456개(99.8 %)의 전사가 train에 그대로 있어(화자만 다름, §5·§7) Zeroth test CER은 "본 문장·새 화자(seen-text)" 조건의 수치이며, ClovaCall 50~67 h from-scratch는 CER 11~22 %(사전학습 후 미세조정 7~8 %), FLEURS 한국어는 0.6B 사전학습 모델도 CER 21.7 %다.
2. 영어 기준점: LibriSpeech 100 h from-scratch Conformer/E-Branchformer(30~38M)는 test-clean/other WER 6.3~6.9 / 17.0~18.1, 960 h에서는 2.1~2.6 / 4.3~6.0. 데이터 10배가 WER을 약 3배 줄인다.
3. 가장 효과가 큰 레시피 선택 세 가지: (a) SpecAugment + 속도 교란(0.9/1.0/1.1) + 충분히 긴 학습(SpecAugment는 LAS test-other 13.3→8.3), (b) 출력 단위는 음절(약 2.3k)로 하고 CTC를 중간층에도 거는 Intermediate/Self-conditioned CTC(81 h WSJ: Conformer-12 CTC 12.4→InterCTC 10.8→+Stochastic depth 9.9; Transformer-18 CTC 14.9→InterCTC 12.7→SC-CTC 11.9), (c) 안정화 세트: warmup(10k~15k step 또는 Eden 500 batch), label smoothing 0.1, dropout 0.1, weight decay 1e-6, 길이 버킷 배치, fp16에서는 ScaledAdam/Balancer 또는 grad-norm 비정상 시 스킵.
4. 65 h 규모의 자기지도 사전학습(SSL)은 비용 대비 효과가 낮다: wav2vec 2.0은 비라벨 108 h에서 dev-other WER 44~50 %(860 h에서 15~16 %), 965 h 한국어로 처음부터 사전학습해도 지도학습 Transformer 대비 CER 7.78→7.16 % 개선에 그친다.
5. 계정 없이 쓸 수 있는 한국어 데이터: Zeroth-Korean 51.6 h + 1.2 h(CC BY 4.0, HF `kresnik/zeroth_korean`/`Bingsu/zeroth-korean`, OpenSLR 40), FLEURS `ko_kr` 약 10 h(CC BY 4.0, `google/fleurs`), Common Voice 17 한국어 미러 `fsicoli/common_voice_17_0`(CC0, 검증분 1.63 h, 전체 5.57 h). 합계 약 63~68 h가 "65 h" 예산과 맞는다. KsponSpeech(969 h)는 AI Hub 승인·내국인 전용, ClovaCall은 학술 신청·비상업이다.
6. 11 GB fp16 실현 가능성: 공개 레시피는 32 GB V100 기준 Zipformer-S(23M) max-duration 1500 s/GPU, 109M Conformer-CTC는 200 s를 권장한다. 11 GB에서는 20~40M 모델에 max-duration 200~400 s가 출발점(추정). fp16 NaN은 (i) CTC/transducer 손실을 fp32로, (ii) attention score 오버플로 방지, (iii) grad-scale 붕괴 시 lr 하향으로 대응한 선례가 있다.

## 2. 소규모 데이터 레시피

| 기법 | 보고된 효과 (데이터, 모델) | 권장 설정 | 출처 |
|---|---|---|---|
| SpecAugment | LibriSpeech 960 h LAS-4-1024(D 스케줄) test clean/other WER 4.4/13.3 → LB 3.4/9.2 → LD 3.4/8.3. SWBD 300 h None 12.1/22.6 → SM 9.5/18.8. KsponSpeech 소형 Transformer CER Dev/clean/other 7.2/8.7/9.7 → 6.5/8.0/9.0(상대 9.7/8.0/7.2 %). ClovaCall 67 h LAS는 15.1→31.1로 악화(DS2는 11.4→10.1) | LB/LD: W 80, F 27, mF 1~2, T 100, mT 1~2. SM: W 40, F 15, mF 2, T 70, p 0.2. ESPnet 100 h 설정: freq [0,27]×2, time ratio [0,0.05]×5, time_warp 5. Conformer: F 27, 시간 마스크 10개, pS 0.05 | arXiv 1904.08779; MDPI app10196936; arXiv 2004.09367; ESPnet 설정; arXiv 2005.08100 |
| 과적합→과소적합 전환 | "SpecAugment turns overfitting into underfitting"; 큰 네트워크·긴 스케줄이 증강이 강할수록 유리 | 소규모 데이터에서도 epoch 수를 아끼지 말 것(ESPnet 100 h: 70 epoch, icefall 100 h: 30~40 epoch) | arXiv 1904.08779; ESPnet/icefall README |
| 속도 교란 | 4개 LVCSR(100~960 h) 평균 상대 WER 4.3 % 개선 | 0.9/1.0/1.1 세 배 복제(ESPnet librispeech_100, icefall, Zipformer 공통) | ISCA ko15_interspeech; ESPnet run.sh; arXiv 2310.11230 |
| 잡음/RIR 증강 | icefall KsponSpeech 레시피 MUSAN 사용, Kaldi Zeroth 레시피 잔향 증강 사용(수치 없음, 확인 필요). ClovaCall 67 h: 잡음 증강 DS2 11.4→10.7, LAS 15.1→18.9(일관된 이득 없음) | 스트리밍 서비스 잡음 환경이면 MUSAN 수준으로 보수적으로 | icefall ksponspeech RESULTS; goodatlas/zeroth; arXiv 2004.09367 |
| Dropout / weight decay | Conformer: dropout 0.1, ℓ2 1e-6, Adam β(0.9,0.98) ε1e-9. E-Branchformer(960 h): enc dropout 0.1, layer drop 0.1, dec dropout 0.2, wd 1e-6. ESPnet 100 h 설정은 layer_drop 0.0 | dropout 0.1, wd 1e-6에서 시작, 100 h급은 layer drop 0 | arXiv 2005.08100; arXiv 2210.00077; ESPnet 설정 |
| Stochastic depth + Intermediate CTC | WSJ 81 h Conformer-12 CTC eval92 WER 12.4 → InterCTC 10.8 → StochDepth 10.8 → 둘 다 9.9. TED-LIUM2 207 h 9.8 → 9.1/10.7/9.9. Transformer-12 WSJ 16.5 → 13.6/16.2/13.7 | CTC 계열이면 InterCTC는 기본 채택, StochDepth는 데이터셋 의존 | arXiv 2102.03216 |
| Self-conditioned CTC | WSJ eval92 CTC 14.9 → InterCTC 12.7 → SC-CTC 11.9; TEDLIUM2 test 12.2 → 10.1 → 9.4 (18층 Transformer, LM 없음) | 비자기회귀 CTC 스트리밍 모델에 적합 | arXiv 2104.02724 |
| 길이 커리큘럼 | DS2 SortaGrad(첫 epoch 길이순): dev WER 11.96→10.83(BN 없음), 9.78→9.52(BN) | icefall DynamicBucketingSampler/ESPnet numel 배치처럼 길이 버킷 배치가 사실상 표준 | arXiv 1512.02595 |
| LR 스케줄 | Conformer: Transformer 스케줄 warmup 10k, peak 0.05/√d. ESPnet 100 h: warmuplr 15k, peak 2e-3. Zipformer Eden: base 0.045, warmup 500 batch; ScaledAdam 2.21/4.79 vs Adam 2.38/5.51 | 단일 GPU 작은 배치면 step 기준 warmup을 데이터 양에 맞춰 축소 | arXiv 2005.08100; ESPnet 설정; arXiv 2310.11230 |
| Label smoothing | SWBD 300 h: None 12.1/22.6→11.2/21.6, SM 9.5/18.8→8.5/16.1(증강과 가산적). 단, 작은 lr에서 학습을 불안정하게 할 수 있어 LibriSpeech는 140k step 이전에만 적용 | 0.1 (ESPnet, KsponSpeech 논문, KoSpeech 모두 0.1) | arXiv 1904.08779; MDPI; arXiv 2009.03092 |
| 출력 단위(한국어) | Zeroth: 음절 CER 1.8/WER 2.6, 자모 6.3/19.9, 음절-서브워드 6k 2.3/2.5, 바이트 2.5/4.4 — CER은 음절이 최저지만 WER은 서브워드 6k(2.5)가 음절(2.6)보다 0.1 낮고 논문 결론은 서브워드 6k 우위(WER은 morfessor 분절 기준, 어절 WER 아님). KsponSpeech: 음절 2,306 ≥ 서브워드 5k~12k(Eval-clean CER 8.0 vs 8.1~8.5, Dev 6.5 vs 6.8~7.2, Eval-other 9.0 vs 9.0~9.7로 10k와 동률; small Transformer+SpecAugment) | 음절(약 2.3k) 또는 음절 기반 소형 BPE | arXiv 1910.11590; MDPI app10196936 |
| 하이브리드 CTC/attention | ESPnet 100 h Conformer ctc_weight 0.3, beam 1 → 20에서 8.4/19.4 → 6.5/17.3 | 스트리밍은 CTC/transducer 쪽, 오프라인 평가는 attention 재점수 | ESPnet librispeech_100 README |

## 3. 데이터 규모별 성능

| 데이터 | 시간 | 모델(파라미터) | from-scratch 결과 | 출처 |
|---|---|---|---|---|
| LibriSpeech train-1h (Libri-light) | 1 h | TDS+CTC grapheme, 4-gram LM | test clean/other WER 78.4/88.0 | arXiv 1912.07875 |
| LibriSpeech train-10h | 10 h | TDS+CTC grapheme, 4-gram LM | 33.5/62.1 (LM 없는 수치 없음) | arXiv 1912.07875 |
| LibriSpeech 100 h | 100 h | ESPnet Conformer(12L d256, bpe5000) | 6.5/17.3 (beam 20, LM 없음), 1×V100-32GB 57,072 s | ESPnet egs2/librispeech_100 |
| LibriSpeech 100 h | 100 h | ESPnet E-Branchformer 38.47M | 6.3/17.0 | 같은 README |
| LibriSpeech 100 h | 100 h | ESPnet Conformer Transducer 30.53M | 6.9/18.1 | 같은 README |
| LibriSpeech 100 h | 100 h | ESPnet E-Branchformer CTC 26.43M(CTC 디코딩) | 9.6/23.1 | 같은 README |
| LibriSpeech 100 h | 100 h | icefall pruned_transducer_stateless2 약 80M, 1 GPU fp16, num-epochs 40 | greedy 6.63/17.39 (epoch 30 avg 10; fp32는 epoch 39 avg 10에서 6.64/17.19, 같은 epoch 29 avg 8 비교는 fp32 6.71/17.77 vs fp16 6.81/17.84) | icefall librispeech RESULTS |
| LibriSpeech 100 h | 100 h | LAS-6-1280 + SpecAugment (32 TPU) | 5.5/16.9 | arXiv 2005.09629 |
| LibriSpeech 100 h | 100 h | Whisper medium 미세조정(참고, 사전학습) | 2.7/5.6 | ESPnet README |
| LibriSpeech 960 h | 960 h | Conformer S 10.3M / M 30.7M / L 118.8M | 2.7/6.3, 2.3/5.0, 2.1/4.3 | arXiv 2005.08100 |
| LibriSpeech 960 h | 960 h | Zipformer S 23.3M / M 65.6M / L 148.4M (transducer) | 2.42/5.73, 2.21/4.79, 2.06/4.63 | arXiv 2310.11230 |
| LibriSpeech 960 h | 960 h | icefall Zipformer CR-CTC S 22.1M / M 64.3M / L 147M | 2.57/5.95, 2.12/4.62, 2.03/4.37 | icefall RESULTS |
| BABEL 10개 언어 | 36~130 h/언어 | wav2vec2 구조 CTC from scratch | 평균 CER 47.1 (언어별 42.7~58.4) | arXiv 2006.13979 |
| 같은 데이터 SSL 비교 | 같은 36~130 h | XLSR-Monolingual(같은 언어로 사전학습 후 미세조정) | 평균 CER 30.5; XLSR-53 22.4 | arXiv 2006.13979 |
| LibriSpeech, 라벨 10 h | 비라벨 108/216/430/860 h | wav2vec2 Base 사전학습 후 미세조정 | dev-other WER 44.3/31.8/23.3/15.9 (Large 49.7/30.3/22.1/14.9) | arXiv 2211.00854 |
| ClovaCall(8 kHz 전화) | 50 h / 67 h | DS2, LAS from scratch | CER DS2 16.7/11.4, LAS 22.1/15.1; AIHub 사전학습+미세조정 8.0/7.0 | arXiv 2004.09367 |
| Zeroth-Korean | 51.6 h | ESPnet Transformer bpe5000 | CER 1.4 / WER 2.0 (test 457발화, seen-text: 본 문장·새 화자) | ESPnet egs2/zeroth_korean |
| Zeroth-Korean | 51.6 h | LAS 하이브리드 39~48M, 음절 | CER 1.8 / WER 2.6 (seen-text; WER은 morfessor 분절 기준) | arXiv 1910.11590 |
| KsponSpeech | 965 h | 대형 Transformer(ESPnet, 음절 2,306) | CER Dev/Eval-clean/Eval-other 6.1/7.6/8.5, sWER 11.9/13.4/15.4 | MDPI app10196936 |
| KsponSpeech | 965 h | Big Transformer(발음 전사) / S-Wav2vec2(965 h 자체 사전학습) | CER clean/other 7.78/8.26 → 7.16/7.64 | arXiv 2110.05172 |

스케일링 법칙(arXiv 2106.09488, 134~23,000 h): 데이터 지수 α_D 0.0195, 손실 5 % 감소에 데이터 14배 필요(손실 기준, WER 아님).

## 4. 한국어 ASR 결과

### 4.1 데이터셋별 결과

| 데이터셋 | 시스템 | 사전학습/scratch | 단위 | CER / WER | 출처 |
|---|---|---|---|---|---|
| Zeroth test(457, seen-text) | Kaldi chain tdnn_opgru1a_sp + 4-gram(fglarge) | scratch(GMM-HMM 계열) | 형태소(data-driven) | WER 9.45 (tdnn1b 10.55) | kaldi egs/zeroth_korean RESULTS |
| Zeroth test(seen-text) | ESPnet Transformer(12L d256) raw / fbank_pitch | scratch | BPE 5000 | CER 1.4/1.3, WER 2.0/2.0 | ESPnet egs2/zeroth_korean |
| Zeroth test(seen-text) | LAS 하이브리드 CTC/attention | scratch | 음절 2,371(빈도 상위 2,350 + `<sp>`/`<unk>`/`<symb>`) | CER 1.8 / WER 2.6 (자모 6.3/19.9, 음절-서브워드 6k 2.3/2.5; WER은 morfessor 분절 기준) | arXiv 1910.11590 |
| Zeroth test(seen-text) | wav2vec2 XLSR 미세조정 | 사전학습 | 문자 | CER 1.78 / WER 4.74 (모델 카드 자가보고) | HF kresnik/wav2vec2-large-xlsr-korean |
| Zeroth test(seen-text) | Whisper medium 미세조정(5k step) | 사전학습 | Whisper BPE | CER 1.48 / WER 3.64 (모델 카드) | HF seastar105/whisper-medium-ko-zeroth |
| KsponSpeech Eval-clean/other | 대형 Transformer + SpecAugment | scratch | 음절 2,306 | CER 7.6/8.5, WER 21.1/25.5, sWER 13.4/15.4 | MDPI app10196936 |
| KsponSpeech | RNN(BLSTM) + SpecAugment | scratch | 음절 | CER 8.6/9.8 | MDPI |
| KsponSpeech | KoSpeech LAS(VGG+MHA) | scratch | 문자 | CER 10.31 (자체 분할, LM 없음) | arXiv 2009.03092 |
| KsponSpeech | SpeechBrain Conformer medium + Transformer LM | scratch | unigram | CER 7.33/7.99 (6×A100 80GB) | HF speechbrain/asr-conformer-transformerlm-ksponspeech |
| KsponSpeech | icefall Zipformer transducer 74.8M, 30 epoch, fp16 | scratch | BPE(크기 미기재) | CER 10.35/11.35 (modified beam) | icefall egs/ksponspeech RESULTS |
| KsponSpeech | icefall 스트리밍 Zipformer 79M, chunk 320 ms / 640 ms | scratch | BPE | CER 10.13/10.88, 9.91/10.72 | 같은 RESULTS |
| KsponSpeech | K-Wav2vec 2.0(영어 w2v2 → Kspon 추가 사전학습, 자모+음절 공동 디코딩) | 사전학습 | 자모/음절 | CER 6.88/7.34 | arXiv 2110.05172 |
| KsponSpeech eval-clean | Whisper medium zero-shot → 미세조정(3 epoch) | 사전학습 | Whisper BPE | CER 11.94 → 7.61 (eval-other 11.96 → 8.36), 공백 무시 | EKSSS pss-15-3-75 |
| KsponSpeech eval-clean | Whisper large-v2 zero-shot → Kspon 미세조정 | 사전학습 | — | CER 14.66 → 10.48, WER 29.97 → 24.05, sWER 20.75 → 16.49 | arXiv 2410.18444 |
| ClovaCall test | LAS / DS2 | scratch(67 h) vs AIHub 사전학습 | 음절 | CER 15.1/11.4 vs 7.0/8.3 | arXiv 2004.09367 |
| FLEURS ko | w2v-BERT-51 0.6B / mSLAM 0.6B 미세조정 | 사전학습 | — | CER 21.7 / 22.4 (CJK 그룹 평균 24.6/25.0) | arXiv 2205.12446 |
| Common Voice ko | — | — | — | 공개 벤치마크 결과 없음(ESPnet commonvoice 레시피에 ko 없음) | ESPnet egs2/commonvoice |

seen-text 표시: Zeroth 공식 test는 457발화 중 456발화(99.8 %, 고유 문장 423 중 422)의 전사가 train에 그대로 있는 "본 문장·새 화자" 조건이다(실측 2026-10-11, `docs/data.md`). 위 Zeroth test 행의 CER/WER은 모두 이 조건의 값이며 텍스트 일반화를 측정하지 않는다.

평가 지표 주의: KsponSpeech 논문 CER은 공백을 문자로 포함하고, EKSSS Whisper 논문은 공백을 제외하고 센다. K-Wav2vec은 발음 전사, KsponSpeech 논문은 철자 전사 기준이다. 시스템 간 CER 직접 비교는 피한다.

### 4.2 한국어 특이 이슈

- 띄어쓰기: 어절 단위 WER은 띄어쓰기 변이로 과대 계산되어 KsponSpeech 논문이 sWER(공백 정규화 WER)을 제안했다. 대형 Transformer WER 21.1 vs sWER 13.4. 서비스 지표는 CER 또는 sWER로 잡아야 한다.
- 숫자·외래어 이중 전사: KsponSpeech는 `(철자)/(발음)` 형식으로 숫자·약어·외래어를 이중 전사한다(예 `(AI/에이아이)`). 숫자는 단위와 띄어 쓰고 발음을 자리수 단위로 띄어 적는다. KoSpeech 전처리는 `phonetic`/`spelling` 모드로 한쪽을 택하고, `%`를 `퍼센트/프로`로 치환한다. Zeroth는 숫자를 한글로 풀어 썼다(예 "이천 십 팔 년", "삼 월").
- 비유창성·비음성 태그: `/`(간투어), `+`(반복·말더듬·자기수정), `*`(발음 불명확), `u/`(미상), `b/ l/ o/ n/`(숨소리·웃음·중첩·잡음). 논문 기준 전처리는 철자 전사 선택 → 문장부호·비음성 기호 제거 → `/`, `+` 기호 제거(단어는 유지). 유창 전사(간투어 제거) 모델과 태그 포함 모델의 sWER은 Eval-clean 16.5 vs 16.7로 비슷하다.
- 출력 단위: 자모는 WER이 크게 나빠지고(19.9), 음절은 Zeroth에서 CER 최저(1.8)이지만 WER은 음절-서브워드 6k(2.5)가 0.1 낮고(논문 결론은 서브워드 6k 우위), KsponSpeech에서는 음절 2,306이 서브워드 확장보다 CER 기준 같거나 낫다(Eval-other는 10k와 동률). 음절 수 2,300~2,400개는 softmax 비용이 작아 11 GB에도 부담이 없다.
- Whisper 계열 특징: 한국어 미세조정 후 영어 WER이 6.74→57.56으로 붕괴하고, ESPnet 디코딩에서는 문장 끝 삽입 오류로 CER 60 % 이상이 나온 사례가 있다(EKSSS). 사전학습 모델을 참조 기준선으로 쓸 때도 디코더 설정에 주의.

## 5. 데이터 목록과 라이선스

| 이름 | ID | 분할·시간 | 샘플레이트·포맷 | 라이선스 | 계정/게이트 | 전사 규약 |
|---|---|---|---|---|---|---|
| Zeroth-Korean | HF `kresnik/zeroth_korean`, `Bingsu/zeroth-korean`; OpenSLR SLR40 | train 22,263발화 51.6 h(실측 51.68 h, 105명, 고유 문장 3,003 실측; 문장당 중앙값 7회 낭독), test 457발화 1.2 h(10명, 고유 문장 423; 그중 422문장 = 456/457발화 99.8 %가 train 전사와 정확히 일치, 화자는 train과 겹치지 않음) | 16 kHz FLAC(HF는 Parquet), 3.3~20.7 s | CC BY 4.0(코드 Apache 2.0) | gated=false, 로그인 불필요, HF config 이름은 `default`(`clean` 아님). OpenSLR tar 10 GB(LM·사전·Morfessor 포함) | 공백 포함 한글 문장, 숫자는 한글로 풀어 씀, 문장부호 없음 |
| Zeroth 확장판 | goodatlas/zeroth(2023-08-23 아카이브) | 95.7 h, 46,347발화, 181명, 고유 문장 27,330 | — | CC BY 4.0 | MoreCoin 앱 기여 후 AWS 임시 자격증명(현재 유효 여부 확인 필요) | — |
| FLEURS ko_kr | HF `google/fleurs` config `ko_kr` | train 2,307 / validation 226 / test 382; 7.93 / 0.77 / 1.33 h(실측; 평균 12.38/12.30/12.58 s) | 16 kHz | CC BY 4.0 | gated=false | `transcription`(정규화) + `raw_transcription`; 문장 단위 낭독(FLoRes 문장). 실측: `transcription`에도 발화의 26.8 %에 아라비아 숫자, 15 %에 라틴 문자가 남아 있음 |
| Common Voice 17 ko (미러) | HF `fsicoli/common_voice_17_0` config `ko` | clip_durations 3,373클립 5.57 h; validated 1,046클립 1.63 h(52명); train 376(0.66 h, 3명)/dev 330(0.47 h)/test 339(0.50 h); other 2,057(3.39 h, 88명, 실측); invalidated 270(0.55 h) — train/dev/test/other 시간은 실측, 나머지는 TSV 집계(계산) | MP3 32 kHz(실측; 16 kHz로 재샘플링 필요) | CC0-1.0 | gated=false, 로딩 스크립트 사용(datasets ≥2.18) | 문장 낭독, 문장부호 포함(실측 85 %), up/down_votes 메타 |
| Common Voice 공식 | HF `mozilla-foundation/common_voice_17_0` | 2025-10부터 비어 있음("exclusively available through Mozilla Data Collective") | — | — | MDC 계정 필요(확인 필요) | — |
| Common Voice Scripted 25.0 ko | Mozilla Data Collective | 7,122클립 10.3 h, 검증 2.51 h, 200명, train/dev/test 678/506/554 (검색 스니펫, 페이지 로드 실패, 확인 필요) | MP3 | CC0-1.0 | 로그인 필요, 재호스팅 금지 조항(확인 필요) | — |
| KsponSpeech | AI Hub dataSetSn=123 | 969 h, 622,545발화, Train 620,000 / Dev 2,545(3.9 h) / Eval-clean 3,000(2.6 h) / Eval-other 3,000(3.8 h) | 16 kHz 16-bit headerless PCM, 전사 EUC-KR | AI Hub 이용정책 | 승인 필요, "내국인만 데이터 신청이 가능" | 이중 전사, 비유창성 태그(§4.2) |
| ClovaCall | github clovaai/ClovaCall | clean 67 h(train 50 h / test 0.88 h), 59,662/1,084발화 | 8 kHz(논문은 16 kHz 업샘플) | 비상업 연구 전용, NAVER 크레딧 | Google Form 신청, 학술기관 한정 | JSON(wav, text, speaker_id), 음절 사전 |
| Pansori-TEDxKR | OpenSLR SLR58 | 약 3 h, 41명 | 16 kHz 16-bit FLAC, 174 MB | CC BY-NC-ND 4.0(파생물 금지) | 불필요 | 자막 경계 분절, 수동 정렬 |
| Deeply Korean read speech(샘플) | OpenSLR SLR97 | 전체의 약 1 % 샘플(281 MB); 전체는 상업 라이선스 | 16 kHz | CC BY-NC-ND 4.0 | 불필요 | JSON 메타 |
| Seoul Corpus | OpenSLR SLR113 | 자발화, 시간 미기재, sound 2.5 GB | FLAC | CC BY-NC 2.0 | 불필요 | Praat TextGrid 음성학 레이블 |
| KSS | HF `Bingsu/KSS_Dataset` | 12,853발화 12+ h, 화자 1명 | 44.1 kHz | CC BY-NC-SA 4.0 | gated=false | TTS용 대본(원문/확장/분해) |
| YODAS ko | HF `espnet/yodas` ko000(수동 자막) 2,774 h, ko100~103(자동 자막) 17,673 h | 유튜브 자막, 품질 보증 없음 | 16 kHz | CC BY 3.0 | gated=false, 로딩 스크립트 | 자막 텍스트(비정규화) |
| Emilia ko | HF `amphion/Emilia-Dataset` | Emilia 0.2k h(CC BY-NC 4.0), Emilia-YODAS 7.3k h(CC BY 4.0) | MP3/WebDataset | 혼합 | 약관 동의·연락처 제출 필요 | ASR 생성 전사(확인 필요) |

주석: NC(비상업) 또는 ND(파생물 금지) 데이터는 서비스 모델 학습에 쓰지 않는 것이 안전하다. 계정 없이 쓸 수 있는 CC BY/CC0 합계는 Zeroth 52.8 h + FLEURS 약 10 h + CV17 검증분 1.6 h ≈ 64 h(계산)이며, CV17 `other` 3.4 h(실측)와 CV25 추가분은 품질 확인 후 선택한다.

## 6. 11 GB fp16 실현 가능성

### 6.1 공개된 메모리·처리량 기준점

| 설정 | 모델 | GPU | 배치 | 시간 | 출처 |
|---|---|---|---|---|---|
| ESPnet librispeech_100 Conformer | 12L d256 ff1024, bpe5000 | 1×V100-32GB, AMP | batch_bins 16M(numel) × accum 4 | 70 epoch 총 57,072 s(약 15.9 h; 속도 교란 3배 포함 약 300 h/epoch → 약 13.6 min/epoch, 계산) | ESPnet README·설정 |
| ESPnet librispeech_100 Transformer | 12L d256 | 1×V100-32GB, AMP | 같은 설정 | 총 42,834 s | ESPnet README |
| icefall librispeech 100 h | pruned_transducer_stateless2 약 80M | 1 GPU(종류 미기재), `--use-fp16 True` | max-duration 300 또는 500 s | num-epochs 40, 보고 WER은 epoch 30 avg 10; fp16 max-duration 300/500 결과 6.63/17.39, 6.89/17.75(500이 더 나쁨). fp32 1 GPU 비교값: epoch 29 avg 8 6.71/17.77, epoch 39 avg 10 6.64/17.19; fp32 2 GPU 6.81/17.66, 4 GPU 6.86/18.29(epoch 30) | icefall RESULTS |
| Zipformer 논문 | S 23.3M / M 65.6M / L 148.4M | 2×/4×/4× V100 32GB, mixed precision | max-duration 1500 / 1000 / 1000 s per GPU | 87 / 69 / 80 min/epoch(960 h × 속도 교란 3배 ≈ 2,880 h/epoch, Table 7·A.4.1 명시) | arXiv 2310.11230 Table 7 |
| icefall Zipformer CR-CTC | S 22.1M / M 64.3M / L 147M | 2×V100 32G / 4×V100 32G / 2×A100 80G, `--use-fp16 1` | max-duration 850 / 700 / 1400 | 50 epoch | icefall RESULTS |
| icefall KsponSpeech Zipformer | 74.8M | 4 GPU, fp16, MUSAN | max-duration 750 | 30 epoch | icefall ksponspeech RESULTS |
| icefall conformer_ctc 튜토리얼 | 109M | V100 | "we recommend you to set it to 200"(max-duration) | 미기재 | icefall docs |
| SpeechBrain KsponSpeech Conformer | medium | 6×A100 80GB | — | 미기재 | HF 모델 카드 |

11 GB 환산(추정): 32 GB 대비 약 1/3 메모리이므로 Zipformer-S급(20~25M)은 max-duration 300~500 s, 60~80M급은 150~250 s에서 시작해 OOM 경계를 찾는다. ESPnet V100 처리량을 그대로 두면 65 h×3(속도 교란) = 195 h/epoch에 약 9 min/epoch(계산)이고, Zipformer-S 기준(2,880 h/epoch에 87 min, 2×V100)으로는 약 6 min/epoch(2 GPU), 1 GPU 환산 약 12 min/epoch(계산)이며, 2080 Ti는 메모리와 fp16 처리량이 낮아 이보다 느리다(실측 필요). 실측 전까지 "70 epoch ≈ 10~20 h" 정도를 예산으로 본다(추정). Fast Conformer(8배 서브샘플링)는 인코더 추론 2.8배 가속, MAC 2.9배 절감으로 스트리밍 서빙에 유리하지만 학습 메모리 수치는 논문에 없다.

### 6.2 fp16 함정과 선례 있는 대응

- CTC/transducer 손실 정밀도: icefall `zipformer/model.py`는 `k2.rnnt_loss_smoothed`, `k2.rnnt_loss_pruned`를 `torch_autocast(enabled=False)` 안에서 `.float()` 입력으로 계산한다. PyTorch CUDA autocast의 fp32 승격 목록에는 `log_softmax`, `softmax`, `layer_norm`, `nll_loss`, `cross_entropy`, `exp`, `pow`, `sum`, `cumsum`이 있고 `ctc_loss`는 없다(CPU 목록에만 있음). 따라서 CTC 입력은 `log_softmax` 결과(fp32)를 그대로 넘기고, 직접 fp16 logits를 만들지 않는다. `zero_infinity=True`와 `(T-2) < len(tokens)`인 발화 제거(icefall 주석: "otherwise inf. in loss appears")를 함께 둔다.
- Attention score 오버플로: WavLM은 대형 모델에서 fp16 상한을 넘는 attention score로 NaN이 났고, query를 1/(c√d)로 나눈 뒤 행 최댓값을 빼고 c(=32)를 곱해 softmax하는 방식으로 해결했다. 우리 규모(≤100M)에서는 발생 빈도가 낮지만, 긴 발화와 큰 lr에서 재현될 수 있으니 `--inf-check`형 훅을 준비한다.
- GradScaler 붕괴: icefall `train.py`는 `init_scale=1.0`으로 시작해 scale이 0.01 미만이면 경고·배드 모델 저장, 1e-5 미만이면 `grad_scale is too small` 오류를 낸다. icefall KsponSpeech `zipformer` 레시피(74.8M, 4 GPU, `--use-fp16 1`, max-duration 750)는 이 오류 때문에 base-lr을 0.045→0.035로 내렸고, 같은 RESULTS의 스트리밍 레시피(79M)는 기본 lr을 유지했다. PyTorch GradScaler는 inf/NaN grad가 있으면 `optimizer.step()`을 건너뛰고, 클리핑 전 `unscale_`를 한 번만 호출해야 한다.
- 비정상 grad 스킵: ESPnet2 trainer는 `torch.isfinite(grad_norm)`이 아니면 "The grad norm is {grad_norm}. Skipping updating the model."로 갱신을 건너뛰고, 모든 step이 비정상이면 학습을 멈춘다. autocast dtype은 `torch.cuda.is_bf16_supported()`가 True면 bf16인데, torch 2.11에서 이 함수는 기본 `including_emulation=True`라 RTX 2080 Ti(cc 7.5)에서도 True를 반환한다. 따라서 ESPnet AMP는 이 GPU에서 에뮬레이션 bf16으로 돌며 fp32보다 느리다(4096² matmul 실측 bf16 5.8 TFLOPS vs fp16 45.4 vs fp32 7.4). 네이티브 지원 여부는 `is_bf16_supported(including_emulation=False)`(=False)로 확인하고 autocast dtype을 fp16으로 명시해야 한다.
- 옵티마이저·정규화: Zipformer는 ScaledAdam + Balancer/Whitener/BiasNorm으로 발산을 막고, Balancer 제거는 발산 위험을 높인다고 보고한다(원문 "especially when employing mixed precision training", A.3.3; Table 6 No Balancer 2.23/4.97 vs 기본 2.21/4.91, 40 epoch — WER 차이가 아니라 발산 위험에 관한 정성적 서술이며 발산 통계는 없다). Adam을 쓰려면 BiasNorm을 모든 모듈에 넣어야 발산을 피했다. Conformer 계열은 LayerNorm·pre-norm, grad clip 5(KsponSpeech 논문), label smoothing은 작은 lr에서 불안정 요인이 될 수 있다.
- fp16 정확도 손실: icefall 100 h 실험의 fp16(max-duration 300) greedy 6.63/17.39(epoch 30 avg 10)와 fp32 6.64/17.19(epoch 39 avg 10)는 체크포인트가 다르다. 같은 epoch 29 avg 8 비교에서는 fp32 6.71/17.77 vs fp16 6.81/17.84로 fp16이 절대값 0.10/0.07 나쁘다. 차이는 작지만 0이 아니며, max-duration을 500으로 늘린 fp16은 6.89/17.75로 더 나빠졌다.

## 7. 우리 조건에서의 권고와 open questions

권고
1. 데이터: Zeroth 51.6 h(train) + FLEURS ko train 7.9 h + CV17 validated 1.6 h를 학습(약 61 h), FLEURS validation·CV dev를 검증, Zeroth test·FLEURS test·CV test를 세 가지 테스트로 분리해 도메인 차이를 그대로 보고한다. NC/ND 데이터(Pansori, KSS, Seoul Corpus)는 제외한다.
2. 텍스트 정규화: 숫자는 한글 표기(Zeroth 방식)로 통일하고, FLEURS는 `transcription`(정규화) 컬럼을 쓴다. 단, 실측 결과 `transcription`에도 26.8 % 발화에 아라비아 숫자가 남아 있어 숫자 발화 처리(제외 또는 변환) 규칙은 `docs/data.md`·`docs/experiments.md`에서 정한다. 평가는 공백 포함/제외 CER을 둘 다 기록하고 sWER을 보조로 둔다.
3. 모델: 20~40M Zipformer 또는 Conformer, 음절 약 2.3k 출력, CTC(+InterCTC) 또는 pruned transducer. 스트리밍은 chunk 320~640 ms(icefall KsponSpeech 스트리밍 결과가 오프라인 대비 큰 손실 없이 CER 10.1~10.9).
4. 학습: 속도 교란 3배 + SpecAugment(F 27 ×2, 시간 마스크 비율 0.05 ×5~10), dropout 0.1, wd 1e-6, label smoothing 0.1(attention 디코더 사용 시), warmup은 데이터 양에 맞춰 축소(ESPnet 100 h 15k step은 300 h/epoch 기준), 70 epoch 이상, 체크포인트 평균(ESPnet 10개, icefall avg 9~26).
5. fp16: autocast dtype을 fp16으로 명시(`is_bf16_supported()` 기본값이 에뮬레이션 포함이라 2080 Ti에서도 True), 손실은 fp32, GradScaler init_scale 낮게, grad-scale 경고 시 lr 하향, 비정상 grad 스킵, 긴 발화(>20 s) 제외, max-duration은 300 s부터 탐색.
6. SSL 자체 사전학습은 보류한다. 65 h 비라벨 데이터는 wav2vec2에서 WER 44~50 % 수준의 사전학습 효과만 보였고, 같은 라벨 데이터를 지도학습에 쓰는 편이 낫다. YODAS ko 수동 자막 2,774 h(CC BY 3.0)를 쓸 경우에만 SSL 또는 준지도(pseudo-label)가 의미를 가진다(자막 품질 확인 필요).

Open questions (확인 필요)
- (확인 완료 2026-10-11) Zeroth test와 train의 문장 겹침: datasets-server /rows 전수 비교와 `scripts/inventory_data.py` 실측으로 457발화 중 456발화(99.8 %, 고유 문장 423 중 422)가 train 3,003 고유 문장과 정확히 일치하고 화자는 겹치지 않는다. Zeroth CER 1.4~2.5 %는 텍스트 일반화를 측정하지 않는 낙관치이므로 문장 단위 홀드아웃을 따로 둔다(`docs/data.md`).
- FLEURS ko `transcription` 정규화 규칙 원문 확인(실측으로는 숫자 26.8 %·라틴 15 %가 남아 있음까지만 확인), CV25 Korean 데이터시트 수치. CV17 ko MP3 샘플레이트는 32 kHz로 확인 완료.
- goodatlas 95.7 h 확장판의 현재 다운로드 가능 여부(아카이브된 저장소, AWS 임시 자격증명 방식).
- 2080 Ti에서 Zipformer-S fp16 실측 max-duration과 epoch 시간. ScaledAdam이 Turing fp16에서도 발산 없이 도는지.
- Korean CER 보고 관행(공백 포함 여부)이 논문마다 달라 외부 수치와 비교할 때 기준을 명시해야 한다.
- 스트리밍 Zipformer가 65 h에서 오프라인 대비 어느 정도 손실인지(KsponSpeech 965 h 결과만 존재).

## 8. 읽은 자료 목록

논문
- SpecAugment, arXiv 1904.08779 (ar5iv HTML)
- Conformer, arXiv 2005.08100 (ar5iv)
- Zipformer, arXiv 2310.11230v4 (arXiv HTML)
- E-Branchformer, arXiv 2210.00077 (ar5iv)
- Intermediate CTC, arXiv 2102.03216 (ar5iv); Self-conditioned CTC, arXiv 2104.02724 (ar5iv)
- Deep Speech 2 SortaGrad, arXiv 1512.02595 (ar5iv)
- Speed perturbation, Ko et al. Interspeech 2015 (isca-archive.org/interspeech_2015/ko15_interspeech.html)
- Noisy Student Training for ASR, arXiv 2005.09629 (ar5iv); Iterative Pseudo-Labeling, arXiv 2005.09267 (ar5iv)
- Libri-light, arXiv 1912.07875 (ar5iv)
- XLSR, arXiv 2006.13979 (ar5iv); More Speaking or More Speakers, arXiv 2211.00854 (ar5iv); SEW, arXiv 2109.06870 (ar5iv)
- Scaling Laws for Acoustic Models, arXiv 2106.09488 (ar5iv)
- WavLM(fp16 안정화 절), arXiv 2110.13900 (ar5iv)
- Fast Conformer, arXiv 2305.05084 (ar5iv)
- FLEURS, arXiv 2205.12446 (ar5iv)
- KsponSpeech, Bang et al., Applied Sciences 10(19):6936, 2020 (mdpi.com, 표는 브라우저로 추출)
- KoSpeech, arXiv 2009.03092 (PDF 텍스트 추출)
- K-Wav2vec 2.0, arXiv 2110.05172v1 (arXiv HTML)
- ClovaCall, arXiv 2004.09367 (ar5iv) 및 github.com/clovaai/ClovaCall
- Lexicon-free units for Korean, arXiv 1910.11590 (ar5iv)
- Korean Whisper 미세조정, 음성·음성학 15(3), eksss.org pss-15-3-75; Korean meteorological ASR, arXiv 2410.18444 (arXiv HTML)
- Data quality in multilingual speech datasets, arXiv 2506.17525v1 (CV17 ko 통계)

레시피·코드·문서
- espnet/egs2/librispeech_100/asr1 README, run.sh, conf/tuning/train_asr_conformer_lr2e-3_warmup15k_amp_nondeterministic.yaml, train_asr_e_branchformer_size256_..._edrop0.0_ddrop0.0.yaml
- espnet/egs2/zeroth_korean/asr1 README, run.sh, conf/tuning/train_asr_transformer5.yaml; espnet/egs2/commonvoice/asr1 README; espnet2/train/trainer.py
- k2-fsa/icefall egs/librispeech/ASR/RESULTS.md, egs/ksponspeech/ASR/RESULTS.md, egs/librispeech/ASR/zipformer/train.py, model.py; icefall docs conformer_ctc.html
- kaldi egs/zeroth_korean/s5/RESULTS; goodatlas/zeroth README
- sooftware/kospeech README, dataset/kspon/preprocess/preprocess.py
- PyTorch docs amp.html(2.14), notes/amp_examples.html

데이터 카드·API
- HF datasets: kresnik/zeroth_korean, Bingsu/zeroth-korean, google/fleurs(+datasets-server size/statistics ko_kr), fsicoli/common_voice_17_0(+transcript/ko TSV 직접 집계), mozilla-foundation/common_voice_17_0, espnet/yodas, amphion/Emilia-Dataset, Bingsu/KSS_Dataset; HF API `gated` 필드 확인
- HF models: speechbrain/asr-conformer-transformerlm-ksponspeech, kresnik/wav2vec2-large-xlsr-korean, seastar105/whisper-medium-ko-zeroth, pyf98 ESPnet 모델(링크만)
- OpenSLR resources.php, SLR40, SLR58, SLR97, SLR113
- AI Hub dataSetSn=123(한국어 음성)
- Mozilla Data Collective: 데이터셋 페이지 로드 실패(검색 스니펫만, 확인 필요)

## 9. 검증 후 정정 (2026-10-11)

검증 C1~C13(`docs/survey/README.md` §2)과 자체 재고 조사(`docs/data.md`, `scripts/inventory_data.py`)를 반영해 아래를 고쳤다. 확정된 수치 자체는 바꾸지 않았다.

- §1-1, §5 Zeroth 행, §7 열린 질문: 고유 문장 3,000 → 3,003(실측), test 457발화 중 456(99.8 %, 고유 문장 423 중 422)의 전사가 train에 있음(화자 비중첩), HF config 이름 `default`; 열린 질문은 확인 완료로 (C8, 실측)
- §3·§4.1 Zeroth test 행 전부에 seen-text(본 문장·새 화자) 라벨과 설명 추가; 음절 2,371 = 빈도 상위 2,350 + 특수 토큰, WER 2.6은 morfessor 분절 기준 (C8, C7)
- §1-3 InterCTC/SC-CTC 요약을 설정별로 분리: Conformer-12 CTC 12.4→InterCTC 10.8→+Stochastic depth 9.9; Transformer-18 CTC 14.9→InterCTC 12.7→SC-CTC 11.9 (C4)
- §2 출력 단위 행, §4.2: KsponSpeech 음절 ">" → "≥"(CER 기준, Eval-other 10k와 동률, small Transformer 조건), Zeroth WER은 서브워드 6k 우위 명시 (C7)
- §3 icefall 100 h 행, §6.1 icefall 행, §6.2 fp16 정확도: fp16 6.63/17.39는 epoch 30 avg 10, fp32 6.64/17.19는 epoch 39 avg 10으로 체크포인트가 다름을 병기; 같은 epoch 29 avg 8 비교 fp32 6.71/17.77 vs fp16 6.81/17.84; max-duration 500 fp16 6.89/17.75 악화; fp32 다중 GPU 비교값 추가 (C2)
- §6.1 Zipformer Table 7 행: epoch 시간은 960 h × 속도 교란 3배(A.4.1 명시); §6.1 환산 문단에 Zipformer-S 기준 계산 추가 (C1)
- §6.2 Balancer: 원문 표현과 Table 6 수치(No Balancer 2.23/4.97 vs 2.21/4.91) 병기, 발산 위험에 관한 정성적 서술임을 명시 (C1)
- §6.2 ESPnet autocast: `is_bf16_supported()`가 torch 2.11 기본값(에뮬레이션 포함)에서 2080 Ti도 True → 에뮬레이션 bf16(실측 5.8 TFLOPS)으로 동작, `including_emulation=False`로 확인하고 dtype 명시; §7 권고 5에 반영 (C9)
- §6.2 GradScaler: KsponSpeech base-lr 0.045→0.035는 `zipformer` 레시피(74.8M, 4 GPU, fp16, max-duration 750) 조건, 스트리밍 레시피는 기본 lr 유지 (C9)
- §5 FLEURS 행: 시간 7.93/0.77/1.33 h(실측), `transcription`에 숫자 26.8 %·라틴 15 % 잔존; §7 권고 2·열린 질문에 반영 (실측)
- §5 CV17 행, §5 주석: MP3 48 kHz 추정 → 32 kHz(실측), other 1,758(2.82 h) → 2,057(3.39 h, 88명), 문장부호 85 %; 열린 질문의 샘플레이트 항목은 확인 완료 (실측)
- 미반영: README §2가 "04 L124"에 지목한 "같은 틀로 재현(4 GPU, icefall RESULTS 기준)" 문구는 이 문서에 없다(01-encoders.md에만 해당) (C1)
