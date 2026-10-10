# 03. 학습 목적 함수 · 디코딩 · 출력 단위 조사 (2020-01 ~ 2026-10)

작성일: 2026-10-10. 검증 후 정정: 2026-10-11 (8절). 대상 조건: 한국어 STT를 사전학습 가중치 없이 처음부터 학습, 데이터 약 65 h (Zeroth-Korean 51.6 h + FLEURS ko + Common Voice ko), GPU 1장 RTX 2080 Ti 11 GB (fp16만 지원, bf16 불가), 최종 목표는 스트리밍 서비스.
모든 수치는 본문에 적은 arXiv id 또는 URL의 페이지에서 직접 읽은 값이다. 직접 확인하지 못한 항목은 (확인 필요)로 표시했다.

## 1. 요약

1. 1순위 목적 함수는 **CTC + 중간층 보조 CTC(InterCTC 또는 Self-conditioned CTC)** 이고, 메모리가 허락하면 **CR-CTC**를 얹는다. CTC는 조인트 네트워크가 없어 11 GB에서 가장 안전하고, 손실을 fp32로 계산해도 비용이 거의 없다. InterCTC는 WSJ 81 h 소량 데이터에서 Conformer CTC WER을 12.4→10.8 (eval92)로 낮췼고 (2102.03216), CR-CTC는 LibriSpeech에서 CTC를 transducer·CTC/AED 수준까지 끌어올렸다 (2410.05101).
2. 2순위는 **pruned RNN-T(stateless decoder) + CTC 보조 손실** (icefall 방식). 스트리밍 정확도는 CTC보다 일관되게 좋지만 (2512.11543, 2011.04785), k2 의존성과 디코딩 복잡도가 늘어난다. pruned 손실은 V100 손실 계산 벤치마크(vocab 500, batch 30)에서 torchaudio RNN-T 대비 피크 메모리 18.48 GB→3.73 GB로 줄였다 (2206.13236; 전체 학습 메모리가 아니라 손실 계산 피크). 단 k2 CUDA는 Windows pip 휠이 없고 순수 PyTorch pruned 구현도 없어 WSL2 없는 Windows에서는 사실상 막힌 경로다 (6절 2순위 약점).
3. 3순위는 **하이브리드 CTC/attention(U2식 2-pass)**. 오프라인 정확도는 좋지만 attention decoder 파라미터와 2-pass 지연이 추가되며, 스트리밍 1-pass는 결국 CTC다.
4. 출력 단위는 **한글 음절(약 2,300~2,500개, 공백 토큰 포함)**. KsponSpeech 논문은 음절 2306이 unigram 서브워드 5k~12k보다 CER·sWER 기준 모든 평가셋에서 같거나 좋았고(Eval-other CER은 10k와 9.0 동률, Eval-other WER은 5k·10k가 26.5로 2306의 26.6보다 0.1 낮음; small Transformer+SpecAugment, LM 없음 조건) (MDPI app10196936), Zeroth 51.6 h에서는 음절 CER 1.8 / 자모 6.3 / 음절 서브워드 6k 2.3 / 바이트 2.5였다 (1910.11590). 단, WER은 음절-서브워드 6k 2.5 < 음절 2.6이며 논문 자체 결론은 "음절 기반 서브워드(6k)가 최고"다. Zeroth WER은 morfessor 형태소 분절 텍스트 기준이라 어절 WER이 아니다. 자모 단위는 가장 나쁘다. 음절 OOV 보완용으로 자모 보조 헤드(K-Wav2vec 2.0식)는 선택지.
5. 디코딩은 CTC prefix beam search + n-gram LM(TLG/HLG 또는 GPU n-gram). 외부 LM은 CTC에서 상대 6~22% 개선이 보고되며 인코더 문맥이 짧을수록(스트리밍) 이득이 커진다 (2505.22857, 2607.05612).
6. fp16 주의: CTC/RNN-T 손실 입력은 반드시 fp32로 올린다. PyTorch autocast는 log_softmax를 fp32로 올리지만 ctc_loss 자체는 CUDA fp32 승격 목록에 없고, cuDNN CTC 경로는 float32를 요구한다 (PyTorch 2.14 docs). ESPnet(ctc.py)과 NeMo는 손실 직전에 `.float()` 캐스트를 두고, icefall은 k2 transducer 손실에만 `.float()`를 두며 CTC 경로(forward_ctc)는 nn.LogSoftmax 출력이 autocast로 fp32가 되는 것에 의존한다.
7. 평가는 음절 단위 CER을 주지표로, 공백 정규화 sWER(KsponSpeech 정의)을 보조로 쓴다. Whisper 팀도 한국어는 띄어쓰기 불일치 때문에 CER로 채점한다 (openai/whisper discussions #1762).

## 2. 목적 함수 비교

| 목적 함수 | 대표 논문 (arXiv) | 핵심 아이디어 | 보고된 효과 | 메모리·연산 | 스트리밍 적합성 | 소량 데이터 근거 |
|---|---|---|---|---|---|---|
| CTC | Graves 2006 (기초); 비교는 2410.05101, 2011.04785 | 프레임별 조건부 독립, blank로 정렬 주변화 | Zipformer-M 64.3M LibriSpeech 2.52/6.02 (CTC) vs pruned transducer 65.6M 2.21/4.79 (2410.05101 Table 1). 스트리밍 7개 언어 벤치마크에서 LF-MMI와 비슷한 WER, RTF 약 30% 개선 (2011.04785) | 출력 B×T×V만 필요. 가장 가볍다 | 1-pass 그리디/prefix beam 가능. 단 emission 지연은 별도 제어 필요 | WSJ 81 h에서 InterCTC로 큰 개선 (아래) |
| InterCTC | 2102.03216 (Lee & Watanabe 2021) | 인코더 중간층(⌊L/2⌋)에 보조 CTC, 가중치 w=0.3 | Conformer 12L WSJ dev93/eval92 15.2/12.4→13.4/10.8; TED-LIUM2 10.5/9.8→9.7/9.1; AISHELL-1 5.4/6.0→5.1/5.6 (CER). 6층 Transformer WSJ 21.1→18.3 | 추가 선형층 1개, 학습·추론 오버헤드 거의 0 | CTC와 동일 | WSJ 81 h, TED 207 h 모두 개선 |
| Self-conditioned CTC | 2104.02724 (Nozaki & Komatsu 2021; 원제 "Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions", "Self-conditioned CTC"는 ESPnet의 후속 명칭) | 중간 CTC 예측을 다음 층 입력에 더해 조건부 독립 완화. L=18, K=5(3층마다), λ=0.5 | WSJ CTC 18.7/14.9 → InterCTC 15.9/12.7 → SC-CTC 14.5/11.9; TED 12.8/12.2→9.9/9.4; AISHELL-1 5.7/6.2→4.9/5.3. RTF 0.037 (AR beam 12.212) | InterCTC와 유사, 아주 작음 | CTC와 동일 (중간 예측도 프레임 동기) | WSJ 81 h 20%+ 상대 개선 |
| CR-CTC | 2410.05101 (ICLR 2025) | 같은 입력의 두 증강 뷰 CTC 분포 간 KL 일관성(α=0.2, time mask 2.5×) | Zipformer-S 22.1M: 2.85/6.89→2.52/5.85; M: 2.52/6.02→2.10/4.61; L: 2.50/5.72→2.02/4.35. AISHELL-1 M: 4.47/4.80→3.72/4.02. transducer에 얹으면 L 1.88/3.95 | **두 증강 뷰를 배치 축으로 이어 2× 배치로 1회 forward(연산 2배)** → 논문은 배치 절반·에폭 절반으로 학습 (Sec 4.1, Table 8: M CTC 100 ep → CR-CTC 50 ep, 80G A100 2장; 공개 icefall 레시피는 M 4×V100 32G, max-duration 700, fp16, 50 ep) | CTC와 동일 (논문에 스트리밍·causal 실험 없음, 추론) | 100 h(LibriSpeech train-clean-100) Zipformer-S CR-CTC가 dev-other 6.02%로 쓰였음 (2606.23306) |
| Bayes Risk CTC | 2210.07499 (ICLR 2023) | 경로군별 위험함수로 정렬 제어(조기 emission·다운샘플) | 오프라인 RTF 최대 47%↓(Aishell-2 transducer+CTC 0.42→0.22). 온라인 Aishell-2: vanilla 431 ms/9.97 CER vs BRCTC 302 ms/11.63, 447 ms/9.65, 532 ms/9.16 (Table 11) | k2 FST 기반, 저자 스스로 "학습 비용 증가" 언급 | 온라인 지연 단축이 목적. ESPnet `ctc_type=brctc`로 사용 가능(k2 필요) | 없음(Aishell-1 178 h 이상) |
| RNN-T (full) | 2005.14327 (Li 2020), 2011.04785 | 예측망+조인트, 레이블 동기 출력 | 65k h: 스트리밍 RNN-T 9.27 vs hybrid 9.34 vs chunk Transformer-AED 9.16 (2005.14327). 7개 언어 스트리밍에서 LF-MMI 대비 2~11% WER↓ (2011.04785) | 조인트 로짓 B×T×U×V. V100 32 GB, vocab 500, batch 30에서 torchaudio 18.48 GB/544 ms (2206.13236 Table 1; 손실 계산 벤치마크 피크이며 전체 학습 메모리 아님, 정밀도 미기재) | 스트리밍 표준. 인코더 초기화(CE init) 없으면 12.16→9.80 차이 (2005.14327) | ESPnet Voxforge 이탈리아어 ~20 h: RNN-T 44.4/42.4, +CTC aux 43.9/42.1, +CTC+LM aux 42.3/41.0 (2201.05420) |
| Pruned RNN-T | 2206.13236 (k2/icefall) | 선형 trivial joiner로 (t,u) 범위 S만 남겨 full joiner 평가. S는 논문이 예시로 든 4~5(Table 3 실험값 미기재; icefall 기본 prune_range 5) | LibriSpeech 2.56/6.27 vs optimized_transducer 2.61/6.46; 에폭당 1.17 h vs 2.33 h | 3.73 GB/64 ms (동일 조건의 손실 계산 벤치마크 피크). 동적 배치 10k 프레임: 2.59 GB/38 ms. k2 필요(Windows CUDA 휠 없음, 6절) | 그대로 스트리밍. 한국어 스트리밍 모델 실례 있음 (kangkyu HF) | 한국어 969 h만 확인, 65 h 근거 없음 |
| Sampled-softmax RNN-T / BAT | 2203.16868; 2305.11571 | V 샘플링 / CIF 경계로 U 제한 | 로짓 메모리 V=2000→300 샘플 시 예제당 850 MB→530 MB (2203.16868). BAT: warp-rnnt 16.9 GB/230 ms → pruned 7.4 GB/94 ms → BAT 6.4 GB/85 ms (AISHELL-1, V100) | pruned와 유사 | BAT는 스트리밍 PR90 230→140 ms, CER 6.96→7.35 | 없음 |
| TDT | 2304.06795 (ICML 2023, NeMo) | 토큰+지속시간 공동 예측, 프레임 건너뛰기 | LibriSpeech test-other RNNT 5.11 → TDT 0-4 5.06 (1.91×), 0-8 5.16 (2.12×). σ=0.05, ω=0.1 | 조인트 출력 V+D, RNN-T와 동급(약 120M 모델 실험) | 스트리밍 미평가. 긴 duration은 emission 지연 유발(부록) | 없음 |
| AED / 하이브리드 CTC/AED | 2005.14327; 2410.05101; 2012.05481 (U2) | 공유 인코더 + attention decoder, CTC 가중 합 | CTC/AED Zipformer-M 90.0M 2.22/4.97 (2410.05101). U2 AISHELL-1: attention 4.92 (RTF 0.197) / CTC prefix beam 4.93 / attention rescoring 4.64 (RTF 0.082) (Table 1의 비스트리밍 full-context·kernel 15 모델; 동적 청크 통합 모델은 full 4.90으로 다른 모델) | decoder 파라미터 추가(S 22.1M→46.3M). 조인트 텐서 없음 | AED 단독 스트리밍 어려움. U2는 1-pass CTC + 2-pass rescoring(Δ 50~100 ms) | KsponSpeech·Zeroth ESPnet 레시피가 모두 이 방식(4절) |
| Paraformer (NAR) | 2206.08317 | CIF predictor + GLM sampler, 단일 스텝 병렬 디코딩 | AISHELL-1 CER 4.6/5.2 vs AR 4.7/5.2, RTF 0.0168 vs 0.21 (12×↑). 20k h 41M: 14.07/7.86 vs AR 13.76/7.75 | AR decoder 유지 + predictor | 논문에 스트리밍 평가 없음. FunASR 논문(2305.11013)도 스트리밍 Paraformer 미기술 → 스트리밍 변형은 (확인 필요) | 없음 |
| HMM vs CTC from scratch | 2210.09951 | full-sum 학습 비교 | 4-gram LM 사용 LibriSpeech CTC(SS) 3.6/8.5. CTC는 blank 자유도로 수렴이 쉽지만 peaky 정렬 | — | — | SWB 300 h 포함 |
| 통합 모델 (2025) | 2512.11543 All-in-One | CTC/AED/HAT를 한 네트워크로 | TLv2 스트리밍 WER: HAT 7.8 / AED 7.6 / CTC 8.9; LibriSpeech 스트리밍 HAT 2.8/6.7 vs CTC 3.2/7.8 (AiO Mamba) | 단일 모델 117.9M | 모든 표에서 CTC가 가장 나쁘고 HAT·AED는 근접 | 없음 |

비고.
- 2025~2026 CTC-vs-transducer 비교: ASR Leaderboard (2510.06961)는 Parakeet CTC 1.1B 평균 WER 7.40/RTFx 2730, TDT 0.6B v2 6.05/3390으로 "CTC·TDT는 처리량, transformer decoder는 정확도"로 요약. 시스템 결합 연구 (2508.09880, LibriSpeech, 4-gram LM, phoneme 단위)는 CTC 6.6 / mRNN-T 6.3 / AED(10K BPE) 5.4 test-other. 2026년 looped-transformer (2606.04678)는 CTC 헤드를 주기적 루프 반복마다 적용하는 변형(검색 요약만 확인, 수치 (확인 필요)).
- Interspeech 2025에서 새 CTC 손실 자체는 찾지 못했다. 관련은 Delayed-KD (비스트리밍→스트리밍 CTC 사후확률 증류, ISCA 아카이브 PDF, 수치 미확인) 정도.
- 정규화: 설문 논문 2510.12827은 label smoothing 전형값 α∈[0.05, 0.2]. KsponSpeech 베이스라인(Transformer 하이브리드)도 label smoothing·warmup·gradient clipping·accumulation을 썼다 (MDPI app10196936). BPE-dropout(2103.07186, Turkish 73 h / Georgian 50 h, α=0.1)은 BPE 500에서 46.1→44.4 / 49.5→46.3, 그리고 어휘 크기에 대한 민감도를 줄였다. 음절 단위에서는 적용 불가하므로 서브워드 선택 시에만 의미.

## 3. 디코딩과 LM 결합

| 방법 | 적용 대상 | 보고된 효과 | 출처 |
|---|---|---|---|
| CTC greedy vs prefix beam search | CTC | icefall CR-CTC Zipformer-S 2.57/5.95 → 2.52/5.85, M 2.12/4.62 → 2.10/4.61 (LM 없음, 차이 미세) | icefall librispeech RESULTS.md |
| CTC + WFST(TLG/HLG) n-gram | CTC | WeNet 문서: aishell/aishell2/librispeech에서 "3%~10%" 일관 이득(상세 수치는 example 디렉터리). icefall Zipformer CTC 86M: ctc-decoding 2.50/5.86, whole-lattice-rescoring 2.44/5.38, attention-rescoring 2.35/5.16 (1best 행 2.01/4.61은 표기 순서가 이상해 (확인 필요)) | wenet docs/lm.md; icefall RESULTS.md |
| GPU n-gram(NGPU-LM) 그리디 결합 | CTC/RNN-T/TDT | LibriSpeech 960 h 모델, 도메인 외: CTC WSJ 7.71→6.47, SPGI 16.09→14.41; RNN-T WSJ 7.25→6.36. 속도 오버헤드 7% 미만. 그리디+LM이 beam과의 격차를 RNN-T는 87% 이상, CTC는 약 35%만 회복 | 2505.22857 (Interspeech 2025) |
| 외부 LM이 CTC에 주는 이득 vs 문맥 길이 | CTC | LibriSpeech CTC + Transformer LM: 5.00→3.88 (22.2% 상대). 인코더 중심 창 0.3 s면 55.8%, 4.8 s 31.8%, 9.6~15.6 s 약 24% → 스트리밍(짧은 문맥)일수록 LM 이득 큼 | 2607.05612 (2026) |
| 소량 데이터 CTC + LM | CTC (100 h) | CR-CTC Zipformer-S(train-clean-100) dev-other 6.02%: 3-gram shallow fusion은 유의미 개선 없음(6.018), RoBERTa PLL+MBR(G=128)로 5.53 (8.2% 상대) | 2606.23306 (2026) |
| Transducer shallow fusion / LODR | RNN-T | icefall Zipformer 65.55M: greedy 2.22/4.87, mbs 2.21/4.79, shallow fusion 2.01/4.37, LODR 1.94/4.17 (에폭·avg 설정 다름) | icefall RESULTS.md; icefall LM 문서 |
| 고속 transducer beam(ALSD++/AES++) + n-gram | RNN-T/TDT | 저자원(960 h) RNN-T: SPGI 14.65 → ALSD++ +LM 11.43, WSJ 급 이득 13.7~22%. beam/greedy 속도 격차 10~20%. blank에 LM 페널티를 주어 삭제 증가 방지 | 2506.00185 (Interspeech 2025) |
| 2-pass attention rescoring | CTC/AED (U2) | AISHELL-1 dynamic chunk 16: CTC prefix beam 6.08 → rescoring 5.33; chunk 4: 6.64 → 5.71; 지연 320 ms + Δ(50~100 ms) | 2012.05481 |
| 자모 보조 헤드 joint decoding | CTC (한국어) | KsponSpeech phonetic eval-clean CER 6.904→6.881, eval-other 7.356→7.339 (미세, OOV 음절 복구 목적) | 2110.05172 |

우리 조건 해석: 외부 텍스트(Zeroth LM 코퍼스 1.09억 문장, goodatlas/zeroth README)로 음절 또는 어절 n-gram을 만들어 CTC prefix beam/TLG에 결합하는 것이 가장 비용 대비 효과가 크다. 소량 데이터에서 3-gram 이득이 미미했다는 2606.23306 결과는 영어 BPE 기준이므로 한국어 음절 단위에서는 재확인이 필요하다.

## 4. 한국어 출력 단위

### 4.1 증거 표

| 시스템 (출처) | 목적 함수·모델 | 단위 / 어휘 크기 | 데이터 | 결과 |
|---|---|---|---|---|
| KsponSpeech 베이스라인 (MDPI app10196936, 2020) | ESPnet 하이브리드 CTC/attention Transformer (CTC weight 0.5/0.4, beam 10/60). 서브워드 비교(Table 4)는 small Transformer(4 heads × 256) + SpecAugment, ESPnet SentencePiece unigram, 4× 1080Ti, 외부 LM 없음 조건이며 large Transformer 베이스라인이 아님 | 음절 2306(= 학습 전사에 등장한 음절 + 공백) vs unigram 서브워드 5000/8000/10000/12000 | KsponSpeech 969 h | Table 4 (CER / WER / sWER): 2306 → dev 6.5/17.1/12.7, eval-clean 8.0/21.9/14.2, eval-other 9.0/26.6/16.5; 5000 → 6.9/17.7/13.4, 8.1/22.0/14.4, 9.1/26.5/16.7; 10,000 → 6.8/17.4/13.2, 8.3/22.3/14.6, 9.0/26.5/16.6; 12,000 → 7.2/18.4/14.0, 8.5/22.7/15.1, 9.7/27.8/18.0. 음절은 CER·sWER 기준 모든 세트에서 같거나 좋음(eval-other CER은 10,000과 9.0 동률, eval-other WER은 5,000·10,000이 0.1 낮음). Large Transformer 음절: eval-clean 7.6/21.1/13.4, eval-other 8.5/25.5/15.4 |
| Lexicon-free units (1910.11590, 2019) | ESPnet 하이브리드 CTC/attention BLSTM, CTC weight 0.2, beam 30, LM 없음 | 음절 2371(= 빈도 상위 2,350 음절 + `<sp>`/`<unk>`/`<symb>`, 관측 음절 전체가 아님) / 자모 88 / 음절-서브워드 3k·6k / 자모-서브워드 2k·3k / 바이트 256 | Zeroth-Korean 51.6 h (test 1.19 h) | CER/WER: 음절 1.8/2.6, 자모 6.3/19.9, 음절SW 3k 3.0/3.2, 6k 2.3/2.5, 자모SW 3k 4.2/4.3, 바이트 2.5/4.4. CER은 음절 최저지만 WER은 서브워드 6k가 0.1 낮아 논문 결론은 "음절 기반 서브워드 6k 최고". WER은 morfessor 형태소 분절 텍스트 기준(어절 WER 아님) |
| ESPnet egs2/zeroth_korean (README) | Transformer (train_asr_transformer5), BPE nbpe=5000 | BPE 5000 | Zeroth (test_clean 457 발화) | CER 1.4 / WER 2.0 (fbank_pitch: 1.3 / 2.0) |
| Pronunciation-aware transducer (2404.04295, 2024) | FastConformer transducer ~50M (논문: 인코더 hidden 640→400; 인용된 NeMo 기본 설정의 인코더 d_model은 512이고 640은 pred/joint hidden이어서 정확한 구조는 재구성 불가) | 문자(음절) 단위(논문 Sec. 4 "character-based representations", ko-pron으로 발음 생성), 어휘 크기 미기재; 디코더 임베딩에 초성(C)·나머지(V) 발음 특성, 최고치는 조이너 임베딩에도 CVW 적용 | Zeroth ~50 h | 베이스라인(W/W) CER 1.51 → 디코더만 V 1.36(Table 3) → 디코더 CV + 조이너 CVW 1.22(Table 5; 본문은 V-VW라 썼으나 표의 V-VW는 1.27). 디코딩 방식·GPU·에폭·정밀도 미기재, 검증 오류 증가 시까지 학습 + 상위 5 체크포인트 평균; 사전학습 가중치 언급 없음(처음부터 학습으로 보이나 명시 안 됨) |
| Romanization encoding (2407.04368, 2024) | FastConformer RNNT(+Roman 경로는 Transformer R2C 디코더, 텍스트 손실 합산), 그리디, LM 없음 | 한국어 문자 어휘 1,202 (로마자화해도 동일 크기) | Zeroth 50 h 전체 + AISHELL-1 무작위 50 h, 중국어-한국어 이중언어 모델(연결 어휘 Char 6,380 / Roman 2,441) | Zeroth test CER 1.40 (Char = Roman; 한국어 로마자화는 거의 1:1이라 어휘 축소 없음). Adam + cosine, warmup 10k, peak LR 1.5e-3→1e-6, SpecAug, 체크포인트 평균; 모델 크기·GPU·에폭·정밀도 미기재, 사전학습 언급 없음 |
| K-Wav2vec 2.0 (2110.05172, 2021) | wav2vec2 + CTC 다중 과제 (음절+자모, λ=0.5) | 음절 2,260(발음 전사)/2,302(철자 전사); 자모 67/104 | KsponSpeech | 음절 단독 eval-clean CER 6.904, WER 19.983, sWER 11.877; joint 6.881/19.895/11.765 |
| icefall egs/ksponspeech (RESULTS.md, prepare.sh) | Zipformer + pruned stateless transducer 74.78M, 30 ep, fp16, GPU 4장 | BPE 5000 | KsponSpeech + MUSAN | 오프라인 mbs CER 10.35/11.35; 스트리밍 79M 640 ms mbs 9.91/10.72 |
| kangkyu/icefall-asr-ko-streaming-zipformer-72m (HF README_ko) | Zipformer2 causal 71.4M, pruned RNN-T + CTC aux 0.2, 40 ep, fp16, ScaledAdam | 한글 음절 2,460 | KsponSpeech 1,000 h | 오프라인 + KenLM CER 7.63/7.72; 스트리밍 int8 chunk 16(320 ms) 8.63/8.64, chunk 64 8.28/8.22 |
| ESPnet egs2/ksponspeech (README, run.sh) | Conformer 112M + Transformer LM 52M | BPE nbpe=2309 (음절 수와 비슷 → 사실상 음절 어휘로 추정, (확인 필요)) | KsponSpeech | CER 8.7/9.4, WER 21.7/26.6 |
| KoSpeech (2009.03092) | LAS, 문자 단위 | 문자(음절); 서브워드·자모는 미실험 | KsponSpeech | CER 10.31 (AM만) |
| Whisper 계열 (2212.04356; 2509.02523; 2410.18444) | 사전학습 AED | GPT-2 다국어 BPE | FLEURS/CV/Zeroth/KsponSpeech | Whisper 논문 Table 13 FLEURS ko large-v2 14.3(표 제목은 WER, 한국어는 CER 목록에 없음). Moonshine 논문 CER: whisper-medium Zeroth 6.66 / FLEURS 6.99 / CV17 9.38, moonshine-tiny Zeroth 5.72. 기상 논문: Whisper large-v2 KsponSpeech CER 14.66, KsponSpeech 파인튜닝 후 10.48 |
| KoALa-Bench (2604.19782, 2026) | 대형 오디오 LLM | — | Zeroth/CV/KsponSpeech | Zeroth CER Qwen3-Omni 3.39, Raon 4.28; CER 전 구두점 제거·공백 축약 |
| BBPE16 (2602.01717, 2026) | 다국어 ASR | BPE/BBPE/BBPE16 3000 | KsponSpeech, Zeroth | 한국어는 WER로 보고: 단일언어 eval-clean 18.5/18.7/18.6; Zeroth continual WER 7.6/7.5 |

해석.
- 위 표의 Zeroth test CER(1910.11590, ESPnet 레시피, 2404.04295, 2407.04368)은 모두 같은 조건을 공유한다: Zeroth test 457발화 중 456개(99.8 %)의 전사가 train에 그대로 있고 화자만 다르다(2026-10-11 HF datasets-server /rows 전수 비교). 즉 "본 문장·새 화자(seen-text)" 조건의 수치이며 텍스트 일반화는 측정하지 않는다.
- 한국어는 음절이 이미 2~3개 자모의 결합이라 영어 문자처럼 서브워드로 묶을 이득이 없다는 것이 KsponSpeech 저자들의 설명이며 (MDPI), Zeroth 51.6 h에서도 음절(CER 1.8)이 음절 서브워드 3k(3.0)·6k(2.3)보다 CER은 좋았지만 WER은 6k(2.5)가 음절(2.6)보다 0.1 낮아 저자들은 서브워드 6k를 최고로 결론지었다 (1910.11590). 65 h 조건에서는 서브워드 어휘가 희소해지므로 음절이 더 안전하다고 보지만, 이 결론은 CER 기준이다.
- 자모(약 70~100개)는 시퀀스가 길어져 CTC/attention 모두 성능이 크게 나빠진다 (CER 6.3 vs 1.8). 단 임베딩 파라미터는 음절 11,172 대비 1% 미만이라는 LM 연구가 있다 (2023.eacl-main.172). 음절 OOV 보완용 보조 헤드로만 고려.
- 전체 현대 한글 음절은 11,172자이지만 (2023.eacl-main.172), 학습 전사에 실제로 나타나는 음절은 KsponSpeech 969 h 2,306개(공백 포함), Zeroth는 빈도 상위 2,350개를 잘라 쓴 값(+`<sp>`/`<unk>`/`<symb>` = 2,371; 관측 음절 전체 수는 미기재, 1910.11590), 사용자 모델 2,460개 수준이다. 65 h 코퍼스에서는 더 적을 수 있으므로 미등장 음절은 `<unk>` 또는 자모 폴백으로 처리해야 한다 (수치 (확인 필요)).
- BPE 5000(icefall KsponSpeech CER 10.35)과 음절 2,460(kangkyu CER 7.63)은 모델·에폭·LM이 달라 직접 비교는 불가하다.

### 4.2 정규화 관행

- KsponSpeech 전사: "(철자)/(발음)" 이중 전사, 숫자는 발음도 함께 기록하고 "년·월·일·분" 등 단위와 띄어 쓰며 숫자 발음은 10진 단위로 띄어 적는다. 띄어쓰기는 표준 어문 규정을 따르되 애매하면 띄운다. 문장부호는 문장 끝. 비유창성 태그(간투어·반복·단어 조각) 포함 (MDPI 2.2.5, 2.2.7). KoSpeech·sooftware/ksponspeech 전처리는 b/ n/ 등 잡음 태그와 '/', '*', '+' 제거 후 발음(phonetic) 또는 철자(spelling) 모드 선택, 숫자만 철자 선택 옵션 제공 (2009.03092; sooftware README).
- Zeroth: Morfessor 형태소 분절기와 LM(109,037,699문장 학습 코퍼스, 3-gram PPL 221.3, 4-gram 187.2)이 함께 배포된다 (goodatlas/zeroth, OpenSLR 40). Moonshine 논문은 Zeroth 정규화 방식을 따라 "모든 구두점 제거, 아라비아 숫자는 발음 철자로 변환"했다 (2509.02523 Appendix B). FLEURS/CV의 숫자 표기가 Zeroth와 같은지는 (확인 필요).
- Whisper 평가: FLEURS는 `transcription` 열(정규화본)을 쓰고 한국어는 선행 단어를 라틴 문자로 반복한 괄호구를 정규식으로 제거했다. 한국어는 띄어쓰기 생략이 흔히 허용되고 CV15·FLEURS 라벨의 띄어쓰기가 불일치해 CER로 채점했다 (openai/whisper discussions #1762).

### 4.3 평가 지표 관행

- CER: 음절 단위 Levenshtein 거리(공백 포함 여부는 논문마다 다름; KsponSpeech는 문자+공백 단위, K-Wav2vec는 음절 단위) (MDPI; 2110.05172).
- WER: 어절(word-phrase) 단위. 한국어는 띄어쓰기 변이 때문에 WER과 sWER 격차가 크다(KsponSpeech large Transformer eval-other WER 25.5 vs sWER 15.4).
- sWER: 기준 전사의 띄어쓰기에 맞춰 가설의 공백만 정규화한 뒤 WER 계산(sclite, Algorithm A2) (MDPI Appendix B; 2110.05172).
- 2026년 벤치마크(KoALa-Bench)는 구두점 제거·공백 축약 후 CER. Whisper 팀도 한국어 CER. 결론: 주지표 CER, 보조 sWER, 정규화 규칙(구두점·숫자·공백)을 명시해 보고.

## 5. fp16 / 11 GB 주의점

1. **CTC 손실은 fp32로.** PyTorch 2.14 autocast(CUDA) fp32 승격 목록에는 log_softmax, softmax, nll_loss, cross_entropy, layer_norm, exp, pow, sum, cumsum이 있지만 **ctc_loss는 CUDA 목록에 없다**(CPU 목록에만). 따라서 autocast 안에서도 log_softmax 출력(fp32)을 그대로 넘기면 안전하지만, fp16 텐서를 직접 넘기면 `"ctc_loss_cuda" not implemented for 'Half'` 류 오류가 난다(ESPnet issue #1836; torch 2.11.0+cu128·RTX 2080 Ti에서 재현 확인). ESPnet2 `ctc.py`는 `th_pred.log_softmax(2).float()`로 명시 캐스트하고 기본 `zero_infinity=True`를 쓴다. icefall의 CTC 경로(forward_ctc)는 명시 캐스트 없이 nn.LogSoftmax 출력이 autocast로 fp32가 되는 것에 의존한다.
2. **cuDNN CTC 경로 조건:** blank=0, 각 타깃 길이 <256 (ATen `_use_cudnn_ctc_loss` 코드 기준 `target_lengths[b] < 256`; 공식 문서는 ≤256으로 적혀 있어 1 차이), 타깃은 1-D 연결(concatenated)·int32·contiguous, log_probs는 3-D CUDA fp32, 모든 input_lengths==T, target_len ≤ input_len. 조건을 벗어나면 네이티브 커널로 떨어지며 결정성은 `cudnn.deterministic`으로 강제 (PyTorch 2.14 CTCLoss docs). Windows CUDA 빌드(torch 2.11.0+cu128, cuDNN 9.19, RTX 2080 Ti)에서 cuDNN 경로가 실제로 실행됨을 프로파일러로 확인(aten::_cudnn_ctc_loss); int64 lengths/targets를 주면 네이티브 커널로 떨어진다(손실값은 동일). `zero_infinity=True`는 입력이 타깃보다 짧아 생기는 inf 손실·기울기를 0으로 만든다.
3. **PyTorch 포럼 경고:** CTC는 "수치 정확도 요구가 큰 함수"로 fp32에서도 문제가 보이며 fp16은 좋은 선택이 아니라는 개발자(Thomas V) 의견. autocast에서 blank만 출력하며 손실이 멈춘 사례에 대해서는 `F.log_softmax(..., dtype=torch.float32)`의 fp32 입력을 확인하라는 답이 달렸을 뿐 해결 여부는 스레드에 없다 (discuss.pytorch.org #148800).
4. **RNN-T 손실도 fp32.** icefall `zipformer/model.py`는 `torch_autocast(enabled=False)` 블록에서 `k2.rnnt_loss_smoothed(lm=lm.float(), am=am.float(), ...)`, `k2.rnnt_loss_pruned(logits=logits.float(), ...)`를 호출하고 prune_range 기본 5. NeMo `RNNTLoss`는 `force_float32=True`가 기본이며 numba fp16 커널이 지원될 때만 fp16 실행 (NeMo rnnt.py). TDT도 `force_float32=True`.
5. **조인트 메모리 추산(본 문서 계산, 논문값 아님):** B=8, T=400(16 s, 4× 서브샘플링 가정), U=80 음절, V=2,500이면 로짓 원소 8×400×80×2,500 = 6.4×10^8 → fp32 2.56 GB, log-softmax·기울기까지 보수적으로 3배면 약 7.7 GB로 11 GB에서 모델·활성화와 공존 불가. pruned S=5면 8×400×5×2,500 = 4×10^7 → 약 160 MB. 실측 비율도 비슷하다: vocab 500, batch 30 손실 계산 벤치마크 피크 18.48 GB→3.73 GB (2206.13236; 전체 학습 메모리 아님). 같은 B·T·U·V로 torchaudio `rnnt_loss`를 이 PC에서 돌린 피크는 fp32 7.24 GB, fp16 3.62 GB로 위 추정과 맞는다(9항). 한국어 음절 2,500 어휘는 영어 BPE 500의 5배라 full RNN-T는 피해야 한다.
6. **CR-CTC는 두 증강 뷰를 배치 축으로 이어 2× 배치로 1회 forward(연산 2배)**하므로 유효 배치가 절반이 된다. 논문도 배치·에폭을 절반으로 줄여 학습했고 (2410.05101), icefall train.py도 "should use half the max-duration compared to regular ctc"라고 안내한다(공개 S 레시피 max-duration 850 vs CTC/AED-S 1700). 11 GB에서는 Zipformer-S급(22M)에서만 현실적.
7. **fp16 학습 불안정 실례:** icefall KsponSpeech `zipformer` 레시피(74.8M, 4 GPU, `--use-fp16 1`, max-duration 750)는 `grad_scale is too small` 오류를 피하려고 base-lr 0.045→0.035로 낮췄고, 같은 RESULTS의 스트리밍 레시피(79M)는 기본 lr을 유지했다 (icefall ksponspeech RESULTS.md). NeMo conformer_ctc_bpe.yaml은 "bf16 권장, 없으면 16, fp16이 불안정하면 fp32" 주석. 로그-소프트맥스 입력 로짓이 fp16에서 65,504를 넘지 않도록 최종 projection 출력도 fp32로 두는 것이 안전하다(autocast가 log_softmax를 fp32로 올리므로 입력 캐스트만 확인).
8. **배치 구성:** KsponSpeech 베이스라인은 GPU 메모리 때문에 3,000프레임 이상 또는 400음절 이상 발화를 제외했다 (MDPI 3.1). 길이 기반 동적 배치(max-duration)와 긴 발화 제외는 11 GB에서 필수.
9. **torchaudio 2.11 `rnnt_loss`(k2 없는 full RNN-T 대안):** 2.10 릴리스 노트에서 C++/CUDA 확장 존치가 확정됐다. fp16 logits를 받지만 bf16은 거부하고("logits must be float32 or float16"), 내부 계산은 fp32이되 비용·기울기는 logits dtype(fp16)으로 반환한다. `reduction='sum'`을 fp16 비용에 그대로 쓰면 65,504 초과 시 inf(실측 B=32·발화당 약 2.6k). autocast 밖에서 쓰면 `reduction='none'` 뒤 `.float().sum()`으로 합산할 것. logits는 contiguous, targets/lengths는 int32 필수. 워크스페이스는 5×B×T×U float(V 무관)이고 B×T×U×V 크기는 호출자가 만드는 logits와 그 기울기다. 실측(B=8, T=400, U=80, V=2,500; torch 2.11.0+cu128, RTX 2080 Ti): fp16 피크 3.62 GB, fp32 7.24 GB(5항의 "약 3배·7.7 GB" 추정과 일치). pruned 손실은 제공하지 않는다.

## 6. 우리 조건에서의 순위와 열린 질문

### 순위 (65 h, 11 GB fp16, 스트리밍 목표)

1. **CTC + InterCTC/Self-conditioned CTC(+ 가능하면 CR-CTC), 음절 2.3~2.5k, prefix beam + n-gram(TLG/HLG)**
   - 근거: 조인트 텐서 없음(5절), 소량 데이터 개선 증거(WSJ 81 h, 2102.03216·2104.02724), CTC 격차를 메우는 CR-CTC(2410.05101), 한국어 음절 단위 우위(4절), 스트리밍은 chunk 인코더 + 1-pass. 손실 fp32 캐스트 비용 무시 가능.
   - 약점: 스트리밍 CTC는 transducer보다 일관되게 나쁘다(2512.11543: TLv2 8.9 vs 7.8; LibriSpeech 3.2/7.8 vs 2.8/6.7). 외부 LM으로 일부 보완 가능(2607.05612: 짧은 문맥에서 LM 이득 큼).
2. **Pruned RNN-T(stateless decoder) + CTC 보조 손실(scale 0.2), 음절 단위, icefall 레시피**
   - 근거: 스트리밍 정확도 우위(2011.04785 RNN-T가 LF-MMI 대비 2~11%↓; 2512.11543), 손실 계산 피크 3.73 GB 실측(2206.13236; 전체 학습 메모리 아님), 한국어 스트리밍 실례(kangkyu 71.4M, CER 8.63 @320 ms), fp16 학습 + 손실 fp32 캐스트 구조 내장.
   - 약점: k2는 Windows용 CPU 휠만 제공(torch 2.11.0·cp311 포함)하고 CUDA는 소스 빌드만이며, 2025-11~12 이슈(#1342, #1343)에서 torch 2.8/2.9 Windows CUDA 빌드 실패·torch 2.6+수동 cmake 수정만 성공, 관리자도 Linux 권장. fast_rnnt는 PyPI sdist만(1.3, 2023-11), `_fast_rnnt` C++/CUDA 확장 필수(Python 대체 없음), Windows CI 없음, 2023-08 이후 커밋 없음. 순수 PyTorch pruned RNN-T 공개 구현은 없음(GitHub 검색 기준). 따라서 순수 PyTorch·Windows·WSL2 없음 조건에서는 k2/fast_rnnt 경로가 사실상 막힌다(torchaudio `rnnt_loss`는 full RNN-T만, 5절 9항). 그 밖에 디코딩·LM 결합이 더 복잡(LODR/shallow fusion), 65 h 소량 근거 없음(Voxforge 20 h RNN-T는 CTC aux로 소폭 개선, 2201.05420). 인코더 초기화 민감(2005.14327).
3. **하이브리드 CTC/attention(U2식 1-pass CTC + 2-pass attention rescoring)**
   - 근거: KsponSpeech·Zeroth 공개 레시피의 표준(4절), rescoring이 chunk 16에서 CER 6.08→5.33(2012.05481), 오프라인 베스트 가능.
   - 약점: decoder 파라미터(Zipformer-S 22.1M→46.3M) 및 Δ 50~100 ms 추가, 스트리밍 1-pass는 결국 CTC. CTC 단독보다 학습 신호가 하나 더 있어 소량 데이터 정규화 효과는 기대 가능.
- 보류: TDT(스트리밍 미평가, duration이 emission 지연, 2304.06795), Paraformer(오프라인 NAR, 스트리밍 변형 미확인), BRCTC(학습 비용 증가, k2 FST), 자모 단위(최악 CER), BPE 5k 이상(한국어에서 손해).

### 열린 질문

1. 65 h 전사에 등장하는 음절 수와 Zeroth test/FLEURS/CV 음절 커버리지. 미등장 음절 처리(자모 보조 헤드 vs `<unk>`) — 2110.05172의 joint decoding 이득은 0.02~0.1 CER 수준으로 작았다.
2. CR-CTC의 2× 배치 forward가 11 GB에서 어느 모델 크기까지 가능한가(Zipformer-S 22M 기준 배치 절반). CR-CTC vs InterCTC 단독 비교는 소량 데이터에서 보고가 없다.
3. 한국어 음절 n-gram vs 어절 n-gram 중 어느 쪽이 CTC prefix beam/TLG에서 유리한가. 2606.23306의 "3-gram 무효" 결과는 영어 BPE 500·100 h 기준.
4. Zeroth(숫자를 한글로 표기)와 FLEURS/CV ko(숫자·구두점 표기 방식 (확인 필요))를 합칠 때의 정규화 통일 규칙. 평가 시 공백 처리(CER에 공백 포함 여부)와 sWER 병기 여부.
5. 스트리밍 CTC emission 지연 제어 수단(BRCTC, Delayed-KD, 또는 transducer FastEmit 2010.11148: LibriSpeech p90 지연 210→30 ms) 중 11 GB에서 구현 가능한 것.
6. fp16 Zipformer/ScaledAdam의 안정성(icefall KsponSpeech `zipformer` 레시피는 LR을 낮췄고 스트리밍 레시피는 기본값 유지). Conformer+AdamW 대안과의 비교는 아키텍처 조사(다른 문서) 결과와 합쳐 결정.

## 7. 읽은 자료 목록

- arXiv 2410.05101 CR-CTC (ICLR 2025) — https://arxiv.org/html/2410.05101
- arXiv 2206.13236 Pruned RNN-T — https://ar5iv.labs.arxiv.org/html/2206.13236
- arXiv 2304.06795 TDT (ICML 2023) — https://ar5iv.labs.arxiv.org/html/2304.06795
- arXiv 2102.03216 Intermediate CTC — https://ar5iv.labs.arxiv.org/html/2102.03216
- arXiv 2104.02724 Self-conditioned CTC — https://ar5iv.labs.arxiv.org/html/2104.02724
- arXiv 2210.07499 Bayes Risk CTC (ICLR 2023) — https://ar5iv.labs.arxiv.org/html/2210.07499
- arXiv 2206.08317 Paraformer — https://ar5iv.labs.arxiv.org/html/2206.08317
- arXiv 2305.11013 FunASR — https://ar5iv.labs.arxiv.org/html/2305.11013
- arXiv 2012.05481 WeNet U2 — https://ar5iv.labs.arxiv.org/html/2012.05481
- arXiv 2005.14327 Li et al. 2020 RNN-T/AED 비교 — https://ar5iv.labs.arxiv.org/html/2005.14327
- arXiv 2011.04785 LF-MMI/CTC/RNN-T 스트리밍 벤치마크 — https://ar5iv.labs.arxiv.org/html/2011.04785
- arXiv 2210.09951 HMM vs CTC from scratch — https://ar5iv.labs.arxiv.org/html/2210.09951
- arXiv 2201.05420 ESPnet transducer 연구 — https://ar5iv.labs.arxiv.org/html/2201.05420
- arXiv 2203.16868 Sampled-softmax RNN-T — https://ar5iv.labs.arxiv.org/html/2203.16868
- arXiv 2305.11571 BAT — https://ar5iv.labs.arxiv.org/html/2305.11571
- arXiv 2010.11148 FastEmit (abstract) — https://arxiv.org/abs/2010.11148
- arXiv 2512.11543 All-in-One ASR — https://arxiv.org/html/2512.11543
- arXiv 2508.09880 시스템 결합 비교 — https://arxiv.org/html/2508.09880
- arXiv 2510.06961 ASR Leaderboard — https://arxiv.org/html/2510.06961v4
- arXiv 2510.12827 ASR 설문 — https://arxiv.org/html/2510.12827
- arXiv 2505.22857 NGPU-LM (Interspeech 2025) — https://arxiv.org/html/2505.22857
- arXiv 2506.00185 Transducer beam search (Interspeech 2025) — https://arxiv.org/html/2506.00185v1
- arXiv 2606.23306 CTC Oracle Gap (2026) — https://arxiv.org/html/2606.23306
- arXiv 2607.05612 LM PPL vs WER (2026) — https://arxiv.org/html/2607.05612
- arXiv 2103.07186 BPE-dropout 저자원 — https://ar5iv.labs.arxiv.org/html/2103.07186
- arXiv 1910.11590 한국어 lexicon-free 단위 — https://ar5iv.labs.arxiv.org/html/1910.11590
- KsponSpeech 논문 (Appl. Sci. 2020, 10, 6936) — https://ksp.etri.re.kr/ksp/article/file/62525.pdf (pdftotext로 본문 확인)
- arXiv 2009.03092 KoSpeech — https://arxiv.org/pdf/2009.03092 (pdftotext로 본문 확인)
- arXiv 2110.05172 K-Wav2vec 2.0 — https://ar5iv.labs.arxiv.org/html/2110.05172
- arXiv 2404.04295 Pronunciation-aware transducer (Zeroth) — https://arxiv.org/html/2404.04295
- arXiv 2407.04368 Romanization encoding (Zeroth) — https://arxiv.org/html/2407.04368v2
- arXiv 2410.18444 한국어 기상 ASR 평가 — https://arxiv.org/html/2410.18444v2
- arXiv 2604.19782 KoALa-Bench — https://arxiv.org/html/2604.19782
- arXiv 2602.01717 BBPE16 — https://arxiv.org/html/2602.01717
- arXiv 2509.02523 Flavors of Moonshine — https://arxiv.org/html/2509.02523
- arXiv 2212.04356 Whisper (Appendix D 표) — https://ar5iv.labs.arxiv.org/html/2212.04356
- openai/whisper discussions #1762 (large-v3, 한국어 CER 채점) — https://github.com/openai/whisper/discussions/1762
- ACL 2023.eacl-main.172 (한글 음절 11,172) — https://aclanthology.org/2023.eacl-main.172/
- goodatlas/zeroth README; OpenSLR 40 — https://github.com/goodatlas/zeroth , https://www.openslr.org/40/
- ESPnet egs2/zeroth_korean README·run.sh; egs2/ksponspeech README·run.sh — https://github.com/espnet/espnet
- ESPnet espnet2/asr/ctc.py; issue #1836 — https://github.com/espnet/espnet
- icefall egs/librispeech/ASR/RESULTS.md; zipformer/model.py; egs/ksponspeech RESULTS.md·prepare.sh; LM 디코딩 문서; conformer_ctc 문서 — https://github.com/k2-fsa/icefall , https://k2-fsa.github.io/icefall/
- kangkyu/icefall-asr-ko-streaming-zipformer-72m README_ko — https://huggingface.co/kangkyu/icefall-asr-ko-streaming-zipformer-72m
- WeNet docs/lm.md — https://github.com/wenet-e2e/wenet/blob/main/docs/lm.md
- NeMo nemo/collections/asr/losses/rnnt.py; examples/asr/conf/conformer/conformer_ctc_bpe.yaml; ASR configs 문서 — https://github.com/NVIDIA/NeMo , https://docs.nvidia.com/nemo-framework/
- PyTorch 2.14 docs: amp (autocast op list), torch.nn.CTCLoss — https://docs.pytorch.org/docs/2.14/amp.html , https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html
- PyTorch forum: CTC loss fp16 — https://discuss.pytorch.org/t/ctc-loss-ctc-loss-not-implemented-for-half/148800
- sooftware/ksponspeech README — https://github.com/sooftware/ksponspeech
- PyTorch ATen `aten/src/ATen/native/cudnn/LossCTC.cpp` (`_use_cudnn_ctc_loss`) — https://github.com/pytorch/pytorch
- torchaudio 2.11 `rnnt_loss` 문서·소스(`src/libtorchaudio/rnnt/`)·릴리스 노트 2.8~2.11 — https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.rnnt_loss.html , https://github.com/pytorch/audio/releases
- k2 설치 문서(from_wheels, CPU/CUDA 휠 인덱스)·이슈 #1263, #1342, #1343; fast_rnnt PyPI·저장소 — https://k2-fsa.github.io/k2/installation/from_wheels.html , https://github.com/k2-fsa/k2 , https://github.com/k2-fsa/fast_rnnt
- 로컬 실측(2026-10-11; torch 2.11.0+cu128, torchaudio 2.11.0+cu128, cuDNN 9.19, RTX 2080 Ti, Windows 11): cuDNN CTC 경로 프로파일, `rnnt_loss` dtype·메모리, ESPnet #1836 재현; Zeroth train/test 전사 전수 비교(HF datasets-server /rows)

## 8. 검증 후 정정 (2026-10-11)

검증 C1~C13(`README.md` 2절, `work/survey_workflow_results.json` → verify) 가운데 이 문서를 겨냥한 항목을 반영했다. 수치 자체가 틀린 곳은 없었고 조건·용어·일반화를 고쳤다.

- C7: §1-4, §4.1 KsponSpeech 행, §4.1 해석 — 음절 2306 "모든 평가셋에서 좋음" → "CER·sWER 기준 같거나 좋음"(Eval-other CER 10k 동률, Eval-other WER 5k·10k 0.1 낮음); small Transformer+SpecAugment·LM 없음 조건과 10,000 행 추가.
- C7: §1-4, §4.1 1910.11590 행, §4.1 해석, §4.1 음절 수 항목 — Zeroth WER은 서브워드 6k 2.5 < 음절 2.6이고 논문 결론은 서브워드 6k 우위; WER은 morfessor 분절 기준; 2371 = 상위 2,350 + `<sp>`/`<unk>`/`<symb>`.
- C8: §4.1 해석 첫 항목 신설 — Zeroth test 456/457발화 전사가 train에 포함된 "본 문장·새 화자(seen-text)" 조건 주석.
- C9: §1-6, §5-1 — `.float()` 캐스트는 ESPnet·NeMo; icefall은 k2 transducer 손실에만, CTC 경로는 autocast log_softmax 승격 의존. ESPnet #1836 오류는 torch 2.11.0+cu128·2080 Ti에서 재현 확인("apex O1 시절" 삭제).
- C9: §5-3 — 포럼 #148800은 "해결"이 아니라 "fp32 입력 확인 답변, 해결 확인 없음".
- C9: §5-7, §6 열린 질문 6 — base-lr 0.045→0.035는 `zipformer` 레시피(74.8M, 4 GPU, max-duration 750) 조건; 스트리밍 레시피는 기본 lr 유지.
- C5: §2 CR-CTC 행, §5-6, §6 열린 질문 2 — "forward 2회" → "두 증강 뷰를 배치 축으로 이어 2× 배치 1회 forward(연산 2배)"; 공개 icefall 레시피(M 4×V100 fp16 50 ep, train.py 안내) 병기; "스트리밍 적합성: CTC와 동일"은 추론임을 명시.
- C6: §2 pruned RNN-T 행 — S(=4~5)는 예시값, 실험값 미기재(icefall 기본 prune_range 5).
- C6: §1-2, §2 RNN-T·pruned 행, §5-5, §6 2순위 근거 — 18.48→3.73 GB는 손실 계산 벤치마크 피크(vocab 500, V100, 정밀도 미기재)이며 전체 학습 메모리가 아님.
- C13: §2 AED/하이브리드 행 — U2 4.92/4.93/4.64는 Table 1 비스트리밍 kernel-15 모델(동적 청크 통합 모델 full 4.90과 다름).
- C10: §4.1 2404.04295 행 — 단위 문자(음절)·ko-pron; 1.22는 디코더 CV + 조이너 CVW(Table 5; V-VW는 1.27), 디코더만 V 1.36; hidden 640 출처 불명; GPU·에폭·정밀도·디코딩 미기재, 사전학습 언급 없음.
- C10: §4.1 2407.04368 행 — 중국어-한국어 이중언어(연결 어휘 6,380/2,441), Adam+cosine warmup 10k peak 1.5e-3, R2C 디코더, 모델 크기·GPU·에폭·정밀도 미기재.
- C11: §5-2 — 타깃 길이 "≤256" → ATen 코드 "<256"(문서는 ≤256), 조건 목록 보강, Windows cu128에서 cuDNN CTC 경로 실행 확인.
- C11: §5-9 신설 — torchaudio 2.11 `rnnt_loss`(fp16 허용·bf16 거부·비용은 logits dtype 반환·reduction='sum' overflow·실측 fp16 3.62 GB / fp32 7.24 GB).
- C11: §1-2, §2 pruned 행, §6 2순위 약점 — "k2 빌드 의존" → k2 Windows CPU 휠만·CUDA 소스 빌드 실패 보고·fast_rnnt 미관리·순수 PyTorch pruned 구현 없음.
- C4: §2 SC-CTC 행 — 2104.02724 원제 병기(표 수치는 수정 불필요 판정).
- §7에 ATen LossCTC.cpp, torchaudio rnnt_loss, k2/fast_rnnt 설치 자료, 로컬 실측 항목 추가. 머리말에 정정일 표기.
