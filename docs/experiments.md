# 측정 규칙과 결과

규칙은 재기 전에 적고 커밋한다. 결과는 각 단계 절의 "결과" 소절에만 적고, 규칙을 바꾸면 "규칙 변경" 절에 이유와 시점을 남긴다. 봉인 test는 7단계에서만, 사전 등록한 구성으로 각 1회 잰다.

## 공통 (2026-10-11 규칙, 측정 전)

### 지표
- **주지표 CER_ns**: 공백 제외 음절 CER. 정규화는 whisper-ko-ft와 같다: 소문자화, 글자·숫자·공백·작은따옴표가 아닌 문자는 공백으로, 한국어는 작은따옴표와 공백을 모두 없앤 뒤 글자 단위 편집 거리. 묶음 전체의 편집 수 합 ÷ 정답 글자 수 합. whisper-ko-ft의 Zeroth test 앵커(turbo 제로샷 4.48 %, turbo+LoRA 1.96 %)가 이 정의다
- **보조 CER_ws**: 공백 포함 음절 CER(공백도 글자로 셈, KsponSpeech 정의)
- 구간: 발화 단위 부트스트랩 1,000회, seed 0, 95 %. 두 구성의 비교는 같은 발화를 다시 뽑는 짝지은 구간
- 모든 수치에 체크포인트(epoch, 평균 개수)와 디코딩 모드(full / simulated chunk / real streaming)와 청크·좌측 문맥을 함께 적는다
- 정답에 아라비아 숫자·라틴 문자가 남는 발화(FLEURS·CV)는 **전체**와 **한글 전용 부분집합** 두 수치를 모두 보고한다. 모델 어휘에 숫자·라틴이 없으므로 전체 수치에서는 그 글자가 오류로 센다

### 텍스트 정규화 (학습 정답·평가 정답 공통, 스크립트 하나)
1. NFC 정규화
2. 한글·라틴 문자·숫자·공백 외 문자(문장부호·기호)는 공백으로
3. 공백 축약, 양끝 제거
4. 숫자가 든 발화는 **학습에서 제외**(한국어 숫자 읽기가 둘이라 자동 변환이 틀리기 쉬움). 평가에서는 위 "두 수치" 규칙
5. 라틴 문자만 든 발화도 학습에서 제외(어휘는 한글 음절만). 제외 수는 세트별로 보고

### 어휘
- blank(0), `<unk>`(1), 공백(2), 그다음 학습 전사(정규화·제외 뒤)에 1회 이상 등장한 한글 음절 전부(빈도순). 평가 세트별 OOV 음절 수를 CER 하한으로 보고

### 데이터 분할 (SHA-256, seed 없음)
- Zeroth 고유 문장 3,003개를 `sha256("zeroth-holdout:" + 문장)`의 앞 8자리 16진수를 정수로 보아 `% 100`이 `< 10`이면 **홀드아웃 H**(약 300문장, 그 문장의 발화 전부 ≈ 2,200발화 ≈ 5 h). H 안에서 `% 100`이 `< 5`면 **dev_H**, 아니면 **test_H**(봉인)
- 학습 = Zeroth train − H + FLEURS train(제외 규칙 뒤) + CV22 validated − CV22 dev − CV22 test(path 기준)
- **dev_mix**(1차 선택 지표) = FLEURS validation(한글 전용) + CV22 dev(한글 전용). 두 세트의 편집 수·글자 수를 합쳐 하나의 CER로 계산
- **dev_H**(2차 가드) = Zeroth 홀드아웃 dev
- **봉인 test**(7단계 1회) = Zeroth 공식 test 457(본 문장·못 본 화자, 표시 붙임), Zeroth test_H(못 본 문장·본 화자), FLEURS test 382, CV22 test 472
- Common Voice `other`(미검증)와 `invalidated`는 쓰지 않는다
- 길이 20 s 초과 발화는 학습에서 제외. 25 fps 출력에서 `T' < U + (연속 같은 토큰 수) + 1`인 발화도 제외(CTC 손실 무한대 방지). 모두 수를 보고

### 선택 규칙 (모든 단계 공통)
- 후보 사이 비교는 dev_mix의 CER_ns로 하고, dev_H가 절대 0.3 넘게 나빠지는 후보는 채택하지 않는다
- 채택 임계는 **max(0.3 절대, 2s)**. s = 1단계에서 같은 설정 3시드의 dev_mix CER_ns 표준편차. 상대 기준을 쓰는 가설은 짝지은 부트스트랩 구간이 0을 제외해야 한다
- 같은 비교 안의 런은 같은 업데이트 step 수·배치 규칙·시드를 쓴다(메모리가 남아도 배치를 키우지 않음)
- 체크포인트는 매 epoch 저장, 평가 모델은 dev_mix 상위 10개 평균

### 예산 규칙
- 0단계에서 잰 epoch 시간 × 100 > 14 h이면 2단계 epoch = floor(14 h ÷ epoch 시간), 하한 60. 적용하면 기록
- 단계마다 GPU 시간을 기록한다. 다른 프로그램이 GPU를 쓰는 동안 잰 속도·지연은 "부하 조건"으로 표시한다

### fp16 규칙
- `torch.autocast(device_type="cuda", dtype=torch.float16)` 명시, 파라미터·옵티마이저 fp32, GradScaler 기본값, `unscale_` 뒤 grad clip 5.0
- CTC와 자기조건 CTC 입력은 `log_softmax(..., dtype=float32)`. 단위 테스트로 고정
- **fp16 발산 판정**: 손실 NaN/Inf 3회 이상, 또는 GradScaler scale < 1.0이 100 step 연속, 또는 3 epoch까지 dev CTC 손실이 내려가지 않음. 발산한 런은 기록하고 LR×0.5로 최대 2회 재시작. 발산 횟수는 최종 보고에 표로 남긴다
- 시작 로그에 `torch.cuda.is_bf16_supported(including_emulation=False)` 값을 남긴다(2080 Ti는 False여야 함)

## 0단계: 데이터·코드·마이크로벤치 (2026-10-11 규칙, 측정 전)

### 할 것
- 정규화 스크립트·어휘·분할 매니페스트를 만들고 `manifests/`에 커밋
- 모델·학습·디코딩 코드와 단위 테스트
  - 미래 누출: 미래 프레임을 난수로 바꿔도 현재 청크까지의 출력이 불변(청크 4/8/16/32, 좌측 64/128)
  - 스트리밍 == 오프라인: 캐시 스트리밍 forward와 전체 발화 + 청크 마스크 forward의 log-prob 최대 차이 fp32 1e-5, fp16 1e-2 이내(fbank 경계 프레임 포함)
  - CTC 입력 dtype == float32
  - 과적합: 발화 20개로 CER 0 수렴
- 마이크로벤치: max-duration {200, 300, 400, 600} s에서 피크 메모리·step 시간·데이터 대기 비율, epoch 시간 추정, SDPA 커널 종류

### 가설
- **H0.1**: 25 fps에서 CTC 길이 제약에 걸리는 학습 발화 ≤ 0.5 %
- **H0.2**: max-duration 300 s 피크 메모리 ≤ 9.5 GB, 학습셋 1 epoch ≤ 8분

### 판정
- 단위 테스트 전부 통과 전에는 학습을 시작하지 않는다
- H0.1 위반 발화는 제외하고 비율을 기록. H0.2 위반이면 예산 규칙 적용
- 데이터 대기 비율 > 20 %면 fbank fp16 사전 계산으로 전환하고 기록

### 0단계 결과 (2026-10-11)
- 데이터(`reports/stage0/data_summary.md`, `scripts/prepare_data.py`): 학습 21,853발화 51.77 h(Zeroth 46.11 + FLEURS 4.75 + CV22 0.91), dev_mix 700발화 1.46 h, dev_h 1,113발화 2.65 h, test_zeroth 457, test_h 1,280발화 2.92 h, test_fleurs 382, test_cv 472. 학습 제외 844발화 3.19 h(숫자 619, 라틴 200, 20 s 초과 24, 모두 FLEURS; CV22 오디오 결손 1). 홀드아웃 H 318문장 2,393발화(10.6 %), dev_h 151문장, test_h 167문장, 서로 겹침 없음. 어휘 1,288(음절 1,285). 평가 세트 OOV 음절 비율 0.005~0.37 %. 학습 토큰률 p50 6.55 / p95 8.01 / 최대 10.64 토큰/s
- **H0.1 충족**: 25 fps CTC 길이 제약 위반 0 / 21,853 (0.000 %)
- 참고: Zeroth 공식 test 423문장 중 367개는 학습 세트에, 55개는 홀드아웃 H에 있다(본 문장 조건은 그대로)
- 단위 테스트: `tests/test_model.py` 21개 + `tests/test_prepare_data.py` 13개 통과(미래 누출 없음, 캐시 스트리밍 == 오프라인 청크 마스크 fp32 최대 차이 < 1e-5, 부분 마지막 청크 포함, 패딩 불변, 스트리밍 fbank == 오프라인, autocast 아래 CTC 입력 fp32, 분할 conv가 독립 unfold 구현과 일치, 합성 20발화 과적합 CER 0)
- 마이크로벤치(`reports/stage0/bench.json`, RTX 2080 Ti, fp16, 동적 청크·속도 교란·SpecAugment 포함, 30 step 평균): max-duration 300 s → 피크 4.31 GB, 0.663 s/step, 697 step/epoch, **epoch 약 7.8분**, 평균 배치 32발화, 데이터 대기 1.1 %; 400 s → 5.62 GB, 0.685 s/step, 6.0분. 200·600 s도 OOM 없음(600 s 8.9 GB). SDPA는 flash 커널 불가(sm_75), memory-efficient 커널 사용. `is_bf16_supported(including_emulation=False)` = False 확인
- **H0.2 충족**(300 s 피크 ≤ 9.5 GB, epoch ≤ 8분). 100 epoch ≈ 13 h로 예산 규칙(14 h) 안이므로 2단계는 100 epoch, max-duration 300 s 그대로
- 측정 중 고친 것: 처음 벤치는 `cudnn.benchmark=True`로 돌려 1.36 s/step(배치 모양이 매번 달라 cuDNN 자동 튜닝이 step마다 돌았음)이었고, 끄자 0.66 s/step. 분할 depthwise conv를 unfold 구현에서 grouped conv1d + 청크 끝 보정으로 바꿨다(출력은 독립 구현과 1e-5 안에서 같음, 테스트 추가). step 시간은 배치 크기(13~55발화)와 무관하게 약 0.65 s로 GPU 연산(약 0.14 s)보다 커널 실행·Python 오버헤드가 지배한다. Windows라 torch.compile(Triton)은 쓸 수 없어 그대로 둔다

## 1단계: fp16 안정성·peak LR (2026-10-11 규칙, 측정 전)

- 후보: peak LR {1e-3, 2e-3} × 시드 {1, 2, 3}, warmup 4,000 step, 최종 구성 그대로 5 epoch
- 측정: 런별 발산 여부(위 정의), epoch 5의 dev_mix·dev_H CER_ns(full greedy), GradScaler 스킵 횟수와 scale 이력
- **H1**: 2e-3에서 발산 ≤ 1/3
- 판정: 발산 0/3인 LR 가운데 epoch 5 dev_mix 평균이 낮은 쪽을 채택. 둘 다 발산이 있으면 5e-4·warmup 8,000을 더해 반복. 채택 LR의 3시드 표준편차를 s로 기록

## 2단계: 주 모델 (2026-10-11 규칙, 측정 전)

- Run A: 동적 청크 통합 모델, 채택 LR, 시드 1, 100 epoch(예산 규칙)
- Run B(통제, 시간이 남을 때 우선순위 1): 청크 마스크 없는 전체 문맥 전용 모델, 같은 step 수·LR·시드
- 측정(매 epoch dev, 최종은 10개 평균): full greedy CER_ns(dev_mix, dev_H), simulated chunk {160, 320, 640, 1,280} ms × 좌측 128 CER_ns, 중간 헤드(3·6·9층)별 CER
- **H2.1**: Run A의 full CER_ns ≤ Run B × 1.05 (통합 학습 비용 ≤ 5 %)
- **H2.2**: Run A의 640 ms ≤ full × 1.25, 320 ms ≤ full × 1.40
- **H2.3**: 9층 헤드 CER > 최종 CER (자기조건화 정상 동작)
- 판정: 서비스 청크는 {320, 640} 중 CER ≤ full × 1.25를 만족하는 가장 짧은 것(둘 다 미달이면 640). H2.1 위반이면 서비스는 Run A를 쓰되 오프라인 비교 보고에 Run B를 병기

## 3단계: 동일 예산 ablation (2026-10-11 규칙, 측정 전)

Run A와 같은 step 수·LR·시드. 우선순위 순서, 예산 초과 시 d→c→b→a 순으로 뺀다.
- (a) InterCTC 단독(6층, w=0.3) vs 자기조건 CTC(Run A). **H3a**: 자기조건 CTC가 dev_mix 상대 5 % 이상 좋고 짝지은 구간이 0을 제외 → 유지, 아니면 InterCTC로 교체
- (b) E-Branchformer 블록(파라미터 ±5 %). **H3b**: |Δ| < 5 % 상대. 교체 조건: dev_mix 개선 ≥ max(0.3, 2s)이고 상대 8 % 이상이며 dev_H 가드 통과
- (c) 폭 384(≈40M): Run A에 과적합 징후(학습 CER < dev_H CER × 0.5)가 없을 때만. 교체 조건 (b)와 같고 메모리·epoch 예산 안
- (d) CR-CTC(α 0.2, time mask 2.5×, max-duration 절반, 같은 벽시계 시간). **H3d**: 이득 ≤ 5 % 상대 → 5 % 초과이고 임계를 넘을 때만 채택

## 4단계: 방출 지연 (2026-10-11 규칙, 측정 전)

- Run A(또는 3단계 승자) 최종 체크포인트에서 15 epoch 이어 학습 3개: 통제, TrimTail(꼬리 0~200 ms 균등 절단), Peak-First CTC(λ는 0단계 소형 실험에서 정규화 항이 CTC 손실의 약 10 %가 되는 값 하나로 고정)
- 기준 정렬: MFA Korean 3.0. 설치 실패 시 Run B의 CTC 강제 정렬로 대체하고 "상대 비교만 유효"로 표시
- 측정: 320·640 ms simulated greedy에서 첫·마지막 토큰 지연 FTD/LTD P50/P90, dev_mix·dev_H CER_ns
- **H4**: 변형 중 하나는 FTD P50을 80 ms 이상 줄이면서 CER_ns 상대 증가 ≤ 3 %
- 판정: CER 증가 ≤ 3 %인 변형 중 지연 감소가 가장 큰 것을 채택, 없으면 통제 유지

## 5단계 (선택): 끝점 뒤 2차 디코딩 (2026-10-11 규칙, 측정 전)

- 후보: greedy / prefix beam 8(LM 없음) / prefix beam 8 + 음절 4-gram(순수 Python, 보간 Kneser-Ney). LM 텍스트는 학습 전사 + Zeroth LM 코퍼스 부분집합(2~5M 문장); 모든 dev·test 정답 문장과 정규화 후 정확히 일치하는 문장은 제거하고 건수를 보고. dev_H는 LM 가중 선택에 쓰지 않는다
- 격자 α∈{0.3, 0.5, 0.8}, β∈{0, 0.5, 1.0}은 dev_mix에서만
- **H5**: dev_mix에서 LM이 greedy 대비 상대 5 % 이상 개선
- 판정: dev_mix 상대 3 % 이상 개선이고 세그먼트 종료 시 2차 디코딩 RTF ≤ 0.3이면 서비스 옵션 기본 켬, 아니면 끔

## 6단계: 스트리밍 서비스 (2026-10-11 규칙, 측정 전)

- 구성: WebSocket 서버(16 kHz PCM 320 ms 청크 수신 → 캐시 스트리밍 인코더 → greedy 부분 자막 → 끝점에서 세그먼트 확정) + 브라우저 페이지(마이크)
- 끝점 규칙: blank 사후확률 ≥ 0.8 프레임을 묵음으로 세고, 토큰 뒤 묵음 {0.8, 1.2} s 가운데 dev에서 "조기 끝점 삭제 CER 증가 ≤ 0.2"를 만족하는 짧은 쪽을 1회 선택해 고정; 무조건 묵음 2.4 s, 세그먼트 20 s. 끝점 뒤 캐시 초기화 vs 유지는 dev에서 1회 비교해 고정
- 측정표(02 §4.5): 청크 {160, 320, 640} ms × 좌측 {64, 128} × {CER_ns, 알고리즘 지연(평균·최대, ms), 청크당 계산 p50/p90(GPU, CPU 1스레드)}, 실제 스트리밍 CER vs simulated CER(dev_H 전수), UPWR/UPSR, EP50/EP90, 마이크→자막 종단 지연 p50/p90(로컬), 10분 연속 입력의 메모리·지연 드리프트
- **H6.1**: 실제 스트리밍 CER_ns − simulated CER_ns ≤ 0.1 절대 → 위반이면 캐시 버그로 보고 수정 전 진행 금지
- **H6.2**: 320 ms 청크 계산 p90 ≤ 50 ms(GPU), ≤ 250 ms(CPU 1스레드)
- **H6.3**: greedy 부분 결과 UPWR = 0
- 서비스 경로 CER == 6단계 측정 CER ±0.1

## 7단계: 봉인 test (2026-10-11 규칙, 측정 전)

- 2~5단계 판정으로 확정한 구성 하나로, Zeroth 공식 test·Zeroth test_H·FLEURS test·CV22 test를 각 1회: full greedy(앵커 비교), 선택 청크 실제 스트리밍 greedy, 5단계 채택 시 2차 디코딩
- 보고: CER_ns·CER_ws·OOV 하한·부트스트랩 구간·체크포인트·디코딩 모드, fp16 발산 횟수 표, GPU 시간 사용 내역
- **H7.1**: Zeroth 공식 test full CER_ns가 4.48(turbo 제로샷)보다 낮고 1.96(turbo+LoRA)보다 높다
- **H7.2**(판정 없음, 범위 보고): FLEURS test 한글 전용 12~30 %, CV22 test 한글 전용 15~40 %, Zeroth test_H는 공식 test보다 높을 것
- test 뒤에는 모델·규칙을 바꾸지 않는다

## 규칙 변경
(없음)
