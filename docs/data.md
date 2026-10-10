# 데이터

계정이나 약관 동의 없이 받을 수 있는 공개 한국어 음성만 쓴다. 받은 파일의 정확한 커밋과 목록은 `reports/data/download-manifest-2026-10-11.json`, 통계는 `reports/data/inventory-2026-10-11.json`(`scripts/inventory_data.py`로 다시 만들 수 있다)에 있다. 데이터 자체는 `data/`에 두고 커밋하지 않는다.

## 출처와 라이선스

| 이름 | 출처 | 라이선스 | 게이트 | 내용 |
|---|---|---|---|---|
| Zeroth-Korean | HF `kresnik/zeroth_korean` (parquet, 16 kHz FLAC) | CC BY 4.0 | 없음 | 낭독 음성. train 22,263발화 51.7 h(105명), test 457발화 1.19 h(10명) |
| FLEURS ko_kr | HF `google/fleurs` `refs/convert/parquet` 브랜치 | CC BY 4.0 | 없음 | 위키 문장 낭독. train 2,307발화 7.93 h, validation 226발화 0.77 h, test 382발화 1.33 h |
| **Common Voice 22 ko** | HF `fsicoli/common_voice_22_0` (미러, mp3 32 kHz tar + tsv, ko 파일만 190 MB) | CC0 1.0 | 없음 | 크라우드소싱 낭독. validated 1,466발화 2.27 h(154명), 그중 train 519발화 0.91 h / dev 474발화 0.69 h / test 472발화 0.67 h, other(미검증) 3,813발화 5.36 h, invalidated 399발화. **이것을 쓴다** |
| Common Voice 17 ko | HF `fsicoli/common_voice_17_0` | CC0 1.0 | 없음 | 22가 못 쓰이게 될 때의 폴백. validated 1,046발화 1.63 h(90명), train 376 / dev 330 / test 339, other 2,057발화 3.39 h |

받는 법: `uv run python scripts/download_data.py` (약 5.5 GB; `--only` 로 일부만). 스크립트가 게이트 여부를 먼저 확인하고 게이트가 있으면 멈춘다.

쓰지 않는 것: KsponSpeech(AI Hub 승인·내국인 전용), ClovaCall(학술 신청), OpenSLR 58/97/113·KSS(NC 또는 ND 라이선스), Common Voice 공식 저장소(2025-10부터 Mozilla Data Collective 계정 필요).

## 재고 조사 결과 (2026-10-11, `scripts/inventory_data.py`)

| 코퍼스 | 분할 | 발화 | 시간 | 고유 문장 | 음절 종류 | 글자/초 p50 / p95 / 최대 | 길이 평균 / p95 / 최대 (s) |
|---|---|---|---|---|---|---|---|
| Zeroth | train | 22,263 | 51.68 | **3,003** | 1,202 | 5.0 / 6.1 / 7.7 | 8.4 / 13.3 / 20.7 |
| Zeroth | test | 457 | 1.19 | 423 | 833 | 4.6 / 5.5 / 7.4 | 9.4 / 14.7 / 20.5 |
| FLEURS | train | 2,307 | 7.93 | 1,347 | 1,022 | 3.6 / 5.1 / 6.9 | 12.4 / 19.1 / 31.0 |
| FLEURS | validation | 226 | 0.77 | 129 | 586 | 3.7 / 5.2 | |
| FLEURS | test | 382 | 1.33 | 270 | 736 | 3.6 / 5.3 / 9.3 | 12.6 / 18.8 / 25.8 |
| CV22 | train | 519 | 0.91 | 519 | 791 | 3.9 / 5.0 | |
| CV22 | dev | 474 | 0.69 | 474 | 772 | 4.4 / 5.7 | |
| CV22 | test | 472 | 0.67 | 472 | 764 | 4.2 / 5.8 | |
| CV22 | validated(전체) | 1,466 | 2.27 | 1,465 | 1,001 | 4.2 / 5.6 | |
| CV22 | other | 3,813 | 5.36 | 3,811 | 1,149 | 3.9 / 7.8 | |

글자/초는 공백을 뺀 글자 수를 길이로 나눈 값이다. FLEURS는 같은 문장을 여러 사람이 읽어 고유 문장이 발화 수보다 적다.

### 문장 겹침 (정확 일치)

| 비교 | 결과 |
|---|---|
| **Zeroth test 문장이 Zeroth train에 있는가** | **423개 중 422개(발화 기준 99.8 %)가 train에 그대로 있다.** train은 3,003개 문장을 105명이 나눠 읽은 것이고, test는 그 가운데 423개 문장을 다른 10명이 읽은 것이다. 즉 공식 test는 "본 문장·못 본 화자" 시험이다 |
| FLEURS test·validation ↔ train | 0개 |
| CV22 test·dev ↔ train | 0개 |

의미: Zeroth test의 공개 수치(ESPnet CER 1.4, Whisper 파인튜닝 1.5~2.0, whisper-ko-ft의 1.96 %)는 문장을 train에서 본 조건의 값이다. 처음부터 학습하는 모델은 3,003개 문장을 외우는 것만으로 이 test에서 높은 점수를 낼 수 있으므로, **Zeroth 공식 test는 비교용으로만 보고(표시를 붙여서), 판정은 문장이 겹치지 않는 평가로 한다**:
- Zeroth 안에서 문장 단위로 떼어 둔 세트(못 본 문장·본 화자): 3,003개 문장 가운데 일부를 통째로 빼서 만든다(규칙은 `docs/experiments.md`에 정한다)
- FLEURS test(못 본 문장·못 본 화자·다른 도메인), CV22 test(못 본 문장·못 본 화자)

### 음절

- 학습 분할(Zeroth train + FLEURS train + CV22 train) 전체의 음절 종류는 **1,321개**(현대 한글 11,172자 가운데). Zeroth train만은 1,202개
- 평가 분할에서 학습에 없는 음절: Zeroth test 0개, FLEURS test 19토큰(16,849개 중, 12종), CV22 test(실측값은 `reports/data/inventory-2026-10-11.json`). 미등장 음절은 `<unk>`로 두어도 CER 손실이 0.1~0.3 % 수준이다
- 글자/초 최대는 Zeroth 7.7, FLEURS 9.3, CV other 12.5(길이 105 s짜리 이상치 포함). 서브샘플링 8배(초당 12.5프레임)에서는 CTC가 같은 글자 반복에 blank를 넣어야 하므로 최대 글자율 근처 발화는 손실이 무한대가 될 수 있다. 4배(초당 25프레임)면 여유가 있다. 8배를 쓰려면 `(프레임 수 − 반복 수) < 글자 수`인 발화를 걸러야 한다(수는 전처리 단계에서 센다)

### 표기 차이

| | 숫자 | 로마자 | 문장부호 |
|---|---|---|---|
| Zeroth | 없음(한글로 풀어 씀, 예 "이천 십 팔 년") | 없음 | 없음 |
| FLEURS `transcription` | **발화의 26.8 %에 아라비아 숫자**(정규화 열인데도 숫자는 남아 있음) | 15.1 % | 15.3 % |
| CV22 | 없음 | 거의 없음 | **대부분 문장부호 포함**(마침표 등; CV17 85 %) |

통일 규칙(초안, `docs/experiments.md`에서 확정): 문장부호는 모두 지운다. 로마자는 그대로 둔다(학습 어휘에 로마자 포함). 숫자는 한국어 읽기가 둘(한자어·고유어)이라 자동 변환이 틀리기 쉬우므로, **숫자가 든 발화는 학습·평가에서 뺀다**(FLEURS train 2,307개 중 약 620개 손실, 약 2 h). 뺀 수는 보고한다.

### 샘플레이트·길이

- Zeroth·FLEURS 16 kHz. Common Voice mp3는 32 kHz라 16 kHz로 다시 샘플링한다
- 20 s가 넘는 발화: Zeroth 최대 20.7 s, FLEURS 최대 31 s, CV other 최대 105 s. 학습에서는 길이 상한(초안 20 s)을 두고 뺀 수를 보고한다

## 학습·검증·평가 분할

규칙은 `docs/experiments.md` "공통"에 있다(2026-10-11 확정).

| 용도 | 구성 |
|---|---|
| 학습 | Zeroth train − 문장 홀드아웃 H(고유 문장 10 %) + FLEURS train(숫자·라틴 발화 제외) + CV22 validated − dev − test ≈ 53 h(제외 수는 0단계에서 보고) |
| dev_mix(1차 선택) | FLEURS validation(한글 전용) + CV22 dev(한글 전용) |
| dev_H(2차 가드) | Zeroth 홀드아웃 H의 절반(문장 해시 기준) |
| 봉인 test(7단계 1회) | Zeroth 공식 test(본 문장·못 본 화자 표시), Zeroth test_H(H의 나머지 절반, 못 본 문장), FLEURS test, CV22 test |

Common Voice `other`(미검증 5.4 h)와 `invalidated`는 쓰지 않는다.
