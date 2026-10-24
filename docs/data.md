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

### 귀속 (2026-10-24 확인)

이 저장소가 다시 배포하는 데이터 파생물은 전사문과 모델 출력이다: `manifests/*.jsonl`(정규화 전사 `text`, 받은 원문 `text_raw`, 화자 id(FLEURS는 없음), 길이)과 `reports/`의 발화별 가설 파일(`reports/final/*.hyps.jsonl`, `reports/followup/hyps/`, `reports/stage2/hyps/`·`reports/stage3/hyps/`·`reports/stage4/hyps/`는 `ref`·`hyp`, `reports/stage6/client-*.jsonl`은 `ref`·`text`; `ref`는 매니페스트의 정규화 전사, `hyp`·`text`는 이 저장소 모델의 출력). 오디오는 다시 배포하지 않는다(`data/`는 커밋하지 않음). CC BY 4.0은 저작자 표시·라이선스 링크·변경 표시를 요구한다(라이선스 요약: "You must give appropriate credit, provide a link to the license, and indicate if changes were made." — https://creativecommons.org/licenses/by/4.0/). 아래 인용 문구는 2026-10-24에 원 출처 페이지와 데이터 카드(Zeroth·Common Voice는 받은 커밋의 카드, FLEURS는 `main` 브랜치의 카드)에서 확인해 그대로 옮긴 것이고, 받은 파일의 저장소 커밋 전체 해시는 `reports/data/download-manifest-2026-10-11.json`에 있다.

- **Zeroth-Korean** — 라이선스 [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). 원 출처 OpenSLR SLR40 (https://www.openslr.org/40/): "Korean Open-source Speech Corpus for Speech Recognition by Zeroth Project (https://github.com/goodatlas/zeroth)", "This project was developed in collaboration between Lucas Jo(@Atlas Guide Inc.) and Wonkyum Lee(@Gridspace Inc.)." 받은 곳은 HF `kresnik/zeroth_korean`(커밋 `1fe9378`, parquet 변환본)이고, 그 데이터 카드는 "The uploader has transferred the publicly available data from openslr.org to the Hugging Face Hub and did not participate in the creation of the dataset."라고 적는다
- **FLEURS (ko_kr)** — 라이선스 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)(데이터 카드 메타데이터 `cc-by-4.0`, 본문 "All datasets are licensed under the Creative Commons license (CC-BY)."). 출처 Google의 HF `google/fleurs` (https://huggingface.co/datasets/google/fleurs; 받은 것은 `refs/convert/parquet` 브랜치 커밋 `168de34`). 데이터 카드: "Fleurs is the speech version of the FLoRes machine translation benchmark." 인용: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, Ankur Bapna, "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", arXiv:2205.12446 (2022)
- **Common Voice Corpus 22.0 (ko)** — 라이선스 [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). CC0은 권리 포기(퍼블릭 도메인 헌정)라 저작자 표시 조건이 없고, 출처는 관례로 적는다. 원 출처 Mozilla Common Voice (https://commonvoice.mozilla.org/). 받은 곳은 HF 미러 `fsicoli/common_voice_22_0`(커밋 `ae911de`; 데이터 카드 "This dataset is an unofficial version of the Mozilla Common Voice Corpus 22. It was downloaded and converted from the project's website https://commonvoice.mozilla.org/.", 메타데이터 `cc0-1.0`, "Licensing Information" 절 "Public Domain, CC-0"). 인용: R. Ardila 외, "Common Voice: A Massively-Multilingual Speech Corpus", LREC 2020, pp. 4211–4215

**변경 표시**: 세 코퍼스의 전사문(매니페스트의 `text`, 가설 파일의 `ref`)은 모두 `docs/experiments.md` "텍스트 정규화"를 거친 것이다: NFC, 한글 음절, 라틴 문자(A–Z, a–z), 숫자(0–9), 공백이 아닌 글자(문장부호·기호와 FLEURS의 한자·악센트 붙은 라틴 문자 등)는 공백으로, 연속 공백은 하나로 줄이고 양끝 공백은 지운다(`stt_scratch.text.normalize_train`). 매니페스트에는 받은 원문도 `text_raw`로 함께 둔다(Zeroth parquet의 `text`, FLEURS parquet의 `transcription`, Common Voice tsv의 `sentence`). 정규화로 실제로 바뀐 행은 Common Voice 1,258/1,466(모두 문장부호), FLEURS 445/2,915, Zeroth 0/22,720이다. 분할도 원본과 다르다: Zeroth train을 문장 홀드아웃 H로 나눠 dev_H·test_H를 만들고, FLEURS train에서 숫자·라틴 문자·20 s 초과 발화를 학습에서 빼며(`manifests/train_excluded.jsonl`), Common Voice는 validated − dev − test를 학습으로 쓴다(아래 "학습·검증·평가 분할"). 가설(`hyp`·`text`)은 이 저장소의 모델이 낸 출력이다. 이 귀속은 데이터에 대한 것이며 이 저장소 코드의 라이선스와는 별개다.

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
| CV22 test·dev ↔ 이 프로젝트 학습 세트, 정규화 뒤 비교 (2026-10-24 추가 확인) | test 1발화(472 중, 다른 화자), dev 0개. 아래 설명 |

2026-10-24 추가 확인: 위 표의 처음 세 행은 `scripts/inventory_data.py`가 각 코퍼스 공식 분할의 받은 전사(Zeroth `text`, FLEURS `transcription`, CV `sentence`)를 이 프로젝트의 정규화 없이 정확 일치로 비교한 값이다(CV22는 공식 `train.tsv` 519발화 ↔ `test.tsv`·`dev.tsv`; 이 프로젝트의 CV22 학습 세트 validated − dev − test는 오디오가 없는 1발화를 뺀 뒤 공식 `train.tsv`와 같은 519클립이다). 매니페스트의 정규화 전사(`docs/experiments.md` "텍스트 정규화", 문장부호 제거)로 비교하면 CV22 test 1발화 `common_voice_ko_37195158`이 학습 세트에 든 다른 화자의 클립 `common_voice_ko_36899449`(공식 `train.tsv`에도 있음)와 같은 문장이다(채점 정규화 `normalize_score`로 비교해도 같은 결과). 두 클립의 원문은 끝 마침표 하나만 달라 Common Voice도 다른 문장(`sentence_id`)으로 다루며, 원문 정확 일치 비교가 0개였던 이유도 이것이다. CV22 dev, FLEURS test·validation, Zeroth dev_H·test_H는 정규화 뒤에도 학습 세트와 겹치는 문장이 없다(Zeroth 공식 test는 홀드아웃 H를 뺀 학습 세트와 397/457발화가 겹쳐 7단계 사후 분할과 같은 수). 영향이 472발화 중 1발화라 봉인 수치(`docs/experiments.md` "7단계 결과")는 그대로 두고 학습 세트도 바꾸지 않는다(dev·test와 정규화 문장이 같은 클립을 학습에서 빼려면 규칙을 먼저 등록하고 다시 학습·측정해야 하는 별도 결정이다).

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

~~통일 규칙(초안): 문장부호는 모두 지운다. 로마자는 그대로 둔다(학습 어휘에 로마자 포함). 숫자가 든 발화는 학습·평가에서 뺀다.~~ **이 초안은 폐기**했다. 확정 규칙은 `docs/experiments.md` "텍스트 정규화"·"어휘"다: 문장부호는 지우고, 어휘는 한글 음절만이며 라틴 문자가 하나라도 든 발화와 숫자가 든 발화는 **학습에서만** 뺀다(FLEURS train에서 숫자 619·라틴 200건). 평가에서는 그대로 두고 전체·한글 전용 두 수치를 보고한다.

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
