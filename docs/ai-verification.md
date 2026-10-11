# AI가 만든 코드의 검증 기록

이 프로젝트의 코드와 문서는 AI 코딩 도구(Claude Code가 설계·구현·측정·문서, Codex가 읽기 전용 독립 검토)가 만들었다. 기능마다 (1) 무엇을 만들었고, (2) 어떻게 검증했고, (3) 검증에서 무엇이 잡혔는지를 날짜와 함께 남긴다. 사람이 정한 것은 따로 적는다.

## 2026-10-10~11 조사 (`docs/survey/`)
- **만든 것**: 네 주제(인코더, 스트리밍, 목적 함수·디코딩·출력 단위, 소규모 데이터·한국어 데이터)의 문헌 조사 4편. 모든 수치는 arXiv 원문·공식 저장소·데이터 카드에서 읽고 출처를 달았다
- **검증**: 결정에 쓰인 핵심 주장 13건(C1~C13)을 별도 검증자가 반박하려는 자세로 원문을 다시 열어 대조했고, 누락 논문을 세 주제로 따로 찾았다(`docs/survey/README.md` §2~3). 설계안 4건을 서로 다른 관점으로 만들고 심사 3건으로 채점했다(§4~5)
- **잡은 것**: 수치 오류는 없었고 조건 표기 13곳을 고쳤다. 예: icefall 스트리밍 수치가 simulated 값이라는 점, E-Branchformer 비교가 dev 수치라는 점과 "발산 다수"가 과장이라는 점, 보조 transducer 지연 342 ms는 CTC 헤드의 값이라는 오독, RC 혼합 학습이 RC0 디코딩을 퇴보시킨다는 점, k2가 Windows에서 사실상 불가하다는 점. 각 문서 끝 "검증 후 정정" 절에 변경 목록이 있다
- **사람이 정한 것**: 논문을 먼저 읽고 구조를 고른다는 순서, 공개 데이터만 쓴다는 것, 스트리밍 서비스까지가 완료 기준이라는 것, 조사 뒤에는 Claude가 구조를 정한다는 것

## 2026-10-11 데이터 재고 (`scripts/inventory_data.py`, `docs/data.md`)
- **만든 것**: Claude Code. 세 코퍼스의 시간·고유 문장·음절 종류·글자율·문장 겹침·OOV를 세는 스크립트
- **검증**: 실제 데이터로 실행해 수치를 `reports/data/`에 남김. HF API로 게이트·라이선스 확인
- **잡은 것**: Zeroth 공식 test 문장의 99.8 %가 train에 있음(조사 문서가 "확인 필요"로 남긴 항목). 이 발견으로 평가 설계를 바꿨다(문장 홀드아웃 + 다른 도메인 test). FLEURS `transcription` 열에 숫자가 27 % 남아 있음. CV mp3는 32 kHz

## 2026-10-11 모델·학습 코드 (`src/stt_scratch/`)
- **만든 것**: Claude Code. 인과 Conv2d 서브샘플링, RoPE + SDPA 청크 마스크 어텐션, BACON식 분할 depthwise conv, 자기조건 CTC 헤드, 동적 청크 샘플링, 캐시 스트리밍 추론, fp32 CTC 손실, greedy 디코딩, 길이 버킷 배치, 학습 루프(fp16 autocast, GradScaler, warmup+cosine, 발산 판정), 평가(full / simulated / real streaming, CER_ns·CER_ws, 부트스트랩)
- **검증**: `tests/test_model.py` 21개(0단계 게이트). 미래 누출(청크 4/8/16/32 × 좌측 64/128에서 미래 프레임을 난수로 바꿔도 현재 청크까지 출력 불변), 스트리밍 == 오프라인 청크 마스크(fp32 최대 차이 1e-5, 부분 마지막 청크 포함, BACON on/off), 배치 패딩이 유효 프레임에 영향 없음, 스트리밍 fbank == 오프라인 fbank, autocast 아래 CTC 입력 dtype float32, 청크 경계를 넘는 greedy 반복 축약, 합성 발화 20개 과적합(CER 0). 서브샘플링 길이 공식은 모듈 출력과 대조
- **잡은 것**
  - 스트리밍 어텐션 캐시가 첫 청크 뒤에도 비어 있는 채로 남는 버그(캐시 길이 0일 때 새 캐시를 0길이로 자름)를 테스트를 쓰기 전 코드 검토에서 발견해 "블록이 전체 K/V 입력을 돌려주고 호출자가 좌측 문맥 길이로 자르는" 구조로 고쳤다
  - 배치 패딩 프레임이 어텐션에서 모든 키가 막혀 NaN이 되고, 그 NaN이 BACON의 청크 내 우측 탭을 통해 유효 프레임으로 새어 들어올 수 있는 경로를 설계 검토에서 찾아, 패딩 쿼리 행은 자기 자신을 보게 하고 conv 입력의 패딩 프레임은 0으로 두게 했다(`test_padding_does_not_change_valid_frames`)
  - `subsampled_length`를 이중 음수 나눗셈으로 쓴 것을 ruff(B002)가 잡아 명시적 올림 식으로 바꿨다
  - 체크포인트 로드가 `weights_only=False` 기본값이라는 보안 경고(플러그인 훅)를 받아 `weights_only=True`로 바꿨다
- **사람이 정한 것**: 구조(docs/survey/README.md "결정"), 허용 오차(fp32 1e-5), 20 s 길이 상한, 판정 규칙 전부(docs/experiments.md)

## 2026-10-11 데이터 준비 (`scripts/prepare_data.py`)
- **만든 것**: Claude Code(배경 에이전트). 정규화·제외·홀드아웃 해시·어휘·PCM 저장소·요약 보고서
- **검증**: `tests/test_prepare_data.py`(해시 규칙 결정성, dev_h/test_h 분리, CTC 프레임 공식, CV 경로 분리) + 실제 실행 결과 `reports/stage0/data_summary.md`
- **잡은 것**: FLEURS `id`는 문장 id라 발화마다 유일하지 않아(최대 3명이 같은 문장) id에 parquet 행 번호를 붙였다. CV22 validated 1행은 어느 tar에도 mp3가 없어 `missing_audio`로 제외했다(규칙에 없던 사유를 추가해 기록). mp3 헤더의 프레임 수가 실제 디코딩 길이와 달라(표본 40/40) 헤더 대신 디코딩 길이를 쓰게 했다. 두 번 실행해 매니페스트·어휘의 SHA-256이 같음을 확인했다

## 2026-10-11 0단계 마이크로벤치 (`scripts/bench_stage0.py`)
- **만든 것**: Claude Code. max-duration별 피크 메모리·step 시간·데이터 대기 비율, SDPA 커널 탐지
- **검증**: 실제 학습 step으로 측정(`reports/stage0/bench.json`), 프로파일러로 GPU 시간과 벽시계를 분리
- **잡은 것**: 첫 측정이 1.36 s/step으로 예상(5~8분/epoch)을 크게 넘겼다. 프로파일러에서 GPU 시간은 0.14 s뿐이고 `convolution_backward`의 CPU 시간이 컸다 → 배치 모양이 매번 달라지는데 `cudnn.benchmark=True`가 step마다 알고리즘 탐색을 돌린 것. 끄자 0.66 s/step. 또 분할 conv의 unfold 구현이 (B, C, T, 15) 텐서를 매 층 만들던 것을 grouped conv1d + 청크 끝 보정으로 바꾸고, 보정 루프의 `.any()` 호스트 동기화 84회/forward를 없앴다. 독립 unfold 구현과 대조하는 테스트를 추가했다
- **사람이 정한 것**: 없음(예산 규칙은 사전 등록대로 적용)

## 2026-10-11 1단계 첫 시도 발산 → CTC 커널 버그 (`src/stt_scratch/model/ctc.py`)
- **만든 것**: Claude Code. 1단계 sweep 첫 시도(LR 1e-3, 시드 1·2·3)가 모두 100~120 step에서 GradScaler scale이 1e-12까지 떨어지며 발산 규칙에 걸렸다. 실패 배치를 저장해(`work/nan_batch.pt`) 원인을 찾았다
- **검증**: 같은 배치를 fp32로 돌려도 16시드 중 14번 NaN이 나와 fp16 문제가 아니었다. CTC 손실만 떼어 발화 하나씩 계산한 기준 기울기와 비교했다
- **잡은 것**: 두 가지. (1) 패딩 프레임(out_length 이후)의 CTC 기울기를 CUDA 커널이 초기화하지 않아 쓰레기 값이 역전파됐다 → `torch.where`로 패딩 프레임을 가려 역전파에서 버린다. (2) `log_probs.transpose(0, 1)`의 **비연속 뷰**를 그대로 넘기면 CUDA CTC backward가 유효 프레임에서도 기준과 최대 55 차이 나는(비결정적) 기울기와 NaN을 냈다 → `.contiguous()`로 (T, B, V) 연속 텐서를 넘기고, 정답은 (B, S) 패딩 int32 CPU 텐서, 길이도 int32 CPU로 넘긴다(1차원 정답 + CUDA int32 길이 조합은 기준과 1e-3 차이, 패딩 레이아웃은 1.2e-7). 수정 뒤 같은 배치 16시드에서 NaN 0회(fp16·fp32 모두). 무효 처리한 세 런의 요약은 `reports/stage1/attempt1-invalid-summary.json`에 남기고 출력은 지웠다. 발산 횟수 표에는 "코드 버그로 무효"로만 적고 H1 판정에 넣지 않는다
- **사람이 정한 것**: 없음

## 2026-10-11 Codex 0단계 독립 검토 반영
- **방식**: Codex 읽기 전용 샌드박스가 로컬 파일을 못 읽어 `work/relay/run-codex-stdin.ps1`로 소스 전문(줄 번호 포함)과 규칙 문서를 stdin으로 넣어 정적 검토를 받았다(`work/relay/2026-10-11-0204-stage0-review-full-codex.md`, 16건: P1 2, P2 11, P3 3)
- **반영(코드)**: ① CUDA fp16 실제 크기(12층 d=256, 1,288 클래스) 스트리밍 == 오프라인 테스트(≤ 1e-2, 25청크로 128프레임 캐시 절단 포함), fp16 미래 불변 테스트, 원시 음성→fbank→인코더 경로 테스트, `allclose`를 최대 절대차로 교체 ② dev_mix CTC 손실을 매 epoch 재서 "epoch 3 ≥ epoch 1"이면 발산(`train.py`) ③ BACON 우측 탭을 "더한 뒤 빼기"에서 "가린 뒤 더하기"로 바꿔 fp16에서도 미래 값이 반올림에 들어가지 않게 함(fp16 테스트는 비트 단위 동일) ④ 실제 스트리밍 평가도 autocast fp16으로 통일하고 결과에 `precision` 기록, `--fp32` 옵션 ⑤ 재개 시 난수 상태·발산 카운터·dev 손실 이력 복원, 중간 epoch 체크포인트는 재개 거부 ⑥ `step`(시도)·`updates`·`skipped` 분리 기록 ⑦ warmup 2,000(규칙 변경 1) ⑧ 첫 패킷 5,360 샘플(평가·클라이언트·브라우저) ⑨ 라틴 규칙 문구 ⑩ 어휘를 CTC 제외 뒤에 생성 ⑪ 평가 세트 누락 시 중단 ⑫ `<unk>` → U+318D ⑬ `--no-sc-feedback`(InterCTC 단독 ablation) ⑭ SpecAugment 시작점 상한 포함(마지막 bin·프레임도 가려짐) ⑮ 좌측 문맥 0 거부 ⑯ 주석·문서 정정(carry 길이, 테스트 수 29, greedy 반복 테스트가 청크 경계를 실제로 가로지르게, data.md 초안 폐기 표시)
- **반영(규칙)**: `docs/experiments.md` "규칙 변경" 1~10
- **검증**: 전체 테스트 56개 통과(`tests/test_model.py` 38, `tests/test_prepare_data.py` 13, `tests/test_service.py` 5). 수정 뒤 학습 루프 12 step 연기 실행과 중간 체크포인트 재개 거부, 체크포인트 난수 상태 왕복(`weights_only=True`) 확인
- **잡은 것(반영 중)**: `weights_only=True` + `map_location=cuda`가 난수 상태 바이트 텐서도 GPU로 옮겨 `set_rng_state`가 실패 → `.cpu()`로 되돌린다. `SplitDepthwiseConv1d.half` 속성이 `nn.Module.half()`를 가려 테스트에서 `.to(torch.float16)`을 쓴다
- **사람이 정한 것**: 없음(모두 검토 지적의 반영이며 판정 규칙의 뜻은 바꾸지 않음)

## 2026-10-11 1단계 2차 시도 전부 발산 → 진단 (`reports/stage1/diag-2026-10-11.md`)
- **만든 것**: Claude Code. 2차 시도 6런이 모두 발산 판정(5런은 dev 손실 규칙, 1런은 scale 붕괴)을 받아 원인을 찾는 탐색 런 7개: 이어 학습, 실제 발화 20개 과적합, 어텐션 로짓·가중치 노름 추적, QK-norm 파일럿, 정규화 끄기(P1)·SpecAugment만 끄기(P5)·커리큘럼(P6). 모두 "판정에 쓰지 않는 진단"으로 표시
- **검증**: PCM 저장소를 원본 parquet 음성과 비트 단위로 대조(37발화 불일치 0), 인덱스·길이 불일치 0 / 21,853. 실제 발화 20개 과적합이 150 step에 CER 0 → 데이터·특징·모델·CTC 경로 정상
- **잡은 것**: ① 규칙 변경 3의 "epoch 3 vs epoch 1" 비교가 warmup 구간과 겹쳐 정체와 미학습을 구분하지 못함 ② 1단계 cosine 스케줄 길이를 5 epoch로 잡아 peak LR 유지 구간이 없었음(`--schedule-steps`로 2단계 스케줄을 잘라 쓰게 고침) ③ LR 7e-4 이상에서 어텐션 로짓 표준편차 2 → 13~27 → 60~111 폭주, 기울기 노름 37k(QK-norm으로 1~3에 묶임) ④ **진짜 원인은 처음부터 건 SpecAugment**: 끄면 epoch 2에 dev 손실 96·CER 0.57로 정렬이 시작되고, 속도 교란·동적 청크는 무관(P5). 처방은 규칙 변경 13(QK-norm 기본, SpecAugment epoch 3부터, 2단계 스케줄 절단) ⑤ 운영: GPU 작업 두 개를 동시에 띄우면 WDDM 페이징으로 둘 다 멈춤 → 한 번에 하나만. 감시 파이프라인 끝의 `cut`이 블록 버퍼링으로 이벤트를 삼킴
- **사람이 정한 것**: 없음(2차 시도 6런은 "처방 전"으로 기록만 하고 H1에 넣지 않음)

## 2026-10-11 Codex 1단계 독립 검토 반영 (`work/relay/2026-10-11-1014-stage1-review-codex.md`)
- **방식**: 1단계 판정 직후, 2단계 Run A가 시작된 상태에서 experiments.md·진단 기록·train.py·sweep·attention·ctc·evaluate 전문을 stdin으로 넣어 정적 검토(263 s). 결과: 3차 결과를 무효화하거나 Run A를 멈출 P1 없음, P2 11건, P3 1건
- **일치 확인**: 2e-3 선택과 s의 대상 선택이 등록대로, 6런 모두 dev 손실 규칙 통과, dev 손실 계산 조건(eval·full·최종 헤드·학습과 같은 autocast), CTC fp32 경로, QK-norm이 캐시 키에도 동일 적용·미래 정보 없음, `updates + skipped = 3485`, 1·2차 시도 제외의 타당성
- **반영(코드, Run A 처방 불변)**: ① 체크포인트에 처방 인자와 dev 손실 규칙 적용 여부 저장, `--resume`가 처방 불일치면 거부(`--allow-recipe-change`로 명시 override, `meta-resume-*.json`에 기록, 원래 `meta.json` 보존) ② `last.pt` 없는 `--resume` 거부, `--resume`+`--init` 거부 ③ dev 손실 비교는 두 값 모두 유한할 때만 "내려감" ④ sweep: 후보 정의 인자의 `--train-args` 중복 거부, 런별 `meta.json` 대조(`recipe_ok`), 3시드·5 epoch 완료·발산 0만 `adoptable`, 표본표준편차 병기 ⑤ `--augment-start-step`(배치 규칙이 다른 arm용) ⑥ 2단계 보고에 QK-norm affine 크기·블록별 로짓 표준편차 기록
- **반영(문서)**: 규칙 변경 13의 시각을 커밋 시각(07:02, 3차 시작 07:23 전)으로 바로잡고 규칙 변경 머리말에 1~12/13/15~16의 작성 시점을 구분; s의 pstdev가 측정 전 커밋(3dc3a1b)의 정의임을 명시하고 표본표준편차(2.28 임계)는 병기만; 진단의 인과 문구를 관측 범위로 축소(SpecAugment 끄기가 "충분"했고 속도 교란·동적 청크는 "방해가 아니었다", 자기조건 피드백·저 LR 대안은 "시험하지 않음", 로짓 성장은 "함께 관측된 불안정"); "전체가 warmup" → 2,000/2,091 step; P6b가 가중치만 이어받았음을 명시; 파일럿 B vs 3차 시드 1의 직접 비교 추가; 규칙 변경 15(재개·sweep 보호, 커리큘럼 step 2,091 정의, 로그 표본 표기)·16(4단계 이어 학습 처방: SpecAugment 처음부터, warmup 500, 15 epoch cosine, LR 1e-3, dev 손실 규칙 미적용)
- **Run A 감시(검토 권고)**: 5 epoch 통과를 100 epoch 안정성으로 보지 않는다. epoch마다 dev 손실/CER·skipped 증가·scale·`low_scale_streak`를 보고, 등록된 재시작 조건(비유한 손실 3회, scale < 1.0 100 step, scratch 런 epoch 3 규칙)만 LR×0.5 재시작의 근거로 쓴다. 기울기 노름·QK affine 성장은 기록만 한다
- **사람이 정한 것**: 없음

## 2026-10-11 밤 3단계 후보 코드 (`src/stt_scratch/model/ebranchformer.py`, `ctc.cr_consistency`, `scripts/stage3.py`)
- **만든 것**: Claude Code. 2단계 학습이 GPU를 쓰는 동안 CPU에서만 작성·시험. E-Branchformer 블록(MHSA | cgMLP 병렬, concat → depthwise conv → 더하기 → linear 병합, 두 conv 모두 Conformer와 같은 BACON 분할 conv와 패딩 0 처리, 스트리밍 캐시 2개), 블록 선택 옵션, 블록별 스트리밍 캐시 초기화(`init_conv_cache`), SpecAugment 시간 마스크 수 옵션, CR-CTC 일관성 손실(두 뷰를 배치 축으로 이어 1회 forward), 3단계 실행 스크립트(후보 순서·예산·(c) 과적합·메모리 조건·평가·짝지은 비교 자동화, `--dry-run`), 2단계 보고의 발화별 가설 저장과 학습셋 CER, 짝지은 비교 스크립트
- **검증**: 기본값이 Run A 처방과 같음(기존 게이트 테스트 그대로 통과). 새 테스트 9개: E-Branchformer 미래 누출 없음(청크 4/8/16 × BACON 켬/끔), 캐시 스트리밍 == 오프라인(캐시 절단 + 부분 마지막 청크, BACON 켬/끔, fp32 1e-5), 패딩 불변, CR 일관성 성질(같은 뷰면 0, 패딩 프레임 무시, 두 뷰 모두로 기울기). 세 후보 각각 작은 모델로 CPU 2 step 학습 연기 시험, 실행 스크립트 dry-run으로 명령 확인. CPU 68개 통과(CUDA 전용 5개는 GPU가 학습 중이라 건너뜀)
- **잡은 것**: 3단계 예산 추정이 체크포인트 첫·끝 시각 차이를 써서 일시 중지·부하 구간까지 포함하던 것을 epoch 간격 중앙값으로 바꿈. 같은 epoch에서 두 번 재개하면 재개 기록을 덮어쓰던 것을 번호 붙이기로 바꿈
- **사람이 정한 것**: 없음(3단계 설정·예산은 규칙 변경 17로 측정 전 등록)

## 2026-10-12 Codex 2단계 독립 검토 반영 (`work/relay/2026-10-12-0830-stage2-review-codex.md`)
- **방식**: 2단계 판정 직후, 3단계 시작 전에 experiments.md·2단계 보고서·짝지은 비교·3/4단계 코드 전문을 stdin으로 넣어 정적 검토(288 s). P1 2, P2 11, P3 4
- **일치 확인**: 2단계 수치·H2.1/H2.2/H2.3·서비스 청크 판정이 보고서와 맞음, Run B 재시작 흐름, 재개 경로, CR-CTC 뷰·정규화, InterCTC 단독 옵션, E-Branchformer 연결, 4단계 처방·지연 정의
- **반영(P1)**: ① 3단계 실행기가 epoch 로그 줄 수로 완료를 판단하던 것을 체크포인트 기준으로(`train.completed_epochs`), 재개 때 체크포인트 뒤 epoch 기록을 따로 옮김 ② 기존 런을 다시 쓰기 전 `meta.json` 처방을 등록 처방과 대조(`train.expected_recipe`, `build_parser`)
- **반영(P2)**: 평가 단계 실패 확인, 실제 사용 시간 기반 예산과 우선순위 끝부터 빼기, (c) 조건은 비율이 있을 때만, 3단계 발산 재시작(LR×0.5 최대 2회), 이미 학습 중인 런은 기다림, 4단계 재개(`--init`+`--resume` 허용, init은 처방으로 대조), 4단계 시작 모델 구조·손실 옵션을 3단계 정의에서 가져오고 결과 파일에 시작 체크포인트 SHA-256, H4 최종 채택 규칙(320 ms dev_mix)과 공통 발화 FTD 보조 보고, H2.1·H2.3·안정성 통계·fp16 귀인 문구를 관측 범위로 축소, H2.2 짝지은 구간 추가, dev_H 경계를 공통 규칙("넘게")에 맞춤, 학습셋 표본 방식 기록. 규칙 변경 19·20
- **추가 확인(지적 13)**: GPU fp32로 같은 200발화를 재서 argmax 100 % 일치, fp16 불일치 4프레임은 모두 상위 두 후보 차 ≤ 0.0078의 동점임을 확인(`work/diag/stage2_streaming_fp32_gpu.json`)
- **사람이 정한 것**: 없음

## 2026-10-12 새벽 4단계 준비 (TrimTail, Peak-First, MFA 기준 정렬, 지연 측정)
- **만든 것**: Claude Code. 2단계 Run B가 GPU를 쓰는 동안 CPU로 작성. 학습 옵션 `--trimtail-ms`(발화 끝 0~N ms 절단, CTC 길이 조건 유지, 잘린 구간 패딩 값)와 `--peak-first-lambda`(KL(sg(p_{t+1})‖p_t)), Peak-First λ 보정 스크립트, MFA 입력 내보내기, 지연 측정 스크립트(`scripts/stage4_latency.py`: 청크 끝 도착 시각 기준 방출 시각, MFA 형태소를 어절로 다시 묶기, 편집 거리 어절 맞춤, FTD/LTD/MAD P50/P90). 측정 정의는 규칙 변경 18
- **도구 설치**: Miniforge 26.7.2(conda-forge 공식 릴리스, GitHub 릴리스 메타데이터의 SHA-256과 대조)를 `D:\dev-tools\miniforge3`에, MFA 3.4.3 환경을 `D:\dev-tools\envs\mfa`에(패키지 캐시 `D:\dev-cache\conda-pkgs`, 모델 `D:\dev-cache\mfa`). PATH·레지스트리는 건드리지 않음. 한국어 토큰화에 `python-mecab-ko`가 필요해 MFA 환경에만 설치
- **검증**: 새 테스트 7개(Peak-First 항의 성질, TrimTail의 길이 조건·패딩, 지연 도우미 4종). MFA 정렬 dev_mix 633/633, dev_H 1,107/1,113 정렬(6 실패), 어절 묶기 성공 dev_mix 633, dev_H 1,106
- **잡은 것**: Peak-First λ를 증강 없는 eval 모드로 재면 학습 문장 암기 때문에 CTC 손실이 발화당 0.27이라 λ가 2.7e-5로 사실상 0이 됨 → 학습과 같은 조건(dropout·SpecAugment·속도 교란·동적 청크)에서 재도록 규칙 변경 18 문구와 스크립트를 바로잡음(시험 실행 3배치: CTC 9.03, 정규화 항 887, λ ≈ 1.0e-3; 실제 λ는 4단계 시작 모델로 20배치에서 정함). MFA 로그가 Windows 콘솔에서 UTF-16으로 깨져 `PYTHONIOENCODING=utf-8`로 다시 실행
- **사람이 정한 것**: 없음(개발 도구 설치는 사용자 상시 허용 범위)

## 2026-10-11 스트리밍 서비스 코드 (`src/stt_scratch/service/`, `web/index.html`, `scripts/stream_client.py`)
- **만든 것**: Claude Code. 규칙 기반 끝점(blank ≥ 0.8 묵음, rule1 2.4 s / rule2 1.2 s / rule3 20 s), 세션(PCM → fbank → 캐시 스트리밍 → append-only greedy → 끝점), FastAPI WebSocket 서버(바이너리 int16 PCM 입력, partial/final JSON 출력, flush 뒤 청크당 계산 시간 통계), 브라우저 마이크 페이지, 실시간 클라이언트(6단계 지연 측정용)
- **검증**: `tests/test_service.py` 8개(끝점 규칙 시점, 끝점 끈 세션의 최종 텍스트 == 배치 청크 마스크 greedy와 정확한 청크 수, 강제 끝점 뒤에도 모든 mel 프레임이 인코더에 들어가고 패킷마다 청크 1개가 나옴, CUDA에서 fp16 autocast로 도는지, WebSocket 왕복이 stats 메시지로 끝남, 끝점 뒤 세그먼트 전진). 6단계 측정표는 학습 뒤
- **잡은 것**: 아래 적대적 검토 절 참조(끝점에서 버퍼를 버리던 P1 결함 등)
- **사람이 정한 것**: 없음

## 2026-10-11 Codex 반영 코드의 적대적 검토 (Claude 워크플로, 1단계 재시작 전)
- **방식**: 7개 관점(분할 conv 수치, 학습 루프, 평가·서비스, CTC·데이터, 테스트 품질, Codex 16건 커버리지, 스트리밍 정확성)으로 읽기 전용 검토자를 병렬로 돌리고, 발견 23건마다 반박자 2명이 "반박 실패"해야 확정하는 방식. 확정 17건(중복 포함), 쟁점 3건, 반박 3건. 결과 JSON은 `work/`에 보관
- **잡은 것(고침)**
  - **P1 서비스 세션의 끝점 처리**: `_new_segment(reset_cache=True)`가 버퍼의 mel 프레임과 fbank 캐리까지 지워, 끝점마다 다음 세그먼트 음성 약 300 ms가 사라지고 그 뒤 모든 패킷이 32프레임 격자에서 벗어나 첫 청크가 다음 패킷까지 320 ms 늦어졌다(검토자가 소형 모델로 798 mel 프레임 중 33개 손실을 재현). 끝점에서는 greedy·끝점 카운터·인코더 캐시만 초기화하고 버퍼는 두도록 고치고, 강제 끝점 뒤 프레임 수 보존·패킷당 청크 1개를 확인하는 테스트를 추가했다
  - **P2 서비스 경로 정밀도**: 세션이 autocast 없이 fp32로 돌아 평가 경로(fp16)와 달랐다 → 세션에 `amp` 옵션(기본 GPU에서 fp16), 서버 `--fp32`, `/info`에 precision, CUDA에서 dtype을 확인하는 테스트
  - **P2 CER_ws의 `<unk>`**: 학습 정규화가 U+318D를 공백으로 바꿔 CER_ws에서는 `<unk>`가 사라졌다 → 가설 쪽 정규화만 U+318D를 남기는 `normalize_hyp_ws`를 두고 두 지표 모두에서 오류 1자로 세게 하고 테스트 추가(규칙 변경 7 문구 보강)
  - 학습 루프: 같은 `--run`에 로그가 있으면 `--resume` 없이 시작 거부(이전 시도 로그 혼입 방지), 발산 판정을 받은 런은 재개 거부, dev 손실이 NaN이면 "내려가지 않음"으로 처리
  - 테스트: 세션 테스트의 항진 단언(`a == b or a > 0`) 교체, WebSocket 테스트가 final이 없으면 무한 대기하던 것을 stats 메시지까지 유한 루프로, 원시 음성 경로 허용 오차를 등록 게이트 1e-5로(실측 2.9e-6 이하), CTC 손실·기울기를 발화별 기준과 대조하는 회귀 테스트 추가
  - 클라이언트: 계산 시간 p50/p90을 이벤트(토큰이 난 청크 + compute 0인 eos)가 아니라 서버가 flush 뒤 보내는 전체 청크 통계로 계산. 브라우저 정지 시 worklet에 남은 320 ms 미만 음성을 drain 뒤 flush
- **반박된 것(기록만)**: fp16 스트리밍 게이트 1e-2는 무작위 초기화에서만 확인했고 학습된 모델은 희박 클래스 log-prob에서 넘을 수 있다는 지적 → 2단계 끝에 확률 최대 절대차·argmax 일치율을 함께 보고하도록 규칙 변경 11에 적었다
- **검증**: 전체 테스트 62개 통과(`test_model` 41, `test_prepare_data` 13, `test_service` 8)
- **사람이 정한 것**: 없음
