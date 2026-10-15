# 계획

한국어 음성 인식 모델을 처음부터 설계·학습해 스트리밍 자막 서비스로 묶는다. 구조 결정과 근거는 `docs/survey/README.md`, 측정 규칙은 `docs/experiments.md`, 데이터는 `docs/data.md`.

## 조건
- 데이터: 공개 한국어 낭독 음성 약 53 h(Zeroth-Korean, FLEURS ko, Common Voice 22 ko). 사전학습 가중치·외부 모델 산출물(의사 라벨, 합성 음성)은 쓰지 않는다
- 장비: RTX 2080 Ti 11 GB(fp16만) 한 장, Windows 11, 순수 PyTorch 2.11 + CUDA 12.8
- 측정: 후보와 판정 규칙을 먼저 커밋하고, dev로 고르고, 봉인 test는 7단계에서 한 번만 잰다

## 구조 (요약)
청크 인과 Conformer-CTC, 약 21M. 12층 d=256, 인과 Conv2d 4배 서브샘플링(look-ahead 0), RoPE + SDPA, conv 모듈 LayerNorm·k=15·BACON 반반 분할, U2식 청크 경계 마스크와 동적 청크 학습, 최종 CTC + 중간 CTC, 한글 음절 출력, 스트리밍 greedy + 규칙형 끝점. 처음 결정(`docs/survey/README.md` "결정")은 자기조건 CTC(3·6·9층)였고, 측정으로 바뀐 최종 모델은 InterCTC 6층(w 0.3, 피드백 없음; 3단계 (a) 판정) + TrimTail 15 epoch 이어 학습(4단계 판정)이다(`docs/experiments.md`).

## 단계

| 단계 | 내용 | 산출물 |
|---|---|---|
| 0 | 데이터·정규화·어휘·분할, 모델 코드, 단위 테스트(미래 누출, 스트리밍==오프라인, CTC dtype, 과적합), 11 GB 마이크로벤치(max-duration별 메모리·step 시간) | `src/stt_scratch/`, `manifests/`, `reports/stage0/`, experiments.md 0단계 결과 |
| 1 | fp16 안정성·peak LR 선택: {1e-3, 2e-3} × 3시드 × 5 epoch. 시드 표준편차 s 추정 | `reports/stage1/` |
| 2 | 주 모델 학습 Run A(동적 청크 통합 모델) 100 epoch(예산 규칙), 통제 Run B(오프라인 전용) | `outputs/`(커밋 안 함), `reports/stage2/` |
| 3 | 동일 예산 ablation: (a) InterCTC 단독 vs 자기조건 CTC, (b) E-Branchformer 블록, (c) 폭 384, (d) CR-CTC. 예산 초과 시 d→c→b→a 순으로 제거 | `reports/stage3/` |
| 4 | 방출 지연 제어: 통제 / TrimTail / Peak-First 15 epoch 이어 학습, 강제 정렬 기준 FTD/LTD | `reports/stage4/` |
| 5 | (선택) 끝점 뒤 2차 디코딩: prefix beam + 음절 4-gram, dev에서만 격자 | 생략(`docs/experiments.md` 5단계 판정) |
| 6 | 스트리밍 서비스: WebSocket 서버 + 브라우저 페이지, 지연·안정성·끝점 측정표 | `service/`, `web/`, `reports/stage6/` |
| 7 | 봉인 test 1회(Zeroth 공식·Zeroth 문장 홀드아웃·FLEURS·CV22), README 정리 | `reports/final/`, README |

각 단계 끝에 Codex 읽기 전용 검토를 받고(`work/relay/`), 지적과 반영을 `docs/ai-verification.md`에 남긴다.

## 하지 않는 것
- Zipformer·pruned RNN-T·AED 2-pass·Mamba 재구현(근거는 `docs/survey/README.md` §4~5)
- 온라인 배포·컨테이너 이미지(서비스는 로컬에서 띄우는 데모)
- 인증(로컬 데모)
