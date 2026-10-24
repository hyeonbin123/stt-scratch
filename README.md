# stt-scratch

한국어 음성 인식(STT) 모델을 처음 설계부터 만들어 공개 데이터로 학습하고, 스트리밍 자막 서비스까지 묶는 프로젝트다. 사전학습 가중치는 쓰지 않는다.

- 모델: 청크 인과 Conformer-CTC 약 21M 파라미터(순수 PyTorch). 인과 Conv2d 서브샘플링, RoPE 어텐션 + QK-norm + 청크 마스크, BACON식 분할 depthwise conv, 동적 청크 학습, 캐시 스트리밍 추론. 최종 모델은 6층 중간 CTC(InterCTC) 헤드와 TrimTail 이어 학습을 쓴다(2단계 자기조건 CTC → 3단계 판정으로 교체, 4단계 TrimTail 채택). 구조 선택의 근거와 기각한 대안은 `docs/survey/README.md`
- 데이터: Zeroth-Korean(CC BY 4.0) + FLEURS 한국어(CC BY 4.0) + Common Voice 22 한국어(CC0). 라이선스·분할·정규화는 `docs/data.md`. `manifests/`와 `reports/`의 전사문·가설은 각 데이터셋의 라이선스를 따른다
- 측정 규칙(사전 등록)과 결과: `docs/experiments.md`. 단계 계획: `docs/plan.md`. AI가 만든 코드의 검증 기록: `docs/ai-verification.md`
- 하드웨어: RTX 2080 Ti 11 GB 한 장(fp16 autocast, bf16 없음), Windows 11

## 결과 (봉인 test, 2026-10-14 1회)

최종 모델 `outputs/stage4/trimtail/avg10.pt`, greedy, CER_ns(공백 제외 문자 오류율, 한글 전용 발화) %. 스트리밍은 서비스 구성(320 ms 청크, 좌측 5.12 s)의 실제 캐시 스트리밍. 괄호는 발화 단위 95 % 부트스트랩 구간(2026-10-24 추가: 같은 화자의 발화를 한 묶음으로 다시 뽑은 군집 구간은 모든 행에서 이보다 넓다. 화자 정보가 없는 FLEURS는 같은 문장으로 묶는다. 사후 보조, `reports/final/cluster_ci.json`). 전체 표·CER_ws·출력 불가능 문자 하한은 `docs/experiments.md` "7단계 결과"와 `reports/final/sealed_test.md`.

| test 세트 | full | 스트리밍 320 ms |
|---|---|---|
| Zeroth 공식 test | 3.67 (2.97-4.42) | 5.35 (4.55-6.26) |
| 　└ 학습 세트에 있는 문장 397발화(저장된 가설의 사후 분할) | 1.85 | 3.31 |
| 　└ 처음 보는 문장 60발화(사후 분할) | 16.01 | 19.25 |
| Zeroth test_H (처음 보는 문장만, 1,280발화) | 15.55 (15.09-16.02) | 18.08 (17.56-18.55) |
| FLEURS 한국어 test | 14.62 (13.47-15.89) | 17.74 (16.59-19.07) |
| Common Voice 22 한국어 test | 28.19 (26.62-29.64) | 33.35 (31.63-34.92) |

- 같은 Zeroth 공식 test에서 whisper-large-v3-turbo는 제로샷 4.48 %, LoRA 파인튜닝 1.96 %다(별도 프로젝트 whisper-ko-ft, 같은 채점 정규화). 3.67 %는 그 사이(등록 가설 H7.1 충족)지만, Zeroth test 문장 대부분이 학습 세트에도 있고 학습 세트에 있는 문장(1.85 %)과 처음 보는 문장(16.01 %)의 격차가 커서 문장 암기와 일관된다. 처음 보는 문장에서는 15~16 %다
  - 2026-10-24 추가 확인(사후 분석, 새 디코딩 없음, `reports/final/zeroth_anchor_notation.json`): 채점 정규화는 같지만 Zeroth 정답은 숫자를 한글로 풀어 쓰고 whisper는 아라비아 숫자를 내서 그 표기 차이가 앵커 쪽 오류로 센다. whisper-ko-ft가 정한 숫자 발화 142개(whisper-small 또는 turbo 가설에 아라비아 숫자가 든 발화)를 빼면 turbo 제로샷 3.29 %, 이 모델 3.69 %(같은 발화 짝 차 +0.39포인트, 95 % −0.39 ~ +1.25)로, 숫자 표기가 걸리지 않는 발화에서는 제로샷과의 차이가 0과 구별되지 않는다. H7.1 판정(등록 지표의 점추정)과 앵커 값은 그대로다. 방법·민감도는 `docs/experiments.md` 7단계 결과
- 스트리밍 서비스(dev 측정): 실제 캐시 스트리밍 CER은 simulated와 0.03포인트 안, 패킷 송신 → 부분 자막 p50 62 ms(로컬, 청크를 모으는 시간 제외), 끝점 확정 EP50 1.3 s, 청크당 계산 p50 62 ms(GPU) / 44 ms(CPU 1스레드), 10분 길이 오디오에서 계산 시간·메모리 증가 없음. GPU 계산 p90 ≤ 50 ms 가설(H6.2)은 미충족(63 ms; 연산 수가 많아 커널 실행 호출이 병목인 것으로 보임)
- 7단계 뒤 후속 측정(dev 전용, 사전 등록, 다른 프로그램이 GPU를 쓰던 부하 조건)으로 서비스 기본값을 바꿨다: 끝점 뒤 같은 모델의 전체 문맥 2차 패스가 확정 자막 CER을 dev_mix 26.86 → 22.85 %, dev_H 18.75 → 16.01 %로 낮췄고(2차 패스 GPU p90 55 ms), 캐시를 유지한 채 끝점 묵음을 0.8 s로 줄여 EP50 0.9 s, 기존 경로와 비트 단위로 같은 출력을 내는 CUDA Graph step으로 GPU 청크 계산 p90 72.5 → 43.0 ms(캐시가 찬 정상 상태 8.5 ms). 봉인 test로는 확인하지 않은 수치이고 봉인 수치는 위 표 그대로다
- 학습: 공개 데이터 51.8 h, RTX 2080 Ti 한 장, 실험 전체 GPU 약 65 h(단계별 기록의 합, 추정 포함; 최종 모델은 3단계 (a) 100 epoch + 4단계 TrimTail 15 epoch 이어 학습). fp16 수치 발산은 규칙 변경 13(SpecAugment 커리큘럼·QK-norm) 뒤 0회

## 설치

```bash
uv sync
```

Python 3.11, PyTorch CUDA 빌드가 설치된다(`pyproject.toml`). 데이터·출력은 `data/`, `outputs/`에 두며 커밋하지 않는다.

## 데이터 준비 (0단계)

```bash
uv run python scripts/download_data.py
uv run python scripts/inventory_data.py
uv run python scripts/prepare_data.py
uv run python scripts/compute_mel_stats.py
```

`manifests/`에 분할 매니페스트·어휘·mel 통계가, `data/pcm/`에 16 kHz int16 오디오 저장소가 생긴다. 요약은 `reports/stage0/data_summary.md`.

## 테스트·린트

```bash
uv run pytest -q
uv run ruff format src scripts tests && uv run ruff check src scripts tests
```

테스트는 0단계 게이트(미래 누출 없음, 캐시 스트리밍 == 오프라인 fp32 1e-5 / fp16 1e-2, CTC 입력 fp32, 과적합), 서비스 경로, 4단계 지연 계산 도우미, 7단계 봉인 test 보호 장치(한 번만 실행·재개 조건·잠금), 후속 옵션(빠른 step 경로가 eager와 비트 단위로 같음, 2차 패스 = 세그먼트의 전체 문맥 디코딩), 서버 상태 보고(`/info`의 어휘 SHA-256·스레드 수·CUDA Graph 캡처 상태, 캡처 실패 시 fast 폴백과 경고, 어휘 크기 검사, CPU 장치의 기본 1스레드, `--step-path`·`--final-pass`가 세션에 실제로 전달됨), 봉인 test 사후 분석(군집이 발화이면 군집 부트스트랩이 등록 부트스트랩과 비트 단위로 같음, 저장된 보고서 재현, whisper 앵커와 짝지을 때 결합이 일대일이고 정답이 같아야 함; 실데이터가 필요한 재현 확인은 `data/raw`와 whisper-ko-ft 체크아웃이 없으면 건너뜀), 브라우저 자막 페이지 스크립트의 연결·오디오 수명 관리(`tests/web_page_harness.js`가 Node.js vm의 가짜 DOM·WebSocket·Web Audio·가상 타이머로 `web/index.html`의 스크립트를 돌림: 멈춤 직후 재시작, 시작 실패, 마이크 트랙 정지, 16 kHz가 아닐 때 전송 안 함, endpoint 처리; `node`가 없으면 건너뜀)를 확인한다. CUDA가 없으면 fp16·CUDA Graph 테스트는 건너뛴다. GitHub Actions(`.github/workflows/ci.yml`)가 push마다 CPU에서 ruff와 pytest를 돈다.

## 학습·평가

```bash
uv run python scripts/bench_stage0.py --durations 200,300,400,600
uv run python scripts/stage1_sweep.py --epochs 5 --lrs 1e-3,2e-3 --seeds 1,2,3 --schedule-steps 69700
uv run python -m stt_scratch.train --run outputs/stage2/runA --epochs 100 --lr 2e-3 --seed 1 --cuda-memory-fraction 0.8
uv run python scripts/average_checkpoints.py --run outputs/stage2/runA --k 10 --out outputs/stage2/runA/avg10.pt
uv run python scripts/stage2_report.py --checkpoint outputs/stage2/runA/avg10.pt --name runA-avg10
uv run python scripts/compare_runs.py --a reports/stage2/hyps/runA-avg10-dev_mix-full.jsonl --b reports/stage2/hyps/runB-lr1e-3-avg10-dev_mix-full.jsonl --name H2.1-dev_mix
uv run python scripts/stage3.py --arms a,b,c,d --budget-hours 45
uv run python scripts/stage4.py --start runA
uv run python scripts/stage6.py grid --checkpoint outputs/stage4/trimtail/avg10.pt   # also h61, session, drift
uv run python scripts/training_summary.py   # fp16 stability and GPU time of every run (final report)
uv run python scripts/sealed_breakdown.py --anchor-report ../whisper-ko-ft/reports/whisper-large-v3-turbo/zeroth-test.json --digit-list ../whisper-ko-ft/reports/digit_utterances/zeroth-test.json
uv run python scripts/cluster_ci.py
uv run python -m stt_scratch.evaluate --checkpoint outputs/stage2/runA/avg10.pt --manifest dev_mix --chunk 8 --left 128 --streaming
uv run python scripts/followup.py gate   # after Stage 7, dev only: gate, timing, exit6, session, final-cpu, summary
```

`train.py`는 매 epoch `epochs.jsonl`에 dev CER과 dev CTC 손실을, `train.jsonl`에 손실·LR·GradScaler scale·GPU 메모리를 남기고 체크포인트를 저장한다. 발산 판정, 재개(`--resume`, 처방 대조), SpecAugment 커리큘럼(epoch 3부터), QK-norm은 `docs/experiments.md` "fp16 규칙"과 "규칙 변경"에 있다. Windows에서는 PyTorch 할당기 조각이 커져 GPU 메모리가 넘치지 않도록 `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256`과 `--cuda-memory-fraction 0.8`을 함께 쓴다.

4단계 지연 측정의 기준 정렬은 Montreal Forced Aligner 3.4.3 + `korean_mfa` 3.0으로 만든다(`scripts/export_mfa_corpus.py`로 내보낸 뒤 `mfa align ... --g2p_model_path korean_mfa --single_speaker --output_format csv`; 한국어 토큰화에 `python-mecab-ko` 필요). 7단계 봉인 test(`scripts/stage7.py`)는 확인 플래그를 요구하고 한 번만 실행된다. 봉인 test 사후 분석(저장된 가설만 읽음, 새 디코딩 없음)은 `scripts/sealed_breakdown.py`(학습 문장 여부 분할; whisper-ko-ft 발화별 보고서와 숫자 발화 목록을 주면 앵커 숫자 표기 분석도 하며, 행 번호 id를 원래 id로 잇는 데 `data/raw`의 Zeroth parquet가 필요)와 `scripts/cluster_ci.py`(문장·화자 군집 부트스트랩 보조 구간)다.

## 스트리밍 서비스 (6단계)

```bash
uv run python -m stt_scratch.service.server
```

기본값은 6단계에서 정한 구성에 7단계 뒤 후속 측정(`docs/experiments.md` "7단계 뒤 후속 측정", 규칙 변경 26)의 판정을 더한 것이다: 최종 모델 `outputs/stage4/trimtail/avg10.pt`(12층), 320 ms 청크, 좌측 문맥 128프레임(5.12 s), GPU fp16 autocast, 토큰 뒤 묵음 0.8 s 끝점, 끝점 뒤 인코더 캐시 유지, 끝점에서 세그먼트를 같은 모델의 전체 문맥으로 다시 디코딩한 확정 자막(`--final-pass full`; `none`이면 6단계까지의 1-pass), 인코더 step 경로 `--step-path auto`(GPU는 정상 상태 step을 CUDA Graph로 재생하는 graph, CPU는 연산 수를 줄인 fast; 둘 다 6·7단계에 쓴 eager 경로와 출력이 비트 단위로 같음을 dev_H 전수로 확인). 시작할 때 부분 청크 길이·2차 패스 길이 묶음마다 한 번씩 예열하고 CUDA Graph를 캡처한다. 캡처가 실패하면 경고를 남기고 이후 세션의 step 경로를 fast로 바꾼다(출력은 같고 청크당 계산만 느려진다). `/info`는 설정과 함께 상태를 준다: 체크포인트와 어휘 파일(`manifests/vocab.json`)의 SHA-256, 실제 step 경로와 CUDA Graph 캡처 상태(`graph_captured`·`graph_failed`), torch 연산 스레드 수(`cpu_threads`), 프레임·세션 상한. 어휘 파일의 토큰 수가 체크포인트의 출력 크기와 다르면 서버와 `stt_scratch.evaluate`·`scripts/followup.py`·`scripts/stage6.py`는 시작하지 않는다. CPU 장치에서는 torch 연산 스레드를 기본 1로 둔다(`--threads`로 바꿀 수 있음; 아래 CPU 수치를 잰 조건과 같다). 보안 점검(OWASP ZAP·bandit·pip-audit·gitleaks, 결과와 고친 것)은 `docs/security.md`. 브라우저에서 `http://127.0.0.1:8765/`를 열면 마이크 입력을 320 ms마다 보내고 부분 자막(320 ms 스트리밍, 덧붙기만 함)과 끝점으로 확정된 세그먼트(2차 패스 결과가 부분 자막을 대체)를 보여 준다. 프로토콜은 `src/stt_scratch/service/server.py` 머리말에 있다. 인증이 없으므로 로컬 데모용이다.

페이지는 잠정 자막(기울임)과 확정 자막을 구분하고 설정 줄에 2차 패스·step 경로를 보인다. 멈춤을 누르면 남은 음성을 보내고 `flush` 응답(마지막 확정 자막·통계)을 받은 뒤 연결을 닫으며, 멈춤 직후 다시 시작해도 새 연결은 이전 연결과 독립이다(이전 연결의 확정 자막은 그대로 붙는다). 시작 실패(마이크 권한 거부, 비보안 출처, 16 kHz가 아닌 AudioContext)와 연결 실패·끊김(서버의 오류 줄·종료 코드)은 상태 줄에 사유를 보이고 시작 버튼을 되살린다. 페이지 스크립트는 `tests/test_web_page.py`가 Node.js의 가짜 브라우저 환경에서 확인하지만 실제 브라우저와 다를 수 있으므로, 페이지를 고치면 브라우저에서 직접 본다: 멈춤 직후 다시 시작해도 '듣는 중'이 유지되고 이전 세그먼트의 확정 자막이 들어오는지, 마이크 권한을 거부하면 상태 줄에 사유가 뜨고 시작 버튼이 돌아오는지, 멈춘 뒤 브라우저 탭의 마이크 사용 표시가 꺼지는지. 다른 PC에서 접속하게 하려면 `--host <IP>`로 열고(인증이 없어 시작 경고가 남는다) 클라이언트가 주소창에 쓰는 이름·IP를 `--allowed-host`로 Host 허용 목록에 더한다(기본 목록은 `127.0.0.1`·`localhost`·`[::1]`과 `--host` 값이고, 목록 밖의 Host로 온 요청은 400). 다만 브라우저는 마이크 API를 보안 출처(https 또는 localhost)에서만 열어 주므로 http로 접속한 다른 PC에서는 페이지가 '이 출처에서는 마이크를 쓸 수 없습니다'를 보인다.

6단계 측정(dev, `docs/experiments.md` "6단계 결과"): 실제 캐시 스트리밍 CER은 simulated와 0.03포인트 안, 청크당 계산 p50 약 62 ms(GPU) / 44 ms(CPU 1스레드), 패킷 송신 → 부분 자막 p50 62 ms(로컬, 청크를 모으는 시간 제외), 끝점 확정 EP50 1.3 s, 10분 길이 오디오를 한 세션으로 처리했을 때(실시간이 아니라 약 2분에 처리) 청크 계산 시간·메모리 증가 없음.

후속 측정(dev, 2026-10-18, 다른 프로그램이 GPU 3D 엔진을 약 40~49 % 쓰던 부하 조건; `reports/followup/summary.md`): 세션(발화 뒤 잡음 3 s, 캐시 유지, rule2 0.8 s)의 확정 자막 CER_ns는 1-pass 대비 dev_mix 26.86 → 22.85 %(−4.02, 95 % −4.55 ~ −3.50), dev_H 18.75 → 16.01 %, 2차 패스 계산 GPU p50/p90 50 / 55 ms·CPU 1스레드 123 / 195 ms, EP50/EP90 905 / 1,065 ms(1.2 s일 때 1,295 / 1,465 ms). 확정 자막은 텍스트가 있는 세그먼트의 약 96 %에서 부분 자막과 다르다(부분 자막을 고쳐 씀). 청크당 계산 p50/p90은 GPU eager 64.7 / 72.5 ms → graph 38.1 / 43.0 ms(정상 상태 7.9 / 8.5 ms), CPU 1스레드 eager 46.8 / 51.3 → fast 40.3 / 44.4 ms. 6층 조기 종료는 보고만 했다(dev_mix 차이 없음, dev_H 14.5 % vs 18.8 %; 서비스는 봉인 test로 잰 12층 그대로).

2026-10-24 추가 확인: 위 두 문단의 CPU 수치는 모두 torch 연산 1스레드에서 잰 값이다(측정 도구가 `torch.set_num_threads(1)`로 고정; 서버도 CPU 장치에서는 기본 1스레드로 돌며 `--threads`로 바꿀 수 있고 실제 값은 `/info`의 `cpu_threads`에 있다). CPU 2차 패스 시간(p50/p90 123 / 195 ms)은 dev_mix 앞 50발화(모두 Common Voice 22 dev, 최대 7 s)의 세션에서 잰 값이라 FLEURS처럼 긴 발화(최대 약 30 s)나 rule3 상한 20 s 세그먼트의 2차 패스 시간은 포함하지 않는다. 수치와 판정은 그대로다.

실시간 클라이언트로 지연을 재려면:

```bash
uv run python scripts/stream_client.py --manifest dev_h --limit 50 --name client-dev_h-mine   # server running
uv run python scripts/stream_client.py --summarize --name client-dev_h-mine
```

## 구조

```
src/stt_scratch/   text(정규화·어휘), metrics(CER·발화·군집 부트스트랩), data(매니페스트·PCM 저장소·배치),
                   model/(features, subsampling, attention, conv, conformer, ctc, decode, streaming),
                   train, evaluate, service/(endpoint, session, server)
scripts/           데이터 내려받기·재고·준비, mel 통계, 벤치, 단계별 실행기(stage1~7), 체크포인트 평균,
                   스트리밍 클라이언트, 진단(diag_*), 학습 요약, 7단계 뒤 후속 측정(followup),
                   봉인 test 사후 분석(sealed_breakdown, cluster_ci)
tests/             게이트 테스트(test_model), 전처리(test_prepare_data), 서비스(test_service),
                   4단계 지연 도우미(test_latency), 봉인 test 보호 장치(test_stage7_guards),
                   빠른 step 경로·조기 종료(test_fast_streaming), 끝점 뒤 2차 패스(test_final_pass),
                   사후 분석(test_sealed_breakdown, test_cluster_bootstrap),
                   브라우저 페이지 스크립트(test_web_page + Node.js 하네스 web_page_harness.js)
docs/              survey/(조사·결정), experiments.md(규칙·결과), plan.md, data.md, ai-verification.md, security.md
manifests/         분할 매니페스트, 어휘, mel 통계 (커밋)
reports/           단계별 측정 결과 (커밋)
web/               브라우저 자막 페이지
```
