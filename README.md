# stt-scratch

한국어 음성 인식(STT) 모델을 처음 설계부터 만들어 공개 데이터로 학습하고, 스트리밍 자막 서비스까지 묶는 프로젝트다. 사전학습 가중치는 쓰지 않는다.

- 모델: 청크 인과 Conformer-CTC 약 21M 파라미터(순수 PyTorch). 인과 Conv2d 서브샘플링, RoPE 어텐션 + QK-norm + 청크 마스크, BACON식 분할 depthwise conv, 동적 청크 학습, 캐시 스트리밍 추론. 최종 모델은 6층 중간 CTC(InterCTC) 헤드와 TrimTail 이어 학습을 쓴다(2단계 자기조건 CTC → 3단계 판정으로 교체, 4단계 TrimTail 채택). 구조 선택의 근거와 기각한 대안은 `docs/survey/README.md`
- 데이터: Zeroth-Korean(CC BY 4.0) + FLEURS 한국어(CC BY 4.0) + Common Voice 22 한국어(CC0). 라이선스·분할·정규화는 `docs/data.md`. `manifests/`와 `reports/`의 전사문·가설은 각 데이터셋의 라이선스를 따른다
- 측정 규칙(사전 등록)과 결과: `docs/experiments.md`. 단계 계획: `docs/plan.md`. AI가 만든 코드의 검증 기록: `docs/ai-verification.md`
- 하드웨어: RTX 2080 Ti 11 GB 한 장(fp16 autocast, bf16 없음), Windows 11

## 결과 (봉인 test, 2026-10-14 1회)

최종 모델 `outputs/stage4/trimtail/avg10.pt`, greedy, CER_ns(공백 제외 문자 오류율, 한글 전용 발화) %. 스트리밍은 서비스 구성(320 ms 청크, 좌측 5.12 s)의 실제 캐시 스트리밍. 괄호는 95 % 부트스트랩 구간. 전체 표·CER_ws·출력 불가능 문자 하한은 `docs/experiments.md` "7단계 결과"와 `reports/final/sealed_test.md`.

| test 세트 | full | 스트리밍 320 ms |
|---|---|---|
| Zeroth 공식 test | 3.67 (2.97-4.42) | 5.35 (4.55-6.26) |
| 　└ 학습 세트에 있는 문장 397발화(저장된 가설의 사후 분할) | 1.85 | 3.31 |
| 　└ 처음 보는 문장 60발화(사후 분할) | 16.01 | 19.25 |
| Zeroth test_H (처음 보는 문장만, 1,280발화) | 15.55 (15.09-16.02) | 18.08 (17.56-18.55) |
| FLEURS 한국어 test | 14.62 (13.47-15.89) | 17.74 (16.59-19.07) |
| Common Voice 22 한국어 test | 28.19 (26.62-29.64) | 33.35 (31.63-34.92) |

- 같은 Zeroth 공식 test에서 whisper-large-v3-turbo는 제로샷 4.48 %, LoRA 파인튜닝 1.96 %다(별도 프로젝트 whisper-ko-ft, 같은 채점 정규화). 3.67 %는 그 사이(등록 가설 H7.1 충족)지만, Zeroth test 문장 대부분이 학습 세트에도 있고 학습 세트에 있는 문장(1.85 %)과 처음 보는 문장(16.01 %)의 격차가 커서 문장 암기와 일관된다. 처음 보는 문장에서는 15~16 %다
- 스트리밍 서비스(dev 측정): 실제 캐시 스트리밍 CER은 simulated와 0.03포인트 안, 패킷 송신 → 부분 자막 p50 62 ms(로컬, 청크를 모으는 시간 제외), 끝점 확정 EP50 1.3 s, 청크당 계산 p50 62 ms(GPU) / 44 ms(CPU 1스레드), 10분 길이 오디오에서 계산 시간·메모리 증가 없음. GPU 계산 p90 ≤ 50 ms 가설(H6.2)은 미충족(63 ms; 연산 수가 많아 커널 실행 호출이 병목인 것으로 보임)
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

테스트는 0단계 게이트(미래 누출 없음, 캐시 스트리밍 == 오프라인 fp32 1e-5 / fp16 1e-2, CTC 입력 fp32, 과적합), 서비스 경로, 4단계 지연 계산 도우미, 7단계 봉인 test 보호 장치(한 번만 실행·재개 조건·잠금)를 확인한다. CUDA가 없으면 fp16 테스트는 건너뛴다.

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
uv run python -m stt_scratch.evaluate --checkpoint outputs/stage2/runA/avg10.pt --manifest dev_mix --chunk 8 --left 128 --streaming
```

`train.py`는 매 epoch `epochs.jsonl`에 dev CER과 dev CTC 손실을, `train.jsonl`에 손실·LR·GradScaler scale·GPU 메모리를 남기고 체크포인트를 저장한다. 발산 판정, 재개(`--resume`, 처방 대조), SpecAugment 커리큘럼(epoch 3부터), QK-norm은 `docs/experiments.md` "fp16 규칙"과 "규칙 변경"에 있다. Windows에서는 PyTorch 할당기 조각이 커져 GPU 메모리가 넘치지 않도록 `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256`과 `--cuda-memory-fraction 0.8`을 함께 쓴다.

4단계 지연 측정의 기준 정렬은 Montreal Forced Aligner 3.4.3 + `korean_mfa` 3.0으로 만든다(`scripts/export_mfa_corpus.py`로 내보낸 뒤 `mfa align ... --g2p_model_path korean_mfa --single_speaker --output_format csv`; 한국어 토큰화에 `python-mecab-ko` 필요). 7단계 봉인 test(`scripts/stage7.py`)는 확인 플래그를 요구하고 한 번만 실행된다.

## 스트리밍 서비스 (6단계)

```bash
uv run python -m stt_scratch.service.server
```

기본값은 6단계에서 정한 구성이다: 최종 모델 `outputs/stage4/trimtail/avg10.pt`, 320 ms 청크, 좌측 문맥 128프레임(5.12 s), GPU fp16 autocast, 토큰 뒤 묵음 1.2 s 끝점, 끝점 뒤 인코더 캐시 유지. 시작할 때 부분 청크 길이마다 한 번씩 예열한다(약 8 s). 보안 점검(OWASP ZAP·bandit·pip-audit·gitleaks, 결과와 고친 것)은 `docs/security.md`. 브라우저에서 `http://127.0.0.1:8765/`를 열면 마이크 입력을 320 ms마다 보내고 부분 자막(수정 없이 덧붙기만 함)과 끝점으로 확정된 세그먼트를 보여 준다. 프로토콜은 `src/stt_scratch/service/server.py` 머리말에 있다. 인증이 없으므로 로컬 데모용이다.

6단계 측정(dev, `docs/experiments.md` "6단계 결과"): 실제 캐시 스트리밍 CER은 simulated와 0.03포인트 안, 청크당 계산 p50 약 62 ms(GPU) / 44 ms(CPU 1스레드), 패킷 송신 → 부분 자막 p50 62 ms(로컬, 청크를 모으는 시간 제외), 끝점 확정 EP50 1.3 s, 10분 길이 오디오를 한 세션으로 처리했을 때(실시간이 아니라 약 2분에 처리) 청크 계산 시간·메모리 증가 없음. 실시간 클라이언트로 지연을 재려면:

```bash
uv run python scripts/stream_client.py --manifest dev_h --limit 50 --name client-dev_h-mine   # server running
uv run python scripts/stream_client.py --summarize --name client-dev_h-mine
```

## 구조

```
src/stt_scratch/   text(정규화·어휘), metrics(CER·부트스트랩), data(매니페스트·PCM 저장소·배치),
                   model/(features, subsampling, attention, conv, conformer, ctc, decode, streaming),
                   train, evaluate, service/(endpoint, session, server)
scripts/           데이터 내려받기·재고·준비, mel 통계, 벤치, 단계별 실행기(stage1~7), 체크포인트 평균,
                   스트리밍 클라이언트, 진단(diag_*), 학습 요약
tests/             게이트 테스트(test_model), 전처리(test_prepare_data), 서비스(test_service),
                   4단계 지연 도우미(test_latency), 봉인 test 보호 장치(test_stage7_guards)
docs/              survey/(조사·결정), experiments.md(규칙·결과), plan.md, data.md, ai-verification.md, security.md
manifests/         분할 매니페스트, 어휘, mel 통계 (커밋)
reports/           단계별 측정 결과 (커밋)
web/               브라우저 자막 페이지
```
