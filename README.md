# stt-scratch

한국어 음성 인식(STT) 모델을 처음 설계부터 만들어 공개 데이터로 학습하고, 스트리밍 자막 서비스까지 묶는 프로젝트다. 사전학습 가중치는 쓰지 않는다.

- 모델: 청크 인과 Conformer-CTC 약 21M 파라미터(순수 PyTorch). 인과 Conv2d 서브샘플링, RoPE 어텐션 + 청크 마스크, BACON식 분할 depthwise conv, 자기조건 CTC, 캐시 스트리밍 추론. 구조 선택의 근거와 기각한 대안은 `docs/survey/README.md`
- 데이터: Zeroth-Korean(CC BY 4.0) + FLEURS 한국어(CC BY 4.0) + Common Voice 22 한국어(CC0). 라이선스·분할·정규화는 `docs/data.md`
- 측정 규칙(사전 등록)과 결과: `docs/experiments.md`. 단계 계획: `docs/plan.md`. AI가 만든 코드의 검증 기록: `docs/ai-verification.md`
- 하드웨어: RTX 2080 Ti 11 GB 한 장(fp16 autocast, bf16 없음), Windows 11

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

테스트는 0단계 게이트(미래 누출 없음, 캐시 스트리밍 == 오프라인 fp32 1e-5 / fp16 1e-2, CTC 입력 fp32, 과적합)와 서비스 경로를 확인한다. CUDA가 없으면 fp16 테스트는 건너뛴다.

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
uv run python scripts/stage6.py grid --checkpoint outputs/stage4/control/avg10.pt
uv run python -m stt_scratch.evaluate --checkpoint outputs/stage2/runA/avg10.pt --manifest dev_mix --chunk 8 --left 128 --streaming
```

`train.py`는 매 epoch `epochs.jsonl`에 dev CER과 dev CTC 손실을, `train.jsonl`에 손실·LR·GradScaler scale·GPU 메모리를 남기고 체크포인트를 저장한다. 발산 판정, 재개(`--resume`, 처방 대조), SpecAugment 커리큘럼(epoch 3부터), QK-norm은 `docs/experiments.md` "fp16 규칙"과 "규칙 변경"에 있다. Windows에서는 PyTorch 할당기 조각이 커져 GPU 메모리가 넘치지 않도록 `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256`과 `--cuda-memory-fraction 0.8`을 함께 쓴다.

4단계 지연 측정의 기준 정렬은 Montreal Forced Aligner 3.4.3 + `korean_mfa` 3.0으로 만든다(`scripts/export_mfa_corpus.py`로 내보낸 뒤 `mfa align ... --g2p_model_path korean_mfa --single_speaker --output_format csv`; 한국어 토큰화에 `python-mecab-ko` 필요). 7단계 봉인 test(`scripts/stage7.py`)는 확인 플래그를 요구하고 한 번만 실행된다.

## 스트리밍 서비스 (6단계)

```bash
uv run python -m stt_scratch.service.server --checkpoint outputs/stage2/runA/avg10.pt --chunk 8 --left 128
```

브라우저에서 `http://127.0.0.1:8765/`를 열면 마이크 입력을 320 ms마다 보내고 부분 자막(수정 없이 덧붙기만 함)과 끝점으로 확정된 세그먼트를 보여 준다. 프로토콜은 `src/stt_scratch/service/server.py` 머리말에 있다. 실시간 클라이언트로 지연을 재려면:

```bash
uv run python scripts/stream_client.py --manifest dev_h --limit 50
```

## 구조

```
src/stt_scratch/   text(정규화·어휘), metrics(CER·부트스트랩), data(매니페스트·PCM 저장소·배치),
                   model/(features, subsampling, attention, conv, conformer, ctc, decode, streaming),
                   train, evaluate, service/(endpoint, session, server)
scripts/           데이터 내려받기·재고·준비, mel 통계, 벤치, 1단계 sweep, 체크포인트 평균, 스트리밍 클라이언트
tests/             게이트 테스트(test_model), 전처리(test_prepare_data), 서비스(test_service)
docs/              survey/(조사·결정), experiments.md(규칙·결과), plan.md, data.md, ai-verification.md
manifests/         분할 매니페스트, 어휘, mel 통계 (커밋)
reports/           단계별 측정 결과 (커밋)
web/               브라우저 자막 페이지
```
