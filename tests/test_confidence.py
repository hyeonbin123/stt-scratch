"""확정 자막 어절 신뢰도(docs/experiments.md 규칙 변경 28) 테스트: 계산 함수, 2차 패스 덤프, 평가 도구.

확인하는 것: (1) 토큰 구간이 greedy CTC 축약과 같은 토큰을 내고 점수·어절 집계가 정의대로다,
(2) 어절 정렬이 편집 수는 jiwer와 같으면서 편집 수가 같은 정렬 가운데 일치가 가장 많은 것을 고른다
(대치·삽입·삭제, 띄어쓰기만 다른 오류 표시), (3) AUROC·ECE·Platt·부트스트랩이 정의(쌍 비교, 손 계산,
알려진 매개변수, 등록 재표본 표)와 같고 결정적이다, (4) 덤프 세션이 서비스 세션과 같은 이벤트를 내면서
2차 패스 로짓의 프레임 통계(top-k 모양·정렬, 엔트로피, blank)를 정확히 남긴다, (5) 덤프 명령이 파일을
쓰고 덮어쓰기·봉인 세트를 거부한다, (6) 평가 도구가 재현 조건을 확인하고 절반 A에서만 고르고 보정한 뒤
절반 B에서 판정한다.
"""

# [방식] 계산 함수는 합성 값으로, 덤프는 tests/test_final_pass.py와 같은 작은 무작위 모델(d=32, 2층)과 8토큰
# 어휘로 CPU에서 본다. 평가 도구는 프레임 통계를 손으로 짠 합성 덤프(어절과 점수를 정해 둔 것)로 끝까지
# 돌린다. 실제 신뢰도 수치는 GPU 덤프 뒤에만 나온다.

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import jiwer
import numpy as np
import pytest
import torch

from stt_scratch import confidence as conf
from stt_scratch.data import PcmStore, Utterance
from stt_scratch.metrics import _resample_indices
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.model.decode import greedy_batch
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.service.endpoint import EndpointConfig
from stt_scratch.service.session import StreamSession
from stt_scratch.text import Vocab

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
followup = importlib.import_module("followup")
stage6 = importlib.import_module("stage6")

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))
UNK, SP, GA, NA, DA = 1, 2, 3, 4, 5


def tiny_model(seed: int = 0) -> ChunkConformerCTC:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=len(VOCAB),
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_dim=64,
        dropout=0.0,
        subsampling_channels=8,
        sc_layers=(1,),
    )
    return ChunkConformerCTC(cfg).eval()


# --- (1) 토큰 구간·점수·어절 --------------------------------------------------------------------------------


# 확인: 토큰 구간의 id 열이 greedy_batch(같은 argmax의 원핫 로짓)와 같고, 같은 토큰이 이어지는 프레임은 한
# 구간, blank를 사이에 둔 같은 토큰은 두 토큰이다.
def test_token_spans_follow_greedy_ctc() -> None:
    argmax = np.array([0, GA, GA, 0, GA, NA, NA, SP, 0, DA, 0, 0])
    spans = conf.token_spans(argmax)
    assert spans == [(GA, 1, 3), (GA, 4, 5), (NA, 5, 7), (SP, 7, 8), (DA, 9, 10)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq = rng.integers(0, len(VOCAB), size=30)
        logits = torch.nn.functional.one_hot(torch.from_numpy(seq), len(VOCAB)).float()[None]
        assert [t for t, _, _ in conf.token_spans(seq)] == greedy_batch(logits, torch.tensor([30]))[0]


# 확인: 토큰 점수는 구간에서 가장 확신한 프레임 값(entropy = 1 − 최소 H / log V, maxprob = exp(최대
# log-prob))이고, 어절은 공백 토큰에서 나뉘며(앞뒤·연속 공백은 빈 어절을 만들지 않음) min·mean 집계가
# 정의대로다.
def test_scores_and_words_follow_the_definitions() -> None:
    log_v = math.log(len(VOCAB))
    argmax = np.array([SP, GA, GA, 0, NA, SP, SP, DA, 0, UNK, SP])
    max_lp = np.log(np.array([0.9, 0.6, 0.8, 0.99, 0.5, 0.9, 0.9, 0.7, 0.99, 0.4, 0.9]))
    ent = np.array([0.1, 0.5, 0.3, 0.0, 1.0, 0.1, 0.1, 0.2, 0.0, 2.0, 0.1]) * log_v / 2.0
    words = conf.segment_words(argmax, max_lp, ent, VOCAB)
    assert [w.text for w in words] == ["가나", "다ㆍ"]
    assert VOCAB.decode([t for t, _, _ in conf.token_spans(argmax)]).split() == [w.text for w in words]
    first, second = words
    np.testing.assert_allclose(first.scores["maxprob"], [0.8, 0.5])
    np.testing.assert_allclose(first.scores["entropy"], [1 - 0.15, 1 - 0.5])
    np.testing.assert_allclose(second.scores["entropy"], [1 - 0.1, 0.0])
    assert conf.word_score(first, "maxprob_min") == pytest.approx(0.5)
    assert conf.word_score(first, "maxprob_mean") == pytest.approx(0.65)
    assert conf.word_score(second, "entropy_min") == pytest.approx(0.0)
    assert conf.word_score(second, "entropy_mean") == pytest.approx(0.45)
    assert list(conf.CANDIDATES) == ["entropy_min", "entropy_mean", "maxprob_min", "maxprob_mean"]
    assert conf.segment_words(np.zeros(5, dtype=np.int64), np.zeros(5), np.zeros(5), VOCAB) == []


# --- (2) 정렬과 라벨 ----------------------------------------------------------------------------------------


# 확인: 대치·삽입·삭제 라벨, jiwer가 일치를 놓치는 동점 사례에서 일치를 살리는 것, 띄어쓰기만 다른 오류 표시,
# 그리고 무작위 어절 열에서 편집 수(대치 + 삽입 + 삭제)가 jiwer와 같고 정렬이 두 열을 빠짐없이 덮는 것.
def test_alignment_labels_and_edit_counts() -> None:
    labels = conf.hyp_word_labels("가나 다라 마바".split(), "가나 마바".split())
    assert labels.correct == [True, True] and labels.counts["delete"] == 1
    labels = conf.hyp_word_labels("가나 다라".split(), "가나 사아 다라".split())
    assert labels.correct == [True, False, True] and labels.counts["insert"] == 1
    labels = conf.hyp_word_labels("가나 다라".split(), "가나 다마".split())
    assert labels.correct == [True, False] and labels.spacing_only == [False, False]
    # 편집 4개짜리 정렬이 여럿인 사례(jiwer 4.0은 어절 4개 대치로 맞춰 '있다'를 놓친다): 일치가 가장 많은
    # 정렬을 고른다.
    ref, hyp = "할 수 있다 가나".split(), "할수 있다 다라 마".split()
    jiwer_out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    labels = conf.hyp_word_labels(ref, hyp)
    assert labels.correct == [False, True, False, False]
    assert labels.spacing_only == [True, False, False, False]  # '할 수' → '할수'
    edits = labels.counts["substitute"] + labels.counts["insert"] + labels.counts["delete"]
    assert edits == jiwer_out.substitutions + jiwer_out.insertions + jiwer_out.deletions
    rng = np.random.default_rng(1)
    pool = ["가", "나", "가나", "다", "라마"]
    for _ in range(200):
        r = [pool[i] for i in rng.integers(0, 5, size=rng.integers(1, 7))]
        h = [pool[i] for i in rng.integers(0, 5, size=rng.integers(0, 7))]
        ops = conf.align_words(r, h)
        assert [i for _, i, _ in ops if i is not None] == list(range(len(r)))
        assert [j for _, _, j in ops if j is not None] == list(range(len(h)))
        out = jiwer.process_words(" ".join(r), " ".join(h)) if h else None
        expected = len(r) if out is None else out.substitutions + out.insertions + out.deletions
        assert sum(op != "equal" for op, _, _ in ops) == expected
        assert sum(op == "equal" for op, _, _ in ops) >= (0 if out is None else out.hits)
        assert all((op == "equal") == (r[i] == h[j]) for op, i, j in ops if i is not None and j is not None)


# --- (3) 지표 -----------------------------------------------------------------------------------------------


def pairwise_auroc(scores, good) -> float:
    pos = [s for s, g in zip(scores, good, strict=True) if g]
    neg = [s for s, g in zip(scores, good, strict=True) if not g]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# 확인: AUROC가 쌍 비교 정의(같은 값 1/2)와 같고, 정수 가중치는 복제와 같으며, 완전 분리 1, 뒤집힘 0, 모두
# 같으면 0.5, 한쪽 부류가 없으면 None이다.
def test_auroc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(2)
    for _ in range(30):
        s = rng.integers(0, 6, size=40) / 5.0  # 같은 값이 많게
        y = rng.random(40) < 0.6
        if y.all() or not y.any():
            continue
        assert conf.auroc(s, y) == pytest.approx(pairwise_auroc(s, y))
        w = rng.integers(0, 3, size=40)
        if (w * y).sum() and (w * ~y).sum():
            expanded_s, expanded_y = np.repeat(s, w), np.repeat(y, w)
            assert conf.auroc(s, y, w) == pytest.approx(pairwise_auroc(expanded_s, expanded_y))
    assert conf.auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert conf.auroc([0.1, 0.2, 0.9], [True, True, False]) == 0.0
    assert conf.auroc([0.5, 0.5, 0.5], [True, False, True]) == 0.5
    assert conf.auroc([0.5, 0.6], [True, True]) is None


# 확인: ECE가 손 계산 값과 같고(1.0은 마지막 구간), 가중치는 복제와 같으며, 완전히 보정된 구간은 0을 낸다.
def test_ece_hand_computed() -> None:
    p = np.array([0.05, 0.05, 0.95, 0.95, 1.0, 0.5])
    y = np.array([False, True, True, True, True, False])
    # 구간(15개): 0.05 → 0, 0.95·1.0 → 14, 0.5 → 7. |1 − 0.1| + |3 − 2.9| + |0 − 0.5| = 1.5 → ÷ 6
    assert conf.ece(p, y) == pytest.approx(1.5 / 6)
    w = np.array([2, 0, 1, 1, 3, 1])
    assert conf.ece(p, y, weights=w) == pytest.approx(conf.ece(np.repeat(p, w), np.repeat(y, w)))
    assert conf.ece(np.full(10, 0.7), np.arange(10) < 7) == pytest.approx(0.0)
    assert conf.ece([], []) is None


# 확인: 알려진 매개변수(a = 1.5, b = −0.5)로 만든 합성 자료에서 Platt 추정이 그 값 가까이 오고, 같은 입력이면
# 비트 단위로 같은 값을 내며, 완전히 갈리는 자료에서도 유한한 값으로 수렴한다.
def test_platt_recovers_parameters_and_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    s = rng.uniform(0.02, 0.98, size=20_000)
    y = rng.random(s.size) < conf.sigmoid(1.5 * conf.logit(s) - 0.5)
    fit = conf.platt_fit(s, y)
    assert (
        fit["converged"]
        and fit["a"] == pytest.approx(1.5, abs=0.05)
        and fit["b"] == pytest.approx(-0.5, abs=0.05)
    )
    assert conf.platt_fit(s, y) == fit
    separable = conf.platt_fit(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True]))
    assert separable["converged"] and math.isfinite(separable["a"]) and separable["a"] > 0
    p = conf.platt_apply(np.array([0.1, 0.9]), separable["a"], separable["b"])
    assert 0.0 < p[0] < 0.5 < p[1] < 1.0


# 확인: 부트스트랩 구간은 등록 재표본 표(metrics._resample_indices)의 발화 뽑힘 횟수를 가중치로 쓴 값의
# 2.5·97.5 백분위이고, 결정적이다. 절반 나누기는 id SHA-256 첫 바이트의 짝·홀이다.
def test_bootstrap_interval_and_half_split() -> None:
    word_utt = np.array([0, 0, 1, 2, 2, 2, 3])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def mean(w: np.ndarray) -> float:
        return float((values * w).sum() / w.sum())

    got = conf.bootstrap_interval(word_utt, 5, mean)  # 발화 4는 어절이 없다
    manual = []
    for row in _resample_indices(5):
        w = np.bincount(row, minlength=5)[word_utt]
        if w.sum():
            manual.append(float((values * w).sum() / w.sum()))
    low, high = np.percentile(manual, [2.5, 97.5])
    assert got == {"low": pytest.approx(low), "high": pytest.approx(high), "resamples": len(manual)}
    assert conf.bootstrap_interval(word_utt, 5, mean) == got
    for uid in ("zeroth/a", "cv22/dev/x.mp3", "fleurs/1"):
        expected = "A" if hashlib.sha256(uid.encode("utf-8")).digest()[0] % 2 == 0 else "B"
        assert conf.half_of(uid) == expected


# --- (4)(5) 덤프 ------------------------------------------------------------------------------------------


def dump_module():
    return importlib.import_module("final_pass_dump")


def noise_wav(seconds: float, seed: int) -> torch.Tensor:
    return torch.randn(int(16000 * seconds), generator=torch.Generator().manual_seed(seed)) * 0.05


# 확인: 덤프 세션의 이벤트(부분·확정·끝점, stream_text 포함)가 같은 설정의 StreamSession과 같고(서비스 동작을
# 바꾸지 않음), 닫힌 세그먼트마다 남긴 프레임 통계가 그 세그먼트 mel을 오프라인 전체 문맥으로 돌린 로짓과
# 같다: argmax의 greedy 축약 = 확정 텍스트, top-k 모양 (프레임, k)·log-prob 내림차순·첫 값 = 최대 log-prob,
# 엔트로피 = 전체 분포에서 계산한 값(0 ~ log V), blank log-prob = id 0의 값.
def test_dump_session_records_full_context_statistics() -> None:
    fpd = dump_module()
    model, logmel = tiny_model(), LogMel()
    # rule3 1 s: 무작위 모델에서도 세그먼트가 여러 번 닫히게 한다(rule2는 토큰·묵음에 달려 있어 보장되지
    # 않음).
    endpoint = EndpointConfig(rule1_s=1e9, rule2_s=0.3, rule3_s=1.0)
    wav = torch.cat([noise_wav(1.3, 1), torch.zeros(16000), noise_wav(1.0, 2)])

    def run(session):
        events = []
        for _pos, piece in fpd.packets(wav):
            events.extend(session.push_samples(piece))
        return events + session.flush()

    kwargs = dict(spec=ChunkSpec(8, 64), logmel=logmel, endpoint=endpoint, reset_cache_on_endpoint=False)
    plain = run(StreamSession(model, VOCAB, final_pass="full", **kwargs))
    session = fpd.DumpSession(model, VOCAB, final_pass="full", top_k=4, keep_mel=True, **kwargs)
    dumped = run(session)
    assert [(e.kind, e.text, e.segment, e.rule, e.stream_text) for e in dumped] == [
        (e.kind, e.text, e.segment, e.rule, e.stream_text) for e in plain
    ]
    closes = [e for e in dumped if e.kind != "partial"]
    assert len(closes) >= 2 and {r["segment"] for r in session.closed} >= {e.segment for e in closes}
    for rec in session.closed:
        if rec["capture"] is None:  # 넣은 mel이 없는 세그먼트(끝점 직후 flush): 2차 패스가 돌지 않았다
            assert rec["text"] == "" and not rec["stream_text"]
            continue
        stats = rec["capture"]["stats"]
        frames = len(stats["argmax"])
        assert stats["topk_ids"].shape == stats["topk_logprobs"].shape == (frames, 4)
        assert stats["topk_ids"].dtype == np.int16 and stats["argmax"].dtype == np.int16
        assert np.all(np.diff(stats["topk_logprobs"], axis=1) <= 0)
        np.testing.assert_array_equal(stats["topk_logprobs"][:, 0], stats["max_logprob"])
        assert VOCAB.decode([t for t, _, _ in conf.token_spans(stats["argmax"])]) == rec["text"]
        mel = rec["capture"]["mel"]
        with torch.no_grad():
            logits, _, lengths = model(mel[None], torch.tensor([mel.shape[0]]), ChunkSpec(0, -1))
        lp = torch.log_softmax(logits[0, : int(lengths[0])].float(), dim=-1)
        assert frames == int(lengths[0]) == rec["capture"]["frames"]
        np.testing.assert_array_equal(stats["argmax"], lp.argmax(-1).numpy())
        np.testing.assert_allclose(stats["blank_logprob"], lp[:, 0].numpy(), atol=1e-6)
        np.testing.assert_allclose(stats["entropy"], -(lp.exp() * lp).sum(-1).numpy(), atol=1e-5)
        assert np.all(stats["entropy"] >= 0) and np.all(stats["entropy"] <= math.log(len(VOCAB)) + 1e-5)


def utt(i: int, text: str, seconds: float = 1.5, hangul: bool = True) -> Utterance:
    return Utterance(id=f"cv/{i}", corpus="cv22", text=text, duration_s=seconds, hangul_only=hangul)


# 도우미: root 아래 manifests/<name>.jsonl과 data/pcm/<name>.{bin,index.jsonl}(seed 고정 난수 PCM)을 만든다.
def fake_set(root: Path, name: str, utts: list[Utterance]) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pcm_dir = root / "data" / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name))
    pieces, offset = [], 0
    with (
        open(root / "manifests" / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as manifest,
        open(pcm_dir / f"{name}.index.jsonl", "w", encoding="utf-8", newline="\n") as index,
    ):
        for u in utts:
            n = int(u.duration_s * 16000)
            row = {"id": u.id, "corpus": u.corpus, "text": u.text, "duration_s": u.duration_s}
            manifest.write(json.dumps(row | {"hangul_only": u.hangul_only}, ensure_ascii=False) + "\n")
            index.write(json.dumps({"id": u.id, "offset": offset, "length": n}) + "\n")
            pieces.append(np.clip(rng.standard_normal(n) * 2000, -32768, 32767).astype("<i2"))
            offset += n
    np.concatenate(pieces).tofile(pcm_dir / f"{name}.bin")


@pytest.fixture
def fake_root(tmp_path) -> Path:
    fake_set(tmp_path, "dev_mix", [utt(i, "가나 다라") for i in range(3)])
    fake_set(tmp_path, "dev_h", [utt(10 + i, "마가") for i in range(2)])
    VOCAB.save(tmp_path / "manifests" / "vocab.json")  # 덤프 메타의 vocab_sha256 대상
    model = tiny_model()
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, tmp_path / "m.pt")
    return tmp_path


def dump_args(**overrides) -> argparse.Namespace:
    values = {
        "manifests": "dev_mix,dev_h",
        "checkpoint": "m.pt",
        "out_dir": "dump",
        "top_k": 3,
        "tail_s": 3.0,
        "limit": 0,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# 확인: 덤프 명령이 세트마다 npz·json을 쓰고, 메타의 npz SHA-256·구성(서버 명령행 기본값: rule2 0.8 s, 캐시
# 유지, CPU fast, 2차 패스 full, 320 ms·좌측 128)·발화 표가 맞으며, 세그먼트 위치가 프레임 배열을 빈틈없이
# 덮고, 발화 확정 가설이 같은 조건의 StreamSession(잡음 seed = 발화 인덱스)으로 흘린 결과와 같다. 있는 결과는
# 덮지 않고, 봉인 세트 이름은 받지 않는다.
def test_dump_command_writes_consistent_files(fake_root, monkeypatch) -> None:
    fpd = dump_module()
    monkeypatch.setattr(followup, "ROOT", fake_root)
    model = tiny_model()
    fpd.cmd_dump(dump_args(), model, VOCAB, torch.device("cpu"))
    meta, arrays = fpd.load_dump(fake_root / "dump", "dev_mix")
    config = meta["config"]
    assert (config["chunk"], config["left"], config["rule2_s"], config["reset_cache_on_endpoint"]) == (
        8,
        128,
        0.8,
        False,
    )
    assert (config["step_path"], config["final_pass"], config["tail_s"], config["top_k"]) == (
        "fast",
        "full",
        3.0,
        3,
    )
    assert meta["provenance"]["checkpoint"] == "m.pt" and len(meta["provenance"]["checkpoint_sha256"]) == 64
    n = len(arrays["argmax"])
    assert arrays["topk_ids"].shape == arrays["topk_logprobs"].shape == (n, 3)
    assert sum(s["frames"] for s in meta["segments"]) == n == meta["frames"]
    offsets = [s["offset"] for s in meta["segments"]]
    assert offsets == list(np.cumsum([0] + [s["frames"] for s in meta["segments"]])[:-1])
    assert [u["id"] for u in meta["utterances"]] == ["cv/0", "cv/1", "cv/2"]
    first = meta["utterances"][0]
    store = PcmStore(fake_root / "data" / "pcm", "dev_mix")
    wav = torch.cat([store.waveform("cv/0"), stage6.noise(3.0, seed=0)])
    session = StreamSession(
        model,
        VOCAB,
        ChunkSpec(8, 128),
        LogMel(),
        EndpointConfig(rule2_s=0.8),
        False,
        step_path="fast",
        final_pass="full",
    )
    closes = []
    for _pos, piece in stage6.packets(wav):
        closes.extend(e for e in session.push_samples(piece) if e.kind != "partial")
    closes.extend(session.flush())
    assert first["hyp_final"] == " ".join(e.text for e in closes if e.kind == "final")
    emitted = [meta["segments"][i] for i in first["segments"] if meta["segments"][i]["emitted"]]
    assert [(s["kind"], s["text"], s["rule"]) for s in emitted] == [(e.kind, e.text, e.rule) for e in closes]
    assert (fake_root / "dump" / "dev_h.npz").exists()
    with pytest.raises(SystemExit):
        fpd.cmd_dump(dump_args(), model, VOCAB, torch.device("cpu"))
    with pytest.raises(SystemExit):
        fpd.cmd_dump(dump_args(manifests="test_zeroth", out_dir="other"), model, VOCAB, torch.device("cpu"))
    (fake_root / "dump" / "dev_mix.npz").write_bytes(b"changed")
    with pytest.raises(SystemExit):
        fpd.load_dump(fake_root / "dump", "dev_mix")


# --- (6) 평가 도구 ------------------------------------------------------------------------------------------


# 합성 덤프: 발화마다 정답 어절 열과 가설 어절 열, 가설 어절마다 신뢰도 하나를 정해 두고 그 신뢰도가 되도록
# 프레임 통계를 만든다(음절 토큰마다 프레임 하나 + blank 하나, 공백 토큰도 프레임 하나; maxprob = 신뢰도,
# entropy 신뢰도 = 신뢰도). 세그먼트는 발화마다 final 하나다.
def synthetic_dump(
    directory: Path, name: str, rows: list[tuple[str, str, str, list[float]]], p1_config: dict
):
    fpd = dump_module()
    ids = {ch: i for i, ch in enumerate(VOCAB.tokens)}
    log_v = math.log(len(VOCAB))
    frames = {k: [] for k in ("argmax", "max_logprob", "entropy", "blank_logprob")}
    segments, utterances, offset = [], [], 0
    for uid, _ref, hyp, scores in rows:
        start = offset
        for w_index, word in enumerate(hyp.split()):
            if w_index:
                frames["argmax"].append(SP)
                frames["max_logprob"].append(math.log(0.99))
                frames["entropy"].append(0.01 * log_v)
                frames["blank_logprob"].append(math.log(0.005))
                offset += 1
            for ch in word:
                for tok, p in ((ids[ch], scores[w_index]), (0, 0.99)):
                    frames["argmax"].append(tok)
                    frames["max_logprob"].append(math.log(p))
                    frames["entropy"].append((1.0 - p) * log_v if tok else 0.01 * log_v)
                    frames["blank_logprob"].append(math.log(p) if tok == 0 else math.log((1 - p) / 2))
                    offset += 1
        segments.append(
            {
                "utt": uid,
                "segment": 0,
                "kind": "final",
                "rule": "eos",
                "emitted": True,
                "text": hyp,
                "stream_text": hyp,
                "offset": start,
                "frames": offset - start,
                "mel_frames": 4 * (offset - start),
            }
        )
        utterances.append(
            {
                "id": uid,
                "corpus": "cv22",
                "hangul_only": True,
                "noise_seed": len(utterances),
                "hyp_final": hyp,
                "hyp_stream": hyp,
                "segments": [len(segments) - 1],
            }
        )
    arrays = {k: np.asarray(v, dtype=np.int16 if k == "argmax" else np.float32) for k, v in frames.items()}
    arrays["topk_ids"] = np.stack([arrays["argmax"], np.zeros_like(arrays["argmax"])], axis=1)
    arrays["topk_logprobs"] = np.stack([arrays["max_logprob"], arrays["blank_logprob"]], axis=1)
    meta = {
        "format": fpd.FORMAT_VERSION,
        "set": name,
        "provenance": {"checkpoint": "m.pt", "checkpoint_sha256": "c" * 64, "git_head": "x", "limit": 0},
        "config": p1_config,
        "vocab_sha256": "v" * 64,
        "frames": offset,
        "segments": segments,
        "utterances": utterances,
    }
    directory.mkdir(parents=True, exist_ok=True)
    fpd.write_dump(directory, name, arrays, meta)


# 평가용 가짜 저장소: 합성 덤프 두 세트, 매니페스트(정답), 그리고 P1 기록(session.json, 확정 가설 파일)을
# 덤프와 같게 만든다. dev_mix는 절반 A·B에 고르게 들도록 id 40개, 가설 어절 신뢰도는 정답 어절이 대체로 높고
# 오류 어절이 대체로 낮게(겹침 있음) 둔다.
def eval_root(tmp_path: Path, p1_config: dict) -> Path:
    eval_mod = importlib.import_module("confidence_eval")
    rng = np.random.default_rng(5)
    sets = {"dev_mix": 40, "dev_h": 12}
    session = {
        "provenance": {"checkpoint_sha256": "c" * 64},
        "step_path": "graph",
        "tail_noise_s": 3.0,
        "runs": {},
    }
    (tmp_path / "reports" / "followup" / "hyps").mkdir(parents=True)
    for name, count in sets.items():
        rows, utts = [], []
        for i in range(count):
            ref = ["가나", "다라", "마가", "나다"][: 2 + i % 3]
            hyp = list(ref)
            scores = []
            for k in range(len(hyp)):
                wrong = rng.random() < 0.4
                if wrong:
                    hyp[k] = "라라"
                scores.append(float(np.clip(rng.normal(0.35 if wrong else 0.8, 0.15), 0.01, 0.99)))
            uid = f"{name}/{i}"
            rows.append((uid, " ".join(ref), " ".join(hyp), scores))
            utts.append(
                Utterance(id=uid, corpus="cv22", text=" ".join(ref), duration_s=1.0, hangul_only=True)
            )
        synthetic_dump(tmp_path / "dump", name, rows, p1_config)
        (tmp_path / "manifests").mkdir(exist_ok=True)
        with open(tmp_path / "manifests" / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for u in utts:
                line = {
                    "id": u.id,
                    "corpus": u.corpus,
                    "text": u.text,
                    "duration_s": 1.0,
                    "hangul_only": True,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        hyps = {uid: hyp for uid, _ref, hyp, _s in rows}
        recorded = eval_mod.cer_ns_hangul(utts, hyps)
        session["runs"][f"{name}_rule2_0.8"] = {"cer_ns_final_pass": recorded}
        path = tmp_path / "reports" / "followup" / "hyps" / f"{name}-session-rule2_0.8-final_pass.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for u in utts:
                f.write(
                    json.dumps({"id": u.id, "ref": u.text, "hyp": hyps[u.id], "hangul_only": True}) + "\n"
                )
    (tmp_path / "reports" / "followup" / "session.json").write_text(json.dumps(session), encoding="utf-8")
    return tmp_path


P1_CONFIG = {
    "chunk": 8,
    "left": 128,
    "rule1_s": 2.4,
    "rule2_s": 0.8,
    "rule3_s": 20.0,
    "reset_cache_on_endpoint": False,
    "step_path": "graph",
    "final_pass": "full",
    "final_pass_bucket": 128,
    "precision": "fp16-autocast",
    "tail_s": 3.0,
    "top_k": 2,
}


def eval_args(**overrides) -> argparse.Namespace:
    values = {"dump_dir": "dump", "out": "out/confidence.json", "overwrite": False}
    values.update(overrides)
    return argparse.Namespace(**values)


# 확인: 재현 조건이 맞으면 평가가 끝까지 돌아 (가) 후보·Platt를 절반 A 어절로만 고르고(절반 B 라벨을 바꿔도
# 선택·매개변수가 그대로), (나) 절반 B 지표가 같은 함수로 B 어절에서 직접 계산한 값과 같으며, (다) 판정이
# 등록 임계로 정해지고, (라) dev_H는 같은 선택·보정을 쓴 참고치다. 재현이 깨지면(기록 가설과 다름) 결과를 쓰지
# 않고 멈추고, 체크포인트가 P1과 다르면(연기 시험) 판정 없이 쓴다. 결과 파일이 있으면 덮지 않는다.
def test_confidence_eval_selects_on_half_a_and_judges_on_half_b(tmp_path, monkeypatch) -> None:
    eval_mod = importlib.import_module("confidence_eval")
    root = eval_root(tmp_path, P1_CONFIG)
    monkeypatch.setattr(eval_mod, "ROOT", root)
    monkeypatch.setattr(eval_mod.followup, "ROOT", root)
    monkeypatch.setattr(eval_mod, "load_vocab", lambda: (VOCAB, "v" * 64))
    eval_mod.cmd_eval(eval_args())
    result = json.loads((root / "out" / "confidence.json").read_text(encoding="utf-8"))
    assert result["reproduction"]["met"] is True
    assert result["reproduction"]["dev_mix"]["exact_match"] == 40
    sel = result["selection"]
    assert sel["chosen"] in conf.CANDIDATES and set(sel["auroc_half_a"]) == set(conf.CANDIDATES)
    # 선택: 절반 A 어절의 후보별 AUROC 가운데 최댓값(동점은 등록 순서)이고 Platt는 그 후보의 A 어절로 맞춘
    # 값이다.
    meta, arrays = dump_module().load_dump(root / "dump", "dev_mix")
    table = eval_mod.word_table(meta, arrays, VOCAB, eval_mod.read_dev_manifest("dev_mix"))
    a = table["half"] == "A"
    a_aurocs = [conf.auroc(table["scores"][c][a], table["correct"][a]) for c in conf.CANDIDATES]
    assert sel["chosen"] == list(conf.CANDIDATES)[int(np.argmax(a_aurocs))]
    assert sel["platt"] == conf.platt_fit(table["scores"][sel["chosen"]][a], table["correct"][a])
    # 절반 B 지표를 같은 함수로 직접 계산해 대조한다.
    b = table["half"] == "B"
    assert 0 < a.sum() and 0 < b.sum() and a.sum() + b.sum() == table["correct"].size
    p = conf.platt_apply(table["scores"][sel["chosen"]][b], sel["platt"]["a"], sel["platt"]["b"])
    half_b = result["dev_mix_half_b"]
    assert half_b["words"] == int(b.sum()) and half_b["auroc"] == pytest.approx(
        conf.auroc(p, table["correct"][b])
    )
    assert half_b["ece"] == pytest.approx(conf.ece(p, table["correct"][b]))
    assert half_b["auroc_ci"]["low"] <= half_b["auroc"] <= half_b["auroc_ci"]["high"]
    verdict = result["verdict"]
    assert verdict["adopted"] == (half_b["auroc"] >= 0.80 and half_b["ece"] <= 0.05)
    assert result["dev_h_reference"]["words"] > 0 and "verdict" not in result["dev_h_reference"]
    with pytest.raises(SystemExit):
        eval_mod.cmd_eval(eval_args())  # 결과가 이미 있음
    # 절반 B 라벨만 바꾼 정답(매니페스트; B 발화의 첫 정답 어절을 바꿈)으로 다시 돌려도 A의 선택·Platt는
    # 그대로다.
    manifest = root / "manifests" / "dev_mix.jsonl"
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if conf.half_of(r["id"]) == "B":
            r["text"] = " ".join(["마마", *r["text"].split()[1:]])
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(eval_mod, "check_reproduction", lambda *a, **k: {"met": True, "applicable": True})
    eval_mod.cmd_eval(eval_args(out="out/changed.json"))
    changed = json.loads((root / "out" / "changed.json").read_text(encoding="utf-8"))
    assert changed["selection"] == sel and changed["dev_mix_half_b"]["auroc"] != half_b["auroc"]


def test_confidence_eval_stops_when_the_dump_does_not_reproduce_p1(tmp_path, monkeypatch) -> None:
    eval_mod = importlib.import_module("confidence_eval")
    root = eval_root(tmp_path, P1_CONFIG)
    monkeypatch.setattr(eval_mod, "ROOT", root)
    monkeypatch.setattr(eval_mod.followup, "ROOT", root)
    monkeypatch.setattr(eval_mod, "load_vocab", lambda: (VOCAB, "v" * 64))
    path = root / "reports" / "followup" / "hyps" / "dev_mix-session-rule2_0.8-final_pass.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in rows[:10]:
        r["hyp"] = "마마"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    session_path = root / "reports" / "followup" / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    utts = eval_mod.read_dev_manifest("dev_mix")
    session["runs"]["dev_mix_rule2_0.8"]["cer_ns_final_pass"] = eval_mod.cer_ns_hangul(
        utts, {r["id"]: r["hyp"] for r in rows}
    )
    session_path.write_text(json.dumps(session), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        eval_mod.cmd_eval(eval_args())
    assert stopped.value.code == eval_mod.REPRO_EXIT == 5
    assert not (root / "out" / "confidence.json").exists()
    # 체크포인트가 P1과 다르면 재현 대조를 하지 않고(연기 시험) 판정 없이 쓴다.
    session["provenance"]["checkpoint_sha256"] = "d" * 64
    session_path.write_text(json.dumps(session), encoding="utf-8")
    eval_mod.cmd_eval(eval_args())
    result = json.loads((root / "out" / "confidence.json").read_text(encoding="utf-8"))
    assert result["reproduction"]["applicable"] is False and result["verdict"] is None
    assert any("checkpoint" in reason for reason in result["reproduction"]["reasons"])
