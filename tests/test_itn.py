"""숫자 역정규화(ITN, docs/experiments.md 규칙 변경 30) 테스트: 변환 규칙, 오탐 방지, 결정성, 평가 도구.

확인하는 것: (1) 표 기반 변환(한자어 수사열·단위·소수·고유어·달 이름, 띄어쓰기와 무관), (2) 오탐 방지(단독
'일'·'이', 수사로 시작하는 흔한 낱말, 지시어 '이', 범위 읽기, 어림말), (3) 결정성과 두 번 적용해도 같음,
숫자·라틴·<unk> 글자는 그대로, (4) 수사 문법(parse_sino)과 표시(format_sino), (5) 평가 도구: 평가 세트의
학습 겹침 제외, parquet에서 PCM 꺼내기, 디코딩 명령, 채점·재현 조건·판정과 결과 파일에 전사문이 없음.
"""

# [방식] 규칙은 손으로 정한 문장 쌍으로 본다(판정 세트 문장이 아니다). 평가 도구는 tmp_path에 가짜 저장소
# (매니페스트, PCM 저장소, FLEURS 모양 parquet, 저장 가설)를 만들고 ROOT를 바꿔 끝까지 돌린다. 디코딩은
# tests/test_confidence.py와 같은 작은 무작위 모델(d=32, 2층)과 8토큰 어휘로 CPU에서 한다.

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch

from stt_scratch.confidence import half_of
from stt_scratch.data import PcmStore
from stt_scratch.evaluate import score
from stt_scratch.itn import format_sino, itn, itn_spans, parse_sino
from stt_scratch.model.config import ModelConfig
from stt_scratch.model.conformer import ChunkConformerCTC
from stt_scratch.text import UNK_CHAR, Vocab

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
followup = importlib.import_module("followup")
prepare_data = importlib.import_module("prepare_data")

VOCAB = Vocab(("<blank>", "<unk>", " ", "가", "나", "다", "라", "마"))


# --- 변환 규칙 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("이천 십 팔 년", "2018년"),  # 기본 예: 띄어 쓴 수사열 + 단위
        ("이천십팔년", "2018년"),
        ("이천 십팔 년도에", "2018년도에"),
        ("사십 명", "40명"),
        ("약 사십 명의 수감자", "약 40명의 수감자"),
        ("천 구백 칠십 팔 년", "1978년"),
        ("천 구백 칠십 년대 후반", "1970년대 후반"),
        ("십 팔 세기 시장", "18세기 시장"),
        ("천 번째 도장", "1000번째 도장"),
        ("사 년 만에", "4년 만에"),
        ("십년간 두는", "10년간 두는"),
        ("오 억 원이었다", "5억원이었다"),
        ("삼억 오천만 원", "3억 5000만원"),
        ("만 오천 원", "15,000원"),
        ("만 명", "1만명"),
        ("십만 원", "10만원"),
        ("삼십 사 쩜 사 퍼센트로", "34.4%로"),  # Zeroth식 소수점 '쩜'
        ("영 쩜 칠 육 퍼센트", "0.76%"),
        ("삼 점 오 배", "3.5배"),
        ("구 쩜 일 일 테러", "9.11 테러"),  # 소수 뒤 '일'은 날로 읽지 않는다
        ("사천 팔백 구십 이 미터입니다", "4892m입니다"),
        ("오백 그램", "500g"),
        ("오십 대", "50대"),
        ("사십 일", "40일"),  # 단위가 있는 해석이 먼저: 41이 아니다
        ("사십 일 명", "41명"),
        ("삼십일 일", "31일"),
        ("삼월 이 일", "3월 2일"),  # 날짜 문맥의 '이 일'
        ("이 일 오후", "2일 오후"),
        ("이천 십 사 년 십 이 월", "2014년 12월"),
        ("유월", "6월"),
        ("시월부터", "10월부터"),
        ("삼 분의 일", "3분의 1"),
        ("이십 여 명의", "20여명의"),
        ("십여 개", "10여개"),
        ("칠십 면을", "70 면을"),  # 단위 없는 두 음절 이상 수
        ("오십 육 까지", "56 까지"),
        ("삼십이 넘는", "30이 넘는"),  # 자릿수 뒤 끝의 '이'는 조사
        ("이십사시간 내내", "24시간 내내"),
        # 고유어 수사 + 명·개·살·시·시간·개월
        ("두 시간 만에", "2시간 만에"),
        ("열한 개와", "11개와"),
        ("두개로", "2개로"),
        ("육 점 두개로", "6점 2개로"),  # '점' 뒤가 숫자가 아니면 단위(점수)
        ("마흔 두 명", "42명"),
        ("스무 살", "20살"),
        ("열 두 시", "12시"),
        ("세 개월", "3개월"),
        ("한 명인", "1명인"),
    ],
)
def test_converts_registered_patterns(text: str, expected: str) -> None:
    assert itn(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "일을 했다",  # 기본 예: 단독 '일'
        "이 사람",  # 기본 예: 지시어 '이'
        "이 분이 오셨다",
        "이 점에서 보면",
        "이번에는 사회 구분이 일시에",
        "만일 그렇다면",
        "천천히 조사를 하던 구조대",
        "오만한 천사가 만세를",
        "사명을 가진 사원",
        "이 세상 세계",
        "오시는 분",
        "한시도 쉬지 않았다",
        "천만에요",
        "백이 넘는",
        "백 사장이 왔다",  # 성 + 사장(대표)
        "사소한 이유로",  # 이유 + 로(유로가 아니다)
        "이 프로에 대해",  # 이 방송
        "이천시 이천 시",  # 지명(값 범위로 시각이 아님)
        "칠팔 년 이삼 일",  # 범위 읽기
        "수천 조 마일이나 몇 십 년",  # 어림말
        "다섯 번째 하는",  # 서수의 고유어는 한글 그대로
        "한시적 오일 일일이",
        "사 주다",
        "세상 두고 네가",
    ],
)
def test_leaves_non_numbers_alone(text: str) -> None:
    assert itn(text) == text
    assert itn_spans(text) == []


def test_deterministic_idempotent_and_passthrough() -> None:
    # 수사·단위·흔한 음절을 섞은 무작위 문장(seed 0)에서 같은 입력 → 같은 출력, 두 번 적용해도 같음.
    pool = list("일이삼사오육칠팔구십백천만억조년월명개분시점쩜영세두한열가나다를의에는") + [" "] * 8
    rng = np.random.default_rng(0)
    for _ in range(2000):
        text = " ".join("".join(rng.choice(pool, size=int(rng.integers(1, 16)))).split())
        once = itn(text)
        assert itn(text) == once
        assert itn(once) == once
    # 숫자·라틴·<unk> 글자는 건드리지 않고, 바꿀 것이 없으면 공백까지 원문 그대로다.
    for text in ("2018년 4892m", "cep다 abc", f"가{UNK_CHAR}나", "", "  두 칸  띄움  "):
        assert itn(text) == text
    spans = itn_spans("그는 천 구백 칠십 팔 년 사십 명")
    assert [(s.start, s.end, s.before, s.after) for s in spans] == [
        (1, 6, "천 구백 칠십 팔 년", "1978년"),
        (6, 8, "사십 명", "40명"),
    ]


def test_parse_and_format_sino() -> None:
    assert parse_sino("이천십팔") == 2018
    assert parse_sino("천구백칠십팔") == 1978
    assert parse_sino("만오천") == 15000
    assert parse_sino("삼억오천만") == 350_000_000
    assert parse_sino("일조이천구백십구억") == 1_291_900_000_000
    assert parse_sino("십") == 10 and parse_sino("만") == 10_000
    for bad in ("", "이삼", "칠팔", "십십", "백천", "만만", "억만", "천천", "가"):
        assert parse_sino(bad) is None, bad
    assert format_sino(2018) == "2018"
    assert format_sino(10_000) == "1만"
    assert format_sino(15_000) == "15,000"
    assert format_sino(350_000_000) == "3억 5000만"
    assert format_sino(1_200_000_000_000) == "1조 2000억"


# --- 평가 도구 ----------------------------------------------------------------------------------------------
def tiny_model() -> ChunkConformerCTC:
    torch.manual_seed(0)
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def manifest_row(utt_id: str, text: str, seconds: float = 0.5, **extra) -> dict:
    corpus = utt_id.split("/")[0]
    hangul = not any(ch.isdigit() for ch in text)
    return {
        "id": utt_id,
        "corpus": corpus,
        "text": text,
        "duration_s": seconds,
        "hangul_only": hangul,
        **extra,
    }


def ids_in_half(prefix: str, half: str, count: int) -> list[str]:
    """SHA-256 절반 규칙으로 절반 half에 드는 id를 count개 만든다."""
    out, k = [], 0
    while len(out) < count:
        candidate = f"{prefix}/{k}"
        if half_of(candidate) == half:
            out.append(candidate)
        k += 1
    return out


def pcm(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0, 3000, int(seconds * 16000)), -32768, 32767).astype("<i2")


def write_store(root: Path, name: str, rows: list[dict], seed: int = 0) -> None:
    (root / "data" / "pcm").mkdir(parents=True, exist_ok=True)
    writer = prepare_data.PcmWriter(root / "data" / "pcm", name)
    for k, r in enumerate(rows):
        writer.write(r["id"], pcm(r["duration_s"], seed + k))
    writer.close()


def write_fleurs_parquet(raw: Path, entries: list[tuple[int, str, np.ndarray]]) -> None:
    """FLEURS 모양 parquet(id, audio{bytes, path}, transcription). 행 순서가 발화 id의 행 번호다."""
    audio = []
    for _sid, _text, samples in entries:
        buf = io.BytesIO()
        sf.write(buf, samples.astype(np.float32) / 32768.0, 16000, format="FLAC", subtype="PCM_16")
        audio.append({"bytes": buf.getvalue(), "path": "x.flac"})
    table = pa.table(
        {"id": [e[0] for e in entries], "audio": audio, "transcription": [e[1] for e in entries]}
    )
    folder = raw / "fleurs_ko" / "ko_kr" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, folder / "0000.parquet")


@pytest.fixture
def itn_eval(monkeypatch, tmp_path):
    """ROOT를 tmp_path로 바꾼 평가 도구 모듈."""
    mod = importlib.import_module("itn_eval")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod.followup, "ROOT", tmp_path)
    return mod


def test_eval_set_excludes_training_overlap(itn_eval, tmp_path, monkeypatch) -> None:
    write_jsonl(
        tmp_path / "manifests" / "train.jsonl",
        [manifest_row("fleurs/train/7/0", "가나 다라"), manifest_row("zeroth/train/0", "마 가")],
    )
    digit = [
        manifest_row("fleurs/train/1/1", "가나 다라", reason="digit"),  # 정규화 문장이 학습 전사와 같음(가정)
        manifest_row("fleurs/train/7/2", "3 가", reason="digit"),  # 같은 문장(id 7)의 다른 낭독이 학습에 있음
        manifest_row("fleurs/train/2/3", "4 나", 25.0, reason="digit"),
        manifest_row("fleurs/train/2/4", "4 나", reason="digit"),
        manifest_row("fleurs/train/3/5", "5 다", reason="latin"),  # 숫자 사유가 아님
    ]
    write_jsonl(tmp_path / "manifests" / "train_excluded.jsonl", digit)
    monkeypatch.setattr(itn_eval, "EXPECTED_DIGIT", 4)
    utts, info = itn_eval.eval_set()
    assert [u.id for u in utts] == ["fleurs/train/2/3", "fleurs/train/2/4"]
    assert info == {
        "digit_excluded": 4,
        "overlap_text": 1,
        "overlap_sentence": 1,
        "kept": 2,
        "sentences": 1,
        "over_20s": 1,
        "hours": round(25.5 / 3600, 3),
    }
    assert [u.id for u in itn_eval.eval_set(limit=1)[0]] == ["fleurs/train/2/3"]
    monkeypatch.setattr(itn_eval, "EXPECTED_DIGIT", 619)
    with pytest.raises(SystemExit, match="expected 619"):
        itn_eval.eval_set()


def test_extract_pcm_matches_the_parquet_audio(itn_eval, tmp_path) -> None:
    samples = [pcm(0.5, 1), pcm(0.25, 2), pcm(0.75, 3)]
    write_fleurs_parquet(
        tmp_path / "raw", [(10, "a", samples[0]), (11, "b", samples[1]), (12, "c", samples[2])]
    )
    utts = [
        itn_eval.Utterance("fleurs/train/12/2", "fleurs", "3 다", 0.75, False),
        itn_eval.Utterance("fleurs/train/10/0", "fleurs", "1 가", 0.5, False),
    ]
    info = itn_eval.extract_pcm(utts, tmp_path / "raw", tmp_path / "pcm", "x")
    assert info["extracted"] == 2
    assert info["parquet"][0]["path"] == "fleurs_ko/ko_kr/train/0000.parquet"
    store = PcmStore(tmp_path / "pcm", "x")
    for utt, original in ((utts[0], samples[2]), (utts[1], samples[0])):
        assert torch.equal(store.waveform(utt.id), torch.from_numpy(original.astype(np.float32) / 32768.0))
    with pytest.raises(SystemExit, match="not in the FLEURS train parquet"):
        itn_eval.extract_pcm(
            [itn_eval.Utterance("fleurs/train/99/9", "fleurs", "1", 0.5, False)],
            tmp_path / "raw",
            tmp_path / "p2",
            "y",
        )
    with pytest.raises(SystemExit, match="samples"):
        itn_eval.extract_pcm(
            [itn_eval.Utterance("fleurs/train/11/1", "fleurs", "2", 0.5, False)],
            tmp_path / "raw",
            tmp_path / "p3",
            "z",
        )


def build_fake_repo(root: Path, checkpoint_sha: str) -> dict:
    """디코딩·채점 끝까지 시험용 가짜 저장소. 반환: 만든 매니페스트 행들."""
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    VOCAB.save(root / "manifests" / "vocab.json")
    write_jsonl(root / "manifests" / "train.jsonl", [manifest_row("zeroth/train/0", "마 가")])
    digit_rows = [
        manifest_row(f"fleurs/train/{20 + k}/{k}", f"{k + 1}명 가", reason="digit") for k in range(3)
    ]
    write_jsonl(root / "manifests" / "train_excluded.jsonl", digit_rows)
    write_fleurs_parquet(
        root / "data" / "raw", [(20 + k, r["text"], pcm(0.5, 50 + k)) for k, r in enumerate(digit_rows)]
    )
    dev_mix = [manifest_row(i, "3명 가") for i in ids_in_half("fleurs/validation/30", "A", 1)]
    dev_mix += [manifest_row(i, "4명 나") for i in ids_in_half("fleurs/validation/31", "B", 2)]
    dev_mix += [manifest_row(f"cv22/dev/{k}", "가 나 다") for k in range(2)]
    dev_h = [manifest_row(f"zeroth/train/{100 + k}", "다 라") for k in range(2)]
    for seed, (name, rows) in enumerate((("dev_mix", dev_mix), ("dev_h", dev_h))):
        write_jsonl(root / "manifests" / f"{name}.jsonl", rows)
        write_store(root, name, rows, seed=100 * (seed + 1))
    (root / "reports" / "followup").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "followup" / "session.json").write_text(
        json.dumps({"provenance": {"checkpoint_sha256": checkpoint_sha}}), encoding="utf-8"
    )
    return {"digit": digit_rows, "dev_mix": dev_mix, "dev_h": dev_h}


def test_decode_then_score_end_to_end(itn_eval, tmp_path, monkeypatch) -> None:
    model = tiny_model()
    ckpt = tmp_path / "outputs" / "tiny.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    rows = build_fake_repo(tmp_path, sha)
    monkeypatch.setattr(itn_eval, "EXPECTED_DIGIT", 3)
    common = [
        "--checkpoint",
        "outputs/tiny.pt",
        "--device",
        "cpu",
        "--sets",
        "fleurs_train_digit,dev_mix,dev_h",
    ]
    itn_eval.main(["--decode", *common])
    out_dir = tmp_path / "outputs" / "followup" / "itn"
    meta = json.loads((out_dir / "decode.json").read_text(encoding="utf-8"))
    assert meta["precision"] == "fp32" and meta["provenance"]["checkpoint_sha256"] == sha
    assert meta["sets"]["fleurs_train_digit"]["kept"] == 3
    assert meta["sets"]["fleurs_train_digit"]["extracted"] == 3
    with pytest.raises(SystemExit, match="exists"):
        itn_eval.main(["--decode", *common])
    # 4단계 저장 가설 자리에 dev 디코딩 결과를 두고, 기록 CER은 evaluate.score로 계산해 둔다(채점 일치 확인).
    stored_sets = {}
    for name in ("dev_mix", "dev_h"):
        hyps = itn_eval.read_jsonl(out_dir / f"{name}-full.jsonl")
        write_jsonl(tmp_path / "reports" / "stage4" / "hyps" / f"trimtail-avg10-{name}-full.jsonl", hyps)
        utts = itn_eval.read_manifest(tmp_path / "manifests" / f"{name}.jsonl")
        stored_sets[name] = {"modes": {"full": score(utts, {r["id"]: r["hyp"] for r in hyps})}}
    (tmp_path / "reports" / "stage4" / "trimtail-avg10.json").write_text(
        json.dumps({"sets": stored_sets}), encoding="utf-8"
    )
    itn_eval.main(["--score"])
    result_text = (tmp_path / "reports" / "followup" / "itn.json").read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert result["reproduction"]["applicable"] and result["reproduction"]["met"]
    assert result["reproduction"]["exact_match"] == len(rows["dev_mix"])
    assert result["scoring_consistency"] == {
        "applicable": True,
        "dev_mix_all": result["scoring_consistency"]["dev_mix_all"],
        "dev_h_hangul_only": result["scoring_consistency"]["dev_h_hangul_only"],
        "met": True,
    }
    assert result["split"] == {"rule": result["split"]["rule"], "dev_mix_digit": 3, "half_a": 1, "half_b": 2}
    assert result["a_fleurs_train_digit"]["utterances"] == 3
    assert result["verdict"]["adopted"] is False  # 무작위 모델 가설에는 바꿀 수 표현이 없다
    for r in rows["digit"] + rows["dev_mix"]:
        assert r["text"] not in result_text  # 결과 파일에 전사문이 없다
    with pytest.raises(SystemExit, match="exists"):
        itn_eval.main(["--score"])


def write_scoring_inputs(root: Path, a_pairs, dev_mix, dev_h, stored, new_mix=None) -> None:
    """디코딩 없이 채점만 시험: decode.json, (가) 가설, 새 dev_mix 가설, 저장 가설(--dev-hyps 형식)."""
    out_dir = root / "outputs" / "followup" / "itn"
    write_jsonl(
        out_dir / "fleurs_train_digit-full.jsonl", [{"id": i, "ref": r, "hyp": h} for i, r, h in a_pairs]
    )
    new_mix = new_mix or stored["dev_mix"]
    write_jsonl(out_dir / "dev_mix-full.jsonl", [{"id": i, "hyp": h} for i, h in new_mix.items()])
    meta = {
        "precision": "fp16-autocast",
        "provenance": {"checkpoint_sha256": "c" * 64, "limit": 0},
        "sets": {"fleurs_train_digit": {}, "dev_mix": {}},
    }
    (out_dir / "decode.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "reports" / "followup").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "followup" / "session.json").write_text(
        json.dumps({"provenance": {"checkpoint_sha256": "c" * 64}}), encoding="utf-8"
    )
    for name, rows in (("dev_mix", dev_mix), ("dev_h", dev_h)):
        write_jsonl(root / "manifests" / f"{name}.jsonl", rows)
        write_jsonl(root / "hyps" / f"{name}.jsonl", [{"id": i, "hyp": h} for i, h in stored[name].items()])


def synthetic_sets():
    a_pairs = [
        (f"fleurs/train/{k}/{k}", ref, hyp)
        for k, (ref, hyp) in enumerate(
            [
                ("그는 1978년에 태어났다", "그는 천 구백 칠십 팔 년에 태어났다"),
                ("약 40명의 사람", "약 사십 명의 사람"),
                ("2시간 만에 3개가", "두 시간 만에 세 개가"),
                ("18세기 시장", "십팔 세기 시장"),
            ]
        )
    ]
    a_ids = ids_in_half("fleurs/validation/1", "A", 2)
    b_ids = ids_in_half("fleurs/validation/2", "B", 3)
    digit_refs = ["1978년에 태어났다", "약 40명의 사람", "18세기 시장"]
    digit_hyps = ["천 구백 칠십 팔 년에 태어났다", "약 사십 명의 사람", "십팔 세기 시장"]
    dev_mix = [manifest_row(i, digit_refs[k % 3]) for k, i in enumerate(a_ids + b_ids)]
    stored_mix = {r["id"]: digit_hyps[k % 3] for k, r in enumerate(dev_mix)}
    hangul = [("가나 다라", "가나 다라"), ("삼십 년 전에", "삼십 년 전에"), ("마 가", "마 가")]
    for k, (ref, hyp) in enumerate(hangul * 40):
        dev_mix.append(manifest_row(f"cv22/dev/{k}", ref))
        stored_mix[f"cv22/dev/{k}"] = hyp
    dev_h = [manifest_row(f"zeroth/x/{k}", ref) for k, (ref, _) in enumerate(hangul * 40)]
    stored_h = {r["id"]: hyp for r, (_, hyp) in zip(dev_h, hangul * 40, strict=True)}
    return a_pairs, dev_mix, dev_h, {"dev_mix": stored_mix, "dev_h": stored_h}


def test_score_judges_on_half_b_and_the_number_free_guard(itn_eval, tmp_path) -> None:
    a_pairs, dev_mix, dev_h, stored = synthetic_sets()
    write_scoring_inputs(tmp_path, a_pairs, dev_mix, dev_h, stored)
    itn_eval.main(["--score", "--dev-hyps", "hyps/{name}.jsonl"])
    result = json.loads((tmp_path / "reports" / "followup" / "itn.json").read_text(encoding="utf-8"))
    a, b = result["a_fleurs_train_digit"], result["b_dev_mix_digit_half_b"]
    assert a["utterances"] == 4 and a["itn"]["edits"] == 0 and a["diff"] < -0.03 and a["ci"][1] < 0
    assert b["utterances"] == 3 and b["diff"] < -0.03 and b["ci"][1] < 0
    assert result["dev_mix_digit_half_a_development"]["utterances"] == 2
    mix = result["guard"]["dev_mix_hangul_only"]
    # '삼십 년 전에'는 정답에도 수 표현이 있어 오탐 분모에서 빠지고, 단순 변경률(1/3)에는 들어간다.
    assert mix["utterances"] == 120 and mix["changed"] == 40 and mix["reference_number_utterances"] == 40
    assert (
        mix["number_free_utterances"] == 80 and mix["false_changes"] == 0 and mix["false_change_rate"] == 0.0
    )
    assert mix["cer_ns_hangul_only"]["diff"] > 0  # 한글로 쓴 정답과는 충돌한다(보고)
    assert result["scoring_consistency"] == {"applicable": False, "met": None}
    v = result["verdict"]
    assert v["a_fleurs_train_digit"] and v["b_dev_mix_digit_half_b"] and v["guard_dev_h"] and v["adopted"]
    assert v["raw_change_guard"] == {"dev_mix_hangul_only": False, "dev_h": False}
    changes = itn_eval.read_jsonl(tmp_path / "outputs" / "followup" / "itn" / "changes.jsonl")
    assert {c["set"] for c in changes} == {
        "fleurs_train_digit",
        "dev_mix_digit",
        "dev_mix_hangul_only",
        "dev_h",
    }


def test_score_guard_counts_changes_on_number_free_references(itn_eval, tmp_path) -> None:
    a_pairs, dev_mix, dev_h, stored = synthetic_sets()
    # 정답에는 수가 없는데 가설에 수 표현이 생긴 발화 2개(인식 오류에 ITN이 걸린 경우) → 2/80 > 1 %.
    stored["dev_h"]["zeroth/x/0"] = "가나 삼 년"
    stored["dev_h"]["zeroth/x/2"] = "마 가 사십 명"
    write_scoring_inputs(tmp_path, a_pairs, dev_mix, dev_h, stored)
    itn_eval.main(["--score", "--dev-hyps", "hyps/{name}.jsonl"])
    result = json.loads((tmp_path / "reports" / "followup" / "itn.json").read_text(encoding="utf-8"))
    h = result["guard"]["dev_h"]
    assert h["false_changes"] == 2 and h["false_change_rate"] == pytest.approx(2 / 80)
    assert result["verdict"]["guard_dev_h"] is False and result["verdict"]["adopted"] is False


def test_score_stops_when_the_dev_mix_redecode_does_not_reproduce(itn_eval, tmp_path) -> None:
    a_pairs, dev_mix, dev_h, stored = synthetic_sets()
    new_mix = {i: ("가" * 30 if k % 2 else h) for k, (i, h) in enumerate(stored["dev_mix"].items())}
    write_scoring_inputs(tmp_path, a_pairs, dev_mix, dev_h, stored, new_mix=new_mix)
    with pytest.raises(SystemExit) as excinfo:
        itn_eval.main(["--score", "--dev-hyps", "hyps/{name}.jsonl"])
    assert excinfo.value.code == itn_eval.REPRO_EXIT
    assert not (tmp_path / "reports" / "followup" / "itn.json").exists()


def test_decode_refuses_sealed_sets(itn_eval) -> None:
    with pytest.raises(SystemExit, match="dev sets only"):
        itn_eval.cmd_decode(argparse.Namespace(out_dir="o", sets="test_zeroth", overwrite=False))
