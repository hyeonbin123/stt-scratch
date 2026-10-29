"""숫자 역정규화(ITN) 표시 옵션의 숫자 발화 평가(docs/experiments.md 규칙 변경 30).

두 단계:
  --decode  (GPU 권장) 평가 세트 (가) = 숫자 때문에 학습에서 뺀 FLEURS train 발화
            (manifests/train_excluded.jsonl의 reason = digit, 등록 619발화) 가운데 학습 전사와 겹치지 않는
            것을 data/raw의 FLEURS train parquet에서 꺼내 16 kHz PCM으로 만들고
            (outputs/followup/itn/pcm/), 최종 모델로 full 문맥 greedy 디코딩한다(4단계 평가와 같은
            evaluate.decode_offline). 재현 확인용으로 dev_mix 700발화도 같은 경로로 다시 디코딩한다.
            결과: outputs/followup/itn/{세트}-full.jsonl, decode.json(출처) — 커밋하지 않는다.
  --score   (CPU) ITN 전후 CER_ns(전체, 숫자를 글자로 셈)와 발화 짝지은 95 % 구간, 숫자 없는 발화의 변경률,
            판정 → reports/followup/itn.json(정답·가설 텍스트 없음). 바뀐 표현 목록은 outputs/ 쪽
            changes.jsonl에(점검용, 커밋 안 함).
판정(등록): (가)와 (나) dev_mix 숫자 발화 절반 B 모두 ITN − 없음 ≤ −3포인트이고 짝지은 구간 상한 < 0,
          숫자 표현이 없는 정답의 발화(dev_mix 한글 전용·dev_H 각각)에서 가설이 바뀐 비율 ≤ 1 %.
          dev_mix 숫자 발화 절반 A는 규칙을 다듬은 세트라 보고만 한다.

사용법: uv run --no-sync python scripts/itn_eval.py --decode   (GPU 약 5분)
        uv run --no-sync python scripts/itn_eval.py --score    (CPU 1분 안)
"""

# [왜 이 세트인가] 모델 어휘에 숫자가 없어 숫자가 든 정답은 '전체' CER에서 그 글자가 늘 틀린다. FLEURS는
# 숫자를 아라비아 숫자로 적는 관례라 (가)·(나)에서 ITN이 그 격차를 얼마나 줄이는지 볼 수 있다. (가)는 학습에
# 쓰지 않은 오디오이고(숫자 사유로 제외), 겹침 검사(같은 정규화 문장, 같은 FLEURS 문장 id)는 등록 때 0이었다.
# [왜 재현 조건인가] (나)·가드 세트는 4단계에 저장된 full 가설(reports/stage4/hyps/)을 그대로 쓰고 (가)만 새로
# 디코딩한다. 같은 체크포인트로 dev_mix를 다시 디코딩해 저장 가설의 CER_ns가 0.02포인트 안으로 재현될 때만 두
# 세트를 같은 디코딩으로 보고 판정한다(어긋나면 종료 코드 5, 판정 없음).
# [가드의 정의] 한글 전용 발화 전체의 변경률(단순 변경률)은 오탐을 재지 못한다: Zeroth·Common Voice 정답은
# 숫자를 한글로 풀어 쓰므로(학습 전사에서 ITN이 바꾸는 발화: Zeroth 28.4 %, CV22 1.2 %, FLEURS 1.4 %) 그
# 비율은 진짜 수의 빈도를 잰다. 그래서 판정 가드는 정답 자체에 ITN이 바꿀 수 표현이 없는 발화(itn(정답) =
# 정답)에서 가설이 바뀐 비율로 두고, 단순 변경률은 함께 보고한다(규칙 변경 30).

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import stt_scratch.itn as itn_rules
from stt_scratch.confidence import half_of
from stt_scratch.data import PcmStore, Utterance, read_manifest
from stt_scratch.evaluate import check_vocab, decode_offline, load_model
from stt_scratch.itn import itn, itn_spans
from stt_scratch.metrics import (
    cluster_difference_interval,
    difference_interval,
    error_rate,
    score_ns,
)
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.text import Vocab, has_digit, normalize_hyp_ws

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import followup  # noqa: E402
import prepare_data  # noqa: E402

# 등록 값(docs/experiments.md 규칙 변경 30). 숫자는 여기 한곳에만 두고 판정 코드가 읽는다.
EVAL_SET = "fleurs_train_digit"  # (가)의 이름(출력 파일 이름에 쓴다)
EXPECTED_DIGIT = 619  # train_excluded.jsonl의 reason = digit 발화 수(0단계 결과). 다르면 멈춘다
THRESHOLD = 0.03  # ITN − 없음 ≤ −3포인트
GUARD = 0.01  # 숫자 표현이 없는 정답 발화의 가설 변경률 ≤ 1 %
REPRO_TOL = followup.REPRO_TOL  # 0.0002 = 0.02포인트(후속 측정의 재현 허용 폭과 같음)
REPRO_EXIT = 5  # 재현 조건 미충족 종료 코드(측정 스크립트가 다른 실패와 구별한다)
DEFAULT_CHECKPOINT = "outputs/stage4/trimtail/avg10.pt"
DEFAULT_DEV_HYPS = "reports/stage4/hyps/trimtail-avg10-{name}-full.jsonl"
STAGE4_RESULT = "reports/stage4/trimtail-avg10.json"  # 저장 가설의 기록 CER(채점 일치 확인용)
LENGTH_TOL = 8  # 디코딩한 표본 수와 매니페스트 길이(소수 셋째 자리 반올림 = 16표본 간격)의 허용 차


def sentence_id(utt_id: str) -> str | None:
    """FLEURS 발화 id(fleurs/<분할>/<문장 id>/<행 번호>)의 문장 id. FLEURS가 아니면 None."""
    parts = utt_id.split("/")
    return parts[2] if parts[0] == "fleurs" and len(parts) == 4 else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_hyps(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- 평가 세트 (가) -----------------------------------------------------------------------------------------
# 학습 제외 목록의 숫자 사유 발화에서, 정규화 정답이 학습 전사와 같거나 같은 FLEURS 문장(다른 화자의 낭독)이
# 학습 세트에 있는 발화를 뺀다. 순서는 매니페스트 순(limit은 앞에서부터, 연기 시험용).
# 반환: (Utterance 목록, 집계). 집계의 수는 결과 파일과 등록 문서에 그대로 적는다.
def eval_set(limit: int = 0) -> tuple[list[Utterance], dict]:
    rows = [r for r in read_jsonl(ROOT / "manifests" / "train_excluded.jsonl") if r["reason"] == "digit"]
    if len(rows) != EXPECTED_DIGIT:
        raise SystemExit(
            f"expected {EXPECTED_DIGIT} digit-excluded utterances (rule change 30), found {len(rows)}"
        )
    train = read_manifest(ROOT / "manifests" / "train.jsonl")
    train_texts = {u.text for u in train}
    train_sentences = {sentence_id(u.id) for u in train if u.corpus == "fleurs"}
    kept, overlap_text, overlap_sentence = [], 0, 0
    for r in rows:
        if r["text"] in train_texts:
            overlap_text += 1
        elif sentence_id(r["id"]) in train_sentences:
            overlap_sentence += 1
        else:
            kept.append(
                Utterance(r["id"], r["corpus"], r["text"], float(r["duration_s"]), False, r.get("speaker"))
            )
    info = {
        "digit_excluded": len(rows),
        "overlap_text": overlap_text,
        "overlap_sentence": overlap_sentence,
        "kept": len(kept),
        "sentences": len({sentence_id(u.id) for u in kept}),
        "over_20s": sum(u.duration_s > 20.0 for u in kept),
        "hours": round(sum(u.duration_s for u in kept) / 3600, 3),
    }
    if limit:
        kept = kept[:limit]
    return kept, info


# FLEURS train parquet를 전처리(scripts/prepare_data.py)와 같은 순서·id 규칙(iter_fleurs)으로 훑어 필요한
# 발화만 같은 디코더(decode_pcm16: mono, 16 kHz, int16)로 풀어 PcmStore 형식으로 쓴다. 길이가 매니페스트와
# 다르거나(다른 파일) 빠진 발화가 있으면 멈춘다. 반환: 출처 집계(parquet 파일 크기·내려받기 기록과의 대조
# 포함).
def extract_pcm(utts: list[Utterance], raw_dir: Path, pcm_dir: Path, name: str) -> dict:
    wanted = {u.id: u for u in utts}
    pcm_dir.mkdir(parents=True, exist_ok=True)
    writer = prepare_data.PcmWriter(pcm_dir, name)
    found: set[str] = set()
    try:
        for src in prepare_data.iter_fleurs(raw_dir, "train"):
            utt = wanted.get(src.id)
            if utt is None:
                continue
            pcm, _rate = prepare_data.decode_pcm16(src.blob)
            expected = round(utt.duration_s * prepare_data.SAMPLE_RATE)
            if abs(len(pcm) - expected) > LENGTH_TOL:
                raise SystemExit(f"{src.id}: decoded {len(pcm)} samples, the manifest says about {expected}")
            writer.write(src.id, pcm)
            found.add(src.id)
    finally:
        writer.close()
    missing = sorted(set(wanted) - found)
    if missing:
        raise SystemExit(
            f"{len(missing)} evaluation utterances are not in the FLEURS train parquet: {missing[:3]}"
        )
    files = sorted((raw_dir / "fleurs_ko" / "ko_kr" / "train").glob("*.parquet"))
    recorded = _recorded_sizes()
    return {
        "extracted": len(found),
        "parquet": [
            {
                "path": f.relative_to(raw_dir).as_posix(),
                "bytes": f.stat().st_size,
                "download_manifest_bytes": recorded.get(f.relative_to(raw_dir / "fleurs_ko").as_posix()),
            }
            for f in files
        ],
    }


# 내려받기 기록(reports/data/download-manifest-*.json)의 FLEURS 파일 크기. 없으면 빈 dict(대조 생략).
def _recorded_sizes() -> dict[str, int]:
    out: dict[str, int] = {}
    for path in sorted((ROOT / "reports" / "data").glob("download-manifest-*.json")):
        for source in json.loads(path.read_text(encoding="utf-8")):
            if source.get("name") == "fleurs_ko":
                out.update({f["path"]: f["bytes"] for f in source.get("files", [])})
    return out


# --- 디코딩 ------------------------------------------------------------------------------------------------
# 디코딩 명령(규칙 변경 30). 세트 이름((가)와 dev 세트만, 봉인 세트는 거부)과 결과 파일(있으면 --overwrite
# 없이 거부)을 계산 전에 확인하고, 어휘 토큰 수를 체크포인트 출력 크기와 대조한 뒤 세트마다 full 문맥
# greedy(evaluate.decode_offline, ChunkSpec(0, -1); cuda면 fp16 autocast, --fp32나 cpu면 fp32)로 디코딩한다.
# (가)는 먼저 parquet에서 PCM을 꺼내 <out-dir>/pcm/에 PcmStore 형식으로 쓴다. 출력:
# <out-dir>/{세트}-full.jsonl(id·ref·hyp)과 decode.json(출처·정밀도, 세트별 발화 수·시간·디코딩 초·가설
# SHA-256, (가)의 겹침 집계·parquet 크기).
def cmd_decode(args) -> None:
    out_dir = ROOT / args.out_dir
    names = [n.strip() for n in args.sets.split(",") if n.strip()]
    for name in names:
        if name != EVAL_SET and name not in followup.DEV_SETS:
            raise SystemExit(f"itn_eval decodes {EVAL_SET} and dev sets only, not {name!r}")
    outputs = [out_dir / f"{name}-full.jsonl" for name in names] + [out_dir / "decode.json"]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing[0]} exists; pass --overwrite to replace it")
    device = torch.device(args.device)
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    check_vocab(model, vocab)  # 어휘 토큰 수 ≠ 체크포인트 출력 크기면 디코딩 전에 멈춘다
    logmel = LogMel().to(device)
    amp = not args.fp32
    sets: dict[str, dict] = {}
    for name in names:
        if name == EVAL_SET:
            utts, info = eval_set(args.limit)
            info.update(extract_pcm(utts, ROOT / args.raw_dir, out_dir / "pcm", name))
            store = PcmStore(out_dir / "pcm", name)
        else:
            utts, store = followup.manifest(name, args.limit)
            info = {}
        start = time.perf_counter()
        hyps = decode_offline(model, vocab, utts, store, ChunkSpec(0, -1), device, logmel, amp=amp)
        elapsed = time.perf_counter() - start
        rows = [{"id": u.id, "ref": u.text, "hyp": hyps[u.id]} for u in utts]
        path = out_dir / f"{name}-full.jsonl"
        write_hyps(path, rows)
        sets[name] = {
            "utterances": len(utts),
            "hours": round(sum(u.duration_s for u in utts) / 3600, 3),
            "decode_s": round(elapsed, 1),
            "hyps_sha256": sha256_file(path),
            **info,
        }
        print(f"{name}: {len(utts)} utterances decoded in {elapsed:.1f} s", flush=True)
    meta = {
        "rule": "docs/experiments.md rule change 30",
        "mode": "full",
        "precision": "fp16-autocast" if amp and device.type == "cuda" else "fp32",
        "provenance": followup.provenance(args, device),
        "sets": sets,
    }
    write_json(out_dir / "decode.json", meta)


# --- 채점 ------------------------------------------------------------------------------------------------
# 발화 하나의 CER_ns 채점. 가설은 정답과 같은 학습 정규화(normalize_hyp_ws: <unk> 글자는 남김)를 거쳐
# whisper-ko-ft 정의로 채점한다. ITN 표시의 %·.·, 는 정답 정규화가 지운 기호라 같이 지워진다.
def _score(utts: list[Utterance], hyps: dict[str, str], apply_itn: bool):
    rows = []
    for u in utts:
        text = itn(hyps[u.id]) if apply_itn else hyps[u.id]
        s = score_ns(u.text, normalize_hyp_ws(text))
        if s is not None:
            rows.append((u, s))
    return rows


# ITN 전후의 짝지은 비교: 같은 발화·같은 정답 길이에서 rate(itn) − rate(없음)과 발화 단위 95 % 구간(등록
# 판정), groups가 있으면 같은 문장 낭독을 한 군집으로 다시 뽑는 사후 보조 구간(metrics의 군집 부트스트랩).
def paired(utts: list[Utterance], hyps: dict[str, str], groups: bool = False) -> dict:
    none, after = _score(utts, hyps, False), _score(utts, hyps, True)
    out: dict = {"utterances": len(none)}
    if not none:
        return {**out, "chars": 0, "none": None, "itn": None, "diff": None, "ci": None}
    e0 = np.array([s.edits for _, s in none], dtype=np.int64)
    e1 = np.array([s.edits for _, s in after], dtype=np.int64)
    lengths = np.array([s.length for _, s in none], dtype=np.int64)
    r0, r1 = error_rate(e0, lengths), error_rate(e1, lengths)
    low, high = difference_interval(e1, e0, lengths)
    out.update(
        {
            "chars": int(lengths.sum()),
            "none": {"rate": r0, "edits": int(e0.sum())},
            "itn": {"rate": r1, "edits": int(e1.sum())},
            "diff": r1 - r0,
            "ci": [low, high],
            "changed_utterances": sum(bool(itn_spans(hyps[u.id])) for u, _ in none),
            "improved_utterances": int((e1 < e0).sum()),
            "worsened_utterances": int((e1 > e0).sum()),
        }
    )
    if groups:
        keys = [sentence_id(u.id) for u, _ in none]
        out["sentences"] = len(set(keys))
        out["cluster_ci_sentence"] = list(cluster_difference_interval(e1, e0, lengths, keys))
    return out


# 숫자 없는 발화 가드. change_rate = 가설이 바뀐 발화 비율(단순 변경률, 보고), reference_number = 정답에 ITN이
# 바꿀 수 표현이 있는 발화, false_change_rate = 그런 표현이 없는 정답의 발화에서 가설이 바뀐 비율(판정
# 가드). CER_ns(한글 전용) 전후는 정답이 숫자를 한글로 쓴 관례와의 충돌 크기(보고).
def guard(utts: list[Utterance], hyps: dict[str, str]) -> dict:
    changed = [bool(itn_spans(hyps[u.id])) for u in utts]
    number_ref = [bool(itn_spans(u.text)) for u in utts]
    free = [c for c, n in zip(changed, number_ref, strict=True) if not n]
    cer = paired(utts, hyps)
    return {
        "utterances": len(utts),
        "changed": sum(changed),
        "change_rate": sum(changed) / len(utts) if utts else None,
        "reference_number_utterances": sum(number_ref),
        "number_free_utterances": len(free),
        "false_changes": sum(free),
        "false_change_rate": sum(free) / len(free) if free else None,
        "cer_ns_hangul_only": {k: cer[k] for k in ("none", "itn", "diff", "ci")},
    }


# 재현 조건(등록). 대조할 수 있는 경우(디코딩 체크포인트 = 저장 가설을 낸 최종 모델, dev_mix 전수 디코딩)에만
# 대조하고, 아니면 applicable False와 사유(판정 없음, 연기 시험). 대조: 같은 발화 집합에서 새 dev_mix 가설의
# CER_ns(전체·한글 전용)가 저장 가설과 각각 0.02포인트 안이면 met. 정확히 같은 가설 수도 적는다.
def check_reproduction(
    meta: dict, new: dict[str, str], stored: dict[str, str], utts: list[Utterance]
) -> dict:
    session = json.loads((ROOT / "reports" / "followup" / "session.json").read_text(encoding="utf-8"))
    prov = meta["provenance"]
    reasons = []
    if prov["checkpoint_sha256"] != session["provenance"]["checkpoint_sha256"]:
        reasons.append("checkpoint differs from the final model that produced the stored hypotheses")
    if prov.get("limit"):
        reasons.append(f"partial decode (--limit {prov['limit']})")
    if "dev_mix" not in meta["sets"]:
        reasons.append("dev_mix was not decoded")
    if reasons:
        return {"applicable": False, "met": None, "reasons": reasons}
    if set(new) != {u.id for u in utts} or not set(new) <= set(stored):
        return {"applicable": True, "met": False, "reasons": ["utterance sets differ"]}
    out: dict = {"applicable": True, "tolerance_pp": REPRO_TOL * 100, "precision": meta["precision"]}
    out["exact_match"] = sum(new[i] == stored[i] for i in new)
    out["utterances"] = len(new)
    met = True
    for subset, rows in (("all", utts), ("hangul_only", [u for u in utts if u.hangul_only])):
        a, b = _score(rows, new, False), _score(rows, stored, False)
        lengths = np.array([s.length for _, s in a], dtype=np.int64)
        rate_new = error_rate(np.array([s.edits for _, s in a]), lengths)
        rate_stored = error_rate(np.array([s.edits for _, s in b]), lengths)
        out[subset] = {"new": rate_new, "stored": rate_stored, "diff": rate_new - rate_stored}
        met = met and abs(rate_new - rate_stored) <= REPRO_TOL
    out["met"] = met
    return out


# 저장 가설의 채점 일치 확인: 이 도구의 ITN 없음 채점이 4단계 기록(reports/stage4/trimtail-avg10.json)의
# dev_mix 전체·dev_H 한글 전용 CER_ns와 같아야 한다(가설 정규화를 더해도 기록 수치가 그대로라는 확인). 기본
# 저장 가설을 읽을 때만 대조한다.
def scoring_consistency(dev_hyps: str, limit: int, dev_utts: dict, stored: dict) -> dict:
    if dev_hyps != DEFAULT_DEV_HYPS or limit:
        return {"applicable": False, "met": None}
    recorded = json.loads((ROOT / STAGE4_RESULT).read_text(encoding="utf-8"))["sets"]
    out: dict = {"applicable": True}
    met = True
    for name, subset in (("dev_mix", "all"), ("dev_h", "hangul_only")):
        rows = dev_utts[name] if subset == "all" else [u for u in dev_utts[name] if u.hangul_only]
        scored = _score(rows, stored[name], False)
        rate = error_rate(
            np.array([s.edits for _, s in scored]), np.array([s.length for _, s in scored], dtype=np.int64)
        )
        want = recorded[name]["modes"]["full"][subset]["cer_ns"]["rate"]
        out[f"{name}_{subset}"] = {"scored": rate, "recorded": want}
        met = met and abs(rate - want) < 1e-12
    out["met"] = met
    return out


# 채점 때의 코드 커밋(측정 스크립트가 커밋된 코드로 돌았는지 결과에서 확인하려고). git이 없으면 None.
def git_state() -> dict:
    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "git_dirty": None}


# 채점 명령(규칙 변경 30). 순서: 결과 파일 확인(있으면 --overwrite 없이 거부) → --decode 결과(decode.json,
# (가) 가설, 새 dev_mix 가설)와 저장 4단계 가설(--dev-hyps) 읽기 → 재현 조건(check_reproduction: 새 dev_mix
# 가설의 CER_ns가 저장 가설과 전체·한글 전용 각각 0.02포인트 안) → 채점 일치(scoring_consistency: ITN 없음
# 채점 = 4단계 기록) → (가)와 (나)(dev_mix 숫자 발화 절반 B)의 ITN 전후 짝지은 비교(판정), 절반 A·dev_mix
# 전체(보고), 가드 세트(dev_mix 한글 전용·dev_H 한글 전용)의 오탐 변경률·단순 변경률 → 판정 → 바뀐 표현 목록
# <out-dir>/changes.jsonl(전사 조각이 들어 있어 outputs/ 쪽)과 reports/followup/itn.json(정답·가설 텍스트
# 없음), 요약 한 줄(표준 출력). 판정은 재현 대조를 할 수 있을 때(최종 모델 체크포인트, --limit 없음, dev_mix를
# 디코딩함)만 적고 아니면 null(연기 시험). 종료 코드: 0 = 결과를 씀, 5(REPRO_EXIT) = 대조했는데 재현 조건
# 미충족(결과를 쓰지 않음), 1 = 그 밖의 실패(결과 파일이 이미 있음, 채점이 4단계 기록과 다름, --decode 결과
# 파일이 없음 등).
def cmd_score(args) -> None:
    out_path = ROOT / args.out
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    decode_dir = ROOT / args.out_dir
    meta = json.loads((decode_dir / "decode.json").read_text(encoding="utf-8"))
    limit = meta["provenance"].get("limit") or 0
    a_rows = read_jsonl(decode_dir / f"{EVAL_SET}-full.jsonl")
    a_utts = [Utterance(r["id"], "fleurs", r["ref"], 0.0, False) for r in a_rows]
    a_hyps = {r["id"]: r["hyp"] for r in a_rows}
    dev_utts, stored = {}, {}
    for name in followup.DEV_SETS:
        manifest_utts = read_manifest(ROOT / "manifests" / f"{name}.jsonl")
        dev_utts[name] = manifest_utts[:limit] if limit else manifest_utts
        stored[name] = {r["id"]: r["hyp"] for r in read_jsonl(ROOT / args.dev_hyps.format(name=name))}
    new_mix = {r["id"]: r["hyp"] for r in read_jsonl(decode_dir / "dev_mix-full.jsonl")}
    repro = check_reproduction(meta, new_mix, stored["dev_mix"], dev_utts["dev_mix"])
    if repro["applicable"] and not repro["met"]:
        print(json.dumps(repro, ensure_ascii=False), flush=True)
        print(
            "the dev_mix re-decode does not reproduce the stored full hypotheses; "
            "not judging (rule change 30)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(REPRO_EXIT)
    consistency = scoring_consistency(args.dev_hyps, limit, dev_utts, stored)
    if consistency["applicable"] and not consistency["met"]:
        raise SystemExit(f"scoring does not reproduce the recorded stage-4 CER: {consistency}")
    digit = [u for u in dev_utts["dev_mix"] if has_digit(u.text)]
    half_a = [u for u in digit if half_of(u.id) == "A"]
    half_b = [u for u in digit if half_of(u.id) == "B"]
    mix_hangul = [u for u in dev_utts["dev_mix"] if u.hangul_only]
    result = {
        "rule": "docs/experiments.md rule change 30",
        "evaluated": {
            **git_state(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "itn_sha256": sha256_file(Path(itn_rules.__file__)),  # 이 채점에 쓴 규칙 파일
        },
        "decode": meta,
        "dev_hypotheses": {
            name: {
                "path": args.dev_hyps.format(name=name),
                "sha256": sha256_file(ROOT / args.dev_hyps.format(name=name)),
            }
            for name in followup.DEV_SETS
        },
        "reproduction": repro,
        "scoring_consistency": consistency,
        "metric": "CER_ns over all characters (digits and Latin letters count), ITN minus none",
        "a_fleurs_train_digit": paired(a_utts, a_hyps, groups=True),
        "b_dev_mix_digit_half_b": paired(half_b, stored["dev_mix"], groups=True),
        "dev_mix_digit_half_a_development": paired(half_a, stored["dev_mix"], groups=True),
        "dev_mix_all": paired(dev_utts["dev_mix"], stored["dev_mix"]),
        "split": {
            "rule": "SHA-256 of the utterance id (UTF-8): first byte even = A (rule development), "
            "odd = B (verdict)",
            "dev_mix_digit": len(digit),
            "half_a": len(half_a),
            "half_b": len(half_b),
        },
        "guard": {
            "dev_mix_hangul_only": guard(mix_hangul, stored["dev_mix"]),
            "dev_h": guard([u for u in dev_utts["dev_h"] if u.hangul_only], stored["dev_h"]),
        },
    }
    result["verdict"] = verdict(result) if repro["applicable"] else None
    _write_changes(decode_dir / "changes.jsonl", a_utts, a_hyps, digit, mix_hangul, dev_utts["dev_h"], stored)
    write_json(out_path, result)
    a, b = result["a_fleurs_train_digit"], result["b_dev_mix_digit_half_b"]
    g = result["guard"]
    print(
        f"(a) {a['utterances']} utts diff {a['diff']} ci {a['ci']} | "
        f"(b) {b['utterances']} utts diff {b['diff']} ci {b['ci']} | "
        f"guard false-change dev_mix {g['dev_mix_hangul_only']['false_change_rate']} "
        f"dev_h {g['dev_h']['false_change_rate']} | verdict {result['verdict']}",
        flush=True,
    )


# 등록 판정. 단순 변경률(한글 전용 발화 전체) ≤ 1 %로 읽은 값도 따로 적는다(판정에는 쓰지 않음).
def verdict(result: dict) -> dict:
    def improves(block: dict) -> bool | None:
        if block["diff"] is None:
            return None
        return block["diff"] <= -THRESHOLD and block["ci"][1] < 0

    def within(value: float | None) -> bool | None:
        return None if value is None else value <= GUARD

    g = result["guard"]
    out = {
        "a_fleurs_train_digit": improves(result["a_fleurs_train_digit"]),
        "b_dev_mix_digit_half_b": improves(result["b_dev_mix_digit_half_b"]),
        "guard_dev_mix_hangul_only": within(g["dev_mix_hangul_only"]["false_change_rate"]),
        "guard_dev_h": within(g["dev_h"]["false_change_rate"]),
    }
    out["adopted"] = all(v is True for v in out.values())
    out["raw_change_guard"] = {
        "dev_mix_hangul_only": within(g["dev_mix_hangul_only"]["change_rate"]),
        "dev_h": within(g["dev_h"]["change_rate"]),
    }
    return out


# 바뀐 표현 목록(점검용, 커밋하지 않는 outputs/ 쪽에 쓴다: 전사 조각이 들어 있다).
def _write_changes(path: Path, a_utts, a_hyps, digit, mix_hangul, dev_h, stored) -> None:
    rows = []
    for name, utts, hyps in (
        (EVAL_SET, a_utts, a_hyps),
        ("dev_mix_digit", digit, stored["dev_mix"]),
        ("dev_mix_hangul_only", mix_hangul, stored["dev_mix"]),
        ("dev_h", dev_h, stored["dev_h"]),
    ):
        for u in utts:
            spans = itn_spans(hyps[u.id])
            if spans:
                rows.append(
                    {
                        "set": name,
                        "id": u.id,
                        "reference_has_number": bool(itn_spans(u.text)),
                        "spans": [[s.before, s.after] for s in spans],
                    }
                )
    write_hyps(path, rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="숫자 역정규화(ITN) 표시 옵션의 숫자 발화 평가(docs/experiments.md 규칙 변경 30). "
        "--decode는 학습에서 뺀 FLEURS 숫자 발화(가)와 재현 확인용 dev_mix를 최종 모델로 full greedy "
        "디코딩하고, --score는 ITN 전후 CER_ns(전체)·숫자 없는 발화 변경률로 판정한다. 둘을 함께 주면 "
        "차례로 한다.",
    )
    p.add_argument("--decode", action="store_true", help="평가 세트를 꺼내 디코딩한다(GPU 권장)")
    p.add_argument("--score", action="store_true", help="디코딩 결과와 저장 가설로 채점·판정한다(CPU)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="디코딩할 체크포인트(저장소 기준)")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="디코딩 장치(cuda면 fp16 autocast, cpu면 fp32). 판정은 재현 조건을 만족할 때만 한다",
    )
    p.add_argument("--fp32", action="store_true", help="cuda에서도 fp16 autocast를 쓰지 않는다")
    p.add_argument(
        "--sets",
        default=f"{EVAL_SET},dev_mix",
        help="디코딩할 세트(쉼표 구분). 연기 시험은 dev_h도 넣어 가드 세트 가설까지 새로 만든다",
    )
    p.add_argument("--limit", type=int, default=0, help="세트마다 앞 N발화만(연기 시험용; 0 = 전부)")
    p.add_argument("--raw-dir", default="data/raw", help="원본 데이터 폴더(FLEURS train parquet가 있는 곳)")
    p.add_argument(
        "--out-dir",
        default="outputs/followup/itn",
        help="디코딩 결과·PCM·decode.json·changes.jsonl 폴더(저장소 기준, 커밋하지 않음)",
    )
    p.add_argument(
        "--dev-hyps",
        default=DEFAULT_DEV_HYPS,
        help="(나)·가드 세트의 full 가설 경로 형식({name} = dev_mix·dev_h). 기본은 4단계 저장 가설",
    )
    p.add_argument("--out", default="reports/followup/itn.json", help="판정 결과 JSON(저장소 기준)")
    p.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어쓴다")
    return p


# 진입점: 표준 출력을 UTF-8로 바꾸고(Windows 콘솔 기본 cp949에서 한글 메시지가 깨지지 않게) --decode·--score를
# 이 순서로 한다(둘 다 없으면 멈춘다; 함께 주면 디코딩 직후 그 결과로 채점한다).
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.decode and not args.score:
        raise SystemExit("pass --decode, --score or both")
    if args.decode:
        cmd_decode(args)
    if args.score:
        cmd_score(args)


if __name__ == "__main__":
    main(sys.argv[1:])
