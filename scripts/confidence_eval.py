"""확정 자막 어절 신뢰도 평가(docs/experiments.md 규칙 변경 28, CPU).

scripts/final_pass_dump.py의 덤프(dev_mix·dev_H)에서 확정 자막의 가설 어절마다 후보 신뢰도 4개
(stt_scratch.confidence CANDIDATES)와 정오 라벨을 만들고, 등록 규칙대로 선택과 판정을 나눈다:
  재현 조건  덤프의 발화별 확정 가설이 후속 측정 P1의 확정 가설(reports/followup/hyps/
             <세트>-session-rule2_0.8-final_pass.jsonl)을 재현하는지(한글 전용 CER_ns가 기록값과
             0.02포인트 안; 정확히 같은 발화 수도 보고). 어긋나면 판정하지 않고 멈춘다. 체크포인트·구성이
             P1과 다르거나 일부 덤프(--limit)면 대조하지 않고 판정 없이 쓴다(연기 시험)
  대상       한글 전용 발화(주지표와 같은 부분집합). dev_mix를 발화 id SHA-256 첫 바이트 짝·홀로
             절반 A·B로 나눈다
  선택       절반 A에서만: 후보마다 AUROC → 가장 높은 후보(동점은 등록 순서), 그 후보로 Platt 보정
  판정       절반 B 한 번: 보정 확률의 오류 어절 검출 AUROC ≥ 0.80이고 ECE(15구간) ≤ 0.05면 채택
             (선택 필드·화면 밑줄 옵션을 별도 커밋으로 제공; 서버 기본값은 그대로). 발화 부트스트랩
             95 % 구간도 적는다
  참고       절반 B의 후보별 AUROC·보정 전 ECE·Brier·밑줄 임계 0.5의 정밀도·재현율·띄어쓰기 민감도,
             숫자·라틴을 포함한 절반 B, dev_H(같은 선택·보정을 그대로 적용)
결과: reports/followup/confidence.json(있으면 --overwrite 없이 거부).

사용법: uv run --no-sync python scripts/confidence_eval.py
  (덤프 뒤 CPU에서; 연기 시험은 --dump-dir·--out을 work/ 아래로)
"""

# [왜 절반을 나누는가] 후보 4개와 보정 매개변수를 같은 자료에서 고르고 판정하면 판정 수치가 낙관적으로
# 치우친다. 절반 A에서 고른 것을 한 번도 보지 않은 절반 B에서 한 번만 재야 등록 판정이 된다(공통 선택 규칙과
# 같은 생각).
# [왜 재현 조건인가] 덤프는 P1 측정 뒤 다시 돌린 GPU 실행이다. 같은 체크포인트·구성인데도 확정 가설이
# 달라졌다면(코드 변경, 비결정성) 그 신뢰도는 측정된 서비스 확정 자막의 것이 아니므로 판정하지 않는다(P1
# (d)와 같은 허용 폭).
# [dev_H] Zeroth 영역은 학습 문장 암기로 과신이 더 클 수 있어, 같은 보정을 그대로 썼을 때의 ECE를 참고로 본다.

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from stt_scratch import confidence as conf
from stt_scratch.data import read_manifest
from stt_scratch.evaluate import score
from stt_scratch.text import Vocab, normalize_hyp_ws, normalize_train

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import followup  # noqa: E402
from final_pass_dump import load_dump  # noqa: E402
from stage6 import rate  # noqa: E402

# 등록 값(docs/experiments.md 규칙 변경 28). 숫자는 여기 한곳에만 두고 판정 코드가 읽는다.
JUDGED_SET = "dev_mix"  # 절반 A(선택)·B(판정)
REFERENCE_SET = "dev_h"  # 참고치(판정 없음)
AUROC_THRESHOLD = 0.80
ECE_THRESHOLD = 0.05
TAU = 0.5  # 화면 밑줄 임계: 보정 확률(정답일 확률) < 0.5인 어절(보고만)
REPRO_TOL = followup.REPRO_TOL  # 0.0002 = 0.02포인트(P1 (d)와 같은 허용 폭)
REPRO_EXIT = 5  # 재현 조건 미충족 종료 코드(측정 스크립트가 다른 실패와 구별한다)
P1_RULE2 = 0.8  # P1 판정 구성의 rule2(확정 가설 파일 이름에 들어감)
# P1 세션 구성 가운데 덤프가 같아야 하는 값(step 경로·잡음 길이는 session.json에서 읽는다).
P1_CONFIG = {
    "chunk": 8,
    "left": 128,
    "rule1_s": 2.4,
    "rule2_s": P1_RULE2,
    "rule3_s": 20.0,
    "reset_cache_on_endpoint": False,
    "final_pass": "full",
    "final_pass_bucket": 128,
    "precision": "fp16-autocast",
}


def load_vocab() -> tuple[Vocab, str]:
    """manifests/vocab.json과 그 SHA-256(덤프 메타의 vocab_sha256과 대조)."""
    path = ROOT / "manifests" / "vocab.json"
    return Vocab.load(path), hashlib.sha256(path.read_bytes()).hexdigest()


# 정답만 필요하므로 PCM 저장소는 열지 않는다(data/가 없는 곳에서도 평가할 수 있게). 봉인 test 세트는 거부한다.
def read_dev_manifest(name: str):
    if name not in followup.DEV_SETS:
        raise SystemExit(f"confidence_eval reads dev sets only, not {name!r}")
    return read_manifest(ROOT / "manifests" / f"{name}.jsonl")


def cer_ns_hangul(utts, hyps: dict[str, str]) -> float:
    """주지표(한글 전용 CER_ns): P1 기록값과 같은 채점(evaluate.score)."""
    return rate(score(utts, hyps))


def git_state() -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "git_dirty": None}
    return {"git_head": head, "git_dirty": bool(status)}


# 재현 조건(등록). 대조할 수 있는 경우(덤프 체크포인트 = P1 체크포인트, 구성 = P1 구성, 전수 덤프)에만
# 대조하고, 아니면 applicable False와 그 사유를 돌려준다(판정 없음). 대조: 발화 집합이 같고 한글 전용 CER_ns
# 차가 0.02포인트 이하이면 within_tolerance. 정확히 같은 확정 가설 발화 수(exact_match)도 함께 적는다(GPU
# 비결정성이 없으면 전수 같다).
def check_reproduction(metas: dict[str, dict], utts_by_set: dict) -> dict:
    session = json.loads((ROOT / "reports" / "followup" / "session.json").read_text(encoding="utf-8"))
    expected = {**P1_CONFIG, "step_path": session["step_path"], "tail_s": session["tail_noise_s"]}
    reasons = []
    for name, meta in metas.items():
        prov = meta["provenance"]
        if prov["checkpoint_sha256"] != session["provenance"]["checkpoint_sha256"]:
            reasons.append(f"{name}: checkpoint differs from the P1 measurement")
        if prov.get("limit"):
            reasons.append(f"{name}: partial dump (--limit {prov['limit']})")
        diff = {k: [meta["config"].get(k), v] for k, v in expected.items() if meta["config"].get(k) != v}
        if diff:
            reasons.append(f"{name}: configuration differs from P1: {diff}")
    if reasons:
        return {"applicable": False, "met": None, "reasons": reasons}
    out: dict = {"applicable": True, "tolerance_pp": REPRO_TOL * 100}
    met = True
    for name, meta in metas.items():
        hyps = {row["id"]: row["hyp_final"] for row in meta["utterances"]}
        path = ROOT / "reports" / "followup" / "hyps" / f"{name}-session-rule2_{P1_RULE2}-final_pass.jsonl"
        recorded = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    recorded[row["id"]] = row["hyp"]
        dump_cer = cer_ns_hangul(utts_by_set[name], hyps)
        recorded_cer = session["runs"][f"{name}_rule2_{P1_RULE2}"]["cer_ns_final_pass"]
        ok = set(hyps) == set(recorded) and abs(dump_cer - recorded_cer) <= REPRO_TOL
        out[name] = {
            "utterances": len(recorded),
            "exact_match": sum(hyps.get(uid) == hyp for uid, hyp in recorded.items()),
            "cer_ns_dump": dump_cer,
            "cer_ns_recorded": recorded_cer,
            "within_tolerance": ok,
        }
        met = met and ok
    out["met"] = met
    return out


# 가설 어절 표. 발화마다 내보낸 final 세그먼트를 순서대로 읽어 어절을 만들고(세그먼트 텍스트가 argmax의 greedy
# 축약과 같은지, 어절 열이 확정 가설의 공백 분리와 같은지 확인; 어긋나면 덤프가 깨진 것이라 멈춘다), 정답(학습
# 정규화)과 정렬해 라벨을 단다. hangul_only면 한글 전용 발화만(판정·참고 지표), 아니면 모든 발화(숫자·라틴
# 포함 참고치). 출력: utt_ids·utt_half(발화별), utt_counts(발화별 정렬 수), word_utt(어절 → 발화 번호),
# correct, spacing_only, scores(후보 → 어절 점수), 모두 numpy 배열(utt_counts는 Counter 목록).
def word_table(meta: dict, arrays: dict, vocab: Vocab, utts, hangul_only: bool = True) -> dict:
    if [row["id"] for row in meta["utterances"]] != [u.id for u in utts]:
        raise SystemExit(f"{meta['set']}: the dump does not cover the manifest utterances in order")
    by_id = {u.id: u for u in utts}
    segs = meta["segments"]
    utt_ids, utt_half, utt_counts = [], [], []
    word_utt, correct, spacing = [], [], []
    scores: dict[str, list[float]] = {c: [] for c in conf.CANDIDATES}
    for row in meta["utterances"]:
        u = by_id[row["id"]]
        if hangul_only and not u.hangul_only:
            continue
        words = []
        for si in row["segments"]:
            seg = segs[si]
            if not (seg["emitted"] and seg["kind"] == "final"):
                continue
            part = slice(seg["offset"], seg["offset"] + seg["frames"])
            argmax = arrays["argmax"][part]
            decoded = vocab.decode([tok for tok, _, _ in conf.token_spans(argmax, vocab.blank_id)])
            if decoded != seg["text"]:
                raise SystemExit(f"{u.id} segment {seg['segment']}: argmax does not decode to the final text")
            words.extend(
                conf.segment_words(argmax, arrays["max_logprob"][part], arrays["entropy"][part], vocab)
            )
        texts = [w.text for w in words]
        if texts != normalize_hyp_ws(row["hyp_final"]).split():
            raise SystemExit(f"{u.id}: dumped words differ from the final hypothesis")
        labels = conf.hyp_word_labels(normalize_train(u.text).split(), texts)
        index = len(utt_ids)
        utt_ids.append(u.id)
        utt_half.append(conf.half_of(u.id))
        utt_counts.append(Counter(labels.counts))
        for word, ok, only_space in zip(words, labels.correct, labels.spacing_only, strict=True):
            word_utt.append(index)
            correct.append(ok)
            spacing.append(only_space)
            for c in conf.CANDIDATES:
                scores[c].append(conf.word_score(word, c))
    return {
        "utt_ids": utt_ids,
        "utt_half": np.asarray(utt_half, dtype=object),
        "utt_counts": utt_counts,
        "word_utt": np.asarray(word_utt, dtype=np.int64),
        "half": np.asarray([utt_half[i] for i in word_utt], dtype=object),
        "correct": np.asarray(correct, dtype=bool),
        "spacing_only": np.asarray(spacing, dtype=bool),
        "scores": {c: np.asarray(v, dtype=np.float64) for c, v in scores.items()},
    }


# 절반('A'·'B', None이면 전체)만 남긴다. 어절이 없는 발화도 발화 수에 세고(부트스트랩은 발화를 뽑는다), 발화
# 번호를 부분집합 안에서 0부터 다시 매긴다.
def subset(table: dict, half: str | None) -> dict:
    keep = [i for i, h in enumerate(table["utt_half"]) if half is None or h == half]
    remap = np.full(len(table["utt_half"]), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    mask = remap[table["word_utt"]] >= 0 if len(table["word_utt"]) else np.zeros(0, dtype=bool)
    counts = Counter()
    for i in keep:
        counts.update(table["utt_counts"][i])
    return {
        "n_utts": len(keep),
        "word_utt": remap[table["word_utt"][mask]],
        "correct": table["correct"][mask],
        "spacing_only": table["spacing_only"][mask],
        "scores": {c: v[mask] for c, v in table["scores"].items()},
        "counts": {k: int(counts.get(k, 0)) for k in ("equal", "substitute", "insert", "delete")},
    }


def _share(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


# 절반 A 선택: 후보마다 원래 점수의 AUROC(높을수록 확신)를 재 가장 높은 것(동점은 CANDIDATES 순서; max는 처음
# 최댓값을 고른다)을 고르고 그 후보로 Platt 보정을 맞춘다. 절반 A에 정답·오류 어절이 함께 있지 않으면(무작위
# 모델의 연기 시험처럼 모두 오류) AUROC가 없으므로 첫 후보를 쓰고 note를 남긴다(Platt는 평활 목표값이라 유한;
# 그런 자료의 판정은 절반 B AUROC도 없어 not_met이 된다).
def select(half_a: dict) -> dict:
    aurocs = {c: conf.auroc(half_a["scores"][c], half_a["correct"]) for c in conf.CANDIDATES}
    chosen = max(conf.CANDIDATES, key=lambda c: -1.0 if aurocs[c] is None else aurocs[c])
    out = {
        "half": "A",
        "utterances": half_a["n_utts"],
        "words": int(half_a["correct"].size),
        "error_words": int((~half_a["correct"]).sum()),
        "auroc_half_a": aurocs,
        "chosen": chosen,
        "platt": conf.platt_fit(half_a["scores"][chosen], half_a["correct"]),
    }
    if all(v is None for v in aurocs.values()):
        out["note"] = "half A has no words of both classes; the first registered candidate is used"
    return out


# 부분집합 하나의 지표(선택한 후보·A의 Platt를 그대로 적용). AUROC·ECE는 보정 확률로 재고(채택 시 화면이 쓰는
# 값; Platt 기울기가 양수면 AUROC는 원래 점수와 같다), 둘 다 발화 부트스트랩 구간을 붙인다. 나머지는 보고만.
def report(sub: dict, chosen: str, platt: dict) -> dict:
    s, y, wu, n = sub["scores"][chosen], sub["correct"], sub["word_utt"], sub["n_utts"]
    p = conf.platt_apply(s, platt["a"], platt["b"])
    errors = ~y
    under = p < TAU
    lenient = y | sub["spacing_only"]
    c = sub["counts"]
    ref_words = c["equal"] + c["substitute"] + c["delete"]
    return {
        "utterances": n,
        "words": int(y.size),
        "error_words": int(errors.sum()),
        "error_word_share": _share(errors),
        "alignment": c,
        "wer": (c["substitute"] + c["delete"] + c["insert"]) / ref_words if ref_words else None,
        "auroc": conf.auroc(p, y),
        "auroc_ci": conf.bootstrap_interval(wu, n, lambda w: conf.auroc(p, y, w)),
        "ece": conf.ece(p, y),
        "ece_ci": conf.bootstrap_interval(wu, n, lambda w: conf.ece(p, y, weights=w)),
        "ece_uncalibrated": conf.ece(s, y),
        "brier": float(np.mean((p - y) ** 2)) if y.size else None,
        "mean_confidence": float(p.mean()) if y.size else None,
        "underline": {
            "tau": TAU,
            "underlined_share": _share(under),
            "precision": _share(errors[under]),
            "recall": _share(under[errors]),
        },
        "spacing_only_errors": int(sub["spacing_only"].sum()),
        "spacing_as_correct": {"auroc": conf.auroc(p, lenient), "ece": conf.ece(p, lenient)},
        "candidates_auroc": {cand: conf.auroc(sub["scores"][cand], y) for cand in conf.CANDIDATES},
    }


# 등록 판정(절반 B 점추정). 구간 조건은 표현에만 쓴다: 채택이면서 AUROC 구간 하한 ≥ 0.80이고 ECE 구간 상한 ≤
# 0.05면 "met", 채택이지만 구간이 임계를 넘으면 "met_by_point_estimate_only", 그 밖은 "not_met".
def verdict(half_b: dict) -> dict:
    au, e = half_b["auroc"], half_b["ece"]
    auroc_met = au is not None and au >= AUROC_THRESHOLD
    ece_met = e is not None and e <= ECE_THRESHOLD
    low, high = half_b["auroc_ci"]["low"], half_b["ece_ci"]["high"]
    intervals = low is not None and low >= AUROC_THRESHOLD and high is not None and high <= ECE_THRESHOLD
    adopted = auroc_met and ece_met
    return {
        "auroc_threshold": AUROC_THRESHOLD,
        "ece_threshold": ECE_THRESHOLD,
        "auroc_met": auroc_met,
        "ece_met": ece_met,
        "adopted": adopted,
        "intervals_within_thresholds": intervals,
        "expression": ("met" if intervals else "met_by_point_estimate_only") if adopted else "not_met",
        "adopted_means": "offer the optional final-event field words [{w, conf}] and an underline display "
        "option in a separate commit; server defaults and the default page stay unchanged",
    }


def cmd_eval(args) -> None:
    out_path = ROOT / args.out
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it")
    vocab, vocab_sha = load_vocab()
    dump_dir = ROOT / args.dump_dir
    dumps, utts = {}, {}
    for name in (JUDGED_SET, REFERENCE_SET):
        meta, arrays = load_dump(dump_dir, name)
        if meta["vocab_sha256"] != vocab_sha:
            raise SystemExit(f"{name}: the dump was decoded with a different manifests/vocab.json")
        dumps[name] = (meta, arrays)
        # 연기 시험의 일부 덤프(--limit N)는 매니페스트 앞 N발화만 덮는다. 그 경우 재현 대조는 하지 않는다.
        limit = meta["provenance"].get("limit") or 0
        manifest_utts = read_dev_manifest(name)
        utts[name] = manifest_utts[:limit] if limit else manifest_utts
    repro = check_reproduction({name: meta for name, (meta, _) in dumps.items()}, utts)
    if repro["applicable"] and not repro["met"]:
        print(json.dumps(repro, ensure_ascii=False), flush=True)
        print(
            "the dump does not reproduce the recorded P1 final hypotheses; not judging (rule change 28)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(REPRO_EXIT)
    mix_meta, mix_arrays = dumps[JUDGED_SET]
    table = word_table(mix_meta, mix_arrays, vocab, utts[JUDGED_SET])
    half_a, half_b = subset(table, "A"), subset(table, "B")
    selection = select(half_a)
    chosen, platt = selection["chosen"], selection["platt"]
    b_all = subset(word_table(mix_meta, mix_arrays, vocab, utts[JUDGED_SET], hangul_only=False), "B")
    p_all = conf.platt_apply(b_all["scores"][chosen], platt["a"], platt["b"])
    ref_meta, ref_arrays = dumps[REFERENCE_SET]
    half_b_report = report(half_b, chosen, platt)
    result = {
        "rule": "docs/experiments.md rule change 28",
        "evaluated": {**git_state(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "dumps": {
            name: {
                "npz_sha256": meta["npz_sha256"],
                "provenance": meta["provenance"],
                "config": meta["config"],
                "graph": meta.get("graph"),
                "frames": meta["frames"],
                "segments": len(meta["segments"]),
                "utterances": len(meta["utterances"]),
            }
            for name, (meta, _) in dumps.items()
        },
        "reproduction": repro,
        "subset": "hangul-only utterances",
        "split": {
            "rule": "SHA-256 of the utterance id (UTF-8): first byte even = A (selection), odd = B (verdict)",
            "half_a_utterances": half_a["n_utts"],
            "half_b_utterances": half_b["n_utts"],
        },
        "selection": selection,
        "dev_mix_half_b": half_b_report,
        "dev_mix_half_b_all_utterances": {
            "utterances": b_all["n_utts"],
            "words": int(b_all["correct"].size),
            "error_words": int((~b_all["correct"]).sum()),
            "auroc": conf.auroc(p_all, b_all["correct"]),
            "ece": conf.ece(p_all, b_all["correct"]),
        },
        "dev_h_reference": report(
            subset(word_table(ref_meta, ref_arrays, vocab, utts[REFERENCE_SET]), None), chosen, platt
        ),
        "verdict": verdict(half_b_report) if repro["applicable"] else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    b = result["dev_mix_half_b"]
    print(
        f"chosen {chosen} (half A AUROC {selection['auroc_half_a'][chosen]}), half B AUROC {b['auroc']}, "
        f"ECE {b['ece']}, verdict {result['verdict']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="확정 자막 어절 신뢰도 평가(docs/experiments.md 규칙 변경 28, CPU). 2차 패스 "
        "덤프의 재현을 확인한 뒤 dev_mix 한글 전용 절반 A에서 후보·Platt 보정을 고르고 절반 B에서 "
        "AUROC ≥ 0.80·ECE ≤ 0.05로 판정한다(dev_H는 참고치)."
    )
    p.add_argument(
        "--dump-dir",
        default="outputs/followup/final_pass_dump",
        help="scripts/final_pass_dump.py 결과 폴더(저장소 기준; dev_mix·dev_h 덤프가 모두 있어야 함)",
    )
    p.add_argument(
        "--out",
        default="reports/followup/confidence.json",
        help="결과 JSON(저장소 기준). 연기 시험은 work/ 아래로",
    )
    p.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어쓴다")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_eval(args)


if __name__ == "__main__":
    main(sys.argv[1:])
