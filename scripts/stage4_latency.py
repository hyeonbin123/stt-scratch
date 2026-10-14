"""MFA 한국어 3.0 정렬을 기준으로 한 4단계 방출 지연 측정(docs/experiments.md 4단계, 규칙 변경 18).

체크포인트 하나와 simulated 청크 크기(기본 320·640 ms, 좌측 128)마다: 청크 마스크 아래에서 greedy로
디코딩하고(simulated 스트리밍. 2단계 점검에서 캐시 스트리밍과 fp16 반올림 범위 안에서 일치했고 프레임
argmax 일치율은 99.99 %였다), 모든 토큰의 방출 프레임(그 토큰 CTC 구간의 첫 프레임)과 그 토큰을 낼 수 있는
시각을 구한다.
    t_emit = min(((4 C (k + 1) - 1) * 160 + 400) / 16000, duration),  k = frame // C
(청크 전체의 음성이 도착해야 하며, 계산 시간은 넣지 않는다). 기준: MFA 단어 구간을 전사의 띄어쓰기 어절로
다시 묶은 것(MFA는 한국어 어절을 형태소로 나눈다). 가설 어절은 편집 거리로 정답 어절과 맞추고, 정확히
맞은 어절의 지연은 t_emit(가설 어절의 마지막 음절) − 정답 어절 끝 시각이다.
  FTD: 첫 정답 어절의 지연, LTD: 마지막 정답 어절의 지연(그 어절을 맞힌 발화만)
  MAD: 한 발화에서 맞은 모든 어절의 평균 지연
발화들에 대한 P50/P90, 포함 발화 수, 같은 디코딩의 CER_ns를 reports/stage4/latency-<name>.json에 쓴다.

사용법: uv run python scripts/stage4_latency.py --checkpoint outputs/stage2/runA/avg10.pt --name runA-avg10
"""

# [왜 필요한가] 측정 정의는 규칙 변경 18에 측정 전에 등록했다. 계산 시간은 빼고 '청크의 음성이 모두 도착해야
# 그 청크의 토큰이 나온다'는 알고리즘 지연만 본다. 그래서 같은 모델·같은 청크에서 학습 기법(TrimTail,
# Peak-First)이 토큰을 얼마나 일찍 내게 하는지만 비교할 수 있다.
# [설계 이유] 실제 캐시 스트리밍 대신 청크 마스크를 건 offline forward(simulated)로 디코딩한다. docstring의
# 2단계 점검 결과처럼 둘이 일치하므로, 배치로 빠르게 잴 수 있는 simulated를 쓴다.
# [연결] scripts/stage4.py가 세 변형마다 부르고, 판정에 쓰는 ftd_by_utt_ms(발화별 FTD)도 여기서 나온다. MFA
# 입력은 scripts/export_mfa_corpus.py가 만들고, 정렬 결과는 work/mfa/aligned/<세트>/*.csv에 있다.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# numpy: 편집 거리 표와 백분위. torch: 모델 forward.
import numpy as np
import torch

# 프로젝트 내부 모듈: 평가와 같은 체크포인트 로더·autocast·길이 버킷 배치·특징 계산·채점, 청크 마스크 설정,
# 어휘(공백 토큰으로 어절을 나눈다).
from stt_scratch.data import PcmStore, read_manifest
from stt_scratch.evaluate import autocast, batch_features, duration_batches, load_model, score
from stt_scratch.model.features import LogMel
from stt_scratch.model.mask import ChunkSpec
from stt_scratch.text import Vocab

# SILENCE: MFA 정렬 결과에서 단어가 아닌 레이블(빈 칸, 묵음 sil·sp, 엡실론, 인식 못 한 구간 spn·<unk>). 어절
# 묶기에서 뺀다.
ROOT = Path(__file__).resolve().parents[1]
SILENCE = {"", "<eps>", "sil", "sp", "spn", "<unk>"}


# MFA CSV(--output_format csv) 하나를 읽어 단어 층(Type == words)의 (시작 s, 끝 s, 레이블) 목록으로 돌려준다.
# 음소 층과 묵음 레이블은 버린다.
def load_alignment(path: Path) -> list[tuple[float, float, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Type", "words") == "words" and r["Label"].strip() not in SILENCE:
                rows.append((float(r["Begin"]), float(r["End"]), r["Label"].strip()))
    return rows


# 왜: MFA 한국어 모델은 어절을 형태소로 나눠 정렬하는데, 지연은 모델 출력과 같은 단위인 어절로 재기로
# 했다(규칙 변경 18). 조각 레이블을 순서대로 이어 붙여 어절과 정확히 같아지면 그 첫 조각 시작~마지막 조각 끝을
# 어절 구간으로 한다.
# 이어 붙인 글자가 어절과 다르거나 조각이 남거나 모자라면 None을 돌려주고, 그 발화는 지연 측정에서 빠진다(빠진
# 수를 보고한다).
def regroup(words: list[str], pieces: list[tuple[float, float, str]]) -> list[tuple[float, float]] | None:
    """MFA 조각을 전사의 띄어쓰기 어절로 다시 묶어 어절별 (시작, 끝) 구간을 돌려준다. 안 맞으면 None."""
    spans, i = [], 0
    for w in words:
        if i >= len(pieces):
            return None
        start, end, acc = pieces[i][0], pieces[i][1], ""
        while i < len(pieces) and len(acc) < len(w):
            acc += pieces[i][2]
            end = pieces[i][1]
            i += 1
        if acc != w:
            return None
        spans.append((start, end))
    return spans if i == len(pieces) else None


# 편집 거리의 삽입·삭제·치환 비용은 모두 1이다. 틀린 어절은 '언제 나왔는가'를 정의할 수 없으므로 지연은
# 정확히 맞힌 어절에서만 잰다.
def match_words(hyp: list[str], ref: list[str]) -> dict[int, int]:
    """가설·정답 어절을 편집 거리로 맞추고, 정확히 같은 어절끼리의 짝 {정답 위치: 가설 위치}를 돌려준다."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    # 표의 오른쪽 아래에서 거꾸로 따라가며 경로를 복원한다. 같은 비용이면 일치(대각선)를 먼저 고른다.
    out, i, j = {}, n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


# 입력은 프레임별 argmax 목록이다. CTC greedy와 같은 규칙(blank가 아니고 직전 프레임과 다른 라벨이 시작될
# 때 토큰 하나)이다(규칙 변경 18).
def emissions(best: list[int], blank: int = 0) -> list[tuple[int, int]]:
    """방출 토큰마다 (토큰, 프레임)을 돌려준다. 프레임은 blank 아닌 라벨이 이어지는 구간의 첫 프레임이다."""
    out, prev = [], blank
    for f, tok in enumerate(best):
        if tok != blank and tok != prev:
            out.append((tok, f))
        prev = tok
    return out


# 인코더 프레임이 속한 청크의 음성이 모두 도착한 시각(초). 청크 k의 마지막 mel 프레임 번호는 4C(k+1) − 1이고
# 그 창(400표본)이 끝나는 표본이 (번호 × 160 + 400)이므로, 정리하면 0.04 × C × (k + 1) + 0.015 s다(규칙 변경
# 18의 식). 발화 길이를 넘지 않게 자른다. 계산 시간은 넣지 않는다.
def emit_time(frame: int, chunk: int, duration: float) -> float:
    k = frame // chunk
    return min(((4 * chunk * (k + 1) - 1) * 160 + 400) / 16000, duration)


# 초 단위 지연 목록을 ms로 바꿔 개수·P50·P90·평균을 돌려준다.
def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "mean": None}
    a = np.asarray(xs) * 1000
    return {
        "n": len(xs),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
    }


# 한 세트·한 청크 크기의 지연과 CER을 잰다.
# 입력: refs = {발화 id: 정답 어절 구간 목록}(어절 묶기에 성공한 발화만). 출력: CER_ns와 구간, FTD·LTD·MAD
# 백분위, 발화 수 집계(정렬 있음·첫 어절 맞음·마지막 어절 맞음), 발화별 FTD·LTD(ms; 4단계 판정의 공통 발화
# 비교에 씀).
@torch.no_grad()
def measure(model, vocab, name, utts, store, refs, chunk, left, device, logmel, amp) -> dict:
    spec = ChunkSpec(chunk, left)
    hyps, ftd, ltd, mad = {}, [], [], []
    ftd_by_utt, ltd_by_utt = {}, {}
    counts = {"utterances": len(utts), "with_alignment": 0, "first_matched": 0, "last_matched": 0}
    for batch in duration_batches(utts, 400.0):
        mel, frames = batch_features(utts, batch, store, device, logmel)
        # 청크 마스크(청크 C, 좌측 left)를 건 simulated forward. 학습·평가와 같은 fp16 autocast다.
        with autocast(device, amp):
            logits, _, lengths = model(mel, frames, spec)
        best = logits.float().argmax(-1).cpu()
        for row, j in enumerate(batch):
            u = utts[j]
            # 패딩을 뺀 유효 프레임만으로 방출을 구하고, CER 채점용 가설 문자열도 같은 토큰으로 만든다.
            em = emissions(best[row, : int(lengths[row])].tolist())
            hyps[u.id] = vocab.decode([t for t, _ in em])
            ref_spans = refs.get(u.id)
            if ref_spans is None:
                continue
            counts["with_alignment"] += 1
            # 가설 어절과, 어절마다 마지막 음절의 방출 프레임을 모은다(공백 토큰에서 어절을 끊는다).
            words, frames_of_words, cur, last = [], [], "", -1
            for tok, f in em:
                if tok == vocab.space_id:
                    if cur:
                        words.append(cur)
                        frames_of_words.append(last)
                    cur = ""
                else:
                    cur += vocab.decode([tok])
                    last = f
            if cur:
                words.append(cur)
                frames_of_words.append(last)
            # 정답 어절과 맞춘 뒤, 맞은 어절마다 지연 = 가설 어절 마지막 음절의 방출 시각 − 정답 어절 끝 시각.
            # 모델이 어절 끝보다 일찍 내면 음수가 될 수 있다.
            ref_words = u.text.split()
            matched = match_words(words, ref_words)
            delays = {
                r: emit_time(frames_of_words[h], chunk, u.duration_s) - ref_spans[r][1]
                for r, h in matched.items()
            }
            # FTD: 첫 정답 어절이 맞았을 때만, LTD: 마지막 정답 어절이 맞았을 때만 그 발화를 센다. 그래서 두
            # 모델의 FTD 표본 집단이 다를 수 있고, 4단계는 공통 발화 비교를 보조로 보고했다.
            if 0 in delays:
                counts["first_matched"] += 1
                ftd.append(delays[0])
                ftd_by_utt[u.id] = delays[0] * 1000
            if len(ref_words) - 1 in delays:
                counts["last_matched"] += 1
                ltd.append(delays[len(ref_words) - 1])
                ltd_by_utt[u.id] = delays[len(ref_words) - 1] * 1000
            # MAD: 맞은 어절이 하나라도 있는 발화의 평균 지연(보조 지표).
            if delays:
                mad.append(float(np.mean(list(delays.values()))))
    scored = score(utts, hyps)
    return {
        "set": name,
        "chunk_frames": chunk,
        "chunk_ms": chunk * 40,
        "left": left,
        "cer_ns": scored["hangul_only"]["cer_ns"]["rate"],
        "cer_ci": scored["hangul_only"]["cer_ns"]["ci"],
        "ftd_ms": percentiles(ftd),
        "ltd_ms": percentiles(ltd),
        "mad_ms": percentiles(mad),
        "counts": counts,
        "ftd_by_utt_ms": ftd_by_utt,
        "ltd_by_utt_ms": ltd_by_utt,
    }


# 세트·청크별로 측정해 JSON 하나로 저장한다. 인자: --checkpoint, --name(결과 파일 이름),
# --sets(dev_mix,dev_h), --chunks(8,16 = 320·640 ms), --left(128), --alignments(MFA 코퍼스·정렬 결과 루트,
# 기본 work/mfa), --report-dir, --fp32, --device.
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--sets", default="dev_mix,dev_h")
    p.add_argument("--chunks", default="8,16")
    p.add_argument("--left", type=int, default=128)
    p.add_argument("--alignments", default="work/mfa")
    p.add_argument("--report-dir", default="reports/stage4")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    amp = not args.fp32 and device.type == "cuda"
    model, _ = load_model(ROOT / args.checkpoint, device)
    vocab = Vocab.load(ROOT / "manifests" / "vocab.json")
    logmel = LogMel().to(device)
    results = []
    for name in args.sets.split(","):
        # 한글 전용 발화만 쓴다(MFA 코퍼스도 같은 기준으로 내보냈다). 등록 수: dev_mix 633, dev_H 1,113발화.
        utts = [u for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl") if u.hangul_only]
        store = PcmStore(ROOT / "data" / "pcm", name)
        # MFA 파일 이름 → 매니페스트 id 대응표를 읽고, 정렬 CSV가 있는 발화마다 어절 구간을 만든다. 정렬이
        # 실패해 CSV가 없는 발화는 건너뛰고, 어절로 묶이지 않는 발화 수는 unregrouped로 보고한다(실측: dev_H
        # 1,107개 정렬 중 1,106개 묶임; docs/ai-verification.md).
        ids = json.loads((ROOT / args.alignments / "corpus" / f"{name}.ids.json").read_text(encoding="utf-8"))
        refs, unregrouped = {}, 0
        text_by_id = {u.id: u.text for u in utts}
        for stem, uid in ids.items():
            path = ROOT / args.alignments / "aligned" / name / f"{stem}.csv"
            if not path.exists():
                continue
            spans = regroup(text_by_id[uid].split(), load_alignment(path))
            if spans is None:
                unregrouped += 1
                continue
            refs[uid] = spans
        for chunk in (int(c) for c in args.chunks.split(",")):
            r = measure(model, vocab, name, utts, store, refs, chunk, args.left, device, logmel, amp)
            r["alignment_files"] = len(ids)
            r["unregrouped"] = unregrouped
            results.append(r)
            print(
                f"{name} chunk {chunk * 40} ms: CER {r['cer_ns'] * 100:.2f} %, FTD P50/P90 "
                f"{r['ftd_ms']['p50']:.0f}/{r['ftd_ms']['p90']:.0f} ms (n {r['ftd_ms']['n']}), LTD P50/P90 "
                f"{r['ltd_ms']['p50']:.0f}/{r['ltd_ms']['p90']:.0f} ms (n {r['ltd_ms']['n']})",
                flush=True,
            )
    # 체크포인트와 정밀도를 함께 기록한다(모든 수치에 체크포인트·디코딩 모드를 적는 규칙). 시작 체크포인트
    # SHA-256은 scripts/stage4.py가 덧붙인다.
    out = ROOT / args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "precision": "fp16-autocast" if amp else "fp32",
        "results": results,
    }
    (out / f"latency-{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
