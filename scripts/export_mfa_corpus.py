"""Export Hangul-only dev utterances as an MFA corpus (Stage 4 reference alignment, experiments.md 4단계).

Writes <out>/<set>/<safe id>.wav (16 kHz int16 mono) and .lab (normalized transcript) plus ids.json
mapping the file stems back to manifest ids. Usage:
  uv run python scripts/export_mfa_corpus.py --sets dev_mix,dev_h --out work/mfa/corpus
"""

# [파일 역할] dev 세트(dev_mix, dev_h)의 한글 전용 발화를 Montreal Forced Aligner(MFA)가 읽는 코퍼스
# 형식(발화마다 .wav + 같은 이름의 .lab 전사)으로 내보낸다. 파일 이름 → 매니페스트 id
# 대응표(<out>/<세트>.ids.json)도 쓴다.
# [왜 필요한가] 4단계 방출 지연(FTD/LTD) 측정의 기준은 '실제로 그 어절이 끝난 시각'인데, 이것을 MFA 3.4.3 +
# korean_mfa 3.0 강제 정렬로 만든다(docs/experiments.md 4단계, 규칙 변경 18). 6단계 끝점 지연(EP50/EP90)의
# 기준 시각도 같은 정렬을 쓴다. 한글 전용 발화만 내보내는 것은 등록 정의가 "한글 전용 dev 발화"이기 때문이다.
# [흐름] 세트마다 매니페스트에서 한글 전용 발화를 골라 → PCM 저장소의 int16 음성을 16 kHz wav로 → 정규화된
# 전사를 .lab으로 → 대응표 저장. 그 뒤 MFA는 README의 명령(mfa align ... --single_speaker --output_format
# csv)으로 따로 돌린다.
# [연결] 정렬 결과(work/mfa/aligned/<세트>/*.csv)와 대응표는 scripts/stage4_latency.py와 scripts/stage6.py가
# 읽는다.

from __future__ import annotations

import argparse
import json
from pathlib import Path

# numpy: 메모리 맵 PCM 저장소에서 int16 표본을 꺼낸다. soundfile: 그 표본을 16 kHz PCM_16 wav로 쓴다(원래 값
# 그대로라 변환 손실이 없다).
import numpy as np
import soundfile as sf

# 프로젝트 내부 모듈: 매니페스트 읽기와 0단계에서 만든 PCM 저장소(학습·평가가 읽는 것과 같은 음성).
from stt_scratch.data import PcmStore, read_manifest

# 저장소 루트. 매니페스트·PCM 저장소·출력 폴더를 이 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]


# 매니페스트 id(예: cv22/dev/common_voice_ko_….mp3, zeroth/train/10024)를 파일 이름으로 쓸 수 있게 바꾼다.
# '/'는 폴더 구분자라 '__'로 바꾸고, CV의 '.mp3'는 지워 '….mp3.wav' 같은 혼란스러운 이름을 피한다. 원래 id는
# 대응표로 되찾는다.
def safe(utt_id: str) -> str:
    return utt_id.replace("/", "__").replace(".mp3", "")


# 세트별로 MFA 코퍼스를 쓴다. 인자: --sets(쉼표로 나눈 매니페스트 이름, 기본 dev_mix,dev_h), --out(코퍼스
# 루트, 기본 work/mfa/corpus; 커밋하지 않는 작업 폴더).
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="dev_mix,dev_h")
    p.add_argument("--out", default="work/mfa/corpus")
    args = p.parse_args()
    for name in args.sets.split(","):
        out = ROOT / args.out / name
        out.mkdir(parents=True, exist_ok=True)
        # 숫자·라틴 문자가 든 발화는 뺀다. 지연 측정의 등록 정의가 한글 전용 dev 발화만 쓴다(규칙 변경 18:
        # dev_mix 633, dev_H 1,113발화).
        utts = [u for u in read_manifest(ROOT / "manifests" / f"{name}.jsonl") if u.hangul_only]
        store = PcmStore(ROOT / "data" / "pcm", name)
        mapping = {}
        for u in utts:
            # PCM 저장소의 이 발화 구간(시작 위치, 표본 수)을 int16 그대로 꺼낸다. 학습·평가와 같은 음성이므로
            # 정렬 시각이 모델 입력과 같은 시간축을 갖는다.
            offset, length = store.index[u.id]
            pcm = np.asarray(store.bin[offset : offset + length], dtype=np.int16)
            stem = safe(u.id)
            sf.write(out / f"{stem}.wav", pcm, 16000, subtype="PCM_16")
            # .lab은 매니페스트의 정규화된 전사(문장부호를 지운 한글 음절과 공백)다. 모델 정답과 같은 어절
            # 경계를 써야 나중에 MFA 결과를 전사의 어절로 다시 묶을 수 있다.
            (out / f"{stem}.lab").write_text(u.text + "\n", encoding="utf-8")
            mapping[stem] = u.id
        # 대응표는 세트 폴더 밖(<out>/<세트>.ids.json)에 쓴다. scripts/stage6.py가 이 위치에서 읽어 MFA 결과
        # 파일 이름을 매니페스트 id로 되돌린다.
        (ROOT / args.out / f"{name}.ids.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        print(f"{name}: {len(utts)} Hangul-only utterances -> {out}", flush=True)


if __name__ == "__main__":
    main()
