"""텍스트 정규화, 채점 정규화, 음절 토큰화(규칙: docs/experiments.md "공통").

텍스트 쪽 규칙을 한곳에 모은다: 학습·평가 정답 정규화(normalize_train), 채점 정규화(normalize_score), 가설용
정규화(normalize_hyp_ws), 숫자·라틴 판별, 음절 어휘(Vocab), CTC 최소 프레임 수(ctc_min_frames).
"""

# [왜 한 파일인가] docs/experiments.md "텍스트 정규화"가 '학습 정답·평가 정답 공통, 스크립트 하나'를 규칙으로
# 정했다. 전처리(scripts/prepare_data.py)와 평가(metrics.py)가 같은 함수를 써야 학습 어휘와 채점 기준이
# 어긋나지 않는다.
# [출력 단위] 한글 음절 + 공백 + <unk>, blank = 0. 조사에서 음절이 CER 기준 최선이었고 학습 전사의 음절은 약
# 1.3k라 어휘가 작다(docs/survey/README.md "결정" 출력 단위).

from __future__ import annotations

# 표준 라이브러리만 쓴다: 어휘 저장(json), 정규식(re), 유니코드 NFC 정규화(unicodedata).
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# 판별용 정규식: 완성형 한글 음절(가~힣, 11,172자), 아라비아 숫자, 라틴 문자. 자모 단독(ㄱ, ㅏ)은 HANGUL에
# 들지 않는다.
HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]")
# 한글 음절, 라틴 문자, 숫자, 공백이 아닌 글자는 모두 공백으로 바꾼다.
_NON_TEXT = re.compile(r"[^가-힣A-Za-z0-9\s]")
# whisper-ko-ft 채점기: 글자·숫자·공백·작은따옴표가 아닌 것(\w가 한글을 포함한다). 밑줄은 \w에 들지만 따로
# 지운다.
_SCORE_PUNCT = re.compile(r"[^\w\s']|_")

# 어휘 앞 세 자리의 특수 토큰. blank는 CTC의 '출력 없음', <unk>는 학습 전사에 없던 음절, 공백은 어절 경계다.
BLANK, UNK, SPACE = "<blank>", "<unk>", " "
# <unk>를 디코딩한 글자: U+318D(한글 자모 아래아). 단어 문자(\w)라 whisper-ko-ft 채점 정규화가 지우지 않으므로
# 모델이 낸 <unk> 하나는 틀린 글자 하나로 센다. 정답에는 절대 나올 수 없다(normalize_train은 한글 음절, 라틴
# 문자, 숫자, 공백만 남긴다).
UNK_CHAR = "\u318d"
# 가설용 normalize_train: 똑같되 UNK_CHAR만 남긴다(CER_ws 채점에 쓴다).
_NON_TEXT_HYP = re.compile("[^\uac00-\ud7a3A-Za-z0-9\\s" + UNK_CHAR + "]")


# docs/experiments.md "텍스트 정규화" 1~3. NFC는 조합형 자모를 완성형 음절로 합치고, 문장부호·기호처럼
# 한글·라틴·숫자·공백이 아닌 문자는 공백이 된다. 숫자·라틴 발화를 학습에서 빼는 일은
# scripts/prepare_data.py가 한다.
def normalize_train(text: str) -> str:
    """학습·평가 정답의 공통 정규화: NFC, 문장부호 → 공백, 연속 공백을 하나로."""
    text = unicodedata.normalize("NFC", text)
    text = _NON_TEXT.sub(" ", text)
    return " ".join(text.split())


# CER_ws 채점에서 가설 쪽에만 쓴다. 처음에는 가설도 normalize_train으로 정규화해 <unk>가 공백으로 사라지면서
# CER_ws에서 오류로 세지 않았다(적대적 검토에서 발견, docs/ai-verification.md).
def normalize_hyp_ws(text: str) -> str:
    """가설용 학습 정규화: normalize_train과 같되 UNK_CHAR를 남긴다.

    그래서 모델이 낸 <unk>가 CER_ws에서도 틀린 글자 하나로 남는다(정답에는 UNK_CHAR가 없다).
    """
    text = unicodedata.normalize("NFC", text)
    text = _NON_TEXT_HYP.sub(" ", text)
    return " ".join(text.split())


# 주지표 CER_ns에 쓴다. 비교 기준(whisper-turbo 제로샷 4.48 %, LoRA 1.96 %)을 낸 whisper-ko-ft와 같은
# 규칙이어야 수치를 나란히 놓을 수 있다. 글자·숫자·공백·작은따옴표 외 문자는 먼저 공백으로 바꾼 뒤
# 지운다(docs/experiments.md "지표").
def normalize_score(text: str) -> str:
    """whisper-ko-ft(한국어)와 같은 채점 정규화: 소문자화, 문장부호 제거, 작은따옴표와 모든 공백 제거.

    그래서 CER을 공백 없는 글자열에서 잰다.
    """
    text = _SCORE_PUNCT.sub(" ", text.lower())
    return "".join(text.replace("'", " ").split())


# 숫자·라틴 포함 여부. 전처리가 학습 제외(숫자 619, 라틴 200건, 모두 FLEURS)에 쓰고, 평가 매니페스트의
# hangul_only 표시도 이것으로 정한다(docs/experiments.md 0단계 결과).
def has_digit(text: str) -> bool:
    return bool(DIGIT.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN.search(text))


# 정규화된 전사에는 한글·라틴·숫자·공백만 남으므로, 비어 있지 않고 숫자·라틴이 없는지만 보면 된다. 모델
# 어휘가 한글 음절뿐이라, 평가는 전체 수치와 함께 이 '한글 전용' 부분집합 수치를 보고한다(주 판정은 한글 전용
# CER_ns).
def hangul_only(text: str) -> bool:
    """정규화된 전사가 비어 있지 않고 한글 음절과 공백만으로 이뤄졌으면 True."""
    return bool(text) and not has_digit(text) and not has_latin(text)


# 3번부터는 학습 전사의 한글 음절이 빈도순으로 온다. frozen이라 만든 뒤 바뀌지 않는다.
# manifests/vocab.json으로 저장되어 학습·평가·서비스가 같은 파일을 읽는다(실제 1,288개).
@dataclass(frozen=True)
class Vocab:
    """음절 어휘: 0 = CTC blank, 1 = <unk>, 2 = 공백, 그다음 한글 음절."""

    tokens: tuple[str, ...]

    # 특수 토큰 순서가 다르면 CTC blank(0) 가정과 디코딩이 깨지므로 생성 때 거부한다.
    def __post_init__(self) -> None:
        if self.tokens[:3] != (BLANK, UNK, SPACE):
            raise ValueError("vocab must start with <blank>, <unk>, space")

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def space_id(self) -> int:
        return 2

    def __len__(self) -> int:
        return len(self.tokens)

    # 빈도 내림차순, 같은 빈도는 글자 순으로 정렬해 같은 입력이면 항상 같은 어휘가 나온다. 어휘는 CTC 길이
    # 제외까지 끝난 뒤의 학습 전사로 만든다(규칙 변경 5).
    # 호출 위치: scripts/prepare_data.py.
    @classmethod
    def build(cls, texts: list[str]) -> Vocab:
        """`texts`(이미 정규화된 학습 전사)에 한 번 이상 나온 한글 음절 전부로 빈도순 어휘를 만든다."""
        counts: dict[str, int] = {}
        for text in texts:
            for ch in HANGUL.findall(text):
                counts[ch] = counts.get(ch, 0) + 1
        ordered = sorted(counts, key=lambda ch: (-counts[ch], ch))
        return cls((BLANK, UNK, SPACE, *ordered))

    # 어휘에 없는 글자는 학습에 없던 음절, 숫자, 라틴 문자 같은 것이다.
    def encode(self, text: str) -> list[int]:
        """정규화된 전사를 글자 단위 id로 바꾼다. 공백 → space id, 어휘에 없는 글자 → <unk>."""
        index = self._index()
        return [index.get(ch, self.unk_id) for ch in text]

    # <unk>를 "<unk>" 글자가 아니라 U+318D 한 글자로 내보내는 것은 규칙 변경 7이다. 이전에는 "unk" 세 글자로
    # 풀려 라틴 정답과 우연히 맞을 수 있었다.
    def decode(self, ids: list[int]) -> str:
        """id 목록을 문자열로 바꾼다. blank는 버리고, <unk>는 UNK_CHAR로 내보내 틀린 글자 하나로 센다."""
        return "".join(UNK_CHAR if i == self.unk_id else self.tokens[i] for i in ids if i != self.blank_id)

    # 글자 → id 사전을 처음 쓸 때 한 번만 만들어 둔다. frozen dataclass라 일반 대입이 막혀
    # object.__setattr__로 캐시 속성을 붙인다.
    def _index(self) -> dict[str, int]:
        cached = getattr(self, "_cached_index", None)
        if cached is None:
            cached = {tok: i for i, tok in enumerate(self.tokens)}
            object.__setattr__(self, "_cached_index", cached)
        return cached

    # 어휘를 {"tokens": [...]} JSON으로 저장·읽기. ensure_ascii=False로 한글을 그대로 쓴다.
    # scripts/prepare_data.py는 줄바꿈을 LF로 고정하려고 같은 형식을 직접 쓰고, 학습·평가·서비스는 load로
    # manifests/vocab.json을 읽는다.
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"tokens": list(self.tokens)}, ensure_ascii=False, indent=0), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Vocab:
        return cls(tuple(json.loads(path.read_text(encoding="utf-8"))["tokens"]))


# 같은 토큰이 연달아 나오려면 사이에 blank 프레임이 하나 있어야 하기 때문에 반복 수만큼 더한다.
# 쓰임: 전처리는 '출력 프레임 ≥ 이 값 + 1'인 발화만 학습에 넣고(docs/experiments.md 데이터 분할), train.py의
# TrimTail은 이 조건 아래로 자르지 않는다.
def ctc_min_frames(ids: list[int]) -> int:
    """CTC가 이 정답 열을 내는 데 필요한 최소 출력 프레임 수: U(토큰 수) + 연속 같은 토큰 수."""
    repeats = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a == b)
    return len(ids) + repeats
