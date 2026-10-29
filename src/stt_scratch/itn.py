"""숫자 역정규화(ITN): 확정 자막에서 한글로 읽은 수를 아라비아 숫자로 바꾸는 표시 전용 규칙.

모델 어휘에는 숫자가 없어(docs/experiments.md "어휘") 모델은 '사십 명', '천 구백 칠십 팔 년'처럼 숫자를 한글
읽기로만 낸다. 이 모듈은 그런 가설을 '40명', '1978년'으로 바꿔 자막을 읽기 쉽게 한다. 측정 규칙은
docs/experiments.md 규칙 변경 30("숫자 역정규화(ITN) 표시 옵션과 숫자 발화 평가" 절)이고, 이 규칙을 그 절의
판정 전에 고정했다.

다루는 것(표로 고정):
  - 한자어 수사열: 일~구, 십·백·천, 만·억·조. 띄어쓰기와 무관하게 이어 읽는다('이천 십 팔' = '이천십팔').
    뒤에 단위(UNITS: 년·월·일·시·분·초·명·개·번·회·퍼센트·원·달러·킬로미터·미터 등)가 오면 바꾸고, 단위가
    없으면 수사 모양이 분명한 두 음절 이상만 바꾼다.
  - 소수: '삼 점 오', '영 쩜 칠 육'(점·쩜 뒤는 한 자리씩) → 3.5, 0.76.
  - 고유어 수사: 한·두·세·네·다섯 … 열 … 아흔(1~99) + 명·개·살·시·시간·개월만.
  - 달 이름 '유월'·'시월'(6월·10월의 불규칙 읽기).
바꾸지 않는 것: 단독 '일'·'이' 같은 한 음절 수사(뒤에 단위가 없을 때), 범위 읽기('칠팔 년'), 어림말('수천
조', '몇 십 년'), 숫자를 한 자리씩 읽은 열('공일공'), 서수 '첫·두 번째'의 고유어, '이 분'·'이 점'처럼 지시어
'이'와 겹치는 표현.

출력 표기: 1만 미만은 숫자 그대로('2018년'), 1만 이상이면서 만 단위로 떨어지면 만·억·조를 섞어 쓰고('5만',
'3억 5000만'), 아니면 세 자리 쉼표('15,000'). 퍼센트·프로는 '%', 길이·무게 단위는 km·m·cm·mm·kg·g·mg 기호,
나머지 단위는 한글 그대로 숫자 뒤에 붙인다.

주의: 표시 전용이다. 학습 정답 정규화(text.normalize_train)와 채점 정규화(text.normalize_score)는 바꾸지
않는다. 규칙형이라 중의적인 표현에서 틀릴 수 있어(예: '사십 일'은 '40일'로 읽고 '41'로 읽지 않는다) 서버
기본값은 끈다(Zeroth 정답처럼 숫자를 한글로 풀어 쓰는 관례와도 충돌한다).
"""

# [왜 규칙형인가] 한국어 ITN 공개 도구가 드물고(NeMo의 WFST 문법은 영어·러시아어 중심), 이 저장소는 외부
# 텍스트 LM·새 의존성을 쓰지 않는다. 표와 작은 문법으로 고정하면 결정적이고, 판정 세트를 보기 전에 규칙을
# 고정할 수 있다.
# [오탐을 막는 장치] 한 음절 수사('일', '이', '사', '구', '만', '조' …)는 흔한 낱말·조사와 겹친다('사회',
# '구분', '이번', '만일'). 그래서 (1) 표현은 어절 시작에서만 시작하고, (2) 단위 뒤 나머지는 조사·접미사
# 표(PARTICLES)로만 이뤄져야 하며('세상'의 '상'은 거부), (3) 한 음절 수사에 단위를 붙여 쓴 어절은
# 붙여 써도 안전한 단위(GLUE_SAFE)만(여러 어절에 걸친 수사라도 단위와 같은 어절의 수사가 한 자리 수
# 하나이면 같음: '백 사장'), (4) 띄어 쓴 단독 '이'는 지시어와 겹치지 않는 단위(TWO_SAFE)만, (5) 어림말
# 뒤의 자릿수 한 음절과 단위 + 조사로 쪼개지는 낱말(COMPOUND_NOUNS)은 받지 않는다. 남는 중의성은
# 규칙 변경 30의 오탐 변경률 가드(정답에 수 표현이 없는 발화에서 가설이 바뀐 비율)가 잰다.
# [연결] scripts/itn_eval.py가 저장된 full 가설과 새 디코딩 가설에 적용해 ITN 전후 CER_ns(전체)를 잰다.
# 서버 옵션(--itn, final 이벤트의 display_text)은 판정 뒤 별도 커밋이고 지금은 연결하지 않는다.

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# --- 한자어 수사 ------------------------------------------------------------------------------------------
# 한 자리 수, 작은 자릿수(그룹 안), 큰 자릿수(만 단위 그룹). 영·공(0)은 소수 부분과 '영 점 …'에만 쓴다(정수
# 자리에서는 '영'이 '영어'·'영화'와 겹치고, 숫자를 한 자리씩 읽은 열은 다루지 않는다).
SINO_DIGITS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SINO_SMALL = {"십": 10, "백": 100, "천": 1000}
SINO_LARGE = {"만": 10**4, "억": 10**8, "조": 10**12}
SINO = frozenset(SINO_DIGITS) | frozenset(SINO_SMALL) | frozenset(SINO_LARGE)
MULTIPLIERS = frozenset(SINO_SMALL) | frozenset(SINO_LARGE)
# 소수점 뒤 한 자리씩 읽는 숫자. Zeroth 전사는 소수점을 '쩜'으로 적는다('삼십 사 쩜 사 퍼센트').
FRACTION_DIGITS = {"영": "0", "공": "0", **{k: str(v) for k, v in SINO_DIGITS.items()}}
DECIMAL_POINTS = ("점", "쩜")

# --- 단위 ---------------------------------------------------------------------------------------------------
# 읽은 단위 → 표시. 긴 것부터 맞춘다('개월' > '개', '시간' > '시', '년대' > '년', '세기' > '세').
UNITS: dict[str, str] = {
    # 때·날짜
    "년": "년",
    "년대": "년대",
    "년간": "년간",
    "년도": "년도",
    "년생": "년생",
    "주년": "주년",
    "개월": "개월",
    "월": "월",
    "주": "주",
    "주일": "주일",
    "주간": "주간",
    "일": "일",
    "일간": "일간",
    "시간": "시간",
    "시": "시",
    "분기": "분기",
    "분": "분",
    "초": "초",
    "세기": "세기",
    # 셈 단위
    "명": "명",
    "인": "인",
    "개국": "개국",
    "개사": "개사",
    "개소": "개소",
    "개": "개",
    "번지": "번지",
    "번": "번",
    "회": "회",
    "차": "차",
    "위": "위",
    "등": "등",
    "층": "층",
    "호": "호",
    "권": "권",
    "대": "대",
    "장": "장",
    "편": "편",
    "건": "건",
    "점": "점",
    "배": "배",
    "세": "세",
    "살": "살",
    "가지": "가지",
    # 돈
    "원": "원",
    "달러": "달러",
    "유로": "유로",
    "엔": "엔",
    "위안": "위안",
    "파운드": "파운드",
    # 비율·길이·무게(기호로 표시: 한국어 위키백과·뉴스 표기 관례)
    "퍼센트": "%",
    "프로": "%",
    "킬로미터": "km",
    "센티미터": "cm",
    "밀리미터": "mm",
    "미터": "m",
    "킬로그램": "kg",
    "밀리그램": "mg",
    "그램": "g",
    # 그 밖의 측정 단위(한글 그대로)
    "킬로": "킬로",
    "톤": "톤",
    "리터": "리터",
    "마일": "마일",
    "피트": "피트",
    "인치": "인치",
    "도": "도",
    "헥타르": "헥타르",
    "평": "평",
}
_UNITS_LONGEST_FIRST = tuple(sorted(UNITS, key=len, reverse=True))

# 한 음절 수사에 붙여 써도(같은 어절) 안전한 단위. 나머지(명·분·시·회·원·번·개·대·점·차·세 …)는 한 음절
# 수사와 붙으면 흔한 낱말이 된다: 사명·구분·일시·사회·사원·이번·조개·일대·이점·오차·만세. '유로'도
# 빼 둔다('이유로' = 이유 + 로).
GLUE_SAFE = frozenset(
    {
        "년",
        "년대",
        "년간",
        "년도",
        "년생",
        "주년",
        "개월",
        "월",
        "일",
        "일간",
        "세기",
        "층",
        "분기",
        "퍼센트",
        "프로",
        "킬로미터",
        "센티미터",
        "밀리미터",
        "미터",
        "킬로그램",
        "밀리그램",
        "그램",
        "킬로",
        "톤",
        "리터",
        "달러",
        "위안",
        "파운드",
        "마일",
        "피트",
        "인치",
        "헥타르",
    }
)
# 띄어 쓴 단독 '이'(2) 뒤에 와도 되는 단위. '이'는 지시어 '이'(this)와 같아 '이 분'·'이 점'·'이 번'·'이 일'·
# '이 시간'·'이 세기'·'이 층'·'이 차'·'이 프로'(이 방송)가 모두 '이 + 명사'로 읽힌다. 지시어 뒤에 오기
# 어려운 단위만 받는다. '이 일'(2일)은 날짜 문맥(앞이 'N월'이거나 뒤가 오전·오후 등)일 때만 받는다.
TWO_SAFE = frozenset(
    {
        "년",
        "년대",
        "년간",
        "년도",
        "주년",
        "개월",
        "월",
        "분기",
        "초",
        "퍼센트",
        "킬로미터",
        "센티미터",
        "밀리미터",
        "미터",
        "킬로그램",
        "밀리그램",
        "그램",
        "킬로",
        "톤",
        "리터",
        "달러",
        "유로",
        "위안",
        "파운드",
        "마일",
        "피트",
        "인치",
        "헥타르",
    }
)
# 붙여 쓴 한 음절 수사 + 안전 단위인데도 흔한 낱말인 것: 오일(oil), 일일(daily·일일이), 만일(if), 이년(욕설).
GLUED_EXCEPTIONS = frozenset({"오일", "일일", "만일", "이년"})
# 띄어 써도 같은 낱말로 읽히는 것: '만 일'(만일).
SPACED_EXCEPTIONS = frozenset({("만", "일")})
# 수사로 읽으면 수가 되지만 흔한 낱말인 수사열: '구조'(9조; 구조대·구조를). 단위를 붙여 쓰면 바꾸지 않는다.
EXCEPTION_NUMERALS = frozenset({"구조"})
# 단위 + 조사로 쪼개지지만 실제로는 한 낱말인 것(이 낱말로 시작하는 나머지는 단위로 읽지 않는다).
COMPOUND_NOUNS = (
    "차이",
    "회의",
    "원인",
    "개인",
    "시인",
    "대전",
    "분당",
    "도로",
    "주의",
    "주가",
    "주로",
    "시도",
    "대로",
    "장인",
    "세상",
    "세계",
)
# 값의 범위가 정해진 단위(벗어나면 그 단위로 읽지 않는다: '이천 시'(이천시)는 2000시가 아니다).
UNIT_RANGE = {"월": (1, 12), "시": (0, 24)}
# 어림말 뒤의 자릿수 한 음절('수천 조 마일', '몇 십 년', '수 백 명')은 어림수의 일부라 바꾸지 않는다(앞
# 어절 전체가 이 낱말일 때).
VAGUE_BEFORE = frozenset(
    {"수", "몇", "여러", "수십", "수백", "수천", "수만", "수억", "몇십", "몇백", "몇천", "몇만"}
)

# --- 고유어 수사 --------------------------------------------------------------------------------------------
# 단위 앞에 오는 꼴(관형사형)만: 한·두·세·네(하나·둘·셋·넷이 아니다). '스무'는 '스무 살'처럼 일의 자리
# 없이만 쓴다. 단위는 명·개·살·시·시간·개월만 받는다('두 번째'·'세 가지'·'한 번'은 한글 그대로가 관례).
NATIVE_ONES = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
NATIVE_TENS = {
    "열": 10,
    "스물": 20,
    "스무": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
    "예순": 60,
    "일흔": 70,
    "여든": 80,
    "아흔": 90,
}
NATIVE_COUNTERS = ("시간", "개월", "명", "개", "살", "시")  # 긴 것부터
NATIVE_RANGE = {"시": (1, 12)}
NATIVE_GLUED_EXCEPTIONS = frozenset({"한시"})  # 한시적·한시도(잠시도)
# 달 이름의 불규칙 읽기.
MONTH_WORDS = {"유월": "6월", "시월": "10월"}

# --- 조사·접미사 --------------------------------------------------------------------------------------------
# 단위(또는 단위 없는 수) 뒤 같은 어절에 올 수 있는 것. 이 조각들을 이어 붙인 것만 받는다('에서도' = 에서 +
# 도). '다'·'고'·'면' 같은 동사 어미는 넣지 않는다('사 주다'를 '4주다'로 읽지 않게).
PARTICLES = (
    "이",
    "가",
    "을",
    "를",
    "은",
    "는",
    "의",
    "에",
    "에서",
    "에게",
    "께",
    "한테",
    "로",
    "으로",
    "와",
    "과",
    "도",
    "만",
    "까지",
    "부터",
    "마다",
    "씩",
    "쯤",
    "째",
    "짜리",
    "이나",
    "나",
    "라도",
    "이라도",
    "밖에",
    "뿐",
    "처럼",
    "보다",
    "만큼",
    "간",
    "경",
    "여",
    "가량",
    "전",
    "후",
    "동안",
    "이상",
    "이하",
    "미만",
    "초과",
    "이후",
    "이전",
    "이내",
    "내",
    "당",
    "들",
    "이다",
    "이었다",
    "였다",
    "입니다",
    "이며",
    "이고",
    "이라는",
    "라는",
    "인",
    "이던",
    "인데",
    "이지만",
    "이라고",
    "라고",
    "이었던",
    "였던",
    "이었고",
    "였고",
    "이었으며",
    "였으며",
    "이란",
    "란",
    "이면",
    "이라면",
    "이므로",
    "이어서",
    "이에요",
    "예요",
)
# '이 일'을 날짜(2일)로 읽게 하는 문맥: 앞 어절이 'N월'(이미 바꾼 결과)이거나 뒤 어절이 때를 나타낼 때.
_MONTH_BEFORE = re.compile(r"\d월$")
DATE_AFTER = ("오전", "오후", "새벽", "아침", "낮", "저녁", "밤", "정오")
# 분수 '삼 분의 일': 앞 어절이 'N분의'이면 뒤의 수사는 한 음절이어도 단위 없이 바꾼다(3분의 1).
_FRACTION_BEFORE = re.compile(r"\d분의$")
# 한 표현이 걸칠 수 있는 최대 어절 수(띄어 쓴 수사 여러 개 + 단위). '천 구백 칠십 팔 년'이 5어절이다.
_LOOKAHEAD = 16


def parse_sino(numeral: str) -> int | None:
    """한자어 수사열(공백 없음)을 값으로. 문법에 맞지 않으면 None.

    문법: [그룹 + 큰 자릿수]… [그룹]. 그룹 = [수]천 [수]백 [수]십 [수]이고 자릿수는 내림차순으로 한 번씩,
    큰 자릿수(조 > 억 > 만)도 내림차순으로 한 번씩. 자릿수 앞의 수가 없으면 1이다(천 = 1000, 만 = 10000).
    두 수가 잇따르면('이삼', '칠팔' = 범위 읽기) 거부한다. 큰 자릿수 앞 그룹이 비는 것은 맨 앞에서만
    받는다('만 오천'은 되고 '억만'은 거부).
    """
    if not numeral:
        return None
    total = 0
    group = 0
    digit: int | None = None
    small_order = 10**4  # 그룹 안에서 직전 작은 자릿수(다음 것은 더 작아야 한다)
    large_order = 10**16  # 직전 큰 자릿수
    for ch in numeral:
        if ch in SINO_DIGITS:
            if digit is not None:
                return None
            digit = SINO_DIGITS[ch]
        elif ch in SINO_SMALL:
            m = SINO_SMALL[ch]
            if m >= small_order:
                return None
            group += (digit if digit is not None else 1) * m
            digit = None
            small_order = m
        elif ch in SINO_LARGE:
            m = SINO_LARGE[ch]
            if m >= large_order:
                return None
            coefficient = group + (digit or 0)
            if coefficient == 0:
                if total:  # '억만'처럼 앞에 다른 큰 자릿수가 있는데 계수가 빈 경우
                    return None
                coefficient = 1
            total += coefficient * m
            group, digit, small_order, large_order = 0, None, 10**4, m
        else:
            return None
    return total + group + (digit or 0)


def format_sino(value: int) -> str:
    """값을 표시 문자열로: 1만 미만은 그대로, 만 단위로 떨어지면 만·억·조 섞어 쓰기, 아니면 쉼표 묶음."""
    if value < 10_000:
        return str(value)
    if value % 10_000 == 0:
        parts = []
        for name, unit in (("조", 10**12), ("억", 10**8), ("만", 10**4)):
            quotient, value = divmod(value, unit)
            if quotient:
                parts.append(f"{quotient}{name}")
        return " ".join(parts)
    return f"{value:,}"


@lru_cache(maxsize=4096)
def is_particles(rest: str) -> bool:
    """rest가 비었거나 PARTICLES 조각을 이어 붙인 것이면 True(동적 계획법)."""
    reachable = [True] + [False] * len(rest)
    for start in range(len(rest)):
        if reachable[start]:
            for piece in PARTICLES:
                if rest.startswith(piece, start):
                    reachable[start + len(piece)] = True
    return reachable[len(rest)]


def _match_unit(text: str, units: tuple[str, ...]) -> tuple[str, str] | None:
    """text가 단위로 시작하고 나머지가 조사이면 (단위, 나머지). 긴 단위부터, 복합 명사는 건너뛴다."""
    for unit in units:
        if text.startswith(unit):
            if any(c.startswith(unit) and text.startswith(c) for c in COMPOUND_NOUNS):
                continue
            rest = text[len(unit) :]
            if is_particles(rest):
                return unit, rest
    return None


@dataclass(frozen=True)
class ItnSpan:
    """바뀐 표현 하나: 어절 [start, end)를 after로 바꿨다(before = 원래 어절들을 공백으로 이은 것)."""

    start: int
    end: int
    before: str
    after: str


class _Flat:
    """어절 start부터 _LOOKAHEAD개 어절의 글자를 한 줄로 펴고, 글자마다 어절 번호·어절 안 위치를 둔다.

    수사는 띄어쓰기와 무관하게 이어 읽어야 해서('천 구백 칠십 팔'), 어절 경계를 건너는 연속 구간을 이 줄에서
    찾는다. 수사 글자만 이어지는 동안만 경계를 건너므로, 경계를 건넌 앞 어절은 늘 수사 글자로만 이뤄져 있다.
    """

    def __init__(self, words: list[str], start: int) -> None:
        self.words = words[start : start + _LOOKAHEAD]
        self.chars: list[str] = []
        self.word: list[int] = []
        self.offset: list[int] = []
        for k, w in enumerate(self.words):
            for c, ch in enumerate(w):
                self.chars.append(ch)
                self.word.append(k)
                self.offset.append(c)

    def run(self, t: int, alphabet) -> int:
        """t부터 alphabet 글자가 이어지는 구간의 끝(어절 경계는 건넌다)."""
        while t < len(self.chars) and self.chars[t] in alphabet:
            t += 1
        return t

    def at_word_end(self, t: int) -> bool:
        """위치 t(글자 t 앞)가 어절 끝이면 True."""
        return t >= len(self.chars) or self.offset[t] == 0

    def tail(self, t: int) -> tuple[str, int]:
        """t가 어절 중간일 때 그 어절의 나머지와 어절 번호."""
        k = self.word[t]
        return self.words[k][self.offset[t] :], k

    def word_index(self, t: int) -> int:
        """위치 t 바로 앞 글자가 속한 어절 번호(t ≥ 1)."""
        return self.word[t - 1]

    def get(self, k: int) -> str | None:
        return self.words[k] if k < len(self.words) else None


# 위치 t 뒤의 '[여] 단위 [조사]'를 찾는다. 반환: (단위, 조사, 마지막 어절 번호, 붙여 씀 여부, '여' 여부,
# 단위와 같은 어절에 있는 수사 글자). 붙여 씀 = 수사와 단위가 같은 어절. '여'(남짓)는 수사에 붙든('십여 명',
# '십여명') 따로 쓰든('이십 여 명', '오백 여명이') 받고, 단위가 수사와 다른 어절이면 띄어 씀으로 본다.
def _unit_after(flat: _Flat, t: int, units: tuple[str, ...]):
    if not flat.at_word_end(t):
        tail, k = flat.tail(t)
        local = "".join(flat.chars[t - flat.offset[t] : t])
        found = _match_unit(tail, units)
        if found:
            return found[0], found[1], k, True, False, local
        if tail.startswith("여") and len(tail) > 1:
            found = _match_unit(tail[1:], units)
            if found:
                return found[0], found[1], k, True, True, local
        if tail == "여":
            nxt = flat.get(k + 1)
            found = _match_unit(nxt, units) if nxt else None
            if found:
                return found[0], found[1], k + 1, False, True, ""
        return None
    k = flat.word_index(t)
    nxt = flat.get(k + 1)
    if nxt is None:
        return None
    found = _match_unit(nxt, units)
    if found:
        return found[0], found[1], k + 1, False, False, ""
    if nxt.startswith("여") and len(nxt) > 1:
        found = _match_unit(nxt[1:], units)
        if found:
            return found[0], found[1], k + 1, False, True, ""
    if nxt == "여":
        after = flat.get(k + 2)
        found = _match_unit(after, units) if after else None
        if found:
            return found[0], found[1], k + 2, False, True, ""
    return None


# 위치 t에서 표현이 끝날 때(단위 없음) 같은 어절의 나머지가 조사이면 (조사, 마지막 어절 번호).
def _bare_after(flat: _Flat, t: int):
    if flat.at_word_end(t):
        return "", flat.word_index(t)
    tail, k = flat.tail(t)
    return (tail, k) if is_particles(tail) else None


def _in_range(unit: str, value: float, ranges: dict[str, tuple[int, int]]) -> bool:
    low, high = ranges.get(unit, (None, None))
    return low is None or low <= value <= high


# 소수: [정수] 점|쩜 [한 자리씩 읽은 수…] [단위]. 정수는 1만 미만만. '점' 뒤에 숫자가 없으면 소수가 아니다
# ('사 점 두 개'의 '점'은 단위). 단위가 붙는 해석을 먼저 보고(소수점 뒤 자리 수는 긴 것부터), 없으면 단위 없는
# 소수를 본다: '삼 쩜 오 퍼센트'는 '3.5%'. 다만 소수 뒤의 '일'은 날(日)로 읽지 않는다: 소수 날수는 드물고
# '구 쩜 일 일 테러'(9·11)처럼 마지막 자리인 경우가 많다.
def _match_decimal(flat: _Flat, e: int, integer: int):
    chars = flat.chars
    if e >= len(chars) or chars[e] not in DECIMAL_POINTS:
        return None
    start = e + 1
    ends = range(flat.run(start, FRACTION_DIGITS), start, -1)
    numbers = {f: f"{integer}." + "".join(FRACTION_DIGITS[c] for c in chars[start:f]) for f in ends}
    for f in ends:
        unit = _unit_after(flat, f, _UNITS_LONGEST_FIRST)
        if unit and unit[0] != "일":
            name, rest, k, _glued, yeo, _local = unit
            return k + 1, numbers[f] + ("여" if yeo else "") + UNITS[name] + rest
    for f in ends:
        bare = _bare_after(flat, f)
        if bare:
            rest, k = bare
            return k + 1, numbers[f] + rest
    return None


# 한자어 수사 표현. 순서: (1) 소수, (2) 단위가 붙는 수, (3) 단위 없는 수. 각 단계 안에서는 수사를 길게 읽는
# 쪽부터 맞춘다. 단위가 있는 해석을 늘 먼저 보므로 '사십 일'은 41이 아니라 '40일'로, '삼십일 일'은 '31일'로
# 읽힌다(뒤에 다른 단위가 있으면 '사십 일 명' = 41명).
def _match_sino(flat: _Flat, prev: str | None):
    chars = flat.chars
    run = flat.run(0, SINO)
    if chars and chars[0] == "영":
        found = _match_decimal(flat, 1, 0)
        if found:
            return found
    cuts = range(run, 0, -1)
    values = {e: parse_sino("".join(chars[:e])) for e in cuts}
    for e in cuts:
        if values[e] is not None and values[e] < 10_000:
            found = _match_decimal(flat, e, values[e])
            if found:
                return found
    for e in cuts:
        numeral, value = "".join(chars[:e]), values[e]
        if value is None:
            continue
        unit = _unit_after(flat, e, _UNITS_LONGEST_FIRST)
        if not unit:
            continue
        name, rest, k, glued, yeo, local = unit
        if not _accept_unit(flat, numeral, value, name, k, glued, local, prev):
            continue
        return k + 1, format_sino(value) + ("여" if yeo else "") + UNITS[name] + rest
    fraction = prev is not None and _FRACTION_BEFORE.search(prev) is not None
    for e in cuts:
        numeral, value = "".join(chars[:e]), values[e]
        if value is None or not _accept_bare(numeral, fraction):
            continue
        bare = _bare_after(flat, e)
        if bare:
            rest, k = bare
            return k + 1, format_sino(value) + rest
    return None


# 단위가 붙는 수를 받을지. 공통: 값 범위, 붙여 쓴 예외 수사('구조대'), 어림말 뒤의 자릿수 한 음절('수천 조').
# 단위와 같은 어절의 수사가 한 자리 수 한 음절이면(수사 전체가 여러 어절이어도) 붙여 쓰기 규칙을 따른다:
# '백 사장이'는 '백' + '사장이'(사장 = 대표)이지 104장이 아니다. 여러 음절 수사는 그 밖에 제한이 없다. 한
# 음절 수사는 붙여 쓰면 GLUE_SAFE 단위만, 띄어 쓴 단독 '이'는 TWO_SAFE 단위만(날짜 문맥의 '이 일'은 받음).
def _accept_unit(
    flat: _Flat, numeral: str, value: int, unit: str, k: int, glued: bool, local: str, prev: str | None
):
    if not _in_range(unit, value, UNIT_RANGE):
        return False
    if glued and numeral in EXCEPTION_NUMERALS:
        return False
    if prev in VAGUE_BEFORE and len(numeral) == 1 and numeral in MULTIPLIERS:
        return False
    if glued and len(local) == 1 and local in SINO_DIGITS:
        if unit not in GLUE_SAFE or local + unit in GLUED_EXCEPTIONS:
            return False
    if len(numeral) > 1:
        return True
    if glued:
        return unit in GLUE_SAFE and numeral + unit not in GLUED_EXCEPTIONS
    if (numeral, unit) in SPACED_EXCEPTIONS:
        return False
    if numeral == "이" and unit not in TWO_SAFE:
        if unit != "일":
            return False
        after = flat.get(k + 1)
        month_before = prev is not None and _MONTH_BEFORE.search(prev) is not None
        return month_before or (after is not None and after.startswith(DATE_AFTER))
    return True


# 단위 없는 수를 받을지: 두 음절 이상, 큰 자릿수(만·억·조) 없음('오만'·'이만'·'조사'·'만사'), 수로 시작하거나
# '십'으로 시작(사십·오백·십팔), 백·천으로 시작하면 세 음절 이상(천사·백사·백구는 낱말). 두 음절 '이천'·
# '사천'·'오천'은 지명과 겹쳐 받지 않는다. 자릿수 뒤 끝의 '이'는 조사로 본다('삼십이 넘는' = 30 + 이).
# 분수 문맥('3분의 일')에서는 한 음절도 받는다.
def _accept_bare(numeral: str, fraction: bool) -> bool:
    if fraction:
        return True
    if len(numeral) < 2 or any(ch in SINO_LARGE for ch in numeral):
        return False
    if numeral[-1] == "이" and numeral[-2] in MULTIPLIERS:
        return False
    if len(numeral) == 2 and numeral[1] == "천":
        return False
    return numeral[0] in SINO_DIGITS or numeral[0] == "십" or len(numeral) >= 3


# flat의 위치 t에서 시작하는 table 낱말 가운데 가장 긴 것(한 어절 안에 다 들어 있어야 한다). 없으면 None.
def _word_at(flat: _Flat, t: int, table: dict[str, int]) -> str | None:
    for key in sorted(table, key=len, reverse=True):
        end = t + len(key)
        if (
            end <= len(flat.chars)
            and "".join(flat.chars[t:end]) == key
            and flat.word[t] == flat.word[end - 1]
        ):
            return key
    return None


# 고유어 수사 + 단위(명·개·살·시·시간·개월). 열·스물 … 아흔 다음 한·두 … 아홉은 붙여 쓰든('열한') 띄어
# 쓰든('마흔 두') 이어 읽는다. 십의 자리가 어절 전체이면 일의 자리는 다음 어절의 시작이다(위치 t가 어절
# 끝이면 flat.chars[t]가 다음 어절 첫 글자). '스무'는 일의 자리 없이만 쓴다. 읽기 후보는 긴 것부터 본다.
def _match_native(flat: _Flat):
    readings = []  # (값, 수사가 끝나는 위치, 수사 글자열)
    tens = _word_at(flat, 0, NATIVE_TENS)
    if tens:
        t = len(tens)
        ones = _word_at(flat, t, NATIVE_ONES) if tens != "스무" else None
        if ones:
            readings.append((NATIVE_TENS[tens] + NATIVE_ONES[ones], t + len(ones), tens + ones))
        readings.append((NATIVE_TENS[tens], t, tens))
    else:
        ones = _word_at(flat, 0, NATIVE_ONES)
        if ones:
            readings.append((NATIVE_ONES[ones], len(ones), ones))
    for value, end, numeral in readings:
        unit = _unit_after(flat, end, NATIVE_COUNTERS)
        if not unit:
            continue
        name, rest, k, glued, yeo, _local = unit
        if yeo or not _in_range(name, value, NATIVE_RANGE):
            continue
        if glued and numeral + name in NATIVE_GLUED_EXCEPTIONS:
            continue
        return k + 1, f"{value}{name}{rest}"
    return None


def _match_month_word(flat: _Flat):
    word = flat.words[0]
    for spoken, shown in MONTH_WORDS.items():
        if word.startswith(spoken) and is_particles(word[len(spoken) :]):
            return 1, shown + word[len(spoken) :]
    return None


def itn_spans(text: str) -> list[ItnSpan]:
    """text(공백으로 나뉜 어절)에서 바꿀 표현들을 왼쪽부터 겹치지 않게 찾는다.

    어절 i에서 달 이름 → 고유어 → 한자어 순으로 맞춰 보고, 맞으면 그 표현이 끝난 어절 다음부터 이어 간다.
    문맥(날짜·분수)은 바로 앞 출력 어절(바꾼 결과 포함)로 본다.
    """
    words = text.split()
    spans: list[ItnSpan] = []
    prev: str | None = None
    i = 0
    while i < len(words):
        flat = _Flat(words, i)
        found = _match_month_word(flat) or _match_native(flat) or _match_sino(flat, prev)
        if found:
            consumed, after = found
            spans.append(ItnSpan(i, i + consumed, " ".join(words[i : i + consumed]), after))
            prev = after
            i += consumed
        else:
            prev = words[i]
            i += 1
    return spans


def itn(text: str) -> str:
    """확정 자막 text의 한글 수 표현을 아라비아 숫자로 바꾼 표시 문자열. 바꿀 것이 없으면 text 그대로.

    바뀐 표현은 한 어절로 합쳐지고(수사·단위·조사 사이 공백 없음, '1억 5000만'처럼 수 안의 공백은 남음)
    나머지 어절은 공백 하나로 다시 잇는다. 결정적이고(같은 입력 → 같은 출력) 두 번 적용해도 같다.
    """
    spans = itn_spans(text)
    if not spans:
        return text
    words = text.split()
    out: list[str] = []
    i = 0
    for span in spans:
        out.extend(words[i : span.start])
        out.append(span.after)
        i = span.end
    out.extend(words[i:])
    return " ".join(out)
