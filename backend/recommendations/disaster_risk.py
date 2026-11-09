import re

from .loaders import load_disaster_risk_zones, normalize_search_text


HIGH_RISK_KEYWORDS = ("추락", "낙석", "급경사", "붕괴", "고립", "범람")
CAUTION_RISK_KEYWORDS = ("위험", "사고", "미끄럼", "침수", "월파", "협곡")
GENERIC_MATCH_TERMS = {"탐방로", "국립공원", "지구", "구간", "코스"}

# district 이름 → 산 이름 매핑 (국립공원 코드명이 산 이름과 다를 때)
DISTRICT_TO_MOUNTAIN = {
    # 설악산
    "남창계곡": "설악산", "남창지구": "설악산",
    "백담": "설악산", "오색": "설악산",
    "외설악": "설악산", "장수대": "설악산",
    "구룡지구": "설악산",  # 사다리병창, 비로봉
    # 지리산
    "금대지구": "지리산", "노고단": "지리산",
    "뱀사골지구": "지리산", "산청지구": "지리산",
    "삼장지구": "지리산", "세석지구": "지리산",
    "청학지구": "지리산", "하동지구": "지리산",
    "함양지구": "지리산",
    # 북한산
    "구기": "북한산", "구기지구": "북한산",
    "북한산성지구": "북한산",
    "산성": "북한산", "산성분소": "북한산", "산성지구": "북한산",
    "새인봉 암장 남서벽": "북한산",
    "새인봉 하단 10m": "북한산", "새인봉 하단 200m": "북한산",
    "수유": "북한산", "우이지구": "북한산", "정릉지구": "북한산",
    # 덕유산
    "구천동지구": "덕유산", "영각지구": "덕유산",
    "용추봉 하단 70m": "덕유산",
    "적상지구": "덕유산", "종주능선 삼도봉": "덕유산",
    "황점지구": "덕유산",
    # 속리산
    "법주사지구": "속리산", "쌍곡지구": "속리산",
    "화북지구": "속리산", "화양지구": "속리산",
    # 무등산
    "백운동지구": "무등산",
    "서석대 전망대": "무등산", "서석대 전망대 하단 50m": "무등산",
    "약사바위": "무등산",
    "탑봉": "무등산", "탑봉 하단 160m": "무등산",
    # 오대산
    "소금강지구": "오대산", "월정지구": "오대산",
    # 월악산
    "송계지구": "월악산", "용하지구": "월악산",
    # 가야산
    "덕산지구": "가야산", "해인사지구": "가야산",
    # 내장산
    "내장지구": "내장산",
    "백양계곡": "내장산", "백양사 지구": "내장산",
    # 계방산
    "계방산지구": "계방산",
    "황골지구": "계방산", "황골(곧은재)": "계방산",
    # 팔영산 (오탈자 포함)
    "팔영산지구(12)": "팔영산", "팔영산지구(13)": "팔영산",
    "팔영산지구(14)": "팔영산", "팔영산지구(15)": "팔영산",
    "팔영산자구(16)": "팔영산", "팔영산지구(17)": "팔영산",
    "팔영산지구(18)": "팔영산", "팔영산지구(19)": "팔영산",
    # 계룡산
    "인계리지구": "계룡산",
    # 도락산
    "단양지구": "도락산",
    # 소백산·토함산 (기존 직접 매칭 보완)
    "소백산 도솔봉지구": "소백산",
    "토함산": "토함산",
}


# 위험지구는 "어느 산(지역)에 있는가"와 "산 안의 어느 지점인가"를 나눠 매칭한다.
#  - 지역만 같음 → AREA_SCOPE: 같은 산의 다른 지점일 수 있으므로 최대 '주의'
#  - 지역이 같고 지점 이름이 코스 이름에 있음 → SPOT_SCOPE: 위험요인 그대로 판정(추락 등은 '비추천')
#  - 지역이 다르면 지점 이름이 같아도(옥녀봉·전망바위 등 전국에 흔한 이름) 매칭하지 않는다.
AREA_SCOPE = "area"
SPOT_SCOPE = "spot"
_WORD_SPLIT = re.compile(r"[\s~/(),·]+")
NON_PLACE_WORDS = GENERIC_MATCH_TERMS | {"등산로", "일원", "정상", "하단", "상단", "상부", "백사장"}
# 주소의 시·군·구·읍·면(종로구, 제천시 등)은 지점이라기엔 너무 넓다. 동·리(구기동, 중산리)는 들머리 마을이라 지점으로 쓴다.
NON_PLACE_SUFFIXES = ("전망대", "탐방로", "해안가", "선착장", "위험", "시", "군", "구", "읍", "면")


def find_course_disaster_risks(course, zones=None, limit=3):
    """코스에 해당하는 위험지구를 지점 일치 우선으로 최대 limit개 돌려준다. 각 항목에 match_scope가 붙는다."""
    zones = zones if zones is not None else load_disaster_risk_zones()
    area_text, spot_text = _course_match_texts(course)
    if not area_text:
        return []

    spots, areas = [], []
    for zone, matcher in zip(zones, _zone_matchers(zones)):
        scope = _match_scope(matcher, area_text, spot_text)
        if scope == SPOT_SCOPE:
            spots.append({**zone, "match_scope": SPOT_SCOPE})
            if len(spots) >= limit:
                break
        elif scope == AREA_SCOPE and len(areas) < limit:
            areas.append({**zone, "match_scope": AREA_SCOPE})
    return (spots + areas)[:limit]


def _course_match_texts(course):
    mountain = str(course.get("mountain") or "")
    name = str(course.get("name") or "")
    area_text = normalize_search_text(" ".join(part for part in (mountain, name, str(course.get("region") or "")) if part))
    # 로컬 등산로는 산 이름 자체가 "북한산_백운대"처럼 봉우리를 포함하고, 코스명도 "북한산_백운대 등산로"처럼
    # 산 이름을 되풀이한다. 이 부분은 지점 정보가 아니므로 지우고 남은 코스명으로만 지점 일치를 본다.
    spot_text = normalize_search_text(name)
    for part in [mountain, *mountain.split("_")]:
        part = normalize_search_text(part)
        if part:
            spot_text = spot_text.replace(part, "")
    return area_text, spot_text


# (zones 리스트, 미리 계산한 매처들). 위험지구마다 단어 추출을 코스 수만큼 반복하지 않도록 한 번만 계산한다.
_COMPILED_ZONES = (None, [])


def _zone_matchers(zones):
    global _COMPILED_ZONES
    compiled_for, matchers = _COMPILED_ZONES
    if compiled_for is not zones or len(matchers) != len(zones):
        matchers = [_zone_matcher(zone) for zone in zones]
        _COMPILED_ZONES = (zones, matchers)
    return matchers


def _zone_matcher(zone):
    """(지역 이름들, 지점 이름들)을 돌려준다."""
    district = zone.get("district", "")
    mapped = DISTRICT_TO_MOUNTAIN.get(district, "")
    if mapped:
        areas = (normalize_search_text(mapped),)
        # 매핑된 district(노고단·장수대·백운동지구 등)는 그 자체가 산 안의 지점 이름이다
        spot_sources = (district, zone.get("location", ""), zone.get("facility", ""))
    else:
        areas = tuple(_place_words(district))
        spot_sources = (zone.get("location", ""), zone.get("facility", ""))
    spots = tuple(word for word in _place_words(*spot_sources) if not any(word in area or area in word for area in areas))
    return areas, spots


def _place_words(*texts):
    words = []
    for text in texts:
        for raw in _WORD_SPLIT.split(str(text or "")):
            word = normalize_search_text(raw)
            if word.endswith("지구"):  # 상주금산지구 → 상주금산, 백운동지구 → 백운동
                word = word[:-2]
            if len(word) < 3 or any(ch.isdigit() for ch in word):
                continue
            if word in NON_PLACE_WORDS or word.endswith(NON_PLACE_SUFFIXES):
                continue
            if word not in words:
                words.append(word)
    return words


def _match_scope(matcher, area_text, spot_text):
    areas, spots = matcher
    if not any(area in area_text for area in areas):
        return None
    if spot_text and any(spot in spot_text for spot in spots):
        return SPOT_SCOPE
    return AREA_SCOPE


def normalize_course_text(course):
    parts = [
        course.get("mountain", ""),
        course.get("name", ""),
        course.get("region", ""),
        " ".join(course.get("highlights", []) or []),
    ]
    return normalize_search_text(" ".join(str(part) for part in parts if part))


def zone_matches_course(zone, course_text):
    return _match_scope(_zone_matcher(zone), course_text, course_text) is not None


def _keyword_level(zones):
    joined = " ".join(zone.get("risk_factor", "") for zone in zones)
    if any(keyword in joined for keyword in HIGH_RISK_KEYWORDS):
        return "high"
    if any(keyword in joined for keyword in CAUTION_RISK_KEYWORDS):
        return "caution"
    return "low"


def disaster_risk_level(zones):
    if not zones:
        return "none"
    # match_scope가 없는 항목(직접 넘긴 위험지구)은 지점 일치로 본다
    spot_zones = [zone for zone in zones if zone.get("match_scope") != AREA_SCOPE]
    area_zones = [zone for zone in zones if zone.get("match_scope") == AREA_SCOPE]
    levels = []
    if spot_zones:
        levels.append(_keyword_level(spot_zones))
    if area_zones:
        # 같은 산이라는 것만 확인된 위험지구는 '주의'까지만 올린다
        levels.append("low" if _keyword_level(area_zones) == "low" else "caution")
    rank = {"low": 1, "caution": 2, "high": 3}
    return max(levels, key=rank.get)


def disaster_risk_messages(zones):
    messages = []
    for zone in zones:
        location = zone.get("location") or zone.get("facility") or zone.get("district")
        factor = zone.get("risk_factor") or "재난위험지구"
        if zone.get("match_scope") == AREA_SCOPE:
            messages.append(f"같은 산 재난위험지구 {location}: {factor} (코스 경유 여부 확인)")
        else:
            messages.append(f"{location}: {factor}")
    return messages
