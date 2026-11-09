import csv
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from .data import COURSES
from .mountain_coordinates import (
    MOUNTAIN_ALIASES,
    MOUNTAIN_COORDINATES,
    AMBIGUOUS_MOUNTAIN_NAMES,
    find_mountain_coordinates,
)

logger = logging.getLogger(__name__)

# 산별 실제 혼잡도 기본값 (데이터 기반 없음 → 통계 기반 추정)
MOUNTAIN_CROWDING = {
    "북한산": 0.82, "도봉산": 0.78, "남산": 0.80, "인왕산": 0.72,
    "관악산": 0.75, "청계산": 0.65, "아차산": 0.70, "용마산": 0.50,
    "수락산": 0.38, "불암산": 0.42, "마니산": 0.55,
    "설악산": 0.68, "한라산": 0.70, "지리산": 0.55, "계룡산": 0.55,
    "내장산": 0.58, "오대산": 0.48, "덕유산": 0.45, "속리산": 0.52,
    "주왕산": 0.48, "가야산": 0.55, "치악산": 0.45, "소백산": 0.40,
    "무등산": 0.58, "월출산": 0.42, "월악산": 0.42, "가리왕산": 0.30,
    "팔공산": 0.60, "두륜산": 0.40, "조계산": 0.45,
    "한려해상": 0.35, "다도해해상": 0.30, "태안해안": 0.35, "변산반도": 0.40,
    "태백산": 0.40, "함백산": 0.32, "두타산": 0.35,
}
_DEFAULT_CROWDING = 0.40


# ── 등산로 노드 유형 판별 ─────────────────────────────────────────────────
# 실제 탐방로 입구로 쓰이는 키워드 (주차장, 매표소, 사찰 입구 등)
_ENTRANCE_KEYWORDS = frozenset({
    "입구", "매표소", "탐방안내소", "탐방지원센터", "탐방로입구",
    "분소", "주차장", "광장", "마을", "집결지", "기점", "들머리",
    "시점", "게이트", "관리소", "휴양림", "야영장", "캠핑장",
    "버스정류장", "터미널", "역전", "ic",
    # 사찰·문화재 입구 (주요 탐방로 시작점으로 쓰임)
    "사입구", "암입구",
})
# 입구 역할을 하는 독립 단어 (완전 매칭)
_ENTRANCE_WORDS = frozenset({
    "소공원", "백무동", "화엄사", "내소사", "개암사", "갑사",
    "동학사", "신원사", "대원사", "거림", "중산리", "성삼재",
    "피아골", "뱀사골", "북한산성", "우이동", "도봉산역",
    "송추", "효자동", "원효광장", "불광계곡", "구기분소",
    "자하교", "탕춘대", "육모정", "증심교",
})
# 산 중간 지점 키워드 (시작점으로 부적절)
_SUMMIT_KEYWORDS = frozenset({
    "정상", "봉", "대피소", "산장", "쉼터", "능선", "갈림길",
    "분기점", "삼거리", "사거리", "전망대", "헬기장", "재",
    "고개", "바위", "평전", "암문", "약수터", "샘터",
})


def _is_entrance(text: str) -> bool:
    """텍스트가 등산로 입구를 나타내는지 판별한다."""
    t = text.strip()
    if not t:
        return False
    if t in _ENTRANCE_WORDS:
        return True
    return any(kw in t for kw in _ENTRANCE_KEYWORDS)


def _is_summit_or_intermediate(text: str) -> bool:
    """텍스트가 정상·능선 등 중간 지점을 나타내는지 판별한다."""
    t = text.strip()
    if not t:
        return False
    return any(kw in t for kw in _SUMMIT_KEYWORDS)


DATA_DIR = settings.BASE_DIR.parent / "data"
# CSV는 data/ 폴더에 있다. (예전엔 저장소 루트를 가리켜 파일을 못 찾고, 재난위험지구가 조용히 0건으로 로드됐음)
TRAIL_CSV_PATH = DATA_DIR / "국립공원공단_탐방로_20240911.csv"
DISASTER_RISK_CSV_PATH = DATA_DIR / "국립공원공단_재난위험지구_20240904.csv"
KEY_FILE_PATH = settings.BASE_DIR.parent / "key.txt"
DATA_LINK_FILE_PATH = DATA_DIR / "데이터 링크.txt"
VWORLD_KEY_FILE_PATH = settings.BASE_DIR.parent / "vworld_key.txt"
KMA_API_HUB_KEY_FILE_PATH = settings.BASE_DIR.parent / "kma_api_hub_key.txt"


@lru_cache(maxsize=1)
def load_public_trail_courses():
    # DB 우선, 없으면 CSV fallback
    base_courses = None
    try:
        from .models import TrailCourse
        qs = TrailCourse.objects.filter(lat__isnull=False, lng__isnull=False)
        if qs.exists():
            base_courses = [
                {
                    "id": tc.course_id,
                    "mountain": tc.mountain,
                    "name": tc.name,
                    "region": tc.region,
                    "difficulty": tc.difficulty,
                    "distance_km": tc.distance_km,
                    "duration_min": tc.duration_min,
                    "elevation_gain_m": tc.elevation_gain_m,
                    "lat": tc.lat,
                    "lng": tc.lng,
                    "crowding": tc.crowding,
                    "highlights": tc.highlights or [],
                    "source": tc.source,
                }
                for tc in qs
            ]
    except Exception:
        pass

    if base_courses is None:
        if TRAIL_CSV_PATH.exists():
            for encoding in ("utf-8-sig", "cp949", "euc-kr"):
                try:
                    base_courses = _read_trail_csv(TRAIL_CSV_PATH, encoding)
                    break
                except UnicodeDecodeError:
                    continue
        if base_courses is None:
            base_courses = []

    # 로컬 SHP 등산로 병합
    try:
        from .local_road_api import load_local_road_trails
        road_trails = load_local_road_trails()
        seen = {
            (str(c.get("mountain", "")).replace(" ", ""), str(c.get("name", "")).replace(" ", ""))
            for c in base_courses
        }
        for t in road_trails:
            key = (str(t.get("mountain", "")).replace(" ", ""), str(t.get("name", "")).replace(" ", ""))
            if key not in seen:
                base_courses.append(t)
                seen.add(key)
    except Exception:
        pass

    return with_seed_courses(base_courses) if base_courses else COURSES


def with_seed_courses(courses):
    merged = list(courses or [])
    seen = {
        (
            str(course.get("mountain", "")).replace(" ", ""),
            str(course.get("name", "")).replace(" ", ""),
        )
        for course in merged
    }
    for seed in COURSES:
        key = (
            str(seed.get("mountain", "")).replace(" ", ""),
            str(seed.get("name", "")).replace(" ", ""),
        )
        if key in seen:
            continue
        item = dict(seed)
        item["id"] = f"sample-{item.get('id')}"
        item.setdefault("source", "seed")
        merged.append(item)
        seen.add(key)
    return merged or COURSES


def _read_trail_csv(path, encoding):
    segments = []
    seen_keys = set()
    with Path(path).open(encoding=encoding, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            course = normalize_trail_row(index, row)
            if course and course["dedupe_key"] not in seen_keys:
                seen_keys.add(course["dedupe_key"])
                course.pop("dedupe_key")
                segments.append(course)

    courses = merge_connected_trail_segments(segments)
    courses = [c for c in courses if c.get("lat") is not None and c.get("lng") is not None]
    return courses or COURSES


@lru_cache(maxsize=1)
def load_disaster_risk_zones():
    # DB 우선, 없으면 CSV fallback
    try:
        from .models import DisasterRiskZone
        qs = DisasterRiskZone.objects.all()
        if qs.exists():
            return [
                {
                    "id": dz.zone_id,
                    "district": dz.district,
                    "location": dz.location,
                    "facility": dz.facility,
                    "risk_factor": dz.risk_factor,
                    "evacuation_place": dz.evacuation_place,
                    "search_text": dz.search_text,
                }
                for dz in qs
            ]
    except Exception:
        pass

    if not DISASTER_RISK_CSV_PATH.exists():
        logger.warning("재난위험지구 CSV를 찾을 수 없습니다: %s (재난위험 경고가 꺼진 상태로 동작)", DISASTER_RISK_CSV_PATH)
        return []

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return _read_disaster_risk_csv(DISASTER_RISK_CSV_PATH, encoding)
        except UnicodeDecodeError:
            continue
    return []


def _read_disaster_risk_csv(path, encoding):
    zones = []
    with Path(path).open(encoding=encoding, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            zone = normalize_disaster_risk_row(index, row)
            if zone:
                zones.append(zone)
    return zones


def normalize_disaster_risk_row(index, row):
    district = clean_text(row.get("지구명"))
    location = clean_text(row.get("위치"))
    facility = clean_text(row.get("시설명"))
    risk_factor = clean_text(row.get("위험요인"))
    if not any([district, location, facility, risk_factor]):
        return None

    return {
        "id": f"disaster-risk-{index}",
        "district": district,
        "location": location,
        "facility": facility,
        "has_signage": parse_int(row.get("표지판설치")),
        "risk_factor": risk_factor,
        "expected_daily_visitors": parse_int(row.get("일최대예상탐방객")),
        "evacuation_capacity": parse_int(row.get("대피계획인원")),
        "evacuation_place": clean_text(row.get("대피장소")),
        "rescue_equipment": clean_text(row.get("구조대편성 및 구조장비현황")),
        "control_facility": clean_text(row.get("통제시설")),
        "source": "국립공원공단_재난위험지구_20240904",
        "search_text": normalize_search_text(" ".join([district, location, facility, risk_factor])),
    }


def normalize_trail_row(index, row):
    name = clean_course_name(row.get("시설물명칭"))
    start = clean_text(row.get("구간_시작지점"))
    waypoint = clean_text(row.get("구간_경유지점"))
    end = clean_text(row.get("구간_종착지점"))
    distance_km = parse_distance_km(row.get("탐방로길이"))

    # 너무 짧거나(100m급 단편) 비현실적으로 긴 구간 제거
    if not name or distance_km < 0.5 or distance_km > 25:
        return None

    # ── 입구 방향 보정 ────────────────────────────────────────────────────
    # 시작이 중간지점이고 종착이 입구이면 방향을 반전한다
    start_is_entrance = _is_entrance(start)
    end_is_entrance = _is_entrance(end)
    start_is_mid = _is_summit_or_intermediate(start)

    if not start_is_entrance and end_is_entrance and start_is_mid:
        # 종착(입구) → 시작(중간)으로 오는 구간을 뒤집는다
        start, end = end, start
        start_is_entrance = True

    has_entrance_start = start_is_entrance

    up_minutes = parse_int(row.get("소요시간_상행"))
    down_minutes = parse_int(row.get("소요시간_하행"))
    duration_min = max(up_minutes, down_minutes, estimate_duration(distance_km))
    # 소요시간 이상값(10시간 초과) → 거리 기반으로 재추정
    if duration_min > 600:
        duration_min = estimate_duration(distance_km)
    difficulty = infer_difficulty(distance_km, duration_min)

    dedupe_key = "|".join([name, start, waypoint, end, str(distance_km)])

    inferred_mountain = infer_mountain_name(name, start, waypoint, end)
    coordinate_mountain, coordinates = find_mountain_coordinates(inferred_mountain, name, start, waypoint, end)
    mountain = coordinate_mountain or inferred_mountain

    return {
        "id": f"public-{index}",
        "mountain": mountain,
        "name": name,
        "region": coordinates["region"] if coordinates else "국립공원 탐방로",
        "difficulty": difficulty,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "elevation_gain_m": infer_elevation_gain(distance_km, difficulty),
        "lat": coordinates["lat"] if coordinates else None,
        "lng": coordinates["lng"] if coordinates else None,
        "crowding": MOUNTAIN_CROWDING.get(mountain, _DEFAULT_CROWDING),
        "highlights": build_highlights(start, waypoint, end),
        "source": "국립공원공단_탐방로_20240911+좌표보강" if coordinates else "국립공원공단_탐방로_20240911",
        "segment_nodes": build_segment_nodes(start, waypoint, end),
        "segment_count": 1,
        "has_entrance_start": has_entrance_start,
        "dedupe_key": dedupe_key,
    }


def merge_connected_trail_segments(segments):
    grouped = {}
    for segment in segments:
        nodes = [normalize_node(node) for node in segment.get("segment_nodes", []) if normalize_node(node)]
        if len(nodes) < 2:
            grouped.setdefault(("single", segment["id"]), []).append(segment)
            continue
        grouped.setdefault((segment.get("mountain") or "국립공원", segment.get("source") or ""), []).append(segment)

    merged_courses = []
    for key, group in grouped.items():
        if key[0] == "single":
            merged_courses.extend(strip_segment_fields(course) for course in group)
            continue
        merged_courses.extend(merge_segment_group(group))
    return merged_courses


def merge_segment_group(group):
    node_to_indexes = {}
    for index, segment in enumerate(group):
        nodes = [normalize_node(node) for node in segment.get("segment_nodes", []) if normalize_node(node)]
        for node in set(nodes):
            node_to_indexes.setdefault(node, set()).add(index)

    visited = set()
    courses = []
    for index in range(len(group)):
        if index in visited:
            continue
        stack = [index]
        component = []
        visited.add(index)
        while stack:
            current = stack.pop()
            component.append(group[current])
            for node in [normalize_node(node) for node in group[current].get("segment_nodes", []) if normalize_node(node)]:
                for next_index in node_to_indexes.get(node, set()):
                    if next_index not in visited:
                        visited.add(next_index)
                        stack.append(next_index)

        if len(component) == 1:
            courses.append(strip_segment_fields(component[0]))
        elif should_merge_trail_component(component):
            courses.append(build_merged_course(component))
        else:
            courses.extend(strip_segment_fields(segment) for segment in component)
    return courses


def should_merge_trail_component(component):
    distance_km = sum(float(segment.get("distance_km") or 0) for segment in component)
    return len(component) <= 8 and distance_km <= 15


def build_merged_course(component):
    first = component[0]
    all_nodes = []
    degree = {}
    for segment in component:
        nodes = [node for node in segment.get("segment_nodes", []) if node]
        all_nodes.extend(nodes)
        if len(nodes) >= 2:
            start = normalize_node(nodes[0])
            end = normalize_node(nodes[-1])
            degree[start] = degree.get(start, 0) + 1
            degree[end] = degree.get(end, 0) + 1

    endpoints = [node for node in all_nodes if degree.get(normalize_node(node), 0) == 1]
    ep0 = endpoints[0] if endpoints else all_nodes[0]
    ep1 = endpoints[-1] if len(endpoints) > 1 else all_nodes[-1]

    # 두 끝점 중 입구에 해당하는 쪽을 시작(start)으로 배치한다
    ep0_entrance = _is_entrance(ep0)
    ep1_entrance = _is_entrance(ep1)
    ep0_mid = _is_summit_or_intermediate(ep0)

    if (ep1_entrance and not ep0_entrance) or (ep0_mid and not _is_summit_or_intermediate(ep1)):
        start, end = ep1, ep0
    else:
        start, end = ep0, ep1

    has_entrance_start = _is_entrance(start)
    waypoints = [node for node in all_nodes if node not in {start, end}]
    distance_km = round(sum(float(segment.get("distance_km") or 0) for segment in component), 2)
    duration_min = sum(int(segment.get("duration_min") or 0) for segment in component)
    difficulty = infer_difficulty(distance_km, duration_min)
    lat_values = [segment["lat"] for segment in component if segment.get("lat") is not None]
    lng_values = [segment["lng"] for segment in component if segment.get("lng") is not None]

    return {
        **strip_segment_fields(first),
        "id": f"public-merged-{stable_text_id(first.get('mountain'))}-{stable_text_id(start)}-{stable_text_id(end)}-{len(component)}",
        "name": f"{start}~{end}",
        "difficulty": difficulty,
        "distance_km": distance_km,
        "duration_min": max(duration_min, estimate_duration(distance_km)),
        "elevation_gain_m": infer_elevation_gain(distance_km, difficulty),
        "lat": round(sum(lat_values) / len(lat_values), 6) if lat_values else None,
        "lng": round(sum(lng_values) / len(lng_values), 6) if lng_values else None,
        "highlights": build_merged_highlights(start, waypoints, end, len(component)),
        "segment_count": len(component),
        "has_entrance_start": has_entrance_start,
    }


def build_merged_highlights(start, waypoints, end, count):
    highlights = [f"출발: {start}"]
    unique_waypoints = []
    for waypoint in waypoints:
        if waypoint and waypoint not in {start, end} and waypoint not in unique_waypoints:
            unique_waypoints.append(waypoint)
    if unique_waypoints:
        highlights.append(f"주요 경유: {', '.join(unique_waypoints[:3])}")
    highlights.append(f"도착: {end}")
    highlights.append(f"{count}개 연결 구간 통합")
    return highlights[:4]


def strip_segment_fields(course):
    clean = dict(course)
    clean.pop("segment_nodes", None)
    return clean


def clean_text(value):
    return (value or "").strip()


def clean_course_name(value):
    text = clean_text(value)
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", text).strip()


def normalize_search_text(value):
    return re.sub(r"\s+", "", str(value or "").lower())


def normalize_node(value):
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def stable_text_id(value):
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(value or ""))[:24] or "course"


def parse_distance_km(value):
    try:
        raw = float(str(value or "0").strip())
    except ValueError:
        return 0

    if raw > 100:
        return round(raw / 1000, 2)
    return round(raw, 2)


def parse_int(value):
    try:
        return int(float(str(value or "0").strip()))
    except ValueError:
        return 0


def estimate_duration(distance_km):
    return max(round(distance_km * 32), 15)


def infer_difficulty(distance_km, duration_min):
    if distance_km >= 6 or duration_min >= 180:
        return "hard"
    if distance_km >= 3 or duration_min >= 90:
        return "medium"
    return "easy"


def infer_elevation_gain(distance_km, difficulty):
    multiplier = {"easy": 35, "medium": 65, "hard": 95}[difficulty]
    return round(distance_km * multiplier)


def infer_mountain_name(*parts):
    """시작·경유·종착 지점 텍스트에서 산 이름을 추론한다.
    MOUNTAIN_COORDINATES와 MOUNTAIN_ALIASES를 단일 소스로 사용한다."""
    joined = " ".join(part for part in parts if part)

    # 1. MOUNTAIN_COORDINATES 이름 직접 매칭 (모호한 이름 제외)
    for mountain in MOUNTAIN_COORDINATES:
        if mountain in AMBIGUOUS_MOUNTAIN_NAMES:
            continue
        if mountain in joined:
            return mountain

    # 2. MOUNTAIN_ALIASES 키워드 매칭
    # 짧은 별칭이 다른 지명 안에 포함될 수 있어(예: "삼도봉" 안의 "도봉"),
    # 먼저 걸린 산이 아니라 전체 별칭 매칭 점수가 가장 높은 산을 선택한다.
    alias_scores = []
    for mountain, aliases in MOUNTAIN_ALIASES.items():
        score = 0
        for alias in aliases:
            count = joined.count(alias)
            if count:
                score += count * len(alias)
        if score:
            alias_scores.append((score, mountain))

    if alias_scores:
        alias_scores.sort(reverse=True)
        return alias_scores[0][1]

    return "국립공원"


def build_highlights(start, waypoint, end):
    highlights = []
    if start:
        highlights.append(f"출발: {start}")
    if waypoint:
        highlights.append(f"경유: {waypoint}")
    if end:
        highlights.append(f"도착: {end}")
    return highlights[:3] or ["탐방로 구간 데이터"]


def build_segment_nodes(start, waypoint, end):
    nodes = [start]
    if waypoint:
        nodes.extend(part.strip() for part in re.split(r"[,/·>]+", waypoint) if part.strip())
    nodes.append(end)
    return [node for node in nodes if node]


@lru_cache(maxsize=1)
def load_public_service_key():
    env_key = os.environ.get("PUBLIC_SERVICE_KEY", "").strip()
    if env_key:
        return env_key

    for path in (KEY_FILE_PATH, DATA_LINK_FILE_PATH):
        key = _load_public_service_key_from_path(path)
        if key:
            return key
    return ""


def _load_public_service_key_from_path(path):
    if not path.exists():
        return ""

    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""

    general_key = re.search(r"일반\s*인증키\s*[:：]\s*([A-Za-z0-9%+/=_-]{20,})", text)
    if general_key:
        return general_key.group(1).strip()

    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in reversed(candidates):
        if len(candidate) >= 20 and " " not in candidate and ":" not in candidate:
            return candidate
    return candidates[-1] if candidates else ""


@lru_cache(maxsize=1)
def load_vworld_api_key():
    env_key = os.environ.get("VWORLD_API_KEY", "").strip()
    if env_key:
        return env_key

    for path in (VWORLD_KEY_FILE_PATH, DATA_LINK_FILE_PATH):
        key = _load_vworld_api_key_from_path(path)
        if key:
            return key
    return ""


def _load_vworld_api_key_from_path(path):
    if not path.exists():
        return ""

    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""

    keyed_patterns = [
        r"VWORLD_API_KEY\s*[:=]\s*([A-Za-z0-9-]{20,})",
        r"VWORLD_KEY\s*[:=]\s*([A-Za-z0-9-]{20,})",
        r"브이월드[^:\n]*키\s*[:=]\s*([A-Za-z0-9-]{20,})",
    ]
    for pattern in keyed_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    uuid_like = re.search(
        r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b",
        text,
    )
    return uuid_like.group(0).strip() if uuid_like else ""


@lru_cache(maxsize=1)
def load_kma_api_hub_key():
    env_key = os.environ.get("KMA_API_HUB_KEY", "").strip()
    if env_key:
        return env_key

    for path in (KMA_API_HUB_KEY_FILE_PATH, DATA_LINK_FILE_PATH):
        key = _load_kma_api_hub_key_from_path(path)
        if key:
            return key
    return load_public_service_key()


def _load_kma_api_hub_key_from_path(path):
    if not path.exists():
        return ""

    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""

    patterns = [
        r"KMA_API_HUB_KEY\s*[:=]\s*([A-Za-z0-9_-]{10,})",
        r"authKey\s*=\s*([A-Za-z0-9_-]{10,})",
        r"산악예보[^:\n]*키\s*[:=]\s*([A-Za-z0-9_-]{10,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""
