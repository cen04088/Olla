from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

from django.core.cache import cache

from .disaster_risk import disaster_risk_level, disaster_risk_messages, find_course_disaster_risks
from .loaders import load_public_trail_courses, load_disaster_risk_zones
from .local_road_api import fetch_local_road_trails, has_road_route, with_route_geometry
from .weather_api import fetch_current_weather
from .vworld_api import fetch_vworld_trails

_WEATHER_CACHE_SECONDS = 600

DIFFICULTY_LEVEL = {"easy": 1, "medium": 2, "hard": 3}
GENERIC_COURSE_NAMES = {"탐방로", "주등산로", "자연관찰로"}


def haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * radius * asin(sqrt(a))


# ── 날씨 안전도 (0~100) ─────────────────────────────────────────────────────────

def weather_safety_score(weather):
    score = 100
    rainfall = float(weather.get("rainfall_mm") or 0)
    wind = float(weather.get("wind_speed_ms") or 0)
    temp = float(weather.get("temperature_c") or 15)
    humidity = float(weather.get("humidity_pct") or 50)

    if rainfall >= 10:
        score -= 45
    elif rainfall > 0:
        score -= 20

    if wind >= 8:
        score -= 30
    elif wind >= 5:
        score -= 15

    if temp <= 0 or temp >= 32:
        score -= 20

    # 비+강풍 복합 위험
    if rainfall >= 5 and wind >= 5:
        score -= 20
    # 영하권 강수 → 결빙 위험
    if temp <= 2 and rainfall > 0:
        score -= 15

    # 불쾌지수 (기온+습도 복합)
    if temp >= 25 and humidity >= 60:
        di = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.99) + 46.3
        if di >= 80:
            score -= 12
        elif di >= 75:
            score -= 6

    # PM2.5 (에어코리아 연계)
    pm25 = float(weather.get("pm25_ugm3") or 0)
    if pm25 >= 75:
        score -= 18
    elif pm25 >= 35:
        score -= 9
    elif pm25 >= 15:
        score -= 4

    return max(score, 0)


# ── 5개 세부 점수 함수 (모두 0.0~1.0) ──────────────────────────────────────────

def _difficulty_fit(course, profile):
    """코스 난이도 vs 사용자 경험 적합도."""
    target = float({"beginner": 1, "intermediate": 2, "advanced": 3}.get(profile.get("experience"), 2))
    course_level = float(DIFFICULTY_LEVEL.get(course.get("difficulty"), 2))
    gap = abs(course_level - target)
    return max(0.0, 1.0 - gap * 0.45)


def _time_fit(course, profile):
    """코스 소요시간 vs 가용 시간 적합도."""
    available = int(profile.get("availableMinutes", 180))
    duration = int(course.get("duration_min") or 0)
    if duration <= 0:
        return 0.5
    if duration > available:
        ratio = (duration - available) / max(available, 1)
        return max(0.0, 0.7 - ratio * 1.4)
    utilization = duration / available
    if utilization >= 0.65:
        return 1.0
    if utilization >= 0.35:
        return 0.70 + (utilization - 0.35) / 0.30 * 0.30
    return max(0.30, utilization / 0.35 * 0.70)


def _access_fit(course, user_lat, user_lng, max_km):
    """코스까지 이동 거리 접근성 점수."""
    if course.get("lat") is None or course.get("lng") is None:
        return 0.45, None
    dist = haversine_km(user_lat, user_lng, course["lat"], course["lng"])
    if dist <= max_km:
        score = 1.0 - (dist / max_km) * 0.30   # max_km 이내 → 0.70~1.0
    else:
        excess = (dist - max_km) / max(max_km, 1)
        score = max(0.0, 0.70 - excess * 0.70)  # 초과 시 급감
    return score, round(dist, 1)


def _daylight_fit(margin_min):
    """일몰 전 하산 여유 점수."""
    if margin_min is None:
        return 0.80
    if margin_min >= 90:
        return 1.0
    if margin_min >= 60:
        return 0.85
    if margin_min >= 30:
        return 0.65
    if margin_min >= 0:
        return 0.35
    if margin_min >= -30:
        return 0.10
    return 0.0


def _data_quality_multiplier(course):
    """데이터 품질 보정 배수 (0.5~1.0)."""
    dist_km = float(course.get("distance_km") or 0)
    if dist_km and dist_km < 0.3:
        return 0.50   # 거의 이동 없는 코스 → 제외에 가까운 감점
    if str(course.get("name", "")).strip() in GENERIC_COURSE_NAMES:
        return 0.82
    if course.get("lat") is None:
        return 0.75
    return 1.0


# ── 동적 가중치 (합계 = 1.0) ────────────────────────────────────────────────────

def compute_weights(profile):
    """목적에 따라 5개 요소 가중치를 동적으로 결정한다."""
    purpose = profile.get("purpose", "balanced")
    if purpose == "healing":
        return {"difficulty": 0.20, "time": 0.20, "access": 0.20, "weather": 0.30, "daylight": 0.10}
    if purpose == "workout":
        return {"difficulty": 0.35, "time": 0.20, "access": 0.15, "weather": 0.20, "daylight": 0.10}
    # balanced / view
    return {"difficulty": 0.30, "time": 0.20, "access": 0.20, "weather": 0.20, "daylight": 0.10}


# ── 안전 판정 ────────────────────────────────────────────────────────────────────

def safety_decision_for_course(course, profile, weather, w_score, diff_fit, time_f, daylight_margin):
    red_flags = []
    yellow_flags = []

    rainfall = float(weather.get("rainfall_mm", 0) or 0)
    wind = float(weather.get("wind_speed_ms", 0) or 0)
    wildfire = weather.get("wildfire_risk", "low")
    disaster_zones = course.get("disaster_risk_zones", [])
    disaster_level = disaster_risk_level(disaster_zones)

    if disaster_level == "high":
        red_flags.extend(disaster_risk_messages(disaster_zones)[:2])
    elif disaster_level == "caution":
        yellow_flags.extend(disaster_risk_messages(disaster_zones)[:2])

    if rainfall >= 10:
        red_flags.append("강수량이 높아 미끄럼 위험이 큽니다")
    elif rainfall > 0:
        yellow_flags.append("비 또는 젖은 노면으로 미끄럼 주의가 필요합니다")

    if wind >= 8:
        red_flags.append("강풍으로 능선부 보행 위험이 높습니다")
    elif wind >= 5:
        yellow_flags.append("풍속이 다소 높아 노출 구간 주의가 필요합니다")

    if daylight_margin is not None:
        if daylight_margin < 0:
            red_flags.append("일몰 전 하산이 불가능한 시간입니다")
        elif daylight_margin < 30:
            red_flags.append("일몰 전 하산 여유가 30분 미만입니다")
        elif daylight_margin < 60:
            yellow_flags.append("일몰 전 하산 여유가 충분하지 않습니다")

    if wildfire == "very_high":
        red_flags.append("산불 위험 단계가 높습니다")
    elif wildfire in {"medium", "high"}:
        yellow_flags.append("산불 위험을 확인하고 입산 안내를 따라야 합니다")

    elevation = int(course.get("elevation_gain_m", 0))
    if elevation >= 900:
        red_flags.append("누적 고도 상승량이 큽니다")
    elif elevation >= 500:
        yellow_flags.append("고도 상승 구간이 있어 체력 안배가 필요합니다")

    if red_flags:
        return {
            "safety_decision": "not_recommended",
            "safety_label": "비추천",
            "safety_color": "red",
            "risk_factors": red_flags + yellow_flags[:2],
        }

    # 재난위험지구 '주의'는 경고 문구가 1개뿐이어도 추천 단계에서 묻히지 않게 '주의'로 판정한다
    if len(yellow_flags) >= 2 or w_score < 75 or diff_fit < 0.40 or disaster_level == "caution":
        return {
            "safety_decision": "caution",
            "safety_label": "주의",
            "safety_color": "yellow",
            "risk_factors": yellow_flags or ["기상, 시간, 체력 조건 확인이 필요합니다"],
        }

    return {
        "safety_decision": "recommend",
        "safety_label": "추천",
        "safety_color": "green",
        "risk_factors": ["현재 조건에서 안정적으로 이용 가능한 코스입니다"],
    }


def safety_rank(decision):
    return {"recommend": 3, "caution": 2, "not_recommended": 1}.get(decision, 0)


def safety_grade(score):
    if score >= 80:
        return "safe"
    if score >= 55:
        return "caution"
    return "danger"


# ── 날씨 캐시 ──────────────────────────────────────────────────────────────────

def _cached_weather(lat, lng):
    cache_key = f"weather:{round(lat, 2)}:{round(lng, 2)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    result = fetch_current_weather(lat, lng)
    if result.get("source") != "mock":
        cache.set(cache_key, result, _WEATHER_CACHE_SECONDS)
    return result


# ── 메인 추천 함수 ──────────────────────────────────────────────────────────────

def recommend_courses(payload):
    profile = payload.get("profile", {})
    location = payload.get("location") or {"lat": 37.5665, "lng": 126.978}
    user_lat, user_lng = location["lat"], location["lng"]
    max_km = max(int(profile.get("maxDistanceKm", 50)), 10)

    mountain_name = (profile.get("mountainName") or "").strip()
    # 캐시된 코스 목록은 읽기만 한다(점수 계산 루프에서 코스별 새 dict를 만듦) → 8천여 개 전체 복사 불필요
    courses = load_public_trail_courses()

    query_lat, query_lng = user_lat, user_lng
    weather_lat, weather_lng = user_lat, user_lng

    if mountain_name:
        from .mountain_coordinates import find_mountain_coordinates
        _, mtn_coords = find_mountain_coordinates(mountain_name)
        if mtn_coords:
            query_lat = mtn_coords["lat"]
            query_lng = mtn_coords["lng"]
            weather_lat = mtn_coords["lat"]
            weather_lng = mtn_coords["lng"]

    weather = _cached_weather(weather_lat, weather_lng)
    w_score = weather_safety_score(weather)
    w_fit = w_score / 100.0
    all_disaster_zones = load_disaster_risk_zones()

    if mountain_name:
        radius_km = max(max_km / 2, 8)

        def _fetch_local():
            return fetch_local_road_trails(query_lat, query_lng, mountain_name, radius_km=radius_km, size=40)

        def _fetch_vworld():
            return fetch_vworld_trails(lat=query_lat, lng=query_lng, mountain_name=mountain_name, radius_km=max(radius_km, 5), size=40)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_fetch_local): "local", pool.submit(_fetch_vworld): "vworld"}
            api_results = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    api_results[key] = future.result()
                except Exception:
                    api_results[key] = {}

        courses = (
            api_results.get("vworld", {}).get("items", [])
            + api_results.get("local", {}).get("items", [])
            + courses
        )

    # 로컬 등산로는 메모리 절약을 위해 좌표 없이 캐시돼 있다 → 경로 보유 여부만 확인하고 좌표는 최종 결과에만 붙인다
    route_courses = [c for c in courses if c.get("route_geometry") or has_road_route(c)]
    if route_courses:
        courses = route_courses

    w = compute_weights(profile)
    HIGH_RISK_KW = {"추락", "낙석", "급경사", "붕괴", "고립"}
    recommendations = []
    # 출발·일몰 시각은 모든 코스에 공통이므로 한 번만 계산한다
    window = departure_window(weather, profile.get("departureDate"), profile.get("departureTime"))

    for course in courses:
        quality = _data_quality_multiplier(course)
        if quality < 0.55:
            continue

        # 원본 코스(워커 캐시)는 건드리지 않고, 요청마다 새 dict 하나만 만든다
        item = {**course, "disaster_risk_zones": _course_disaster_zones(course, all_disaster_zones)}

        diff_fit = _difficulty_fit(course, profile)
        time_f = _time_fit(course, profile)
        access_f, distance = _access_fit(course, user_lat, user_lng, max_km)
        daylight_margin = _margin_in_window(course, window)
        day_f = _daylight_fit(daylight_margin)

        # 5개 요소 가중 합산 → 0.0~1.0
        raw = (
            diff_fit  * w["difficulty"]
            + time_f  * w["time"]
            + access_f * w["access"]
            + w_fit   * w["weather"]
            + day_f   * w["daylight"]
        )

        # 데이터 품질 보정
        raw *= quality

        # 재난위험지구 감점 (최대 -0.15). 같은 산이라는 것만 확인된 위험지구(area)는 '주의' 수준으로만 감점한다.
        zones = item["disaster_risk_zones"]
        if zones:
            high = sum(
                1 for z in zones
                if z.get("match_scope") != "area" and any(kw in (z.get("risk_factor") or "") for kw in HIGH_RISK_KW)
            )
            caution_z = len(zones) - high
            raw -= min(high * 0.06 + caution_z * 0.02, 0.15)

        total = round(min(max(raw, 0.0), 1.0) * 100, 1)

        safety = safety_decision_for_course(
            item, profile, weather, w_score, diff_fit, time_f, daylight_margin
        )

        # reason·agent_briefing·safe_link_preview는 최종 노출 코스에만 만든다(_finalize_recommendation)
        item.update(safety)
        item.update({
            "score": total,
            "scores": {
                "fitness": round(diff_fit * 100, 1),
                "weather": round(w_score, 1),
                "accessibility": round(access_f * 100, 1),
                "time": round(time_f * 100, 1),
                "daylight": round(day_f * 100, 1),
            },
            "distance_from_user_km": distance,
            "daylight_margin_min": daylight_margin,
            "safety_grade": safety_grade(w_score),
            "weather": weather,
        })
        recommendations.append(item)

    recommendations.sort(
        key=lambda item: (safety_rank(item["safety_decision"]), item["score"]),
        reverse=True,
    )

    # 지정 산 코스 최상위 배치
    if mountain_name and not is_generic_mountain_name(mountain_name):
        target = mountain_name.replace(" ", "")
        matched = [
            item for item in recommendations
            if target in str(item.get("mountain", "")).replace(" ", "")
            or target in str(item.get("name", "")).replace(" ", "")
        ]
        matched_ids = {id(item) for item in matched}
        others = [item for item in recommendations if id(item) not in matched_ids]
        recommendations = matched + others

    top_items = recommendations[:3]
    alternative_items = select_alternatives(recommendations, top_items, profile)
    no_safe = bool(recommendations) and all(
        item["safety_decision"] == "not_recommended" for item in recommendations[:8]
    )

    # LLM 브리핑은 상위 3개에만(병렬 호출), 대안 코스는 템플릿 문장으로 만든다
    def _finalize_top(item):
        return _finalize_recommendation(item, profile, w_score, weather, use_llm=True)

    if len(top_items) > 1:
        with ThreadPoolExecutor(max_workers=len(top_items)) as pool:
            top3 = list(pool.map(_finalize_top, top_items))
    else:
        top3 = [_finalize_top(item) for item in top_items]
    alternatives = [
        _finalize_recommendation(item, profile, w_score, weather, use_llm=False)
        for item in alternative_items
    ]

    return {
        "result_state": "no_safe_course" if no_safe else "has_recommendations",
        "recommendations": top3,
        "alternatives": alternatives,
        "weather": weather,
        "agent_summary": (
            build_no_safe_course_message(weather) if no_safe
            else top3[0]["agent_briefing"] if top3 else ""
        ),
        "alternative_actions": (
            ["다른 날짜로 다시 보기", "검색 반경 넓히기", "쉬운 산책 코스만 보기", "실내 대체 활동 보기"]
            if no_safe else []
        ),
    }


def _finalize_recommendation(item, profile, w_score, weather, use_llm):
    """최종 노출 코스에만 경로 좌표·설명·브리핑을 붙인다."""
    item = with_route_geometry(item)
    distance = item.get("distance_from_user_km")
    daylight_margin = item.get("daylight_margin_min")
    item["reason"] = build_reason(item, profile, w_score, distance, daylight_margin)
    item["agent_briefing"] = build_agent_briefing(item, item, daylight_margin, weather, use_llm=use_llm)
    item["safe_link_preview"] = build_safe_link_preview(item, item)
    return item


# 코스별 재난위험지구 매칭 결과 캐시. 코스·위험지구 데이터 모두 워커 수명 동안 고정이라
# 요청마다 (코스 수 × 위험지구 수) 문자열 비교를 반복할 필요가 없다.
_ZONE_MATCH_CACHE = {"zones_ref": None, "matches": {}}
_ZONE_MATCH_CACHE_LIMIT = 100_000


def _course_disaster_zones(course, all_zones):
    if not all_zones:
        return []
    if _ZONE_MATCH_CACHE["zones_ref"] is not all_zones:
        _ZONE_MATCH_CACHE["zones_ref"] = all_zones
        _ZONE_MATCH_CACHE["matches"] = {}
    matches = _ZONE_MATCH_CACHE["matches"]
    key = (course.get("id"), course.get("mountain"), course.get("name"), course.get("region"))
    zones = matches.get(key)
    if zones is None:
        zones = find_course_disaster_risks(course, all_zones)
        if len(matches) < _ZONE_MATCH_CACHE_LIMIT:
            matches[key] = zones
    return list(zones)


# ── 브리핑·설명 빌더 ────────────────────────────────────────────────────────────

def build_agent_briefing(course, safety, daylight_margin=None, weather=None, use_llm=True):
    if use_llm:
        from .llm_briefing import generate_briefing
        llm = generate_briefing(course, safety, weather or {}, daylight_margin)
        if llm:
            return llm
    factors = ", ".join(safety["risk_factors"][:2])
    if safety["safety_decision"] == "recommend":
        return f"{course['name']}은 현재 기상과 하산 시간을 고려했을 때 안정적으로 이용 가능한 코스입니다."
    if safety["safety_decision"] == "caution":
        return f"{course['name']}은 주의가 필요합니다. {factors}"
    return f"{course['name']}은 현재 조건에서 비추천입니다. {factors}"


def build_safe_link_preview(course, safety):
    return {
        "enabled": True,
        "status": "정상 이동" if safety["safety_decision"] == "recommend" else "주의 모니터링",
        "last_sync": "방금 전",
        "alerts": [
            {"type": "risk_zone", "label": "보행 주의 구간 진입", "message": safety["risk_factors"][0]}
        ] if safety["safety_decision"] != "recommend" else [],
    }


def build_no_safe_course_message(weather):
    return (
        "현재 조건에서 안전하게 추천할 수 있는 코스가 없습니다. "
        f"강수 {weather.get('rainfall_mm', 0)}mm, 풍속 {weather.get('wind_speed_ms', 0)}m/s, "
        "일몰 여유를 고려해 산행을 연기하거나 쉬운 산책 코스를 확인하세요."
    )


def build_reason(course, profile, w_score, distance, daylight_margin=None):
    purpose = profile.get("purpose", "balanced")
    reasons = []

    if course.get("difficulty") == "easy":
        reasons.append("안전하게 다녀오기 좋은 짧은 코스를 우선했습니다")
    else:
        reasons.append("경험 수준에 맞춰 적합한 난이도의 코스를 골랐습니다")

    if w_score >= 80:
        reasons.append("강수와 풍속 위험이 낮아 날씨 안전도가 좋습니다")
    else:
        reasons.append("날씨 위험을 반영해 하산 부담이 낮은 코스를 우선했습니다")

    if purpose == "healing":
        reasons.append("힐링 목적에 맞춰 짧고 부담이 낮은 코스를 우선했습니다")
    elif purpose == "workout":
        reasons.append("운동 목적에 맞춰 적당한 활동량을 확보했습니다")
    elif purpose == "view":
        reasons.append("전망과 지도 확인이 쉬운 코스를 우선했습니다")

    if distance is None:
        reasons.append("좌표 데이터가 보강되면 현재 위치 기준 접근성을 더 정확히 반영할 수 있습니다")
    else:
        reasons.append(f"현재 위치에서 약 {distance}km 떨어져 있습니다")

    if daylight_margin is not None:
        if daylight_margin < 0:
            reasons.append("예상 하산 시간이 일몰 이후라 더 짧은 코스를 권장합니다")
        elif daylight_margin < 60:
            reasons.append("일몰 전 하산 여유가 크지 않아 출발 시간을 앞당기는 것이 좋습니다")
        else:
            reasons.append(f"일몰 전 약 {daylight_margin}분의 하산 여유가 있습니다")

    return " ".join(reasons) + "."


def is_generic_mountain_name(name):
    return str(name or "").replace(" ", "") in {"국립공원", "등산로", "브이월드등산로"}


def select_alternatives(recommendations, top_courses, profile=None):
    profile = profile or {}
    purpose = profile.get("purpose", "balanced")
    available = int(profile.get("availableMinutes", 180))
    top_ids = {course["id"] for course in top_courses}
    candidates = [c for c in recommendations if c["id"] not in top_ids]

    purpose_difficulty = {
        "workout": {"hard": 40, "medium": 30, "easy": 10},
        "healing": {"easy": 40, "medium": 20, "hard": 0},
        "view":    {"easy": 30, "medium": 30, "hard": 15},
    }.get(purpose, {"easy": 30, "medium": 25, "hard": 10})

    def alt_rank(course):
        daylight = course.get("daylight_margin_min")
        day_score = max(0, daylight) if daylight is not None else 0
        distance = course.get("distance_from_user_km")
        access_score = max(0, 80 - distance * 2) if distance is not None else 40
        diff_score = purpose_difficulty.get(course.get("difficulty"), 20)
        duration = int(course.get("duration_min", 0))
        time_score = max(0, 50 - abs(duration - available) * 0.5)
        return diff_score + time_score + day_score * 0.25 + access_score

    candidates.sort(key=alt_rank, reverse=True)
    return candidates[:2]


# ── 일몰 여유 계산 ─────────────────────────────────────────────────────────────

def daylight_margin_minutes(course, weather, now=None, departure_date=None, departure_time=None):
    return _margin_in_window(course, departure_window(weather, departure_date, departure_time, now))


def departure_window(weather, departure_date=None, departure_time=None, now=None):
    """(출발 시각, 일몰 시각). 일몰 정보가 없으면 None."""
    if not weather or not weather.get("sunset"):
        return None
    current = parse_departure_datetime(departure_date, departure_time, now)
    current = current or now or datetime.now(ZoneInfo("Asia/Seoul"))
    sunset = parse_departure_datetime(departure_date, weather["sunset"], current)
    if not sunset:
        return None
    return current, sunset


def _margin_in_window(course, window):
    if window is None:
        return None
    current, sunset = window
    finish = current + timedelta(minutes=int(course.get("duration_min", 0)))
    return round((sunset - finish).total_seconds() / 60)


def parse_departure_datetime(date_value=None, time_value=None, now=None):
    try:
        hour, minute = [int(part) for part in str(time_value).split(":")[:2]]
    except (TypeError, ValueError):
        return None
    current = now or datetime.now(ZoneInfo("Asia/Seoul"))
    date = current.date()
    if date_value:
        try:
            from datetime import date as date_cls
            date = datetime.strptime(str(date_value), "%Y-%m-%d").date()
        except ValueError:
            date = current.date()
    return datetime.combine(date, datetime.min.time(), tzinfo=current.tzinfo).replace(hour=hour, minute=minute)


def parse_today_time(value, now=None):
    return parse_departure_datetime(None, value, now)


def data_quality_adjustment(course):
    penalty = 0
    name = (course.get("name") or "").strip()
    mountain = (course.get("mountain") or "").strip()
    lat = course.get("lat")
    lng = course.get("lng")
    if name in GENERIC_COURSE_NAMES:
        penalty -= 18
    if mountain == "국립공원":
        penalty -= 5
    if lat is None or lng is None:
        penalty -= 10
    return penalty


def purpose_bonus(course, profile):
    purpose = profile.get("purpose", "balanced")
    duration = int(course.get("duration_min") or 0)
    difficulty = course.get("difficulty")
    bonus = 0
    if purpose == "healing":
        if difficulty == "easy":
            bonus += 15
        elif difficulty == "medium":
            bonus += 5
        if duration and duration <= 90:
            bonus += 10
    elif purpose == "workout":
        if difficulty == "hard":
            bonus += 15
        elif difficulty == "medium":
            bonus += 5
        if duration and duration >= 180:
            bonus += 10
    return bonus


def time_buffer_range(duration_min, profile):
    companion = profile.get("companion")
    if companion == "vulnerable":
        return (duration_min - 30, duration_min + 15)
    return (duration_min - 30, duration_min + 30)



