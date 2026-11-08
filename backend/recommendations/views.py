import json
import re
from functools import lru_cache

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.gzip import gzip_page
from django.views.decorators.http import require_GET, require_POST

from .landslide_api import fetch_landslide_prediction
from .loaders import load_public_service_key, load_public_trail_courses, load_disaster_risk_zones
from .mountain_coordinates import MOUNTAIN_COORDINATES
from .mountain_identity import mountain_key, with_mountain_identity
from .services import recommend_courses
from .mountain_recommend import recommend_mountains
from .mountain_data import MOUNTAINS
from .sun_api import fetch_sun_times
from .weather_api import fetch_current_weather
from .wildfire_api import fetch_wildfire_risk
from .vworld_api import fetch_vworld_trails
from . import safe_links as safe_link_store
from .osm_trail_api import fetch_osm_trails
from .local_road_api import fetch_local_road_trails


@require_GET
def health(request):
    key = load_public_service_key()
    return JsonResponse(
        {
            "status": "ok",
            "service": "올라 API",
            "public_trail_courses": len(load_public_trail_courses()),
            "public_service_key_loaded": bool(key),
            "public_service_key_prefix": (key[:6] + "...") if key else None,
            "fallback_mountain_coordinates": len(MOUNTAIN_COORDINATES),
        }
    )


# 프론트(산 목록·탐방로 목록)에 필요한 필드만 내려준다. 경로 좌표(route_geometry)를 포함하면 응답이 30MB를 넘는다.
COURSE_SUMMARY_FIELDS = (
    "id", "mountain", "name", "region", "difficulty", "distance_km", "duration_min",
    "elevation_gain_m", "lat", "lng", "has_entrance_start",
)


@lru_cache(maxsize=1)
def _course_summaries_json():
    summaries = [
        {field: course[field] for field in COURSE_SUMMARY_FIELDS if field in course}
        for course in load_public_trail_courses()
    ]
    return json.dumps({"courses": summaries}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@require_GET
@gzip_page
def courses(request):
    return HttpResponse(_course_summaries_json(), content_type="application/json; charset=utf-8")


@require_GET
def data_sources(request):
    key = load_public_service_key()
    return JsonResponse(
        {
            "connected_sources": [
                {
                    "id": "public_trails",
                    "name": "National park trail courses",
                    "status": "loaded",
                    "count": len(load_public_trail_courses()),
                },
                {
                    "id": "disaster_risk_zones",
                    "name": "Disaster risk zones",
                    "status": "loaded",
                    "count": len(load_disaster_risk_zones()),
                },
                {
                    "id": "weather",
                    "name": "KMA nowcast and public safety APIs",
                    "status": "configured" if key else "mock",
                },
            ],
            "service_key_loaded": bool(key),
        },
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def vworld_trails(request):
    mountain_name = request.GET.get("mountain", "")
    lat = request.GET.get("lat")
    lng = request.GET.get("lng")
    radius_km = request.GET.get("radius", 5)
    page_no = request.GET.get("page", 1)
    num_of_rows = request.GET.get("size", 50)
    return JsonResponse(
        fetch_vworld_trails(lat, lng, mountain_name, radius_km, page_no, num_of_rows),
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def local_road_trails(request):
    mountain_name = request.GET.get("mountain", "")
    lat = request.GET.get("lat")
    lng = request.GET.get("lng")
    radius_km = request.GET.get("radius", 8)
    size = request.GET.get("size", 80)
    return JsonResponse(
        fetch_local_road_trails(lat, lng, mountain_name, radius_km, size),
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
def mountain_story(request):
    mountain_name = request.GET.get("mountain", "")
    region = request.GET.get("region", "")
    elevation_m = request.GET.get("elevation_m") or None

    intro = _seeded_intro_for_mountain(mountain_name, region, elevation_m)
    if not intro:
        return JsonResponse(
            {"ok": True, "items": [], "source": "seed_mountain_descriptions"},
            json_dumps_params={"ensure_ascii": False},
        )

    return JsonResponse(
        {
            "ok": True,
            "items": [
                {
                    "mountain_id": f"intro-{mountain_name}",
                    "mountain": mountain_name,
                    "summary": intro,
                    "detail": intro,
                    "intro": intro,
                    "selection_reason": "",
                    "source": "seed_mountain_descriptions",
                }
            ],
            "source": "seed_mountain_descriptions",
        },
        json_dumps_params={"ensure_ascii": False},
    )


def _get_seeded_mountain_intro(mountain_name):
    name = (mountain_name or "").strip()
    if not name:
        return ""

    from .management.commands.seed_mountain_descriptions import DESCRIPTIONS
    return (DESCRIPTIONS.get(name) or "").strip()


_REGION_GROUP_ALIASES = {
    "seoul": ("서울", "서울특별시"),
    "busan": ("부산", "부산광역시"),
    "daegu": ("대구", "대구광역시"),
    "incheon": ("인천", "인천광역시"),
    "gwangju": ("광주", "광주광역시"),
    "daejeon": ("대전", "대전광역시"),
    "ulsan": ("울산", "울산광역시"),
    "sejong": ("세종", "세종특별자치시"),
    "gyeonggi": ("경기", "경기도"),
    "gangwon": ("강원", "강원도", "강원특별자치도"),
    "chungbuk": ("충북", "충청북도"),
    "chungnam": ("충남", "충청남도"),
    "jeonbuk": ("전북", "전라북도", "전북특별자치도"),
    "jeonnam": ("전남", "전라남도"),
    "gyeongbuk": ("경북", "경상북도"),
    "gyeongnam": ("경남", "경상남도"),
    "jeju": ("제주", "제주도", "제주특별자치도"),
}


def _seeded_intro_for_mountain(mountain_name, region="", elevation_m=None):
    intro = _get_seeded_mountain_intro(mountain_name)
    if not intro:
        return ""
    if not _seed_intro_matches_mountain(intro, region, elevation_m):
        return ""
    return intro


def _seed_intro_matches_mountain(intro, region="", elevation_m=None):
    intro_regions = _region_groups_in_text(intro)
    mountain_regions = _region_groups_in_text(region)
    if mountain_regions and intro_regions:
        if not (mountain_regions & intro_regions):
            return False

    intro_height = _extract_height_m(intro)
    if intro_height and elevation_m:
        try:
            if abs(float(intro_height) - float(elevation_m)) > 120:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _extract_height_m(text):
    match = re.search(
        r"(?:해발\s*)?([\d,]+(?:\.\d+)?)\s*(?:m|미터)", text or "", re.IGNORECASE
    )
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _region_group_for_text(text):
    haystack = text or ""
    for group, aliases in _REGION_GROUP_ALIASES.items():
        if any(alias in haystack for alias in aliases):
            return group
    return ""


def _region_groups_in_text(text):
    haystack = text or ""
    return {
        group
        for group, aliases in _REGION_GROUP_ALIASES.items()
        if any(alias in haystack for alias in aliases)
    }


@require_GET
def weather(request):
    from django.core.cache import cache
    from .airquality_api import fetch_air_quality
    lat = float(request.GET.get("lat", 37.5665))
    lng = float(request.GET.get("lng", 126.978))
    mountain = request.GET.get("mountain", "")
    cache_key = f"weather:{round(lat, 2)}:{round(lng, 2)}:{mountain}"
    cached = cache.get(cache_key)
    if cached:
        return JsonResponse(cached, json_dumps_params={"ensure_ascii": False})
    result = fetch_current_weather(lat, lng)
    # 미세먼지 통합 (에러 시 무시)
    try:
        aq = fetch_air_quality(mountain, lat=lat, lng=lng)
        if aq.get("ok"):
            result["pm10_ugm3"] = aq.get("pm10_ugm3")
            result["pm25_ugm3"] = aq.get("pm25_ugm3")
            result["grade_pm10"] = aq.get("grade_pm10")
            result["grade_pm25"] = aq.get("grade_pm25")
            result["air_station"] = aq.get("station_name")
    except Exception:
        pass
    if result.get("source") != "mock":
        cache.set(cache_key, result, 600)
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})


@require_GET
def sun_times(request):
    lat = float(request.GET.get("lat", 37.5665))
    lng = float(request.GET.get("lng", 126.978))
    return JsonResponse(fetch_sun_times(lat, lng), json_dumps_params={"ensure_ascii": False})


@require_GET
def wildfire(request):
    return JsonResponse(fetch_wildfire_risk(), json_dumps_params={"ensure_ascii": False})


@require_GET
def landslide(request):
    sgg = request.GET.get("sgg", "")
    forecast_name = request.GET.get("forecast", "")
    page_no = request.GET.get("page", 1)
    num_of_rows = request.GET.get("size", 10)
    return JsonResponse(
        fetch_landslide_prediction(sgg, forecast_name, page_no, num_of_rows),
        json_dumps_params={"ensure_ascii": False},
    )


@require_GET
@gzip_page
def mountains(request):
    return JsonResponse({"mountains": _get_mountains()}, json_dumps_params={"ensure_ascii": False})


_NATIONAL_PARKS = {
    "설악산","지리산","한라산","북한산","오대산","소백산","태백산","치악산",
    "월악산","속리산","계룡산","덕유산","가야산","내장산","무등산","주왕산",
    "월출산","팔공산",
}
_CROWDING_MAP = {
    "북한산": 0.82, "도봉산": 0.78, "남산": 0.80, "인왕산": 0.72,
    "관악산": 0.75, "청계산": 0.65, "아차산": 0.70, "수락산": 0.38,
    "불암산": 0.42, "설악산": 0.68, "한라산": 0.70, "지리산": 0.55,
    "계룡산": 0.55, "내장산": 0.58, "오대산": 0.48, "덕유산": 0.45,
    "속리산": 0.52, "주왕산": 0.48, "가야산": 0.55, "치악산": 0.45,
    "소백산": 0.40, "무등산": 0.58, "월출산": 0.42, "월악산": 0.42,
    "팔공산": 0.60, "태백산": 0.40, "함백산": 0.32, "두타산": 0.35,
    "금정산": 0.55, "마이산": 0.50, "황매산": 0.35,
}
_HARD_NAMES = {
    "설악산","지리산","한라산","덕유산","치악산","두타산","함백산","점봉산",
    "방태산","가리왕산","민주지산","화악산","청옥산","대암산","백덕산",
    "황석산","운문산","재약산","신불산","가지산","응봉산",
}
_EASY_NAMES = {
    "인왕산","아차산","수리산","칠갑산","천성산","불암산","선자령","삼악산",
    "수락산","달마산","연화산","팔영산",
}
_HEALING_NAMES = {
    "오대산","내장산","두륜산","천관산","조계산","달마산","연화산","팔영산",
    "선자령","장안산","천성산",
}
_VIEW_NAMES = {
    "설악산","한라산","지리산","북한산","팔공산","무등산","월출산","마이산",
    "황매산","비슬산","팔영산","두타산","금정산",
}
_EASY_ACCESS = {
    "인왕산","아차산","수리산","선자령","삼악산","불암산","수락산","청계산",
    "달마산","두륜산","천관산","금정산","토함산","조계산","연화산","팔영산",
    "천성산","내장산","주왕산","공작산","천마산","도봉산","관악산",
}

_KW_MAP = [
    ("폭포", "폭포 경관"), ("단풍", "단풍 명소"), ("눈", "설경 명소"),
    ("설", "설경 명소"), ("암릉", "암릉 경관"), ("바위", "암릉 경관"),
    ("계곡", "계곡 트레킹"), ("사찰", "사찰 탐방"), ("절", "사찰 탐방"),
    ("전망", "정상 전망 우수"), ("조망", "정상 전망 우수"),
    ("철쭉", "철쭉 명소"), ("동백", "동백 명소"), ("억새", "억새 군락"),
]


def _infer_difficulty(height, name):
    if name in _HARD_NAMES or height >= 1400:
        return "hard"
    if name in _EASY_NAMES or height <= 400:
        return "easy"
    return "medium"


def _infer_companion_fit(difficulty, name):
    if name in _EASY_ACCESS or difficulty == "easy":
        return ["vulnerable", "family", "solo"]
    if difficulty == "medium":
        return ["family", "solo"]
    return ["solo"]


def _infer_purpose_fit(height, name):
    fits = ["balanced"]
    if name in _HEALING_NAMES:
        fits.append("healing")
    if name in _VIEW_NAMES:
        fits.append("view")
    if height >= 1200:
        fits.append("workout")
    if len(fits) == 1:
        fits.append("healing" if height < 800 else "view")
    return list(dict.fromkeys(fits))


def _infer_walk_time(height, difficulty):
    if difficulty == "easy":
        return (60, 180)
    if difficulty == "medium":
        return (90, 240) if height < 800 else (150, 360)
    if height >= 1700:
        return (360, 720)
    if height >= 1400:
        return (270, 540)
    return (200, 450)


def _infer_highlights(name, summary):
    tags = []
    if name in _NATIONAL_PARKS:
        tags.append("국립공원")
    seen = set()
    for kw, label in _KW_MAP:
        if kw in summary and label not in seen:
            tags.append(label)
            seen.add(label)
    if not tags:
        tags.append("자연 경관 우수")
    return tags[:4]


def _get_mountains():
    from .models import MountainKnowledge, MountainTags
    from .mountain_data import MOUNTAINS as _STATIC

    tags_map = {t.mountain_name: t.tags for t in MountainTags.objects.all()}

    static_mountains = []
    for mountain in _STATIC:
        item = dict(mountain)
        name = item["name"]
        seed_intro = _seeded_intro_for_mountain(name, item.get("region", ""), item.get("elevation_m"))
        intro = seed_intro or (item.get("description") or "").strip()
        item["description"] = intro
        item["intro"] = intro
        item["mountain_key"] = mountain_key(name, item.get("region", ""))
        item["description_source"] = (
            "seed_mountain_descriptions"
            if seed_intro
            else ("mountain_data" if intro else "")
        )
        item["tags"] = tags_map.get(name, [])

        static_mountains.append(item)

    static_keys = {mountain_key(m.get("name", ""), m.get("region", "")) for m in static_mountains}

    # MountainKnowledge 중 좌표 있는 것을 동적으로 추가
    db_mountains = []
    seen = set(static_keys)
    qs = (
        MountainKnowledge.objects
        .exclude(lat__isnull=True)
        .exclude(lng__isnull=True)
        .order_by("mountain_name", "source")
    )
    for mk in qs:
        name = mk.mountain_name
        key = mountain_key(name, mk.region or "")
        if key in seen:
            continue
        seen.add(key)
        height = mk.height_m or 0
        difficulty = _infer_difficulty(height, name)
        walk_min, walk_max = _infer_walk_time(height, difficulty)
        seed_intro = _seeded_intro_for_mountain(name, mk.region or "", height)
        db_intro = (mk.summary or mk.detail or "").strip()
        intro = seed_intro or db_intro
        db_mountains.append(with_mountain_identity({
            "id": f"m-{name}",
            "name": name,
            "region": mk.region or "",
            "lat": mk.lat,
            "lng": mk.lng,
            "elevation_m": height if height > 0 else None,
            "difficulty": difficulty,
            "walk_time_min": walk_min,
            "walk_time_max": walk_max,
            "trail_count": 0,
            "companion_fit": _infer_companion_fit(difficulty, name),
            "purpose_fit": _infer_purpose_fit(height, name),
            "crowding": _CROWDING_MAP.get(name, 0.35),
            "national_park": name in _NATIONAL_PARKS,
            "highlights": _infer_highlights(name, ""),
            "description": intro,
            "intro": intro,
            "description_source": (
                "seed_mountain_descriptions"
                if seed_intro
                else (mk.source if intro else "")
            ),
            "tags": tags_map.get(name, []),
        }))

    return static_mountains + db_mountains


@csrf_exempt
@require_POST
def recommend_mountains_view(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    # 일몰 시각 조회 — 위치 없으면 서울 기본좌표 사용 (채점에서 제외되면 안 됨)
    loc = payload.get("location") or {}
    lat = float(loc["lat"]) if loc.get("lat") else 37.5665
    lng = float(loc["lng"]) if loc.get("lng") else 126.978
    try:
        sun_data = fetch_sun_times(lat, lng) or {}
        payload["sun_times"] = sun_data
    except Exception:
        pass

    result = recommend_mountains(payload)

    # 산 소개: 시드 파일(최신) 우선, 없으면 MountainIntro DB 폴백
    from .models import MountainIntro, MountainTags
    db_intros = {obj.mountain_name: obj.intro for obj in MountainIntro.objects.all()}
    tags_map = {t.mountain_name: t.tags for t in MountainTags.objects.all()}
    for m in result.get("mountains", []) + result.get("alternatives", []):
        seed = _seeded_intro_for_mountain(m["name"], m.get("region", ""), m.get("elevation_m"))
        if seed:
            m["intro"] = seed
        else:
            db_intro = db_intros.get(m["name"], "")
            m["intro"] = db_intro if _seed_intro_matches_mountain(
                db_intro, m.get("region", ""), m.get("elevation_m"),
            ) else ""
        m["tags"] = tags_map.get(m["name"], [])

    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def recommendations(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    return JsonResponse(recommend_courses(payload))


@require_GET
def osm_trails(request):
    lat = request.GET.get("lat")
    lng = request.GET.get("lng")
    if not lat or not lng:
        return JsonResponse({"items": [], "error": "lat/lng required"}, status=400)
    mountain = request.GET.get("mountain", "")
    radius = int(request.GET.get("radius", 3000))
    result = fetch_osm_trails(float(lat), float(lng), mountain, radius)
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})


@require_GET
def disaster_zones(request):
    mountain = request.GET.get("mountain", "").strip()
    zones = load_disaster_risk_zones()
    if mountain:
        from .loaders import normalize_search_text
        needle = normalize_search_text(mountain)
        zones = [z for z in zones if needle in z.get("search_text", "")]
    return JsonResponse({"zones": zones[:30]}, json_dumps_params={"ensure_ascii": False})


@require_GET
def safe_link_by_code(request):
    code = request.GET.get("code", "").strip().upper()
    if not code:
        return JsonResponse({"error": "코드를 입력해 주세요."}, status=400)
    try:
        session = safe_link_store.get_by_code(code)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("safe_link_by_code 오류: %s", exc, exc_info=True)
        return JsonResponse({"error": f"서버 오류: {exc}"}, status=500)
    if not session:
        return JsonResponse({"error": f"코드 '{code}'를 찾을 수 없습니다. (DB에 없음)"}, status=404)
    return JsonResponse(session, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
def safe_link_create(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    try:
        course = body.get("course", {})
        session = safe_link_store.create(course)
        return JsonResponse({"id": session["id"], "share_code": session.get("share_code", "")}, status=201)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("safe_link_create 오류: %s", exc, exc_info=True)
        return JsonResponse({"error": f"세이프 링크 생성 실패: {exc}"}, status=500)


@csrf_exempt
def safe_link_detail(request, session_id):
    if request.method == "GET":
        session = safe_link_store.get(session_id)
        if not session:
            return JsonResponse({"error": "Not found"}, status=404)
        return JsonResponse(session, json_dumps_params={"ensure_ascii": False})

    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        action = body.get("action")
        if action == "end":
            session = safe_link_store.end_session(session_id)
            return JsonResponse({"ok": True}) if session else JsonResponse({"error": "Not found"}, status=404)
        lat = body.get("lat")
        lng = body.get("lng")
        if lat is None or lng is None:
            return JsonResponse({"error": "lat and lng required"}, status=400)
        session = safe_link_store.update_location(session_id, float(lat), float(lng))
        if not session:
            return JsonResponse({"error": "Not found"}, status=404)
        return JsonResponse({"ok": True, "location_ts": session["location_ts"]})

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
@require_POST
def chat_view(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid request body"}, status=400)

    messages = body.get("messages", [])
    context = body.get("context", {})
    if not messages:
        return JsonResponse({"error": "messages required"}, status=400)

    try:
        context = _enrich_chat_context(context)
        from .chat_ai import get_chat_response
        response_text = get_chat_response(messages, context)
        return JsonResponse({"response": response_text})
    except Exception as exc:
        import traceback, logging
        logging.getLogger(__name__).error("chat_view error: %s", traceback.format_exc())
        return JsonResponse({"error": str(exc), "response": "일시적인 오류가 발생했습니다."}, status=500)


def _enrich_chat_context(context: dict) -> dict:
    """실시간 안전 데이터 5종을 ThreadPoolExecutor로 병렬 조회 (직렬 대비 ~5배 단축)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .nifos_api import fetch_nifos_mountain_weather, fetch_nifos_fine_dust
    from .airquality_api import fetch_air_quality
    from .forest_flux_api import fetch_forest_flux

    context = dict(context)
    mountain = context.get("mountain") or {}
    mountain_name = mountain.get("name", "")
    region = (mountain.get("region") or "").replace("국립공원", "").strip()
    m_lat = mountain.get("lat") or mountain.get("course_lat")
    m_lng = mountain.get("lng") or mountain.get("course_lng")
    lat = float(m_lat) if m_lat else None
    lng = float(m_lng) if m_lng else None

    tasks = {
        "wildfire":    lambda: fetch_wildfire_risk(),
        "landslide":   lambda: fetch_landslide_prediction(region, "", 1, 5) if region else None,
        "nifos_weather": lambda: fetch_nifos_mountain_weather(mountain_name),
        "nifos_dust":  lambda: fetch_nifos_fine_dust(),
        "air_quality": lambda: fetch_air_quality(mountain_name, lat=m_lat, lng=m_lng),
        "forest_flux": lambda: fetch_forest_flux(mountain_name=mountain_name, lat=lat, lng=lng),
    }

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                if result is not None:
                    context[key] = result
            except Exception:
                pass

    return context


@require_GET
def nifos_mountain_weather(request):
    station_code = request.GET.get("station", "")
    from .nifos_api import fetch_nifos_mountain_weather
    return JsonResponse(fetch_nifos_mountain_weather(station_code), json_dumps_params={"ensure_ascii": False})


@require_GET
def air_quality(request):
    mountain = request.GET.get("mountain", "")
    from .airquality_api import fetch_air_quality
    return JsonResponse(fetch_air_quality(mountain), json_dumps_params={"ensure_ascii": False})


@require_GET
def nifos_fine_dust(request):
    station_code = request.GET.get("station", "")
    from .nifos_api import fetch_nifos_fine_dust
    return JsonResponse(fetch_nifos_fine_dust(station_code), json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def safety_advice_view(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    mountain = body.get("mountain", {})
    weather = body.get("weather", {})
    profile = body.get("profile", {})

    # Fetch sun times from mountain coordinates
    sun_data = {}
    lat = mountain.get("lat") or mountain.get("latitude")
    lng = mountain.get("lng") or mountain.get("longitude")
    if lat and lng:
        try:
            sun_data = fetch_sun_times(lat, lng) or {}
        except Exception:
            pass

    from .safety_advice_ai import generate_safety_advice
    advice = generate_safety_advice(mountain, weather, profile, sun_data)
    return JsonResponse({"advice": advice or ""})


def _cached_weather(lat, lng, mountain=""):
    from django.core.cache import cache
    cache_key = f"weather:{round(lat, 2)}:{round(lng, 2)}:{mountain}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    result = fetch_current_weather(lat, lng)
    if result:
        cache.set(cache_key, result, 600)
    return result


@require_GET
def ml_risk_view(request):
    """지정 시각 및 날씨 기반 소방청 사고 ML 위험 예측 (위험 지수 보정 및 24시간 추이 반환)."""
    import datetime
    from .accident_model import predict_accident_risk, predict_hourly_risk_indexes

    # 1. 날짜 및 시간 파라미터 파싱
    date_str = request.GET.get("date")
    time_str = request.GET.get("time")

    now = datetime.datetime.now()
    month = now.month
    hour = now.hour
    weekday = now.weekday()

    if date_str:
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            month = dt.month
            weekday = dt.weekday()
        except ValueError:
            pass

    if time_str:
        try:
            hour = int(time_str.split(":")[0])
        except (ValueError, IndexError):
            pass

    # 2. 날씨 정보 및 점수 산출
    lat_str = request.GET.get("lat")
    lng_str = request.GET.get("lng")
    mountain = request.GET.get("mountain", "")
    weather_score = 1.0
    weather = None

    if lat_str and lng_str:
        try:
            lat = float(lat_str)
            lng = float(lng_str)
            weather = _cached_weather(lat, lng, mountain)
            if weather:
                from .mountain_recommend import _weather_score
                weather_score = _weather_score(weather)
        except Exception:
            pass

    # 3. ML 기본 예측
    temp = 15.0
    rain = 0.0
    wind = 1.5
    humidity = 50.0
    if weather:
        try:
            temp = float(weather.get("temperature_c", 15.0) or 15.0)
            rain = float(weather.get("rainfall_mm", 0.0) or 0.0)
            wind = float(weather.get("wind_speed_ms", 1.5) or 1.5)
            humidity = float(weather.get("humidity_pct", 50.0) or 50.0)
        except Exception:
            pass

    result = predict_accident_risk(
        month=month, hour=hour, weekday=weekday,
        temp=temp, rain=rain, wind=wind, humidity=humidity
    )

    # 4. 날씨 보정 적용
    base_risk = result["risk_index"]
    weather_penalty = 1.0 - weather_score
    # 날씨 안 좋을수록 사고 지수 가산 (최대 +0.50)
    adjusted_risk = min(1.0, base_risk + weather_penalty * 0.50)
    result["risk_index"] = round(adjusted_risk, 3)
    result["ml_safety_score"] = round(1.0 - adjusted_risk, 3)

    # 날씨 반영 경고 조합
    if weather_penalty > 0.15:
        w_reasons = []
        if weather:
            try:
                rainfall = float(weather.get("rainfall_mm", 0) or 0)
                wind = float(weather.get("wind_speed_ms", 0) or 0)
                if rainfall > 0:
                    w_reasons.append(f"비({rainfall}mm)")
                if wind >= 5:
                    w_reasons.append(f"강풍({wind}m/s)")
            except Exception:
                pass
        w_msg = f"기상 악화 요인({', '.join(w_reasons) if w_reasons else '비/바람'}) 반영으로 위험도가 크게 상승했습니다."
        result["warning"] = w_msg + " " + result.get("warning", "")

    # 5. 24시간 추이 계산 (날씨 보정 일괄 적용) — 24개 시간대를 한 번에 예측
    hourly_base = predict_hourly_risk_indexes(
        month=month, weekday=weekday,
        temp=temp, rain=rain, wind=wind, humidity=humidity
    )
    result["hourly_risks"] = [
        {"hour": h, "risk_index": round(min(1.0, h_base + weather_penalty * 0.50), 3)}
        for h, h_base in enumerate(hourly_base)
    ]

    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})


@require_GET
def forest_flux_view(request):
    """산행 환경 지수 — 날씨·대기질·산불 실시간 조합."""
    mountain_name = request.GET.get("mountain", "")
    lat_str = request.GET.get("lat", "")
    lng_str = request.GET.get("lng", "")
    lat = float(lat_str) if lat_str else None
    lng = float(lng_str) if lng_str else None

    from .forest_flux_api import fetch_forest_flux
    from .weather_api import fetch_current_weather
    from .airquality_api import fetch_air_quality
    from .wildfire_api import fetch_wildfire_risk
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_weather = ex.submit(fetch_current_weather, lat, lng) if lat and lng else None
        f_aq = ex.submit(fetch_air_quality, mountain_name, lat=lat, lng=lng)
        f_wf = ex.submit(fetch_wildfire_risk)

    try:
        weather = f_weather.result() if f_weather else None
    except Exception:
        weather = None
    try:
        aq = f_aq.result()
    except Exception:
        aq = {}
    try:
        wf = f_wf.result()
    except Exception:
        wf = {}

    data = fetch_forest_flux(
        mountain_name=mountain_name, lat=lat, lng=lng,
        weather=weather, air_quality=aq, wildfire=wf,
    )
    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def mountain_intro_view(request):
    """seed_mountain_descriptions로 저장한 산 소개문만 반환."""
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = (body.get("name") or "").strip()
    if not name:
        return JsonResponse({"intro": ""})

    return JsonResponse(
        {"intro": _get_seeded_mountain_intro(name)},
        json_dumps_params={"ensure_ascii": False},
    )
