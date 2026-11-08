import gzip
import json
import logging
import math
import struct
import sys
from functools import lru_cache

from django.conf import settings

from .loaders import clean_course_name
from .mountain_zip_api import fetch_mountain_zip_trails
from .vworld_api import route_length_km


logger = logging.getLogger(__name__)

ROAD_DIR = settings.BASE_DIR.parent / "road" / "WG_MT_WAY"
ROAD_SHP_PATH = ROAD_DIR / "WG_MT_WAY.shp"
ROAD_DBF_PATH = ROAD_DIR / "WG_MT_WAY.dbf"

# SHP(약 150MB)를 미리 가공해 둔 경량 캐시. `python manage.py build_road_trails_cache`로 생성한다.
ROAD_CACHE_PATH = settings.BASE_DIR.parent / "data" / "road_trails.json.gz"
ROAD_CACHE_VERSION = 1
ROAD_CACHE_FIELDS = [
    "id", "mountain", "name", "difficulty", "distance_km", "duration_min",
    "elevation_gain_m", "lat", "lng", "highlights", "road_risk", "route_polyline",
]
ROAD_REGION = "브이월드 등산로 로컬 SHP"
ROAD_SOURCE = "road/WG_MT_WAY.shp"


def fetch_local_road_trails(lat=None, lng=None, mountain_name="", radius_km=8, size=30):
    archive_result = fetch_mountain_zip_trails(lat, lng, mountain_name, radius_km, size)
    archive_items = archive_result.get("items") or []
    if archive_items:
        return archive_result

    records = load_local_road_trails()
    if not records and not archive_items:
        return {"ok": False, "source": "local_road_shp", "items": [], "error": "Local road shapefile not found"}

    candidates = records
    target = normalize_text(mountain_name)
    if target:
        matched = [
            record
            for record in candidates
            if target in normalize_text(record.get("mountain", ""))
            or target in normalize_text(record.get("name", ""))
            or target in normalize_text(record.get("region", ""))
        ]
        if not matched and (lat is None or lng is None):
            return {"ok": True, "source": "local_road_shp", "total_count": 0, "items": []}
        if matched:
            candidates = matched

    if lat is not None and lng is not None:
        lat = float(lat)
        lng = float(lng)
        candidates = [
            record
            for record in candidates
            if haversine_km(lat, lng, record["lat"], record["lng"]) <= float(radius_km)
        ] or candidates

    items = archive_items + [item for item in candidates if item.get("id") not in {a.get("id") for a in archive_items}]
    return {
        "ok": True,
        "source": "forest_mountain_zip+local_road_shp" if archive_items else "local_road_shp",
        "total_count": len(items),
        "items": [with_route_geometry(item) for item in items[: int(size)]],
    }


def load_local_road_trails():
    """로컬 등산로 목록. 메모리 절약을 위해 route_geometry는 빠져 있다.

    좌표가 필요하면 with_route_geometry(course)로 해당 코스만 복원한다.
    """
    return _road_store()[0]


def has_road_route(course):
    return bool(course.get("id")) and course["id"] in _road_store()[1]


def with_route_geometry(course):
    """로컬 등산로 코스에 route_geometry를 붙인 사본을 돌려준다. 원본(캐시)은 건드리지 않는다."""
    if course.get("route_geometry"):
        return course
    encoded = _road_store()[1].get(course.get("id"))
    if not encoded:
        return course
    return {**course, "route_geometry": decode_polyline(encoded)}


@lru_cache(maxsize=1)
def _road_store():
    """(코스 목록, {코스 id: 인코딩된 경로}) — 워커당 한 번만 만든다."""
    rows = read_road_cache()
    if rows is None:
        if not ROAD_SHP_PATH.exists() or not ROAD_DBF_PATH.exists():
            return [], {}
        logger.warning(
            "%s 이(가) 없어 SHP를 직접 읽습니다(수십 초, 1GB 이상 메모리 사용). "
            "`python manage.py build_road_trails_cache`로 캐시를 만들어 두세요.",
            ROAD_CACHE_PATH,
        )
        rows = build_road_cache_rows()

    courses = []
    routes = {}
    intern = sys.intern
    for (course_id, mountain, name, difficulty, distance_km, duration_min,
         elevation_gain_m, lat, lng, highlights, road_risk, route_polyline) in rows:
        courses.append({
            "id": course_id,
            "mountain": intern(mountain),
            "name": intern(name),
            "region": ROAD_REGION,
            "difficulty": intern(difficulty),
            "distance_km": distance_km,
            "duration_min": duration_min,
            "elevation_gain_m": elevation_gain_m,
            "lat": lat,
            "lng": lng,
            "crowding": 0.28,
            "highlights": [intern(text) for text in highlights],
            "source": ROAD_SOURCE,
            "road_risk": intern(road_risk),
        })
        routes[course_id] = route_polyline
    return courses, routes


def read_road_cache(path=None):
    path = path or ROAD_CACHE_PATH
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, ValueError) as exc:
        logger.warning("등산로 캐시를 읽지 못했습니다(%s): %s", path, exc)
        return None
    if payload.get("version") != ROAD_CACHE_VERSION or payload.get("fields") != ROAD_CACHE_FIELDS:
        logger.warning("등산로 캐시 형식이 맞지 않습니다(%s). 캐시를 다시 생성하세요.", path)
        return None
    return payload.get("trails") or []


def build_road_cache_rows():
    """SHP·DBF를 파싱해 캐시 한 줄씩(ROAD_CACHE_FIELDS 순서)의 리스트로 만든다."""
    rows = read_dbf_records(ROAD_DBF_PATH)
    shapes = read_polyline_shapes(ROAD_SHP_PATH)
    cache_rows = []
    for index, (row, shape) in enumerate(zip(rows, shapes), start=1):
        course = normalize_road_record(index, row, shape)
        if course:
            cache_rows.append([
                course["id"],
                course["mountain"],
                course["name"],
                course["difficulty"],
                course["distance_km"],
                course["duration_min"],
                course["elevation_gain_m"],
                round(course["lat"], 6),
                round(course["lng"], 6),
                course["highlights"],
                course["road_risk"],
                encode_polyline(course["route_geometry"]),
            ])
    return cache_rows


def write_road_cache(rows, path=None):
    path = path or ROAD_CACHE_PATH
    payload = {
        "version": ROAD_CACHE_VERSION,
        "source": ROAD_SOURCE,
        "fields": ROAD_CACHE_FIELDS,
        "trails": rows,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw_file:
        # mtime=0: 같은 입력이면 같은 파일이 나오도록(불필요한 git 변경 방지)
        with gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=9, mtime=0) as cache_file:
            cache_file.write(body)
    return path


def encode_polyline(points, precision=5):
    """[{lat, lng}, ...] → Google Encoded Polyline 문자열 (1e-5도 ≈ 1.1m 정밀도)."""
    factor = 10 ** precision
    chunks = []
    prev_lat = prev_lng = 0
    for point in points:
        lat = round(point["lat"] * factor)
        lng = round(point["lng"] * factor)
        for delta in (lat - prev_lat, lng - prev_lng):
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                chunks.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            chunks.append(chr(value + 63))
        prev_lat, prev_lng = lat, lng
    return "".join(chunks)


def decode_polyline(encoded, precision=5):
    factor = 10 ** precision
    points = []
    index = lat = lng = 0
    length = len(encoded)
    while index < length:
        deltas = []
        for _ in range(2):
            shift = result = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        lat += deltas[0]
        lng += deltas[1]
        points.append({"lat": lat / factor, "lng": lng / factor})
    return points


def normalize_road_record(index, row, points):
    route_geometry = simplify_points([korea_2000_to_wgs84(x, y) for x, y in points], max_points=96)
    if len(route_geometry) < 2:
        return None

    distance_km = parse_float(row.get("PMNTN_LT")) or route_length_km(route_geometry)
    if distance_km and distance_km > 100:
        distance_km = distance_km / 1000
    distance_km = round(distance_km or route_length_km(route_geometry), 2)
    duration_min = max(round(distance_km * 34), 20)
    difficulty = normalize_difficulty(row.get("PMNTN_DFFL"), distance_km, duration_min)
    trailhead = route_geometry[0]  # 첫 번째 좌표 = 등산로 입구

    mountain = row.get("MNTN_NM") or "등산로"
    name = clean_course_name(row.get("PMNTN_NM") or row.get("PMNTN_MAIN") or f"{mountain} 등산로")
    risk = row.get("PMNTN_RISK", "")

    return {
        "id": f"local-road-{index}-{row.get('MNTN_ID') or row.get('PMNTN_SN') or name}",
        "mountain": mountain,
        "name": name,
        "region": "브이월드 등산로 로컬 SHP",
        "difficulty": difficulty,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "elevation_gain_m": round(distance_km * {"easy": 35, "medium": 65, "hard": 95}[difficulty]),
        "lat": trailhead["lat"],
        "lng": trailhead["lng"],
        "route_geometry": route_geometry,
        "crowding": 0.28,
        "highlights": build_road_highlights(row),
        "source": "road/WG_MT_WAY.shp",
        "road_risk": risk,
    }


def read_polyline_shapes(path):
    data = path.read_bytes()
    offset = 100
    shapes = []
    while offset + 8 <= len(data):
        content_length = struct.unpack(">i", data[offset + 4 : offset + 8])[0] * 2
        content_offset = offset + 8
        content = data[content_offset : content_offset + content_length]
        offset = content_offset + content_length
        if len(content) < 44:
            continue
        shape_type = struct.unpack("<i", content[0:4])[0]
        if shape_type != 3:
            continue
        num_parts = struct.unpack("<i", content[36:40])[0]
        num_points = struct.unpack("<i", content[40:44])[0]
        parts_offset = 44
        points_offset = parts_offset + num_parts * 4
        points = []
        for point_index in range(num_points):
            point_offset = points_offset + point_index * 16
            x, y = struct.unpack("<dd", content[point_offset : point_offset + 16])
            points.append((x, y))
        shapes.append(points)
    return shapes


def read_dbf_records(path):
    data = path.read_bytes()
    record_count = struct.unpack("<I", data[4:8])[0]
    header_length = struct.unpack("<H", data[8:10])[0]
    record_length = struct.unpack("<H", data[10:12])[0]
    field_count = (header_length - 33) // 32
    fields = []
    for index in range(field_count):
        offset = 32 + index * 32
        raw_name = data[offset : offset + 11].split(b"\x00", 1)[0]
        fields.append({"name": raw_name.decode("ascii", errors="ignore"), "length": data[offset + 16]})

    records = []
    for record_index in range(record_count):
        offset = header_length + record_index * record_length
        if offset + record_length > len(data) or data[offset : offset + 1] == b"*":
            continue
        cursor = offset + 1
        row = {}
        for field in fields:
            value = data[cursor : cursor + field["length"]]
            cursor += field["length"]
            row[field["name"]] = value.decode("cp949", errors="ignore").strip()
        records.append(row)
    return records


def korea_2000_to_wgs84(x, y):
    a = 6378137.0
    inv_f = 298.257222101
    f = 1 / inv_f
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0 = 0.9996
    x0 = 1000000.0
    y0 = 2000000.0
    lon0 = math.radians(127.5)
    lat0 = math.radians(38.0)

    def meridional_arc(phi):
        return a * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
            - (35 * e2**3 / 3072) * math.sin(6 * phi)
        )

    m0 = meridional_arc(lat0)
    m = m0 + (y - y0) / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)
    n1 = a / math.sqrt(1 - e2 * sin_phi1**2)
    r1 = a * (1 - e2) / (1 - e2 * sin_phi1**2) ** 1.5
    t1 = tan_phi1**2
    c1 = ep2 * cos_phi1**2
    d = (x - x0) / (n1 * k0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    lon = lon0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cos_phi1

    return {"lat": math.degrees(lat), "lng": math.degrees(lon)}


def simplify_points(points, max_points=96):
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    simplified = points[::step]
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def normalize_difficulty(value, distance_km, duration_min):
    text = str(value or "")
    if "어려" in text or "상" in text or distance_km >= 6 or duration_min >= 180:
        return "hard"
    if "보통" in text or "중" in text or distance_km >= 3 or duration_min >= 90:
        return "medium"
    return "easy"


def build_road_highlights(row):
    highlights = []
    if row.get("PMNTN_MAIN"):
        highlights.append(row["PMNTN_MAIN"])
    if row.get("PMNTN_MTRQ"):
        highlights.append(row["PMNTN_MTRQ"])
    if row.get("PMNTN_RISK"):
        highlights.append(f"위험요인: {row['PMNTN_RISK']}")
    return highlights[:3] or ["로컬 등산로 선형 데이터"]


def parse_float(value):
    try:
        return float(str(value or "").replace(",", ""))
    except ValueError:
        return None


def haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def normalize_text(value):
    return "".join(str(value or "").split()).lower()
