import gzip
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from unittest.mock import patch

from django.test import Client, TestCase

from .llm_briefing import _cache_key as briefing_cache_key
from .rag_retriever import retrieve_trail_context
from .services import (
    recommend_courses,
    data_quality_adjustment,
    daylight_margin_minutes,
    purpose_bonus,
    select_alternatives,
    time_buffer_range,
)
from .loaders import clean_course_name, infer_mountain_name


class RecommendationApiTests(TestCase):
    def test_health_endpoint(self):
        response = Client().get("/api/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_data_sources_endpoint_does_not_expose_key(self):
        response = Client().get("/api/data-sources/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("connected_sources", data)
        self.assertIsInstance(data.get("service_key_loaded"), bool)
        self.assertNotIn("73ce", str(data))

    def test_homonymous_mountain_uses_verified_location(self):
        from .views import _get_mountains

        bakdalsan = next(
            item for item in _get_mountains()
            if item["name"] == "\ubc15\ub2ec\uc0b0"
        )
        sangwonsan = next(
            item for item in _get_mountains()
            if item["name"] == "\uc0c1\uc6d0\uc0b0"
        )

        self.assertEqual(bakdalsan["region"], "\ucda9\ubd81 \uad34\uc0b0\uad70")
        self.assertAlmostEqual(bakdalsan["lat"], 36.836526)
        self.assertAlmostEqual(bakdalsan["lng"], 127.920915)
        self.assertIn("\uad34\uc0b0", bakdalsan["description"])
        self.assertNotIn("\uc591\ud3c9", bakdalsan["description"])
        self.assertNotIn("\uc81c\ucc9c", bakdalsan["description"])
        self.assertNotIn("\ub2e8\uc591", bakdalsan["description"])

        self.assertEqual(sangwonsan["region"], "\uac15\uc6d0 \uc815\uc120\uad70")
        self.assertAlmostEqual(sangwonsan["lat"], 37.5083548)
        self.assertAlmostEqual(sangwonsan["lng"], 128.6767948)
        self.assertEqual(sangwonsan["elevation_m"], 1421)
        self.assertIn("\uc815\uc120", sangwonsan["description"])
        self.assertNotIn("\uc591\ud3c9", sangwonsan["description"])

    def test_mountain_story_rejects_seed_when_region_or_height_conflicts(self):
        response = Client().get(
            "/api/mountain-story/",
            {
                "mountain": "\ubc15\ub2ec\uc0b0",
                "region": "\ucda9\ubd81 \uad34\uc0b0\uad70",
                "elevation_m": 825,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["items"]), 1)
        self.assertIn("\uad34\uc0b0", data["items"][0]["summary"])

    def test_recommendations_returns_top_three_courses(self):
        payload = {
            "profile": {
                "ageGroup": "30s",
                "experience": "beginner",
                "condition": 3,
                "intensity": "moderate",
                "availableMinutes": 150,
                "companion": "solo",
                "purpose": "balanced",
                "transport": "public",
                "maxDistanceKm": 30,
                "departureTime": "17:00",
            },
            "location": {"lat": 37.5665, "lng": 126.978},
        }

        response = Client().post(
            "/api/recommendations/",
            data=json.dumps(payload),
            content_type="application/json",
        )

        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data["recommendations"]), 3)
        self.assertGreaterEqual(data["recommendations"][0]["score"], data["recommendations"][1]["score"])
        self.assertIn("weather", data)

    def test_llm_briefing_runs_only_for_top_three(self):
        payload = {"profile": {"departureTime": "09:00"}, "location": {"lat": 37.5665, "lng": 126.978}}
        with patch("recommendations.llm_briefing.generate_briefing", return_value="LLM 브리핑") as generate:
            result = recommend_courses(payload)

        self.assertLessEqual(generate.call_count, 3)
        self.assertEqual(generate.call_count, len(result["recommendations"]))
        self.assertTrue(all(item["agent_briefing"] == "LLM 브리핑" for item in result["recommendations"]))
        self.assertTrue(all(item["agent_briefing"] != "LLM 브리핑" for item in result["alternatives"]))
        self.assertTrue(all(item.get("route_geometry") for item in result["recommendations"]))

    def test_llm_briefing_cache_key_is_per_course(self):
        safety = {"safety_decision": "recommend", "risk_factors": []}
        weather = {"rainfall_mm": 0, "wind_speed_ms": 1}
        key_a = briefing_cache_key({"id": "a", "name": "A코스", "difficulty": "easy"}, safety, weather, 120)
        key_b = briefing_cache_key({"id": "b", "name": "B코스", "difficulty": "easy"}, safety, weather, 120)

        self.assertNotEqual(key_a, key_b)

    def test_trail_context_prefers_requested_mountain(self):
        passages = retrieve_trail_context("초보 코스 추천", "북한산")

        self.assertTrue(passages)
        self.assertIn("북한산", passages[0])

    def test_courses_endpoint_returns_light_summaries(self):
        response = Client().get("/api/courses/", HTTP_ACCEPT_ENCODING="gzip")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get("Content-Encoding"), "gzip")
        courses = json.loads(gzip.decompress(response.content))["courses"]
        self.assertGreater(len(courses), 0)
        self.assertNotIn("route_geometry", courses[0])
        for field in ("id", "mountain", "name", "lat", "lng", "difficulty", "distance_km", "duration_min"):
            self.assertIn(field, courses[0])

    def test_recommendations_do_not_mutate_public_course_cache(self):
        client = Client()
        before = client.get("/api/courses/").json()["courses"][0]

        client.post(
            "/api/recommendations/",
            data=json.dumps({"profile": {"departureTime": "09:00"}, "location": {"lat": 37.5665, "lng": 126.978}}),
            content_type="application/json",
        )
        after = client.get("/api/courses/").json()["courses"][0]

        self.assertNotIn("disaster_risk_zones", before)
        self.assertNotIn("disaster_risk_zones", after)

    def test_ml_risk_without_params(self):
        response = Client().get("/api/ml-risk/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_index", data)
        self.assertIn("hourly_risks", data)
        self.assertEqual(len(data["hourly_risks"]), 24)

    def test_ml_risk_with_bad_weather_increases_risk(self):
        from unittest.mock import patch
        with patch("recommendations.views._cached_weather") as mock_weather:
            mock_weather.return_value = {
                "temperature_c": 20,
                "rainfall_mm": 0,
                "wind_speed_ms": 1.0,
                "humidity_pct": 50,
                "wildfire_risk": "low",
            }
            res_good = Client().get("/api/ml-risk/", {"lat": "37.5", "lng": "127.0", "date": "2026-06-25", "time": "12:00"})
            self.assertEqual(res_good.status_code, 200)
            good_risk = res_good.json()["risk_index"]

        with patch("recommendations.views._cached_weather") as mock_weather:
            mock_weather.return_value = {
                "temperature_c": 15,
                "rainfall_mm": 15,
                "wind_speed_ms": 12.0,
                "humidity_pct": 95,
                "wildfire_risk": "low",
            }
            res_bad = Client().get("/api/ml-risk/", {"lat": "37.5", "lng": "127.0", "date": "2026-06-25", "time": "12:00"})
            self.assertEqual(res_bad.status_code, 200)
            bad_risk = res_bad.json()["risk_index"]
            warning = res_bad.json().get("warning", "")
            
            self.assertGreater(bad_risk, good_risk)
            self.assertIn("기상 악화 요인", warning)




class RecommendationServiceTests(TestCase):
    def test_select_alternatives_prefers_short_easy_daylight_safe_courses(self):
        courses = [
            {
                "id": "top",
                "difficulty": "medium",
                "duration_min": 150,
                "daylight_margin_min": 20,
                "distance_from_user_km": 10,
            },
            {
                "id": "hard",
                "difficulty": "hard",
                "duration_min": 230,
                "daylight_margin_min": 10,
                "distance_from_user_km": 2,
            },
            {
                "id": "easy",
                "difficulty": "easy",
                "duration_min": 50,
                "daylight_margin_min": 120,
                "distance_from_user_km": 8,
            },
        ]

        alternatives = select_alternatives(courses, [courses[0]])

        self.assertEqual(alternatives[0]["id"], "easy")

    def test_data_quality_adjustment_penalizes_generic_unknown_courses(self):
        adjustment = data_quality_adjustment({"name": "주등산로", "mountain": "국립공원", "lat": None, "lng": None})

        self.assertLess(adjustment, -20)

    def test_purpose_bonus_prefers_healing_short_easy_course(self):
        bonus = purpose_bonus({"difficulty": "easy", "duration_min": 60, "lat": 37.5, "lng": 127.0}, {"purpose": "healing"})

        self.assertGreaterEqual(bonus, 10)

    def test_time_buffer_range_shrinks_upper_bound_for_vulnerable_mode(self):
        self.assertEqual(time_buffer_range(120, {"companion": "solo"}), (90, 150))
        self.assertEqual(time_buffer_range(120, {"companion": "vulnerable"}), (90, 135))

    def test_daylight_margin_uses_selected_departure_date(self):
        margin = daylight_margin_minutes(
            {"duration_min": 90},
            {"sunset": "18:30"},
            now=datetime(2026, 5, 15, 8, 0, tzinfo=ZoneInfo("Asia/Seoul")),
            departure_date="2026-05-17",
            departure_time="16:00",
        )

        self.assertEqual(margin, 60)

    def test_clean_course_name_removes_number_prefix(self):
        self.assertEqual(clean_course_name("27.U2기지초입부~서이말등대(거제)"), "U2기지초입부~서이말등대(거제)")
        self.assertEqual(clean_course_name("17. 영원사~삼불사~약수암"), "영원사~삼불사~약수암")

    def test_infer_mountain_prefers_stronger_alias_match(self):
        self.assertEqual(
            infer_mountain_name(
                "\ub178\uace0\ub2e8\uace0\uac1c~\uc0bc\ub3c4\ubd09",
                "\ub178\uace0\ub2e8\uace0\uac1c",
                "\uc784\uac78\ub839",
                "\uc0bc\ub3c4\ubd09",
            ),
            "\uc9c0\ub9ac\uc0b0",
        )

