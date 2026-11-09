from django.test import TestCase

from .disaster_risk import disaster_risk_level, find_course_disaster_risks
from .loaders import (
    DISASTER_RISK_CSV_PATH,
    TRAIL_CSV_PATH,
    load_disaster_risk_zones,
    normalize_disaster_risk_row,
)
from .services import safety_decision_for_course


class DisasterRiskTests(TestCase):
    def test_bundled_csv_files_are_found_and_loaded(self):
        # 테스트 DB는 비어 있으므로 CSV fallback 경로가 실제 파일을 가리켜야 위험지구가 로드된다
        self.assertTrue(TRAIL_CSV_PATH.exists(), TRAIL_CSV_PATH)
        self.assertTrue(DISASTER_RISK_CSV_PATH.exists(), DISASTER_RISK_CSV_PATH)
        load_disaster_risk_zones.cache_clear()
        try:
            self.assertGreater(len(load_disaster_risk_zones()), 0)
        finally:
            load_disaster_risk_zones.cache_clear()

    def test_normalize_disaster_risk_row(self):
        row = {
            "지구명": "북한산 탐방로",
            "위치": "백운대 하단",
            "시설명": "백운대 하단",
            "표지판설치": "1",
            "위험요인": "추락위험",
            "일최대예상탐방객": "120",
            "대피계획인원": "30",
            "대피장소": "탐방지원센터",
        }

        zone = normalize_disaster_risk_row(1, row)

        self.assertEqual(zone["risk_factor"], "추락위험")
        self.assertEqual(zone["expected_daily_visitors"], 120)

    def test_find_course_disaster_risks_matches_course_text(self):
        zones = [
            {
                "district": "북한산 탐방로",
                "location": "백운대 하단",
                "facility": "백운대 하단",
                "risk_factor": "추락위험",
                "search_text": "북한산탐방로백운대하단추락위험",
            }
        ]

        matches = find_course_disaster_risks({"mountain": "북한산", "name": "백운대 코스"}, zones)

        self.assertEqual(len(matches), 1)
        self.assertEqual(disaster_risk_level(matches), "high")

    def test_generic_trail_word_does_not_match_unrelated_course(self):
        zones = [
            {
                "district": "가라산 탐방로",
                "location": "가라산 전망대",
                "facility": "가라산 전망대",
                "risk_factor": "추락위험",
                "search_text": "가라산탐방로가라산전망대추락위험",
            }
        ]

        matches = find_course_disaster_risks({"mountain": "국립공원", "name": "탐방로"}, zones)

        self.assertEqual(matches, [])

    SAMO_ZONE = {"district": "구기지구", "location": "사모바위 하단", "facility": "사모바위", "risk_factor": "추락위험"}

    def test_same_mountain_only_is_area_match_with_caution(self):
        matches = find_course_disaster_risks({"mountain": "북한산_백운대", "name": "수유동구간"}, [self.SAMO_ZONE])

        self.assertEqual([zone["match_scope"] for zone in matches], ["area"])
        self.assertEqual(disaster_risk_level(matches), "caution")

    def test_spot_in_course_name_keeps_high_risk(self):
        matches = find_course_disaster_risks({"mountain": "북한산_백운대", "name": "사모바위구간"}, [self.SAMO_ZONE])

        self.assertEqual([zone["match_scope"] for zone in matches], ["spot"])
        self.assertEqual(disaster_risk_level(matches), "high")

    def test_mountain_name_repeated_in_course_name_is_not_a_spot(self):
        zone = {"district": "우이지구", "location": "백운대 하단", "facility": "백운대", "risk_factor": "추락"}

        matches = find_course_disaster_risks({"mountain": "북한산_백운대", "name": "북한산_백운대 등산로"}, [zone])

        self.assertEqual([zone["match_scope"] for zone in matches], ["area"])

    def test_same_spot_name_on_other_mountain_does_not_match(self):
        zone = {"district": "금오도지구(8)", "location": "여수금오도 함구미옥녀봉", "facility": "옥녀봉", "risk_factor": "추락"}

        self.assertEqual(find_course_disaster_risks({"mountain": "옥녀봉", "name": "구복리구간"}, [zone]), [])
        self.assertEqual(len(find_course_disaster_risks({"mountain": "금오도", "name": "옥녀봉구간"}, [zone])), 1)

    def test_area_match_makes_course_caution_not_rejected(self):
        zones = find_course_disaster_risks({"mountain": "북한산_백운대", "name": "수유동구간"}, [self.SAMO_ZONE])
        course = {"mountain": "북한산_백운대", "name": "수유동구간", "elevation_gain_m": 100, "disaster_risk_zones": zones}
        weather = {"rainfall_mm": 0, "wind_speed_ms": 1, "wildfire_risk": "low"}

        safety = safety_decision_for_course(course, {}, weather, 90, 1.0, 1.0, 200)

        self.assertEqual(safety["safety_decision"], "caution")
        self.assertTrue(any("사모바위" in factor for factor in safety["risk_factors"]))
