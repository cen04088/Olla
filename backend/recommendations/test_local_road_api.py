from django.test import Client, TestCase

from .local_road_api import (
    decode_polyline,
    encode_polyline,
    fetch_local_road_trails,
    has_road_route,
    korea_2000_to_wgs84,
    load_local_road_trails,
    with_route_geometry,
)


class LocalRoadApiTests(TestCase):
    def test_polyline_round_trip_keeps_meter_precision(self):
        points = [
            {"lat": 37.5665123, "lng": 126.9780456},
            {"lat": 37.5601, "lng": 126.99},
            {"lat": 33.1234567, "lng": 130.9999999},
        ]

        decoded = decode_polyline(encode_polyline(points))

        self.assertEqual(len(decoded), len(points))
        for original, restored in zip(points, decoded):
            self.assertAlmostEqual(original["lat"], restored["lat"], delta=1e-5)
            self.assertAlmostEqual(original["lng"], restored["lng"], delta=1e-5)

    def test_cached_trails_are_stored_without_geometry_but_can_restore_it(self):
        trail = load_local_road_trails()[0]

        self.assertNotIn("route_geometry", trail)
        self.assertTrue(has_road_route(trail))
        restored = with_route_geometry(trail)
        self.assertGreaterEqual(len(restored["route_geometry"]), 2)
        self.assertNotIn("route_geometry", load_local_road_trails()[0])

    def test_korea_2000_to_wgs84_projects_expected_korea_bounds(self):
        point = korea_2000_to_wgs84(953901.0, 1952032.0)

        self.assertGreater(point["lat"], 37)
        self.assertLess(point["lat"], 38)
        self.assertGreater(point["lng"], 126)
        self.assertLess(point["lng"], 127)

    def test_load_local_road_trails_reads_shp_geometry(self):
        trails = load_local_road_trails()

        self.assertGreater(len(trails), 1000)
        self.assertIn("lat", trails[0])
        self.assertIn("lng", trails[0])

    def test_fetch_local_road_trails_filters_by_mountain(self):
        result = fetch_local_road_trails(mountain_name="건봉산", size=5)

        self.assertTrue(result["ok"])
        self.assertLessEqual(len(result["items"]), 5)
        self.assertTrue(any("건봉산" in item["mountain"] or "건봉산" in item["name"] for item in result["items"]))

    def test_local_road_trails_endpoint_returns_geometry(self):
        response = Client().get("/api/local-road-trails/?lat=37.7&lng=128.45&radius=20&size=3")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertLessEqual(len(data["items"]), 3)
        if data["items"]:
            self.assertIn("route_geometry", data["items"][0])
