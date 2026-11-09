from django.test import TestCase

from .accident_model import (
    get_accident_model_training_summary,
    predict_accident_risk,
    predict_hourly_risk_indexes,
)


class AccidentModelTests(TestCase):
    def test_hourly_batch_matches_single_hour_predictions(self):
        conditions = {"month": 7, "weekday": 2, "temp": 30.0, "rain": 12.5, "wind": 6.0, "humidity": 90.0}

        batched = predict_hourly_risk_indexes(**conditions)
        one_by_one = [predict_accident_risk(hour=hour, **conditions)["risk_index"] for hour in range(24)]

        self.assertEqual(batched, one_by_one)

    def test_training_uses_added_accident_datasets_except_2023_2024(self):
        summary = get_accident_model_training_summary()

        self.assertTrue(summary["trained"])
        self.assertNotIn("2023-2024.csv", summary["sources"])
        self.assertIn("전국 산악사고 구조활동현황(2010_2016).csv", summary["sources"])
        self.assertIn("전국 산악사고 구조활동현황(2017~2021).csv", summary["sources"])
        self.assertIn("전국 산악사고 현황(2022~2023년).csv", summary["sources"])
        self.assertGreater(summary["rows"], 10000)

    def test_prediction_returns_training_metadata(self):
        result = predict_accident_risk(month=10, hour=12, weekday=5)

        self.assertIn("risk_index", result)
        self.assertIn("training", result)
        self.assertTrue(result["training"]["trained"])
