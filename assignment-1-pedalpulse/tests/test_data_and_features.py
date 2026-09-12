from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pedalpulse.data import chronological_split, safe_predictor_frame, validate_predictor_schema
from pedalpulse.features import CalendarFeatureEngineer
from pedalpulse.preprocessing import SuspiciousZeroHandler, build_preparation_pipeline


def predictor_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2012-01-07 00:00:00", "2012-01-09 08:00:00", "2012-01-09 17:00:00"],
            "season": [1, 1, 1],
            "holiday": [0, 0, 0],
            "workingday": [0, 1, 1],
            "weather": [1, 2, 1],
            "temp": [10.0, 11.0, 12.0],
            "atemp": [9.0, 10.0, 11.0],
            "humidity": [70, 60, 50],
            "windspeed": [0.0, 10.0, 20.0],
        }
    )


class DataAndFeatureTests(unittest.TestCase):
    def test_schema_rejects_leakage_columns(self) -> None:
        for forbidden in ("casual", "registered", "count"):
            with self.subTest(forbidden=forbidden):
                frame = predictor_rows().assign(**{forbidden: 1})
                with self.assertRaisesRegex(ValueError, "Forbidden outcome/leakage"):
                    validate_predictor_schema(frame)

    def test_safe_predictor_frame_excludes_target_components(self) -> None:
        raw = predictor_rows().assign(casual=[1, 2, 3], registered=[4, 5, 6], count=[5, 7, 9])
        result = safe_predictor_frame(raw)
        self.assertNotIn("count", result.columns)
        self.assertNotIn("casual", result.columns)
        self.assertNotIn("registered", result.columns)

    def test_missing_required_column_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_predictor_schema(predictor_rows().drop(columns="humidity"))

    def test_unsorted_time_fails(self) -> None:
        unsorted = predictor_rows().iloc[[1, 0, 2]].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "ordered strictly forward"):
            validate_predictor_schema(unsorted)

    def test_duplicate_time_fails(self) -> None:
        duplicated = pd.concat([predictor_rows(), predictor_rows().iloc[[2]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_predictor_schema(duplicated)

    def test_calendar_features_are_exact(self) -> None:
        transformed = CalendarFeatureEngineer().fit_transform(predictor_rows())
        self.assertEqual(transformed.loc[0, "weekday"], 5)
        self.assertEqual(transformed.loc[0, "weekend"], 1)
        self.assertEqual(transformed.loc[0, "rush_hour"], 0)
        self.assertEqual(transformed.loc[1, "hour"], 8)
        self.assertEqual(transformed.loc[1, "rush_hour"], 1)
        self.assertEqual(transformed.loc[2, "rush_hour"], 1)
        self.assertAlmostEqual(transformed.loc[0, "hour_sin"], 0.0, places=12)
        self.assertAlmostEqual(transformed.loc[0, "hour_cos"], 1.0, places=12)
        self.assertEqual(transformed.loc[0, "windspeed_zero_flag"], 1)

    def test_suspicious_replacements_use_fit_data_only(self) -> None:
        train = CalendarFeatureEngineer().fit_transform(predictor_rows())
        handler = SuspiciousZeroHandler("replace_with_train_median_and_flags").fit(train)
        self.assertEqual(handler.replacement_values_["windspeed"], 15.0)
        validation = train.iloc[[0]].copy()
        validation["windspeed"] = 0.0
        replaced = handler.transform(validation)
        self.assertEqual(replaced.loc[0, "windspeed"], 15.0)
        self.assertEqual(replaced.loc[0, "windspeed_zero_flag"], 1)

    def test_pipeline_is_deterministic(self) -> None:
        frame = predictor_rows()
        first = build_preparation_pipeline().fit_transform(frame)
        second = build_preparation_pipeline().fit_transform(frame)
        np.testing.assert_allclose(first, second)

    def test_chronological_split_is_nonoverlapping(self) -> None:
        dates = [
            "2012-06-19 12:00:00",
            "2012-07-01 00:00:00",
            "2012-09-19 23:00:00",
            "2012-10-01 00:00:00",
        ]
        base = pd.concat([predictor_rows().iloc[[0]]] * len(dates), ignore_index=True)
        base["datetime"] = dates
        base["casual"] = 1
        base["registered"] = 2
        base["count"] = 3
        splits = chronological_split(base)
        self.assertEqual([len(splits[x]) for x in ("train", "validation", "test")], [1, 2, 1])
        train_max = pd.to_datetime(splits["train"]["datetime"]).max()
        validation_min = pd.to_datetime(splits["validation"]["datetime"]).min()
        test_min = pd.to_datetime(splits["test"]["datetime"]).min()
        self.assertLess(train_max, validation_min)
        self.assertLess(pd.to_datetime(splits["validation"]["datetime"]).max(), test_min)


if __name__ == "__main__":
    unittest.main()
