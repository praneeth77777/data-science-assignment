from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pedalpulse.validation import expanding_quarter_folds, regression_metrics, rolling_seasonal_naive


class TemporalValidationTests(unittest.TestCase):
    def test_folds_end_before_locked_test(self) -> None:
        timestamps = [
            "2011-01-01 00:00:00",
            "2011-07-01 00:00:00",
            "2011-10-01 00:00:00",
            "2012-01-01 00:00:00",
            "2012-04-01 00:00:00",
            "2012-07-01 00:00:00",
            "2012-10-01 00:00:00",
        ]
        raw = pd.DataFrame({"datetime": timestamps})
        folds = expanding_quarter_folds(raw)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            validation_times = pd.to_datetime(raw.iloc[fold.validation_indices]["datetime"])
            self.assertLess(validation_times.max(), pd.Timestamp("2012-10-01"))

    def test_seasonal_baseline_uses_realized_past_validation_only(self) -> None:
        train = pd.DataFrame(
            {
                "datetime": ["2012-01-01 00:00:00", "2012-01-07 01:00:00"],
                "workingday": [0, 0],
                "count": [10, 20],
            }
        )
        validation = pd.DataFrame(
            {
                "datetime": ["2012-01-08 00:00:00", "2012-01-15 00:00:00"],
                "workingday": [0, 0],
                "count": [99, 123],
            }
        )
        predictions, tiers = rolling_seasonal_naive(train, validation)
        self.assertEqual(predictions.tolist(), [10.0, 99.0])
        self.assertEqual(tiers, ["t_minus_168", "t_minus_168"])

    def test_metrics_clip_negative_predictions(self) -> None:
        metrics = regression_metrics(np.array([1.0, 2.0]), np.array([-5.0, 2.0]))
        self.assertTrue(all(np.isfinite(list(metrics.values()))))
        self.assertGreaterEqual(metrics["rmsle"], 0)


if __name__ == "__main__":
    unittest.main()
