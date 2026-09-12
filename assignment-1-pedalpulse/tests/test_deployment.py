import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pedalpulse.data import validate_predictor_schema
from pedalpulse.modeling import load_bundle, make_final_estimator, predict_demand

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "pedalpulse_rf.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "pedalpulse_rf_metadata.json"


class DeploymentTests(unittest.TestCase):
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "datetime": "2013-01-07 08:00:00",
            "season": 1,
            "holiday": 0,
            "workingday": 1,
            "weather": 1,
            "temp": 12.3,
            "atemp": 13.6,
            "humidity": 55,
            "windspeed": 10.0,
        }])

    def test_locked_estimator_configuration(self):
        model = make_final_estimator()
        self.assertEqual(model.n_estimators, 180)
        self.assertEqual(model.max_depth, 18)
        self.assertEqual(model.random_state, 42)

    def test_model_artifact_contract_and_prediction(self):
        self.assertTrue(MODEL_PATH.is_file())
        bundle = load_bundle(MODEL_PATH)
        prediction = predict_demand(bundle, self.sample())
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction[0]), 0.0)

    def test_metadata_confirms_development_only_training(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["training_rows"], 9519)
        self.assertEqual(metadata["locked_test_rows_used_for_training"], 0)
        self.assertFalse(metadata["locked_test_metrics_computed"])

    def test_inference_rejects_outcome_columns(self):
        unsafe = self.sample().assign(count=100)
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            validate_predictor_schema(unsafe)


if __name__ == "__main__":
    unittest.main()
