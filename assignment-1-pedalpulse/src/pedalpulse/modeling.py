"""Locked PedalPulse model configuration and deployment artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from .data import LEAKAGE_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN, chronological_split, validate_predictor_schema
from .preprocessing import build_preparation_pipeline

SEED = 42
MODEL_CONTRACT_VERSION = 1
SELECTED_CANDIDATE = "rf_raw_depth18_leaf1"


def make_final_estimator() -> RandomForestRegressor:
    """Return the exact candidate locked during temporal model selection."""

    return RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )


def _predictor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate inference input and select only the published predictor contract."""

    validate_predictor_schema(frame)
    return frame.loc[:, PREDICTOR_COLUMNS].copy()


def train_deployment_bundle(raw: pd.DataFrame, *, input_sha256: str) -> dict[str, Any]:
    """Fit the locked model on development data only; never score the test period."""

    splits = chronological_split(raw)
    development = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    predictors = development.loc[:, PREDICTOR_COLUMNS].copy()
    validate_predictor_schema(predictors)
    target = development[TARGET_COLUMN].to_numpy(dtype=float)

    preprocessor = build_preparation_pipeline("preserve_with_flags")
    matrix = preprocessor.fit_transform(predictors, target)
    feature_names = [
        str(name)
        for name in preprocessor.named_steps["column_transformer"].get_feature_names_out()
    ]
    forbidden = [
        name for name in feature_names
        if any(token in name for token in (*LEAKAGE_COLUMNS, TARGET_COLUMN))
    ]
    if forbidden:
        raise RuntimeError(f"Leakage audit failed before model fit: {forbidden}")

    model = make_final_estimator()
    model.fit(matrix, target)
    metadata = {
        "contract_version": MODEL_CONTRACT_VERSION,
        "candidate": SELECTED_CANDIDATE,
        "random_seed": SEED,
        "training_rows": int(len(development)),
        "training_start": str(pd.to_datetime(development["datetime"]).min()),
        "training_end": str(pd.to_datetime(development["datetime"]).max()),
        "locked_test_rows_used_for_training": 0,
        "locked_test_metrics_computed": False,
        "input_sha256": input_sha256,
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "output_features": int(matrix.shape[1]),
        "feature_names": feature_names,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    return {"preprocessor": preprocessor, "model": model, "metadata": metadata}


def save_bundle(bundle: dict[str, Any], model_path: Path, metadata_path: Path) -> None:
    """Persist a model bundle plus human-readable metadata."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    metadata_path.write_text(json.dumps(bundle["metadata"], indent=2), encoding="utf-8")


def load_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate the serialized deployment contract."""

    bundle = joblib.load(model_path)
    required = {"preprocessor", "model", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Model artifact does not satisfy the PedalPulse bundle contract")
    if bundle["metadata"].get("contract_version") != MODEL_CONTRACT_VERSION:
        raise ValueError("Model artifact contract version is unsupported")
    return bundle


def predict_demand(bundle: dict[str, Any], predictors: pd.DataFrame) -> np.ndarray:
    """Generate finite, nonnegative demand predictions."""

    safe = _predictor_frame(predictors)
    matrix = bundle["preprocessor"].transform(safe)
    predictions = np.clip(bundle["model"].predict(matrix), 0, None)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return predictions


def empirical_tree_interval(
    bundle: dict[str, Any], predictors: pd.DataFrame, lower: float = 0.10, upper: float = 0.90
) -> tuple[np.ndarray, np.ndarray]:
    """Return an uncalibrated interval from individual Random Forest trees."""

    if not 0 <= lower < upper <= 1:
        raise ValueError("Interval quantiles must satisfy 0 <= lower < upper <= 1")
    safe = _predictor_frame(predictors)
    matrix = bundle["preprocessor"].transform(safe)
    predictions = np.vstack([tree.predict(matrix) for tree in bundle["model"].estimators_])
    low, high = np.quantile(predictions, [lower, upper], axis=0)
    return np.clip(low, 0, None), np.clip(high, 0, None)
