"""Leakage-safe scikit-learn preparation pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CalendarFeatureEngineer

SUSPICIOUS_ZERO_COLUMNS = ("atemp", "humidity", "windspeed")
CATEGORICAL_COLUMNS = (
    "season",
    "holiday",
    "workingday",
    "weather",
    "hour",
    "weekday",
    "month",
    "year",
    "weekend",
    "rush_hour",
)
NUMERIC_COLUMNS = (
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "atemp_zero_flag",
    "humidity_zero_flag",
    "windspeed_zero_flag",
)


class SuspiciousZeroHandler(BaseEstimator, TransformerMixin):
    """Either retain flagged zeros or replace them using training-only medians."""

    def __init__(self, strategy: str = "preserve_with_flags") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: object = None) -> "SuspiciousZeroHandler":
        if self.strategy not in {"preserve_with_flags", "replace_with_train_median_and_flags"}:
            raise ValueError(f"Unknown suspicious-zero strategy: {self.strategy}")
        self.replacement_values_: dict[str, float] = {}
        if self.strategy == "replace_with_train_median_and_flags":
            for column in SUSPICIOUS_ZERO_COLUMNS:
                nonzero = pd.to_numeric(X[column], errors="coerce").replace(0, np.nan)
                value = float(nonzero.median())
                if not np.isfinite(value):
                    raise ValueError(f"Cannot learn a finite nonzero median for {column}")
                self.replacement_values_[column] = value
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        if self.strategy == "replace_with_train_median_and_flags":
            for column, value in self.replacement_values_.items():
                frame.loc[frame[column] == 0, column] = value
        return frame


def build_preparation_pipeline(strategy: str = "preserve_with_flags") -> Pipeline:
    """Build a deterministic pipeline; callers must fit it on training only."""

    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("calendar_features", CalendarFeatureEngineer()),
            ("suspicious_zeros", SuspiciousZeroHandler(strategy=strategy)),
            ("column_transformer", columns),
        ]
    )
