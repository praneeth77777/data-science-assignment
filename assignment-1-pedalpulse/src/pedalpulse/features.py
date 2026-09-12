"""Deterministic calendar and cyclical feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import PREDICTOR_COLUMNS, parse_timestamps, validate_predictor_schema


class CalendarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create operational calendar features without using any outcome columns."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "CalendarFeatureEngineer":
        validate_predictor_schema(X)
        self.feature_names_in_ = np.asarray(PREDICTOR_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_predictor_schema(X)
        frame = X.loc[:, PREDICTOR_COLUMNS].copy().reset_index(drop=True)
        timestamp = parse_timestamps(frame["datetime"])

        frame["hour"] = timestamp.dt.hour.astype("int8")
        frame["weekday"] = timestamp.dt.weekday.astype("int8")
        frame["month"] = timestamp.dt.month.astype("int8")
        frame["year"] = timestamp.dt.year.astype("int16")
        frame["weekend"] = (frame["weekday"] >= 5).astype("int8")
        morning = frame["hour"].between(7, 9)
        evening = frame["hour"].between(16, 19)
        frame["rush_hour"] = ((frame["workingday"] == 1) & (morning | evening)).astype("int8")

        frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
        frame["weekday_sin"] = np.sin(2 * np.pi * frame["weekday"] / 7)
        frame["weekday_cos"] = np.cos(2 * np.pi * frame["weekday"] / 7)
        frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
        frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)

        for column in ("atemp", "humidity", "windspeed"):
            frame[f"{column}_zero_flag"] = (frame[column] == 0).astype("int8")

        return frame.drop(columns="datetime")
