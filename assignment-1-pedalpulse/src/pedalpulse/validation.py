"""Temporal folds, baselines, and regression metrics for PedalPulse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TARGET_COLUMN, parse_timestamps


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


FOLD_BOUNDARIES = (
    ("fold_1", "2011-06-19 23:59:59", "2011-07-01 00:00:00", "2011-09-19 23:59:59"),
    ("fold_2", "2011-09-19 23:59:59", "2011-10-01 00:00:00", "2011-12-19 23:59:59"),
    ("fold_3", "2011-12-19 23:59:59", "2012-01-01 00:00:00", "2012-03-19 23:59:59"),
    ("fold_4", "2012-03-19 23:59:59", "2012-04-01 00:00:00", "2012-06-19 23:59:59"),
    ("fold_5", "2012-06-19 23:59:59", "2012-07-01 00:00:00", "2012-09-19 23:59:59"),
)


def expanding_quarter_folds(raw: pd.DataFrame) -> list[TemporalFold]:
    """Build five explicit expanding folds ending before the locked test."""

    timestamps = parse_timestamps(raw["datetime"].reset_index(drop=True))
    if not timestamps.is_monotonic_increasing:
        raise ValueError("raw rows must be ordered forward in time")
    folds: list[TemporalFold] = []
    for name, train_end_text, validation_start_text, validation_end_text in FOLD_BOUNDARIES:
        train_end = pd.Timestamp(train_end_text)
        validation_start = pd.Timestamp(validation_start_text)
        validation_end = pd.Timestamp(validation_end_text)
        train_indices = np.flatnonzero((timestamps <= train_end).to_numpy())
        validation_indices = np.flatnonzero(((timestamps >= validation_start) & (timestamps <= validation_end)).to_numpy())
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"{name} has an empty training or validation partition")
        if timestamps.iloc[train_indices].max() >= timestamps.iloc[validation_indices].min():
            raise ValueError(f"{name} overlaps in time")
        if timestamps.iloc[validation_indices].max() >= pd.Timestamp("2012-10-01"):
            raise ValueError(f"{name} reaches the locked test period")
        folds.append(TemporalFold(name, train_indices, validation_indices, train_end, validation_start, validation_end))
    return folds


def regression_metrics(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate the declared metrics after enforcing nonnegative counts."""

    truth = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(predictions, dtype=float), 0, None)
    if not np.isfinite(predicted).all():
        raise ValueError("predictions contain non-finite values")
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "rmsle": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(truth)) ** 2))),
        "r2": float(r2_score(truth, predicted)),
    }


def constant_predictions(train: pd.DataFrame, validation: pd.DataFrame, statistic: str) -> np.ndarray:
    if statistic == "mean":
        value = float(train[TARGET_COLUMN].mean())
    elif statistic == "median":
        value = float(train[TARGET_COLUMN].median())
    else:
        raise ValueError("statistic must be mean or median")
    return np.full(len(validation), value, dtype=float)


def rolling_seasonal_naive(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Sequential one-hour-ahead seasonal baseline with audited fallbacks."""

    train_time = parse_timestamps(train["datetime"].reset_index(drop=True))
    validation_time = parse_timestamps(validation["datetime"].reset_index(drop=True))
    if train_time.max() >= validation_time.min():
        raise ValueError("seasonal baseline requires training strictly before validation")
    if not validation_time.is_monotonic_increasing:
        raise ValueError("validation must be ordered forward in time")

    history = {timestamp: float(value) for timestamp, value in zip(train_time, train[TARGET_COLUMN], strict=True)}
    train_context = train.assign(_hour=train_time.dt.hour.to_numpy())
    grouped_median = train_context.groupby(["_hour", "workingday"], observed=True)[TARGET_COLUMN].median().to_dict()
    global_median = float(train[TARGET_COLUMN].median())

    predictions: list[float] = []
    tiers: list[str] = []
    for position, timestamp in enumerate(validation_time):
        weekly = timestamp - pd.Timedelta(hours=168)
        daily = timestamp - pd.Timedelta(hours=24)
        if weekly in history:
            prediction, tier = history[weekly], "t_minus_168"
        elif daily in history:
            prediction, tier = history[daily], "t_minus_24"
        else:
            key = (timestamp.hour, int(validation.iloc[position]["workingday"]))
            if key in grouped_median:
                prediction, tier = float(grouped_median[key]), "train_hour_workingday_median"
            else:
                prediction, tier = global_median, "train_global_median"
        predictions.append(prediction)
        tiers.append(tier)
        # At the next forecast origin this realized outcome is historical.
        history[timestamp] = float(validation.iloc[position][TARGET_COLUMN])
    return np.asarray(predictions), tiers
