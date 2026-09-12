"""Schema, leakage, and chronological-boundary checks for PedalPulse."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TARGET_COLUMN = "count"
LEAKAGE_COLUMNS = ("casual", "registered")
PREDICTOR_COLUMNS = (
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the published timestamp format and reject silent coercion."""

    parsed = pd.to_datetime(values, format=TIMESTAMP_FORMAT, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"datetime contains {bad} unparseable value(s)")
    round_trip = parsed.dt.strftime(TIMESTAMP_FORMAT)
    if not round_trip.equals(values.astype(str).reset_index(drop=True)):
        raise ValueError("datetime values do not exactly match YYYY-MM-DD HH:MM:SS")
    return parsed


def assert_no_leakage_columns(frame: pd.DataFrame) -> None:
    forbidden = sorted(set(frame.columns).intersection((*LEAKAGE_COLUMNS, TARGET_COLUMN)))
    if forbidden:
        raise ValueError(f"Forbidden outcome/leakage columns entered the feature pipeline: {forbidden}")


def validate_predictor_schema(frame: pd.DataFrame, *, require_time_order: bool = True) -> pd.Series:
    """Validate predictor inputs and return parsed timestamps."""

    assert_no_leakage_columns(frame)
    missing = sorted(set(PREDICTOR_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictor columns: {missing}")
    parsed = parse_timestamps(frame["datetime"].reset_index(drop=True))
    if parsed.duplicated().any():
        raise ValueError("datetime must be unique")
    if require_time_order and not parsed.is_monotonic_increasing:
        raise ValueError("predictor rows must be ordered strictly forward in time")
    if not set(frame["season"].dropna().unique()).issubset({1, 2, 3, 4}):
        raise ValueError("season contains an out-of-domain value")
    if not set(frame["holiday"].dropna().unique()).issubset({0, 1}):
        raise ValueError("holiday contains an out-of-domain value")
    if not set(frame["workingday"].dropna().unique()).issubset({0, 1}):
        raise ValueError("workingday contains an out-of-domain value")
    if not set(frame["weather"].dropna().unique()).issubset({1, 2, 3, 4}):
        raise ValueError("weather contains an out-of-domain value")
    return parsed


def safe_predictor_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create X while making the target-component exclusion explicit."""

    required = set(PREDICTOR_COLUMNS) | {TARGET_COLUMN, *LEAKAGE_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw labeled data are missing required columns: {missing}")
    predictors = raw.loc[:, PREDICTOR_COLUMNS].copy()
    validate_predictor_schema(predictors)
    return predictors


def chronological_split(
    raw: pd.DataFrame,
    *,
    train_end: str = "2012-06-19 23:59:59",
    validation_start: str = "2012-07-01 00:00:00",
    validation_end: str = "2012-09-19 23:59:59",
    test_start: str = "2012-10-01 00:00:00",
) -> Mapping[str, pd.DataFrame]:
    """Return locked chronological train/validation/test labeled partitions."""

    predictors = safe_predictor_frame(raw)
    parsed = parse_timestamps(predictors["datetime"].reset_index(drop=True))
    boundaries = [pd.Timestamp(train_end), pd.Timestamp(validation_start), pd.Timestamp(validation_end), pd.Timestamp(test_start)]
    if not (boundaries[0] < boundaries[1] <= boundaries[2] < boundaries[3]):
        raise ValueError("chronological split boundaries are not ordered")

    masks = {
        "train": parsed <= boundaries[0],
        "validation": (parsed >= boundaries[1]) & (parsed <= boundaries[2]),
        "test": parsed >= boundaries[3],
    }
    splits: dict[str, pd.DataFrame] = {}
    for name, mask in masks.items():
        part = raw.loc[mask.to_numpy()].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{name} split is empty")
        splits[name] = part

    maxima = {name: pd.to_datetime(part["datetime"]).max() for name, part in splits.items()}
    minima = {name: pd.to_datetime(part["datetime"]).min() for name, part in splits.items()}
    if not (maxima["train"] < minima["validation"] <= maxima["validation"] < minima["test"]):
        raise ValueError("chronological partitions overlap or are out of order")
    return splits
