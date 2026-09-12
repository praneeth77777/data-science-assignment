#!/usr/bin/env python3
"""Reproducible raw-data audit for PedalPulse Chunk 2.

This script reads the supplied CSV files without altering them. It creates
derived audit tables, figures, a data dictionary, and a Markdown report. It
does not clean, impute, transform for modeling, split targets, or train models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp


EXPECTED_TRAIN_COLUMNS = [
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "casual",
    "registered",
    "count",
]
EXPECTED_TEST_COLUMNS = EXPECTED_TRAIN_COLUMNS[:-3]
TARGET_COMPONENTS = ["casual", "registered"]
TARGET = "count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat(sep=" ")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def display_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if value != 0 and abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    table = frame.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    columns = [str(column) for column in table.columns]
    rows = []
    for row in table.itertuples(index=False, name=None):
        rows.append(
            [
                display_value(value).replace("|", "\\|").replace("\n", " ")
                for value in row
            ]
        )
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(part for part in [header, divider, body] if part)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def file_inventory(paths: dict[str, Path], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for label, path in paths.items():
        raw = path.read_bytes()
        try:
            raw.decode("ascii")
            encoding = "ASCII-compatible"
        except UnicodeDecodeError:
            encoding = "non-ASCII UTF-8/unknown"
        records.append(
            {
                "dataset": label,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": len(frames[label]),
                "columns": frames[label].shape[1],
                "encoding_check": encoding,
                "line_endings": "CRLF" if b"\r\n" in raw else "LF",
            }
        )
    return pd.DataFrame(records)


def build_schema(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            records.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "pandas_dtype": str(frame[column].dtype),
                    "non_null_count": int(frame[column].notna().sum()),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": float(frame[column].isna().mean() * 100),
                    "unique_count": int(frame[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(records)


def build_samples(train: pd.DataFrame) -> pd.DataFrame:
    head = train.head(5).copy()
    head.insert(0, "sample_position", "head")
    head.insert(1, "csv_line_number", head.index + 2)
    tail = train.tail(5).copy()
    tail.insert(0, "sample_position", "tail")
    tail.insert(1, "csv_line_number", tail.index + 2)
    return pd.concat([head, tail], ignore_index=True)


def descriptive_statistics(train: pd.DataFrame) -> pd.DataFrame:
    numeric = train.select_dtypes(include=[np.number])
    stats = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats = stats.reset_index(names="column")
    stats["missing_count"] = [int(train[column].isna().sum()) for column in stats["column"]]
    stats["zero_count"] = [int((train[column] == 0).sum()) for column in stats["column"]]
    stats["negative_count"] = [int((train[column] < 0).sum()) for column in stats["column"]]
    return stats


def build_cardinalities(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in train.columns:
        series = train[column]
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_count": int(series.nunique(dropna=True)),
                "unique_pct_of_rows": float(series.nunique(dropna=True) / len(train) * 100),
                "most_frequent_value": series.value_counts(dropna=False).index[0],
                "most_frequent_count": int(series.value_counts(dropna=False).iloc[0]),
            }
        )
    return pd.DataFrame(records)


def categorical_distributions(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in train.columns:
        if column == "datetime" or train[column].nunique(dropna=True) > 50:
            continue
        counts = train[column].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            records.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "percentage": float(count / len(train) * 100),
                }
            )
    return pd.DataFrame(records)


def build_zero_rates(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset, frame in [("train", train), ("competition_test", test)]:
        for column in frame.select_dtypes(include=[np.number]).columns:
            count = int((frame[column] == 0).sum())
            records.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "zero_count": count,
                    "zero_pct": float(count / len(frame) * 100),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": float(frame[column].isna().mean() * 100),
                }
            )
    return pd.DataFrame(records)


def quality_checks(train: pd.DataFrame, parsed_datetime: pd.Series) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(check_id: str, field: str, rule: str, mask: pd.Series, severity: str, note: str) -> None:
        violations = int(mask.fillna(True).sum())
        records.append(
            {
                "check_id": check_id,
                "field": field,
                "rule": rule,
                "severity": severity,
                "violations": violations,
                "status": "PASS" if violations == 0 else "REVIEW",
                "note": note,
            }
        )

    add("Q001", "datetime", "parseable timestamp", parsed_datetime.isna(), "error", "Raw values are retained; parsing is diagnostic only.")
    add("Q002", "datetime", "unique timestamp", train["datetime"].duplicated(keep=False), "error", "Duplicate hours would make lag lookup ambiguous.")
    add("Q003", "all columns", "no fully duplicated row", train.duplicated(keep=False), "error", "Exact duplicate records are not silently removed.")
    add("Q004", "season", "value in {1,2,3,4}", ~train["season"].isin([1, 2, 3, 4]), "error", "Competition category codes.")
    add("Q005", "holiday", "value in {0,1}", ~train["holiday"].isin([0, 1]), "error", "Binary indicator.")
    add("Q006", "workingday", "value in {0,1}", ~train["workingday"].isin([0, 1]), "error", "Binary indicator.")
    add("Q007", "weather", "value in {1,2,3,4}", ~train["weather"].isin([1, 2, 3, 4]), "error", "Competition category codes.")
    add("Q008", "humidity", "0 <= humidity <= 100", ~train["humidity"].between(0, 100), "error", "Percentage range; exact zero is separately flagged as suspicious.")
    for index, column in enumerate(["temp", "atemp", "windspeed", "casual", "registered", "count"], start=9):
        add(f"Q{index:03d}", column, f"{column} is finite and nonnegative", (~np.isfinite(train[column])) | (train[column] < 0), "error", "No correction is applied in this phase.")
    add("Q015", "count", "count == casual + registered", train["count"] != train["casual"] + train["registered"], "critical", "Any mismatch would require investigation; components remain forbidden predictors.")
    weekend = parsed_datetime.dt.dayofweek >= 5
    expected_workingday = (~weekend & (train["holiday"] == 0)).astype(int)
    add("Q016", "workingday", "workingday matches weekday and non-holiday rule", train["workingday"] != expected_workingday, "review", "Validates the documented logical definition against parsed dates.")
    for index, column in enumerate(["casual", "registered", "count"], start=17):
        add(f"Q{index:03d}", column, f"{column} has integer-valued observations", train[column] % 1 != 0, "error", "Count variables should be whole numbers.")
    return pd.DataFrame(records)


def timestamp_audit(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    parsed = pd.to_datetime(frame["datetime"], errors="coerce")
    valid_sorted = parsed.dropna().sort_values()
    unique_sorted = pd.DatetimeIndex(valid_sorted.drop_duplicates())
    diffs = pd.Series(unique_sorted).diff().dt.total_seconds().div(3600).dropna()
    gap_distribution = (
        diffs.value_counts().sort_index().rename_axis("gap_hours").reset_index(name="transition_count")
    )
    gap_distribution.insert(0, "dataset", dataset)
    if len(unique_sorted):
        expected = pd.date_range(unique_sorted.min(), unique_sorted.max(), freq="h")
        missing_hours = expected.difference(unique_sorted)
        exact_week = (parsed - pd.Timedelta(hours=168)).isin(set(unique_sorted))
        exact_day = (parsed - pd.Timedelta(hours=24)).isin(set(unique_sorted))
    else:
        expected = pd.DatetimeIndex([])
        missing_hours = pd.DatetimeIndex([])
        exact_week = pd.Series(False, index=frame.index)
        exact_day = pd.Series(False, index=frame.index)
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "row_count": len(frame),
                "parse_failures": int(parsed.isna().sum()),
                "duplicate_timestamps": int(frame["datetime"].duplicated(keep=False).sum()),
                "monotonic_in_file_order": bool(parsed.is_monotonic_increasing),
                "minimum_timestamp": valid_sorted.min(),
                "maximum_timestamp": valid_sorted.max(),
                "unique_timestamps": len(unique_sorted),
                "full_span_expected_hours": len(expected),
                "missing_hours_in_full_span": len(missing_hours),
                "missing_hours_pct": float(len(missing_hours) / len(expected) * 100) if len(expected) else np.nan,
                "one_hour_transitions": int((diffs == 1).sum()),
                "transitions_over_one_hour": int((diffs > 1).sum()),
                "maximum_gap_hours": float(diffs.max()) if len(diffs) else np.nan,
                "exact_t_minus_168_coverage_count": int(exact_week.sum()),
                "exact_t_minus_168_coverage_pct": float(exact_week.mean() * 100),
                "exact_t_minus_24_coverage_count": int(exact_day.sum()),
                "exact_t_minus_24_coverage_pct": float(exact_day.mean() * 100),
                "weekly_or_daily_coverage_count": int((exact_week | exact_day).sum()),
                "weekly_or_daily_coverage_pct": float((exact_week | exact_day).mean() * 100),
            }
        ]
    )
    calendar = pd.DataFrame({"datetime": parsed.dropna()})
    calendar["year_month"] = calendar["datetime"].dt.to_period("M").astype(str)
    calendar["date"] = calendar["datetime"].dt.date
    calendar["day_of_month"] = calendar["datetime"].dt.day
    monthly = (
        calendar.groupby("year_month")
        .agg(
            rows=("datetime", "size"),
            unique_dates=("date", "nunique"),
            minimum_day=("day_of_month", "min"),
            maximum_day=("day_of_month", "max"),
        )
        .reset_index()
    )
    monthly.insert(0, "dataset", dataset)
    return summary, gap_distribution, monthly, parsed


def build_target_summary(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for column in ["casual", "registered", "count"]:
        series = train[column]
        records.append(
            {
                "variable": column,
                "rows": len(series),
                "minimum": float(series.min()),
                "q01": float(series.quantile(0.01)),
                "q05": float(series.quantile(0.05)),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
                "q99": float(series.quantile(0.99)),
                "maximum": float(series.max()),
                "standard_deviation": float(series.std(ddof=1)),
                "skewness": float(series.skew()),
                "zero_count": int((series == 0).sum()),
                "zero_pct": float((series == 0).mean() * 100),
            }
        )
    summary = pd.DataFrame(records)
    log_count = np.log1p(train["count"])
    log_summary = pd.DataFrame(
        [
            {
                "variable": "log1p(count)",
                "minimum": float(log_count.min()),
                "median": float(log_count.median()),
                "mean": float(log_count.mean()),
                "maximum": float(log_count.max()),
                "standard_deviation": float(log_count.std(ddof=1)),
                "skewness": float(pd.Series(log_count).skew()),
            }
        ]
    )
    counts, edges = np.histogram(train["count"], bins=20)
    histogram = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "row_count": counts,
        }
    )
    histogram["bin_label"] = histogram.apply(
        lambda row: f"{row['bin_left']:.0f}–{row['bin_right']:.0f}", axis=1
    )
    return summary, log_summary, histogram


def standardized_mean_difference(train: pd.Series, test: pd.Series) -> float:
    pooled = np.sqrt((train.var(ddof=1) + test.var(ddof=1)) / 2)
    return float((test.mean() - train.mean()) / pooled) if pooled else 0.0


def continuous_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in ["temp", "atemp", "humidity", "windspeed"]:
        train_values = train[column].dropna().astype(float)
        test_values = test[column].dropna().astype(float)
        ks_result = ks_2samp(train_values, test_values, alternative="two-sided", method="auto")
        smd = standardized_mean_difference(train_values, test_values)
        records.append(
            {
                "feature": column,
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "mean_difference_test_minus_train": float(test_values.mean() - train_values.mean()),
                "train_std": float(train_values.std(ddof=1)),
                "test_std": float(test_values.std(ddof=1)),
                "standardized_mean_difference": smd,
                "absolute_smd": abs(smd),
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
                "train_zero_pct": float((train_values == 0).mean() * 100),
                "test_zero_pct": float((test_values == 0).mean() * 100),
                "zero_pct_difference": float(((test_values == 0).mean() - (train_values == 0).mean()) * 100),
                "screening_flag_abs_smd_ge_0_10": bool(abs(smd) >= 0.10),
            }
        )
    return pd.DataFrame(records)


def categorical_drift(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dt = pd.to_datetime(train["datetime"], errors="coerce")
    test_dt = pd.to_datetime(test["datetime"], errors="coerce")
    train_augmented = train.copy()
    test_augmented = test.copy()
    for frame, parsed in [(train_augmented, train_dt), (test_augmented, test_dt)]:
        frame["year"] = parsed.dt.year
        frame["month"] = parsed.dt.month
        frame["hour"] = parsed.dt.hour
        frame["weekday"] = parsed.dt.dayofweek
        frame["day_of_month"] = parsed.dt.day
    features = ["season", "holiday", "workingday", "weather", "year", "month", "hour", "weekday", "day_of_month"]
    summary_records = []
    share_records = []
    for feature in features:
        categories = sorted(set(train_augmented[feature].dropna()) | set(test_augmented[feature].dropna()))
        train_shares = train_augmented[feature].value_counts(normalize=True, dropna=False)
        test_shares = test_augmented[feature].value_counts(normalize=True, dropna=False)
        differences = []
        for category in categories:
            train_share = float(train_shares.get(category, 0.0))
            test_share = float(test_shares.get(category, 0.0))
            difference = test_share - train_share
            differences.append((category, difference))
            share_records.append(
                {
                    "feature": feature,
                    "category": category,
                    "train_count": int((train_augmented[feature] == category).sum()),
                    "test_count": int((test_augmented[feature] == category).sum()),
                    "train_pct": train_share * 100,
                    "test_pct": test_share * 100,
                    "difference_percentage_points": difference * 100,
                }
            )
        tvd = 0.5 * sum(abs(difference) for _, difference in differences)
        largest_category, largest_difference = max(differences, key=lambda pair: abs(pair[1]))
        summary_records.append(
            {
                "feature": feature,
                "source": "raw" if feature in ["season", "holiday", "workingday", "weather"] else "timestamp_derived",
                "total_variation_distance": float(tvd),
                "largest_gap_category": largest_category,
                "largest_gap_percentage_points": float(largest_difference * 100),
                "screening_flag_tvd_ge_0_10": bool(tvd >= 0.10),
            }
        )
    return pd.DataFrame(summary_records), pd.DataFrame(share_records)


def build_data_dictionary(train: pd.DataFrame, test: pd.DataFrame, parsed: pd.Series) -> pd.DataFrame:
    definitions = {
        "datetime": ("timestamp text; parsed diagnostically", "Hourly date and time", "temporal key", "Known before prediction", "No"),
        "season": ("categorical integer", "1=spring, 2=summer, 3=fall, 4=winter", "candidate predictor", "Calendar-derived and known", "No"),
        "holiday": ("binary integer", "Whether the day is a holiday", "candidate predictor", "Known from holiday calendar", "No"),
        "workingday": ("binary integer", "Whether the day is neither weekend nor holiday", "candidate predictor", "Known from calendar", "No"),
        "weather": ("categorical integer", "1=clear/few clouds; 2=mist/cloudy; 3=light precipitation; 4=severe precipitation/fog", "candidate predictor", "Requires target-hour weather forecast in deployment", "No direct leakage"),
        "temp": ("continuous float", "Temperature in degrees Celsius", "candidate predictor", "Requires a forecast-time value", "No"),
        "atemp": ("continuous float", "Feels-like temperature in degrees Celsius", "candidate predictor", "Requires a forecast-time value", "No"),
        "humidity": ("integer percentage", "Relative humidity", "candidate predictor", "Requires a forecast-time value", "No"),
        "windspeed": ("continuous float", "Wind speed; unit not stated in supplied CSV", "candidate predictor", "Requires a forecast-time value", "No"),
        "casual": ("nonnegative integer count", "Rentals by non-registered users", "outcome component; audit only", "Known only after/during target hour", "PROHIBITED: direct target leakage"),
        "registered": ("nonnegative integer count", "Rentals by registered users", "outcome component; audit only", "Known only after/during target hour", "PROHIBITED: direct target leakage"),
        "count": ("nonnegative integer count", "Total rentals", "supervised target", "Unknown at prediction time", "Target; never contemporaneous input"),
    }
    expected_values = {
        "season": "{1,2,3,4}",
        "holiday": "{0,1}",
        "workingday": "{0,1}",
        "weather": "{1,2,3,4}",
        "humidity": "0 to 100",
        "temp": ">= 0 for this audit",
        "atemp": ">= 0 for this audit",
        "windspeed": ">= 0",
        "casual": "integer >= 0",
        "registered": "integer >= 0",
        "count": "integer >= 0 and casual + registered",
        "datetime": "parseable, unique, chronologically orderable",
    }
    records = []
    for column in EXPECTED_TRAIN_COLUMNS:
        expected_type, meaning, role, availability, leakage = definitions[column]
        if column == "datetime":
            observed = f"{parsed.min()} to {parsed.max()}; {parsed.nunique()} unique"
        else:
            observed = (
                f"min={train[column].min()}, max={train[column].max()}, "
                f"unique={train[column].nunique()}, missing={train[column].isna().sum()}"
            )
        records.append(
            {
                "field": column,
                "raw_train_dtype": str(train[column].dtype),
                "expected_logical_type": expected_type,
                "competition_documentation_meaning": meaning,
                "expected_values_or_rule": expected_values[column],
                "model_role": role,
                "forecast_time_availability": availability,
                "leakage_status": leakage,
                "present_in_competition_test": column in test.columns,
                "observed_raw_train": observed,
            }
        )
    return pd.DataFrame(records)


def create_figures(
    train: pd.DataFrame,
    test: pd.DataFrame,
    zero_rates: pd.DataFrame,
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(train["count"], bins=30, color="#2563EB", edgecolor="white")
    axes[0].axvline(train["count"].median(), color="#DC2626", linestyle="--", label=f"Median = {train['count'].median():.0f}")
    axes[0].set(title="Raw hourly rental count", xlabel="count", ylabel="Rows")
    axes[0].legend()
    log_count = np.log1p(train["count"])
    axes[1].hist(log_count, bins=30, color="#0F766E", edgecolor="white")
    axes[1].axvline(log_count.median(), color="#DC2626", linestyle="--", label=f"Median = {log_count.median():.2f}")
    axes[1].set(title="log1p(count)", xlabel="log1p(count)", ylabel="Rows")
    axes[1].legend()
    fig.suptitle("Target distribution before any transformation")
    fig.savefig(figures_dir / "raw_target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    train_dt = pd.to_datetime(train["datetime"], errors="coerce")
    test_dt = pd.to_datetime(test["datetime"], errors="coerce")
    months = sorted(set(train_dt.dt.to_period("M").astype(str)) | set(test_dt.dt.to_period("M").astype(str)))
    source_grid = np.zeros((len(months), 31), dtype=int)
    for code, parsed in [(1, train_dt), (2, test_dt)]:
        # Aggregate hourly records to unique calendar days before assigning a
        # source code. Otherwise, summing one code per hour incorrectly clips
        # ordinary single-source days into the "both sources" color.
        for timestamp in parsed.dropna().dt.floor("D").drop_duplicates():
            row = months.index(str(timestamp.to_period("M")))
            col = timestamp.day - 1
            source_grid[row, col] = source_grid[row, col] + code
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#F3F4F6", "#2563EB", "#F59E0B", "#7C3AED"])
    image = ax.imshow(source_grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(31), labels=range(1, 32))
    ax.set_yticks(range(len(months)), labels=months)
    ax.set(xlabel="Day of month", ylabel="Month", title="Raw timestamp coverage: train days 1–19 and competition-test later days")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="No row", markerfacecolor="#F3F4F6", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Train", markerfacecolor="#2563EB", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Competition test", markerfacecolor="#F59E0B", markersize=10),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(figures_dir / "timestamp_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    train_zeros = zero_rates[zero_rates["dataset"] == "train"].sort_values("zero_pct")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(train_zeros["column"], train_zeros["zero_pct"], color="#7C3AED")
    ax.set(xlabel="Rows equal to zero (%)", title="Raw train zero rates by numeric field")
    for index, value in enumerate(train_zeros["zero_pct"]):
        ax.text(value + 0.15, index, f"{value:.2f}%", va="center", fontsize=8)
    fig.savefig(figures_dir / "raw_zero_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drift_plot = pd.concat(
        [
            continuous[["feature", "absolute_smd"]].rename(columns={"absolute_smd": "drift_magnitude"}).assign(metric="Absolute SMD"),
            categorical[["feature", "total_variation_distance"]].rename(columns={"total_variation_distance": "drift_magnitude"}).assign(metric="Total variation distance"),
        ],
        ignore_index=True,
    ).sort_values("drift_magnitude")
    fig, ax = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
    colors = ["#DC2626" if value >= 0.10 else "#0F766E" for value in drift_plot["drift_magnitude"]]
    ax.barh(drift_plot["feature"], drift_plot["drift_magnitude"], color=colors)
    ax.axvline(0.10, color="#111827", linestyle="--", linewidth=1, label="0.10 screening threshold")
    ax.set(xlabel="Screening magnitude", title="Raw train vs competition-test distribution differences")
    ax.legend(loc="lower right")
    fig.savefig(figures_dir / "train_test_drift_screen.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    paths = {
        "train": args.train.resolve(),
        "competition_test": args.test.resolve(),
        "sample_submission": args.sample_submission.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output.resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["competition_test"])
    sample = pd.read_csv(paths["sample_submission"])
    frames = {"train": train, "competition_test": test, "sample_submission": sample}

    inventory = file_inventory(paths, frames)
    schema = build_schema(frames)
    samples = build_samples(train)
    stats = descriptive_statistics(train)
    cardinalities = build_cardinalities(train)
    category_counts = categorical_distributions(train)
    zero_rates = build_zero_rates(train, test)

    train_time_summary, train_gaps, train_monthly, train_dt = timestamp_audit(train, "train")
    test_time_summary, test_gaps, test_monthly, test_dt = timestamp_audit(test, "competition_test")
    timestamp_summary = pd.concat([train_time_summary, test_time_summary], ignore_index=True)
    gap_distribution = pd.concat([train_gaps, test_gaps], ignore_index=True)
    monthly_coverage = pd.concat([train_monthly, test_monthly], ignore_index=True)

    checks = quality_checks(train, train_dt)
    target_summary, log_summary, target_histogram = build_target_summary(train)
    numeric_drift = continuous_drift(train, test)
    category_drift, category_shares = categorical_drift(train, test)
    dictionary = build_data_dictionary(train, test, train_dt)

    train_only = [column for column in train.columns if column not in test.columns]
    test_only = [column for column in test.columns if column not in train.columns]
    shared = [column for column in train.columns if column in test.columns]
    overlap = set(train["datetime"]).intersection(set(test["datetime"]))
    sample_alignment = bool(
        len(sample) == len(test)
        and list(sample.columns) == ["datetime", "count"]
        and sample["datetime"].equals(test["datetime"])
    )
    leakage_mismatches = int((train["count"] != train["casual"] + train["registered"]).sum())
    leakage_validation = pd.DataFrame(
        [
            {
                "rule": "count equals casual plus registered",
                "rows_checked": len(train),
                "mismatch_count": leakage_mismatches,
                "maximum_absolute_difference": int((train["count"] - train["casual"] - train["registered"]).abs().max()),
                "result": "PASS" if leakage_mismatches == 0 else "FAIL",
                "modeling_consequence": "casual and registered are prohibited predictors regardless of result",
            }
        ]
    )
    schema_comparison = pd.DataFrame(
        [
            {
                "train_shape": f"{train.shape[0]} x {train.shape[1]}",
                "competition_test_shape": f"{test.shape[0]} x {test.shape[1]}",
                "sample_submission_shape": f"{sample.shape[0]} x {sample.shape[1]}",
                "shared_columns": ", ".join(shared),
                "train_only_columns": ", ".join(train_only),
                "test_only_columns": ", ".join(test_only) if test_only else "None",
                "overlapping_timestamps": len(overlap),
                "sample_submission_aligns_to_test": sample_alignment,
                "sample_submission_unique_count_values": int(sample["count"].nunique(dropna=False)),
                "sample_submission_all_counts_zero": bool((sample["count"] == 0).all()),
            }
        ]
    )

    monthly_target = pd.DataFrame({"datetime": train_dt, "count": train["count"]})
    monthly_target["year_month"] = monthly_target["datetime"].dt.to_period("M").astype(str)
    monthly_target = (
        monthly_target.groupby("year_month")["count"]
        .agg(rows="size", mean_count="mean", median_count="median", minimum_count="min", maximum_count="max")
        .reset_index()
    )

    tables = {
        "file_inventory.csv": inventory,
        "raw_schema.csv": schema,
        "raw_sample_rows.csv": samples,
        "descriptive_statistics.csv": stats,
        "cardinalities.csv": cardinalities,
        "categorical_distributions.csv": category_counts,
        "missing_and_zero_rates.csv": zero_rates,
        "quality_checks.csv": checks,
        "timestamp_summary.csv": timestamp_summary,
        "timestamp_gap_distribution.csv": gap_distribution,
        "monthly_coverage.csv": monthly_coverage,
        "monthly_target_summary.csv": monthly_target,
        "target_distribution_summary.csv": target_summary,
        "log_target_summary.csv": log_summary,
        "target_histogram_bins.csv": target_histogram,
        "leakage_validation.csv": leakage_validation,
        "schema_comparison.csv": schema_comparison,
        "train_test_numeric_drift.csv": numeric_drift,
        "train_test_categorical_drift.csv": category_drift,
        "train_test_category_shares.csv": category_shares,
        "DATA_DICTIONARY.csv": dictionary,
    }
    for filename, frame in tables.items():
        save_table(frame, tables_dir / filename)

    create_figures(train, test, zero_rates, numeric_drift, category_drift, figures_dir)

    count_row = target_summary[target_summary["variable"] == "count"].iloc[0]
    log_row = log_summary.iloc[0]
    humidity_zeros = int((train["humidity"] == 0).sum())
    wind_zeros = int((train["windspeed"] == 0).sum())
    weather_four = int((train["weather"] == 4).sum())
    atemp_test_zero_times = test.loc[test["atemp"] == 0, "datetime"].tolist()
    humidity_zero_dates = sorted(pd.to_datetime(train.loc[train["humidity"] == 0, "datetime"]).dt.date.astype(str).unique().tolist())

    report_lines = [
        "# PedalPulse Raw Data Quality Report",
        "",
        "**CRISP-DM phase:** Data Understanding  ",
        "**Evidence boundary:** Raw CSVs were read and fingerprinted. No cleaning, imputation, deletion, capping, feature engineering, modeling, or target-based split selection was performed.",
        "",
        "## 1. File inventory",
        "",
        markdown_table(inventory[["dataset", "filename", "size_bytes", "rows", "columns", "encoding_check", "line_endings", "sha256"]]),
        "",
        "## 2. Raw shapes and schemas",
        "",
        f"- `train.csv`: **{len(train):,} rows × {train.shape[1]} columns**.",
        f"- Competition test: **{len(test):,} rows × {test.shape[1]} columns**.",
        f"- Sample submission: **{len(sample):,} rows × {sample.shape[1]} columns**.",
        f"- Train-only columns: `{', '.join(train_only)}`.",
        f"- Shared train/test dtypes match: **{bool(all(train[column].dtype == test[column].dtype for column in shared))}**.",
        f"- Sample-submission timestamps align exactly with competition test: **{sample_alignment}**.",
        "- All sample-submission `count` values are placeholders equal to zero; they are not observed outcomes.",
        "",
        markdown_table(schema[schema["dataset"] == "train"][["column", "pandas_dtype", "non_null_count", "missing_count", "unique_count"]]),
        "",
        "## 3. Raw sample rows",
        "",
        "The first five records, including the original CSV line numbers:",
        "",
        markdown_table(samples[samples["sample_position"] == "head"]),
        "",
        "The last five records are preserved in `tables/raw_sample_rows.csv`.",
        "",
        "## 4. Completeness, duplicates, and validity",
        "",
        f"- Missing cells in train: **{int(train.isna().sum().sum()):,}**.",
        f"- Fully duplicated train rows: **{int(train.duplicated(keep=False).sum()):,}**.",
        f"- Duplicate train timestamps: **{int(train['datetime'].duplicated(keep=False).sum()):,}**.",
        f"- Timestamp parse failures: **{int(train_dt.isna().sum()):,}**.",
        f"- Range/logical checks requiring error-level review: **{int(((checks['severity'].isin(['error', 'critical'])) & (checks['violations'] > 0)).sum())}**.",
        "",
        markdown_table(checks[["check_id", "field", "rule", "severity", "violations", "status"]]),
        "",
        "## 5. Cardinalities and ranges",
        "",
        markdown_table(cardinalities[["column", "dtype", "unique_count", "most_frequent_value", "most_frequent_count"]]),
        "",
        "Complete numerical statistics—including 1st, 5th, 95th, and 99th percentiles—are stored in `tables/descriptive_statistics.csv`.",
        "",
        "## 6. Target and leakage audit",
        "",
        f"- `count` range: **{int(count_row['minimum'])} to {int(count_row['maximum'])}** rentals per hour.",
        f"- Mean/median: **{count_row['mean']:.3f} / {count_row['median']:.0f}**.",
        f"- Raw skewness: **{count_row['skewness']:.3f}**; `log1p(count)` skewness: **{log_row['skewness']:.3f}**.",
        f"- Zero target rows: **{int(count_row['zero_count']):,}**.",
        f"- Identity rows checked: **{len(train):,}**; mismatches in `count = casual + registered`: **{leakage_mismatches:,}**.",
        "- `casual` and `registered` are therefore exact outcome components and remain prohibited from every feature or clustering pipeline.",
        "",
        markdown_table(target_summary),
        "",
        "![Raw target distribution](figures/raw_target_distribution.png)",
        "",
        "**Interpretation:** The raw target is right-skewed. `log1p` reduces the measured skew, but this is only a documented candidate transformation; no transformation has been applied to the raw data.",
        "",
        "## 7. Timestamp coverage",
        "",
        markdown_table(timestamp_summary),
        "",
        "![Timestamp coverage](figures/timestamp_coverage.png)",
        "",
        "**Interpretation:** The Kaggle competition split is organized by day of month: training observations occupy days 1–19 and test observations occupy later days. The test file is not one future holdout and contains no target labels. PedalPulse must construct its own chronological development and final-test periods from `train.csv`.",
        "",
        "## 8. Suspicious zeros",
        "",
        f"- Train humidity equals zero in **{humidity_zeros:,}** rows, concentrated on date(s): **{', '.join(humidity_zero_dates) if humidity_zero_dates else 'none'}**.",
        f"- Train windspeed equals zero in **{wind_zeros:,}** rows (**{wind_zeros / len(train) * 100:.3f}%**).",
        f"- Weather code 4 appears **{weather_four:,}** time(s) in train and is too rare for reliable standalone inference.",
        f"- Competition-test `atemp` equals zero at **{len(atemp_test_zero_times)}** timestamp(s): {', '.join(atemp_test_zero_times) if atemp_test_zero_times else 'none'}.",
        "- These values are flagged, not corrected. Zero could represent a real observation, a sensor/code artifact, or missingness encoded as zero.",
        "",
        "![Raw zero rates](figures/raw_zero_rates.png)",
        "",
        "**Interpretation:** Zero prevalence differs sharply by field. A zero is structurally plausible for rental components, but humidity, feels-like temperature, and windspeed require domain-specific investigation before any handling decision.",
        "",
        "## 9. Train-versus-competition-test drift screen",
        "",
        "Continuous features:",
        "",
        markdown_table(numeric_drift[["feature", "train_mean", "test_mean", "standardized_mean_difference", "ks_statistic", "ks_pvalue", "train_zero_pct", "test_zero_pct"]]),
        "",
        "Categorical and calendar features:",
        "",
        markdown_table(category_drift[["feature", "source", "total_variation_distance", "largest_gap_category", "largest_gap_percentage_points", "screening_flag_tvd_ge_0_10"]]),
        "",
        "![Train/test drift screen](figures/train_test_drift_screen.png)",
        "",
        "**Interpretation:** Shared exogenous distributions are mostly similar, with humidity producing the largest continuous mean shift. Day-of-month has total-variation distance 1.0 because the competition split is deliberately disjoint by calendar day. KS p-values are treated as screening evidence, not practical effect sizes or proof of future drift.",
        "",
        "## 10. Principal data-quality findings",
        "",
        "1. Direct target leakage is confirmed structurally and empirically: `count` equals `casual + registered` for every raw training row.",
        "2. No ordinary null cells, duplicate rows, duplicate timestamps, or timestamp parse failures were measured.",
        "3. The training timeline is not a continuous hourly sequence because later days of every month are withheld for the competition test.",
        "4. Seasonal-naive evaluation must use exact datetime lookups and report fallback coverage; a positional row shift would be wrong.",
        "5. Humidity zeros, windspeed zeros, competition-test `atemp` zeros, and the extremely rare weather code 4 require review rather than automatic deletion or imputation.",
        "6. The competition test cannot measure model performance because `count` is absent and submission zeros are placeholders.",
        "7. The attached data contain no station geography, inventory, or unserved-demand measure; interpretations remain system-wide rental forecasts.",
        "",
        "## 11. Deferred decisions",
        "",
        "No quality flag has been cleaned in Chunk 2. Treatment alternatives will be compared in later training-only experiments. Exact chronological split boundaries will be selected from documented timestamp coverage before modeling and without consulting performance on the future test period.",
    ]
    report_path = output / "DATA_QUALITY_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    dictionary_report = [
        "# PedalPulse Raw Data Dictionary",
        "",
        "Competition meanings are documented semantics; observed ranges and counts come from the attached raw files.",
        "",
        markdown_table(dictionary),
        "",
        "Source: https://www.kaggle.com/competitions/bike-sharing-demand/data",
    ]
    (output / "DATA_DICTIONARY.md").write_text("\n".join(dictionary_report) + "\n", encoding="utf-8")

    runtime_seconds = time.perf_counter() - started
    summary = {
        "train_shape": list(train.shape),
        "competition_test_shape": list(test.shape),
        "sample_submission_shape": list(sample.shape),
        "train_missing_cells": int(train.isna().sum().sum()),
        "train_full_duplicate_rows": int(train.duplicated(keep=False).sum()),
        "train_duplicate_timestamps": int(train["datetime"].duplicated(keep=False).sum()),
        "train_timestamp_parse_failures": int(train_dt.isna().sum()),
        "target_identity_mismatches": leakage_mismatches,
        "target_min": int(train["count"].min()),
        "target_max": int(train["count"].max()),
        "target_mean": float(train["count"].mean()),
        "target_median": float(train["count"].median()),
        "target_skewness": float(train["count"].skew()),
        "log1p_target_skewness": float(pd.Series(np.log1p(train["count"])).skew()),
        "target_zero_count": int((train["count"] == 0).sum()),
        "humidity_zero_count": humidity_zeros,
        "windspeed_zero_count": wind_zeros,
        "weather_code_4_count": weather_four,
        "competition_test_atemp_zero_count": len(atemp_test_zero_times),
        "sample_submission_aligns": sample_alignment,
        "competition_test_target_available": "count" in test.columns,
        "train_test_timestamp_overlap_count": len(overlap),
        "maximum_continuous_abs_smd_feature": str(numeric_drift.loc[numeric_drift["absolute_smd"].idxmax(), "feature"]),
        "maximum_continuous_abs_smd": float(numeric_drift["absolute_smd"].max()),
        "day_of_month_tvd": float(category_drift.loc[category_drift["feature"] == "day_of_month", "total_variation_distance"].iloc[0]),
        "runtime_seconds": runtime_seconds,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    manifest = {
        "script": Path(__file__).name,
        "command_contract": "Provide --train, --test, --sample-submission, and --output paths.",
        "inputs": {
            label: {"filename": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for label, path in paths.items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "runtime_seconds": runtime_seconds,
        "randomness": "No stochastic methods used.",
        "raw_data_modified": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
