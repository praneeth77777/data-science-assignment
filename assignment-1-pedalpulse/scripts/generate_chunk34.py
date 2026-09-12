#!/usr/bin/env python3
"""Generate PedalPulse Chunk 3 EDA and Chunk 4 preparation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats

from pedalpulse.data import LEAKAGE_COLUMNS, TARGET_COLUMN, chronological_split, safe_predictor_frame
from pedalpulse.features import CalendarFeatureEngineer
from pedalpulse.preprocessing import SUSPICIOUS_ZERO_COLUMNS, SuspiciousZeroHandler, build_preparation_pipeline

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_NAMES = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_NAMES = {1: "Clear/partly cloudy", 2: "Mist/cloudy", 3: "Light rain/snow", 4: "Heavy weather"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_ci(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    grouped = frame.groupby(group, observed=True)["count"].agg(["count", "mean", "std", "median"])
    grouped = grouped.rename(columns={"count": "n", "mean": "mean_count", "std": "std_count", "median": "median_count"}).reset_index()
    standard_error = grouped["std_count"] / np.sqrt(grouped["n"])
    critical = stats.t.ppf(0.975, grouped["n"] - 1)
    margin = critical * standard_error
    grouped["standard_error"] = standard_error
    grouped["ci95_low"] = grouped["mean_count"] - margin
    grouped["ci95_high"] = grouped["mean_count"] + margin
    return grouped


def relationship_bins(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    bins = pd.qcut(frame[feature], q=10, duplicates="drop")
    grouped = frame.assign(_bin=bins).groupby("_bin", observed=True).agg(
        n=("count", "size"),
        feature_mean=(feature, "mean"),
        feature_min=(feature, "min"),
        feature_max=(feature, "max"),
        mean_count=("count", "mean"),
        std_count=("count", "std"),
    ).reset_index(drop=True)
    grouped.insert(0, "feature", feature)
    grouped.insert(1, "bin_number", np.arange(1, len(grouped) + 1))
    se = grouped["std_count"] / np.sqrt(grouped["n"])
    critical = stats.t.ppf(0.975, grouped["n"] - 1)
    grouped["ci95_low"] = grouped["mean_count"] - critical * se
    grouped["ci95_high"] = grouped["mean_count"] + critical * se
    return grouped


def label_group_tables(tables: dict[str, pd.DataFrame]) -> None:
    tables["weekday"]["label"] = tables["weekday"]["weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    tables["month"]["label"] = tables["month"]["month"].map(dict(enumerate(MONTH_NAMES, start=1)))
    tables["season"]["label"] = tables["season"]["season"].map(SEASON_NAMES)
    tables["holiday"]["label"] = tables["holiday"]["holiday"].map({0: "Not holiday", 1: "Holiday"})
    tables["workingday"]["label"] = tables["workingday"]["workingday"].map({0: "Non-working day", 1: "Working day"})
    tables["weather"]["label"] = tables["weather"]["weather"].map(WEATHER_NAMES)


def errorbar_panel(ax: plt.Axes, table: pd.DataFrame, x: str, labels: list[str] | None, title: str) -> None:
    x_values = np.arange(len(table)) if labels is not None else table[x].to_numpy()
    lower = table["mean_count"] - table["ci95_low"]
    upper = table["ci95_high"] - table["mean_count"]
    ax.errorbar(x_values, table["mean_count"], yerr=np.vstack([lower, upper]), marker="o", linewidth=2, capsize=3)
    if labels is not None:
        ax.set_xticks(x_values, labels=labels, rotation=30, ha="right")
    ax.set(title=title, xlabel="", ylabel="Mean hourly count")


def create_figures(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], relationships: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    errorbar_panel(axes[0, 0], tables["hour"], "hour", None, "Demand by hour")
    axes[0, 0].set_xticks(range(0, 24, 3))
    errorbar_panel(axes[0, 1], tables["weekday"], "weekday", tables["weekday"]["label"].tolist(), "Demand by weekday")
    errorbar_panel(axes[1, 0], tables["month"], "month", tables["month"]["label"].tolist(), "Demand by month of year")
    trend = tables["year_month"]
    axes[1, 1].plot(range(len(trend)), trend["mean_count"], marker="o")
    axes[1, 1].fill_between(range(len(trend)), trend["ci95_low"], trend["ci95_high"], alpha=0.2)
    axes[1, 1].set_xticks(range(0, len(trend), 3), labels=trend["year_month"].iloc[::3], rotation=30, ha="right")
    axes[1, 1].set(title="Chronological monthly demand", ylabel="Mean hourly count")
    fig.suptitle("Temporal demand patterns with descriptive 95% confidence intervals")
    fig.savefig(figures / "temporal_demand_patterns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, key, title in zip(
        axes.ravel(),
        ["season", "holiday", "workingday", "weather"],
        ["Season", "Holiday", "Working day", "Weather"],
    ):
        table = tables[key]
        yerr = np.vstack([table["mean_count"] - table["ci95_low"], table["ci95_high"] - table["mean_count"]])
        ax.bar(range(len(table)), table["mean_count"], yerr=yerr, capsize=4)
        ax.set_xticks(range(len(table)), table["label"], rotation=25, ha="right")
        ax.set(title=f"Demand by {title.lower()}", ylabel="Mean hourly count")
    fig.suptitle("Calendar and weather groups")
    fig.savefig(figures / "group_comparisons.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), ["temp", "atemp", "humidity", "windspeed"]):
        table = relationships[relationships["feature"] == feature]
        ax.plot(table["feature_mean"], table["mean_count"], marker="o")
        ax.fill_between(table["feature_mean"], table["ci95_low"], table["ci95_high"], alpha=0.2)
        ax.set(title=f"Binned demand relationship: {feature}", xlabel=feature, ylabel="Mean hourly count")
    fig.suptitle("Weather-variable relationships using equal-frequency bins")
    fig.savefig(figures / "weather_relationships.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = tables["hour_weekday_matrix"]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(24), labels=range(24))
    ax.set_yticks(range(7), labels=WEEKDAY_NAMES)
    ax.set(xlabel="Hour", ylabel="Weekday", title="Mean hourly demand by weekday and hour")
    fig.colorbar(image, ax=ax, label="Mean count")
    fig.savefig(figures / "hour_weekday_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(frame["count"], bins=30, edgecolor="white")
    axes[0].axvline(frame["count"].median(), linestyle="--", color="#DC2626", label=f"Median {frame['count'].median():.0f}")
    axes[0].set(title="Raw count", xlabel="count", ylabel="Rows")
    axes[0].legend()
    logged = np.log1p(frame["count"])
    axes[1].hist(logged, bins=30, edgecolor="white", color="#0F766E")
    axes[1].axvline(logged.median(), linestyle="--", color="#DC2626", label=f"Median {logged.median():.2f}")
    axes[1].set(title="log1p(count)", xlabel="log1p(count)", ylabel="Rows")
    axes[1].legend()
    fig.suptitle("Target scale comparison")
    fig.savefig(figures / "target_scale_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    correlation_features = ["count", "temp", "atemp", "humidity", "windspeed", "hour", "weekday", "month", "year"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, method, title in zip(axes, ["pearson", "spearman"], ["Pearson", "Spearman"]):
        corr = frame[correlation_features].corr(method=method)
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)), corr.index)
        for row in range(len(corr)):
            for column in range(len(corr)):
                ax.text(column, row, f"{corr.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{title} correlation")
    fig.colorbar(image, ax=axes, shrink=0.78, label="Correlation")
    fig.savefig(figures / "correlation_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_eda_report(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], correlations: dict[str, pd.DataFrame]) -> str:
    hour_high = tables["hour"].loc[tables["hour"]["mean_count"].idxmax()]
    hour_low = tables["hour"].loc[tables["hour"]["mean_count"].idxmin()]
    weekday_high = tables["weekday"].loc[tables["weekday"]["mean_count"].idxmax()]
    month_high = tables["month"].loc[tables["month"]["mean_count"].idxmax()]
    month_low = tables["month"].loc[tables["month"]["mean_count"].idxmin()]
    season_high = tables["season"].loc[tables["season"]["mean_count"].idxmax()]
    weather_table = tables["weather"]
    matrix = tables["hour_weekday_matrix"]
    max_position = np.unravel_index(np.nanargmax(matrix.to_numpy()), matrix.shape)
    max_day = matrix.index[max_position[0]]
    max_hour = matrix.columns[max_position[1]]
    max_value = matrix.iloc[max_position]
    pearson_target = correlations["pearson"].loc["count"].drop("count")
    spearman_target = correlations["spearman"].loc["count"].drop("count")
    strongest_p = pearson_target.abs().idxmax()
    strongest_s = spearman_target.abs().idxmax()
    holiday = tables["holiday"].set_index("holiday")
    working = tables["workingday"].set_index("workingday")

    return f"""# PedalPulse Textbook-Quality EDA

**CRISP-DM phase:** Data Understanding
**Evidence boundary:** All detailed summaries use the **9,519-row development pool** (chronological train plus validation). The locked October–December 2012 test targets are excluded. `casual` and `registered` are excluded because they exactly compose the target. No cleaning, causal estimation, feature selection, or model fitting was performed in this EDA.

## Figure 1 — Temporal demand patterns

![Temporal demand patterns](figures/temporal_demand_patterns.png)

**Observed evidence:** Mean demand is highest at hour **{int(hour_high['hour']):02d}:00** ({hour_high['mean_count']:.2f}) and lowest at hour **{int(hour_low['hour']):02d}:00** ({hour_low['mean_count']:.2f}). The highest weekday mean is **{weekday_high['label']}** ({weekday_high['mean_count']:.2f}). Across month-of-year groups, **{month_high['label']}** is highest ({month_high['mean_count']:.2f}) and **{month_low['label']}** is lowest ({month_low['mean_count']:.2f}). The chronological panel also shows that the two years should not be collapsed into a purely seasonal interpretation.

**Causal caution:** These patterns describe recorded rentals, not the isolated effect of clock time or month. Weather, year, working-day composition, supply availability, and unmeasured events may contribute.

## Figure 2 — Season, holiday, working-day, and weather groups

![Group comparisons](figures/group_comparisons.png)

**Observed evidence:** **{season_high['label']}** has the highest seasonal mean ({season_high['mean_count']:.2f}). Holiday and non-holiday means are {holiday.loc[1, 'mean_count']:.2f} and {holiday.loc[0, 'mean_count']:.2f}; working-day and non-working-day means are {working.loc[1, 'mean_count']:.2f} and {working.loc[0, 'mean_count']:.2f}. Weather-group means range from {weather_table['mean_count'].min():.2f} to {weather_table['mean_count'].max():.2f}. Weather code 4 has only **{int(weather_table.loc[weather_table['weather'] == 4, 'n'].iloc[0])}** observation, so its confidence interval and mean are not stable.

**Causal caution:** Group differences are unadjusted. The descriptive 95% intervals use a conventional independent-observation formula, while hourly time-series observations are autocorrelated; the intervals must not be interpreted as causal or fully time-series-correct uncertainty.

## Figure 3 — Temperature, feels-like temperature, humidity, and windspeed

![Weather relationships](figures/weather_relationships.png)

**Observed evidence:** Pearson correlations with count are temp **{correlations['pearson'].loc['count', 'temp']:.3f}**, atemp **{correlations['pearson'].loc['count', 'atemp']:.3f}**, humidity **{correlations['pearson'].loc['count', 'humidity']:.3f}**, and windspeed **{correlations['pearson'].loc['count', 'windspeed']:.3f}**. Equal-frequency bins expose nonlinearity that a single correlation coefficient can conceal.

**Causal caution:** These are marginal associations. Temperature and season are related, windspeed contains many zeros, and weather may interact with hour and working-day status.

## Figure 4 — Hour-by-weekday heatmap

![Hour-by-weekday heatmap](figures/hour_weekday_heatmap.png)

**Observed evidence:** The largest cell mean occurs on **{max_day} at {int(max_hour):02d}:00**, with mean count **{max_value:.2f}**. Weekday profiles show commuting-shaped peaks, while weekend profiles are distributed differently across the day.

**Causal caution:** The heatmap supports operational scheduling hypotheses, but it does not prove commuting purpose because trip-purpose labels are absent.

## Figure 5 — Raw target and `log1p` scale

![Target comparison](figures/target_scale_comparison.png)

**Observed evidence:** Raw `count` has mean **{frame['count'].mean():.2f}**, median **{frame['count'].median():.0f}**, and skewness **{frame['count'].skew():.3f}**. `log1p(count)` has skewness **{np.log1p(frame['count']).skew():.3f}**.

**Modeling caution:** Reduced skew does not establish better forecasts. Raw-target and log-target approaches must be compared using temporal validation, with log predictions converted back using `expm1`.

## Figure 6 — Correlation analysis

![Correlation analysis](figures/correlation_analysis.png)

**Observed evidence:** The strongest absolute Pearson association with count among the displayed non-target variables is **{strongest_p}** ({pearson_target[strongest_p]:.3f}); the strongest Spearman association is **{strongest_s}** ({spearman_target[strongest_s]:.3f}).

**Warnings:** Correlation is not causation; it does not represent nonlinear interactions or temporal dependence. Codes such as weekday and month are cyclic categories, so treating their integer labels as distances is mathematically crude. `temp` and `atemp` are strongly related, which may affect linear-model coefficient stability. Outcome components `casual` and `registered` were deliberately omitted.

## Confidence-interval note

The interval tables are useful descriptive summaries. They are not a substitute for blocked temporal uncertainty estimates because repeated hourly observations are not independent. Later model comparisons will use fold-to-fold temporal variation.
"""


def feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "season": "Published season code (1–4)", "holiday": "Published holiday flag", "workingday": "Published working-day flag",
        "weather": "Published weather-severity code", "temp": "Published temperature", "atemp": "Published feels-like temperature",
        "humidity": "Published relative humidity", "windspeed": "Published windspeed", "hour": "Hour of day extracted from datetime",
        "weekday": "Monday=0 through Sunday=6", "month": "Calendar month 1–12", "year": "Calendar year",
        "weekend": "1 for Saturday or Sunday", "rush_hour": "1 on working days at 07–09 or 16–19",
        "hour_sin": "Sine encoding of hour on a 24-hour cycle", "hour_cos": "Cosine encoding of hour on a 24-hour cycle",
        "weekday_sin": "Sine encoding of weekday on a 7-day cycle", "weekday_cos": "Cosine encoding of weekday on a 7-day cycle",
        "month_sin": "Sine encoding of month on a 12-month cycle", "month_cos": "Cosine encoding of month on a 12-month cycle",
        "atemp_zero_flag": "1 when raw atemp equals zero", "humidity_zero_flag": "1 when raw humidity equals zero",
        "windspeed_zero_flag": "1 when raw windspeed equals zero",
    }
    return pd.DataFrame({"feature": frame.columns, "dtype": [str(frame[c].dtype) for c in frame], "description": [descriptions[c] for c in frame.columns]})


def preparation_evidence(raw: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    splits = chronological_split(raw)
    split_rows = []
    for name, part in splits.items():
        timestamp = pd.to_datetime(part["datetime"])
        split_rows.append({
            "split": name,
            "rows": len(part),
            "minimum_timestamp": timestamp.min(),
            "maximum_timestamp": timestamp.max(),
            "target_accessed_for_preparation": name == "train",
        })
    split_manifest = pd.DataFrame(split_rows)
    split_manifest.to_csv(tables_dir / "chronological_split_manifest.csv", index=False)

    X = {name: safe_predictor_frame(part) for name, part in splits.items()}
    y_train = splits["train"][TARGET_COLUMN].copy()
    feature_engineer = CalendarFeatureEngineer().fit(X["train"])
    engineered = {name: feature_engineer.transform(part) for name, part in X.items()}
    feature_dictionary(engineered["train"]).to_csv(tables_dir / "engineered_feature_dictionary.csv", index=False)

    strategy_rows = []
    feature_name_rows = []
    for strategy in ["preserve_with_flags", "replace_with_train_median_and_flags"]:
        handler = SuspiciousZeroHandler(strategy).fit(engineered["train"])
        for split_name, frame in engineered.items():
            changed = handler.transform(frame)
            for column in SUSPICIOUS_ZERO_COLUMNS:
                strategy_rows.append({
                    "strategy": strategy,
                    "fit_partition": "train only",
                    "split_transformed": split_name,
                    "feature": column,
                    "original_zero_count": int((frame[column] == 0).sum()),
                    "post_handler_zero_count": int((changed[column] == 0).sum()),
                    "training_replacement_value": handler.replacement_values_.get(column, np.nan),
                    "indicator_retained": True,
                })

        pipeline = build_preparation_pipeline(strategy)
        train_array = pipeline.fit_transform(X["train"], y_train)
        validation_array = pipeline.transform(X["validation"])
        test_array = pipeline.transform(X["test"])
        names = pipeline.named_steps["column_transformer"].get_feature_names_out()
        for position, name in enumerate(names):
            feature_name_rows.append({"strategy": strategy, "position": position, "output_feature_name": name})
        for split_name, array in [("train", train_array), ("validation", validation_array), ("test", test_array)]:
            strategy_rows.append({
                "strategy": strategy,
                "fit_partition": "train only",
                "split_transformed": split_name,
                "feature": "__pipeline_output__",
                "original_zero_count": np.nan,
                "post_handler_zero_count": np.nan,
                "training_replacement_value": np.nan,
                "indicator_retained": True,
                "output_rows": array.shape[0],
                "output_columns": array.shape[1],
                "all_output_values_finite": bool(np.isfinite(array).all()),
            })

    strategy_table = pd.DataFrame(strategy_rows)
    strategy_table.to_csv(tables_dir / "suspicious_value_strategy_comparison.csv", index=False)
    feature_names = pd.DataFrame(feature_name_rows)
    feature_names.to_csv(tables_dir / "preprocessor_output_features.csv", index=False)

    train_row = split_manifest.set_index("split").loc["train"]
    validation_row = split_manifest.set_index("split").loc["validation"]
    test_row = split_manifest.set_index("split").loc["test"]
    preserve_columns = int(strategy_table.query("strategy == 'preserve_with_flags' and feature == '__pipeline_output__'")["output_columns"].iloc[0])
    replace_values = strategy_table.query("strategy == 'replace_with_train_median_and_flags' and split_transformed == 'train' and feature != '__pipeline_output__'")
    values_text = ", ".join(f"{row.feature}={row.training_replacement_value:g}" for row in replace_values.itertuples())
    report = f"""# PedalPulse Data Preparation Report

**CRISP-DM phase:** Data Preparation
**Status:** Deterministic feature and preprocessing components implemented and executed. No supervised model was fitted.

## Locked chronological partitions

| Partition | Rows | Minimum timestamp | Maximum timestamp |
|---|---:|---|---|
| Train | {int(train_row['rows']):,} | {train_row['minimum_timestamp']} | {train_row['maximum_timestamp']} |
| Validation | {int(validation_row['rows']):,} | {validation_row['minimum_timestamp']} | {validation_row['maximum_timestamp']} |
| Untouched labeled test | {int(test_row['rows']):,} | {test_row['minimum_timestamp']} | {test_row['maximum_timestamp']} |

The boundaries were chosen before model evaluation. Detailed EDA uses only train plus validation; the locked test target is neither summarized nor passed into preprocessing. The Kaggle competition test remains inference-only. Later cross-validation will occur inside the development period without accessing locked-test outcomes for feature or model decisions.

## Features implemented

- Parsed and strictly validated timestamp.
- Hour, weekday, month, year, weekend, and working-day rush-hour flag.
- Sine/cosine encodings for hour, weekday, and month.
- Explicit zero flags for `atemp`, `humidity`, and `windspeed`.
- Original shared exogenous predictors retained.

Rush hour is defined deterministically as 07:00–09:59 or 16:00–19:59 on a published working day. This is a declared operational feature definition, not an observed causal result.

## Leakage and ordering controls

- `safe_predictor_frame` selects only the nine shared predictor fields.
- Schema validation raises an exception if `count`, `casual`, or `registered` reaches the feature pipeline.
- Duplicate, malformed, or out-of-order timestamps raise exceptions.
- The chronological split verifies strictly ordered, non-overlapping periods.

## Pipeline design

The scikit-learn `Pipeline` performs calendar feature engineering, suspicious-zero handling, and a `ColumnTransformer`. Continuous and cyclical fields receive median imputation followed by standardization. Categorical fields receive most-frequent imputation followed by one-hot encoding with unknown-category tolerance.

Both candidate strategies produced **{preserve_columns}** finite output columns and were fitted only on the training partition before transforming validation and test partitions.

## Suspicious-value comparison

1. `preserve_with_flags`: retain zeros and add explicit zero indicators.
2. `replace_with_train_median_and_flags`: replace flagged zeros using nonzero medians learned from training only, while retaining the indicators.

Measured training-only replacement values were: **{values_text}**. No strategy has been declared superior; later temporal validation must compare them. Preserving flagged zeros is the conservative default until that sensitivity experiment is complete.

## Determinism

No randomized transformation is used. Feature definitions, column lists, category handling, and split boundaries are explicit. The executed unit suite checks schema failure, leakage prevention, timestamp ordering, feature values, training-only suspicious-value medians, deterministic transformation, and split non-overlap.
"""
    return split_manifest, strategy_table, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output / "tables"
    figures_dir = args.output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    before_hash = sha256(args.train)
    raw = pd.read_csv(args.train)
    safe_predictor_frame(raw)
    locked_splits = chronological_split(raw)
    development = pd.concat([locked_splits["train"], locked_splits["validation"]], ignore_index=True)
    frame = development.drop(columns=list(LEAKAGE_COLUMNS)).copy()
    timestamp = pd.to_datetime(frame["datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    frame["hour"] = timestamp.dt.hour
    frame["weekday"] = timestamp.dt.weekday
    frame["month"] = timestamp.dt.month
    frame["year"] = timestamp.dt.year
    frame["year_month"] = timestamp.dt.to_period("M").astype(str)

    tables = {key: mean_ci(frame, key) for key in ["hour", "weekday", "month", "season", "holiday", "workingday", "weather", "year_month"]}
    label_group_tables(tables)
    matrix = frame.pivot_table(index="weekday", columns="hour", values="count", aggfunc="mean").reindex(index=range(7), columns=range(24))
    matrix.index = WEEKDAY_NAMES
    matrix.index.name = "weekday"
    tables["hour_weekday_matrix"] = matrix

    relationship_tables = [relationship_bins(frame, feature) for feature in ["temp", "atemp", "humidity", "windspeed"]]
    relationships = pd.concat(relationship_tables, ignore_index=True)
    correlation_features = ["count", "temp", "atemp", "humidity", "windspeed", "hour", "weekday", "month", "year"]
    correlations = {method: frame[correlation_features].corr(method=method) for method in ["pearson", "spearman"]}

    for name, table in tables.items():
        if name == "hour_weekday_matrix":
            table.to_csv(tables_dir / "hour_weekday_mean_matrix.csv")
        else:
            table.to_csv(tables_dir / f"demand_by_{name}.csv", index=False)
    relationships.to_csv(tables_dir / "weather_binned_relationships.csv", index=False)
    for method, table in correlations.items():
        table.rename_axis("feature").reset_index().to_csv(tables_dir / f"correlations_{method}.csv", index=False)
    target_summary = pd.DataFrame([
        {"scale": "count", "minimum": frame["count"].min(), "median": frame["count"].median(), "mean": frame["count"].mean(), "maximum": frame["count"].max(), "std": frame["count"].std(), "skewness": frame["count"].skew()},
        {"scale": "log1p(count)", "minimum": np.log1p(frame["count"]).min(), "median": np.log1p(frame["count"]).median(), "mean": np.log1p(frame["count"]).mean(), "maximum": np.log1p(frame["count"]).max(), "std": np.log1p(frame["count"]).std(), "skewness": np.log1p(frame["count"]).skew()},
    ])
    target_summary.to_csv(tables_dir / "target_scale_summary.csv", index=False)

    create_figures(frame, tables, relationships, figures_dir)
    eda_report = build_eda_report(frame, tables, correlations)
    (args.output / "EDA_REPORT.md").write_text(eda_report, encoding="utf-8")
    split_manifest, strategy_table, prep_report = preparation_evidence(raw, tables_dir)
    (args.output / "DATA_PREPARATION_REPORT.md").write_text(prep_report, encoding="utf-8")

    after_hash = sha256(args.train)
    if before_hash != after_hash:
        raise RuntimeError("Raw train.csv hash changed during execution")
    summary = {
        "raw_rows": len(raw),
        "raw_columns": raw.shape[1],
        "eda_development_rows": len(frame),
        "eda_figures": len(list(figures_dir.glob("*.png"))),
        "tables": len(list(tables_dir.glob("*.csv"))),
        "train_rows": int(split_manifest.query("split == 'train'")["rows"].iloc[0]),
        "validation_rows": int(split_manifest.query("split == 'validation'")["rows"].iloc[0]),
        "locked_test_rows": int(split_manifest.query("split == 'test'")["rows"].iloc[0]),
        "locked_test_target_accessed": False,
        "leakage_columns_in_feature_dictionary": 0,
        "strategy_count": int(strategy_table["strategy"].nunique()),
        "raw_sha256_unchanged": before_hash == after_hash,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "chunk34_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "input": {"path_argument": str(args.train), "sha256": before_hash},
        "runtime": {
            "python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "randomness": "None used in EDA or preprocessing",
        "evidence_boundary": "Detailed EDA used train+validation only; locked test target not summarized; preprocessing fitted on train only",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
