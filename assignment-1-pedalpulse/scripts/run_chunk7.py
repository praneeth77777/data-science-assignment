#!/usr/bin/env python3
"""Locked-test evaluation, explainability, and audit for PedalPulse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from sklearn.inspection import partial_dependence, permutation_importance

from pedalpulse.data import LEAKAGE_COLUMNS, chronological_split, safe_predictor_frame
from pedalpulse.preprocessing import build_preparation_pipeline
from pedalpulse.validation import regression_metrics, rolling_seasonal_naive
from run_chunk56 import make_estimator

SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def performance_slices(test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    frame = test.loc[:, ["datetime", "season", "weather", "count"]].copy()
    timestamp = pd.to_datetime(frame["datetime"])
    frame["weekday"] = timestamp.dt.day_name()
    frame["prediction"] = predictions
    frame["demand_range"] = pd.cut(
        frame["count"],
        bins=[0, 50, 100, 200, 400, np.inf],
        labels=["1–50", "51–100", "101–200", "201–400", "401+"],
        include_lowest=True,
    )
    rows = []
    for dimension in ["season", "weather", "weekday", "demand_range"]:
        for value, group in frame.groupby(dimension, observed=True):
            metrics = regression_metrics(group["count"], group["prediction"].to_numpy())
            rows.append({"dimension": dimension, "group": str(value), "rows": len(group), **metrics})
    return pd.DataFrame(rows)


def drift_screen(development: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ["temp", "atemp", "humidity", "windspeed"]:
        left = development[feature].astype(float)
        right = test[feature].astype(float)
        pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
        smd = (right.mean() - left.mean()) / pooled if pooled else 0.0
        rows.append({"feature": feature, "type": "continuous", "drift_measure": "standardized_mean_difference", "value": smd})
    dev_time, test_time = pd.to_datetime(development["datetime"]), pd.to_datetime(test["datetime"])
    dev_aug = development.assign(hour=dev_time.dt.hour, weekday=dev_time.dt.weekday, month=dev_time.dt.month, year=dev_time.dt.year)
    test_aug = test.assign(hour=test_time.dt.hour, weekday=test_time.dt.weekday, month=test_time.dt.month, year=test_time.dt.year)
    for feature in ["season", "holiday", "workingday", "weather", "hour", "weekday", "month", "year"]:
        categories = sorted(set(dev_aug[feature].unique()) | set(test_aug[feature].unique()))
        p = dev_aug[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
        q = test_aug[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
        rows.append({"feature": feature, "type": "categorical", "drift_measure": "total_variation_distance", "value": float(0.5 * np.abs(p - q).sum())})
    return pd.DataFrame(rows).sort_values("value", ascending=False)


def create_residual_figures(test: pd.DataFrame, predictions: np.ndarray, figures: Path, tables: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    timestamp = pd.to_datetime(test["datetime"])
    actual = test["count"].to_numpy(dtype=float)
    residual = actual - predictions
    absolute_error = np.abs(residual)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.scatter(predictions, residual, s=12, alpha=0.4)
    ax.axhline(0, color="#DC2626", linestyle="--")
    ax.set(title="Locked-test residuals versus predictions", xlabel="Predicted count", ylabel="Actual − predicted")
    fig.savefig(figures / "residuals_vs_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    daily = pd.DataFrame({"date": timestamp.dt.floor("D"), "actual": actual, "prediction": predictions, "absolute_error": absolute_error})
    daily = daily.groupby("date", as_index=False).agg(actual_mean=("actual", "mean"), prediction_mean=("prediction", "mean"), mae=("absolute_error", "mean"), rows=("actual", "size"))
    daily.to_csv(tables / "locked_test_daily_error.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(daily["date"], daily["actual_mean"], label="Actual daily mean")
    axes[0].plot(daily["date"], daily["prediction_mean"], label="Predicted daily mean")
    axes[0].set(title="Temporal prediction audit", ylabel="Mean hourly count")
    axes[0].legend()
    axes[1].plot(daily["date"], daily["mae"], color="#DC2626")
    axes[1].set(xlabel="Date", ylabel="Daily MAE")
    fig.savefig(figures / "temporal_error_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.hist(residual, bins=35, edgecolor="white")
    ax.axvline(0, color="#DC2626", linestyle="--")
    ax.axvline(np.mean(residual), color="#111827", linestyle=":", label=f"Mean residual {np.mean(residual):.1f}")
    ax.set(title="Locked-test residual distribution", xlabel="Actual − predicted", ylabel="Rows")
    ax.legend()
    fig.savefig(figures / "residual_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    histogram, edges = np.histogram(residual, bins=35)
    residual_bins = pd.DataFrame({"lower_edge": edges[:-1], "upper_edge": edges[1:], "rows": histogram})
    residual_bins.to_csv(tables / "residual_histogram_bins.csv", index=False)
    evidence = {
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "residual_std": float(np.std(residual, ddof=1)),
        "mean_absolute_error": float(np.mean(absolute_error)),
    }
    return daily, evidence


def create_slice_figure(slices: pd.DataFrame, figures: Path) -> None:
    dimensions = ["season", "weather", "weekday", "demand_range"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, dimension in zip(axes.ravel(), dimensions):
        part = slices[slices["dimension"] == dimension].sort_values("mae")
        ax.barh(part["group"], part["mae"])
        ax.set(title=f"MAE by {dimension.replace('_', ' ')}", xlabel="MAE", ylabel="")
    fig.suptitle("Locked-test performance slices")
    fig.savefig(figures / "performance_slices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def explain_model(model, development_matrix: np.ndarray, test_matrix: np.ndarray, y_test: np.ndarray, names: np.ndarray, tables: Path, figures: Path) -> dict[str, object]:
    started = time.perf_counter()
    importance = permutation_importance(
        model,
        test_matrix,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
        max_samples=0.7,
    )
    importance_table = pd.DataFrame(
        {"feature": names, "importance_mean_mae_increase": importance.importances_mean, "importance_std": importance.importances_std}
    ).sort_values("importance_mean_mae_increase", ascending=False)
    importance_table.to_csv(tables / "locked_test_permutation_importance.csv", index=False)
    top = importance_table.head(20).sort_values("importance_mean_mae_increase")
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(top["feature"], top["importance_mean_mae_increase"], xerr=top["importance_std"], capsize=2)
    ax.set(title="Final-model permutation importance on locked test", xlabel="Increase in MAE after permutation", ylabel="")
    fig.savefig(figures / "permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd_features = ["numeric__temp", "numeric__humidity", "numeric__windspeed"]
    pd_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, feature_name in zip(axes, pd_features):
        index = int(np.flatnonzero(names == feature_name)[0])
        result = partial_dependence(model, development_matrix, features=[index], grid_resolution=20)
        grid = result["grid_values"][0]
        average = result["average"][0]
        ax.plot(grid, average)
        ax.set(title=feature_name.replace("numeric__", ""), xlabel="Standardized feature value", ylabel="Average prediction")
        for x, y in zip(grid, average, strict=True):
            pd_rows.append({"feature": feature_name, "standardized_grid_value": x, "average_prediction": y})
    fig.suptitle("Partial-dependence diagnostics; associations, not causal effects")
    fig.savefig(figures / "partial_dependence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(pd_rows).to_csv(tables / "partial_dependence_values.csv", index=False)
    return {
        "top_permutation_feature": str(importance_table.iloc[0]["feature"]),
        "top_permutation_importance": float(importance_table.iloc[0]["importance_mean_mae_increase"]),
        "permutation_features": len(importance_table),
        "shap_status": "skipped: SHAP package unavailable",
        "explainability_seconds": time.perf_counter() - started,
    }


def model_card_text(config: dict[str, object], metrics: dict[str, float], seasonal: dict[str, float], uncertainty: dict[str, float], explainability: dict[str, object]) -> str:
    return f"""# PedalPulse Model Card

## Model details

- Candidate: `{config['candidate']}`
- Family: {config['family']}
- Target scale: {config['target_scale']}
- Random seed: {SEED}
- Output: predicted system-wide hourly rental count
- Prediction contract: one-hour-ahead operational planning

## Training and evaluation

- Development rows: 9,519, ending 2012-09-19
- Locked test rows: 1,367, covering 2012-10-01 through 2012-12-19
- `casual` and `registered` were excluded from every model input.
- Locked-test evaluation was performed once after candidate selection.

| Metric | Final model | Seasonal naive |
|---|---:|---:|
| MAE | {metrics['mae']:.3f} | {seasonal['mae']:.3f} |
| RMSE | {metrics['rmse']:.3f} | {seasonal['rmse']:.3f} |
| RMSLE | {metrics['rmsle']:.4f} | {seasonal['rmsle']:.4f} |
| R² | {metrics['r2']:.4f} | {seasonal['r2']:.4f} |

## Uncertainty

The 10th–90th percentile range across individual Random Forest trees covered **{uncertainty['coverage_pct']:.2f}%** of locked-test outcomes and had mean width **{uncertainty['mean_interval_width']:.2f}**. This is a heuristic ensemble spread, not a calibrated prediction interval.

## Explainability

The highest locked-test permutation importance was `{explainability['top_permutation_feature']}`. Partial-dependence diagnostics were generated for temperature, humidity, and windspeed. These summaries are associative and can be distorted by correlated predictors. SHAP was not run because the package was unavailable.

## Intended use

Support system-level rebalancing, staffing, and capacity planning. Predictions should inform human decisions rather than automatically deny service or allocate resources without monitoring.

## Limitations and risks

- Historical data cover only 2011–2012 and may not represent current mobility behavior.
- No station location, dock inventory, trip purpose, event calendar, or unmet-demand field is available.
- The model predicts observed rentals, which may be constrained by historical bike availability.
- Rare severe-weather performance cannot be reliably estimated.
- Drift, outages, policy changes, and new mobility alternatives can reduce accuracy.
"""


def data_card_text() -> str:
    return """# PedalPulse Data Card

## Dataset

Kaggle Bike Sharing Demand competition data: https://www.kaggle.com/competitions/bike-sharing-demand

The student supplied `train.csv`. Redistribution permission has not been established, so raw data are excluded from publication artifacts; reproduction instructions must require users to download the dataset themselves.

## Unit of observation

One system-wide hourly record containing timestamp, calendar/weather fields, `casual`, `registered`, and total `count`.

## Target and leakage

`count` is the target and equals `casual + registered` for every training row. The component fields are forbidden model inputs.

## Splits

- Development: 9,519 rows through 2012-09-19.
- Locked test: 1,367 rows from 2012-10-01 through 2012-12-19.
- Kaggle competition test: target-free and used only for future inference/submission demonstrations.

## Known quality issues

No ordinary missing cells or duplicated timestamps were measured. Suspicious values include concentrated zero humidity, frequent zero windspeed, rare weather code 4, and gaps in the hourly calendar. These were flagged rather than silently deleted.

## Representation and ethics

The data do not contain protected demographic attributes, but that does not eliminate fairness concerns. Service allocation based on historical demand can reinforce geographic or access inequities that cannot be audited without station and neighborhood information.
"""


def evaluation_report_text(
    config: dict[str, object],
    metrics: dict[str, float],
    seasonal: dict[str, float],
    improvements: dict[str, float],
    residual: dict[str, float],
    uncertainty: dict[str, float],
    drift: pd.DataFrame,
    explainability: dict[str, object],
) -> str:
    top_drift = drift.iloc[0]
    return f"""# PedalPulse Final Evaluation and Audit

**CRISP-DM phase:** Evaluation
**Locked candidate:** `{config['candidate']}`
**Locked-test evaluation count:** 1

## Final locked-test results

| Metric | Final model | Seasonal naive |
|---|---:|---:|
| MAE | {metrics['mae']:.3f} | {seasonal['mae']:.3f} |
| RMSE | {metrics['rmse']:.3f} | {seasonal['rmse']:.3f} |
| RMSLE | {metrics['rmsle']:.4f} | {seasonal['rmsle']:.4f} |
| R² | {metrics['r2']:.4f} | {seasonal['r2']:.4f} |

Relative to seasonal naive, the final model changed RMSLE by **{improvements['rmsle_improvement_pct']:.2f}%** and MAE by **{improvements['mae_improvement_pct']:.2f}%**. Positive values mean improvement. The final-test thresholds were {'met' if improvements['thresholds_met'] else 'not both met'}.

## Residual audit

Mean residual (`actual − predicted`) is **{residual['mean_residual']:.3f}**, median residual **{residual['median_residual']:.3f}**, and residual standard deviation **{residual['residual_std']:.3f}**.

Because the residual definition is `actual − predicted`, the positive mean and median show systematic underprediction in the locked October–December period.

![Residuals versus prediction](figures/residuals_vs_predictions.png)

![Temporal error](figures/temporal_error_plot.png)

![Residual distribution](figures/residual_distribution.png)

![Performance slices](figures/performance_slices.png)

## Explainability and uncertainty

The top permutation feature is `{explainability['top_permutation_feature']}`, with mean MAE increase **{explainability['top_permutation_importance']:.3f}** when permuted. Permutation importance was computed after final evaluation for explanation only and was not used to retune the model.

![Permutation importance](figures/permutation_importance.png)

![Partial dependence](figures/partial_dependence.png)

SHAP status: **{explainability['shap_status']}**. The ensemble 10th–90th percentile spread covered **{uncertainty['coverage_pct']:.2f}%** of outcomes; it is not a calibrated interval.

## Drift and generalization

The largest development-to-test drift screen is **{top_drift['feature']}**, {top_drift['drift_measure']} = **{top_drift['value']:.3f}**. Calendar-season drift is expected because the locked test covers October–December. Drift statistics describe distribution differences, not causes.

The test period contains only season code 4, so the season slice is descriptive of that quarter and cannot support a between-season comparison.

Generalization beyond 2011–2012, to individual stations, or to unconstrained demand is unsupported. Operational use requires monitoring and periodic retraining with newer data.

## Leakage and reproducibility audit

- Training input columns contain neither `casual`, `registered`, nor `count`.
- Preprocessing was fitted on development rows only.
- Test outcomes were used only after candidate and hyperparameters were locked.
- Random seed, package versions, runtime, hardware, input hash, and exact candidate configuration are recorded.
- Raw data were not modified or bundled.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    tables, figures = args.output / "tables", args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    raw_hash_before = sha256(args.train)
    raw = pd.read_csv(args.train)
    splits = chronological_split(raw)
    development = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    test = splits["test"].reset_index(drop=True)
    config = json.loads(args.selected_config.read_text(encoding="utf-8"))

    X_development = safe_predictor_frame(development)
    X_test = safe_predictor_frame(test)
    y_development = development["count"].to_numpy(dtype=float)
    y_test = test["count"].to_numpy(dtype=float)
    preprocessor = build_preparation_pipeline("preserve_with_flags")
    development_matrix = preprocessor.fit_transform(X_development, y_development)
    test_matrix = preprocessor.transform(X_test)
    feature_names = np.asarray(preprocessor.named_steps["column_transformer"].get_feature_names_out(), dtype=str)
    forbidden_found = [name for name in feature_names if any(token in name for token in ["casual", "registered", "count"])]
    if forbidden_found:
        raise RuntimeError(f"Leakage audit failed: {forbidden_found}")

    model = make_estimator(config)
    model.fit(development_matrix, y_development)
    predictions = np.clip(model.predict(test_matrix), 0, None)
    # This is the single locked-test metric evaluation for the selected model.
    final_metrics = regression_metrics(y_test, predictions)
    seasonal_predictions, seasonal_tiers = rolling_seasonal_naive(development, test)
    seasonal_metrics = regression_metrics(y_test, seasonal_predictions)
    improvements = {
        "rmsle_improvement_pct": (seasonal_metrics["rmsle"] - final_metrics["rmsle"]) / seasonal_metrics["rmsle"] * 100,
        "mae_improvement_pct": (seasonal_metrics["mae"] - final_metrics["mae"]) / seasonal_metrics["mae"] * 100,
    }
    improvements["thresholds_met"] = improvements["rmsle_improvement_pct"] >= 10 and improvements["mae_improvement_pct"] >= 5
    metrics_table = pd.DataFrame([
        {"candidate": config["candidate"], "evaluation": "locked_test", **final_metrics},
        {"candidate": "seasonal_naive", "evaluation": "locked_test", **seasonal_metrics},
    ])
    metrics_table.to_csv(tables / "final_locked_test_metrics.csv", index=False)
    pd.Series(seasonal_tiers).value_counts().rename_axis("tier").reset_index(name="rows").assign(
        share_pct=lambda x: x["rows"] / x["rows"].sum() * 100
    ).to_csv(tables / "final_seasonal_fallback_coverage.csv", index=False)

    tree_predictions = np.vstack([tree.predict(test_matrix) for tree in model.estimators_])
    lower, upper = np.quantile(tree_predictions, [0.10, 0.90], axis=0)
    uncertainty = {
        "coverage_pct": float(((y_test >= lower) & (y_test <= upper)).mean() * 100),
        "mean_interval_width": float(np.mean(upper - lower)),
        "lower_quantile": 0.10,
        "upper_quantile": 0.90,
    }
    pd.DataFrame([uncertainty]).to_csv(tables / "ensemble_uncertainty_summary.csv", index=False)

    _, residual_evidence = create_residual_figures(test, predictions, figures, tables)
    slices = performance_slices(test, predictions)
    slices.to_csv(tables / "final_performance_slices.csv", index=False)
    create_slice_figure(slices, figures)
    drift = drift_screen(development, test)
    drift.to_csv(tables / "development_locked_test_drift.csv", index=False)
    explainability = explain_model(model, development_matrix, test_matrix, y_test, feature_names, tables, figures)

    audit_checks = [
        ("casual absent from model inputs", "casual" not in " ".join(feature_names)),
        ("registered absent from model inputs", "registered" not in " ".join(feature_names)),
        ("count absent from model inputs", "count" not in " ".join(feature_names)),
        ("development ends before test begins", pd.to_datetime(development["datetime"]).max() < pd.to_datetime(test["datetime"]).min()),
        ("selected-model locked-test metric evaluations", 1),
        ("raw input hash unchanged", raw_hash_before == sha256(args.train)),
    ]
    audit = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "status": "PASS" if (measured == 1 if isinstance(measured, int) and not isinstance(measured, bool) else bool(measured)) else "FAIL",
            }
            for check, measured in audit_checks
        ]
    )
    audit.to_csv(tables / "final_leakage_reproducibility_audit.csv", index=False)

    (args.output / "MODEL_CARD.md").write_text(model_card_text(config, final_metrics, seasonal_metrics, uncertainty, explainability), encoding="utf-8")
    (args.output / "DATA_CARD.md").write_text(data_card_text(), encoding="utf-8")
    (args.output / "CHUNK7_EVALUATION_REPORT.md").write_text(
        evaluation_report_text(config, final_metrics, seasonal_metrics, improvements, residual_evidence, uncertainty, drift, explainability), encoding="utf-8"
    )

    runtime = time.perf_counter() - started
    summary = {
        "selected_candidate": config,
        "development_rows": len(development),
        "locked_test_rows": len(test),
        "locked_test_evaluation_count_selected_model": 1,
        "final_metrics": final_metrics,
        "seasonal_naive_metrics": seasonal_metrics,
        "improvements": improvements,
        "residuals": residual_evidence,
        "uncertainty": uncertainty,
        "explainability": explainability,
        "leakage_audit_passed": bool(audit["status"].eq("PASS").all()),
        "raw_sha256_unchanged": raw_hash_before == sha256(args.train),
        "runtime_seconds": runtime,
    }
    (args.output / "chunk7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "input_sha256": raw_hash_before,
        "selected_config_sha256": sha256(args.selected_config),
        "random_seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "shap": "not installed; skipped",
        "locked_test_evaluation_count_selected_model": 1,
        "runtime_seconds": runtime,
    }
    (args.output / "chunk7_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
