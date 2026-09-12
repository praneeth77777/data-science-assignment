#!/usr/bin/env python3
"""PedalPulse Chunks 5–6: robust preparation and temporal modeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
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
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pedalpulse.data import LEAKAGE_COLUMNS, TARGET_COLUMN, chronological_split, safe_predictor_frame
from pedalpulse.features import CalendarFeatureEngineer
from pedalpulse.preprocessing import build_preparation_pipeline
from pedalpulse.validation import constant_predictions, expanding_quarter_folds, regression_metrics, rolling_seasonal_naive

SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hgb_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=SEED,
    )


def write_fold_manifest(raw: pd.DataFrame, tables: Path) -> pd.DataFrame:
    timestamps = pd.to_datetime(raw["datetime"])
    rows = []
    for fold in expanding_quarter_folds(raw):
        train_time = timestamps.iloc[fold.train_indices]
        validation_time = timestamps.iloc[fold.validation_indices]
        rows.append(
            {
                "fold": fold.name,
                "train_rows": len(fold.train_indices),
                "train_min": train_time.min(),
                "train_max": train_time.max(),
                "validation_rows": len(fold.validation_indices),
                "validation_min": validation_time.min(),
                "validation_max": validation_time.max(),
                "locked_test_rows_used": 0,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(tables / "temporal_fold_manifest.csv", index=False)
    return result


def domain_rule_audit(train: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("season domain", ~train["season"].isin([1, 2, 3, 4])),
        ("holiday domain", ~train["holiday"].isin([0, 1])),
        ("workingday domain", ~train["workingday"].isin([0, 1])),
        ("weather domain", ~train["weather"].isin([1, 2, 3, 4])),
        ("humidity range", ~train["humidity"].between(0, 100)),
        ("negative temperature", train["temp"] < 0),
        ("negative feels-like temperature", train["atemp"] < 0),
        ("negative windspeed", train["windspeed"] < 0),
        ("nonpositive target", train["count"] <= 0),
        ("target identity", train["count"] != train["casual"] + train["registered"]),
    ]
    return pd.DataFrame(
        {"rule": [name for name, _ in rules], "violations": [int(mask.sum()) for _, mask in rules], "action": "review; no automatic deletion"}
    )


def outlier_sensitivity(raw: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fold_rows: list[dict[str, object]] = []
    for fold in expanding_quarter_folds(raw):
        train = raw.iloc[fold.train_indices].reset_index(drop=True)
        validation = raw.iloc[fold.validation_indices].reset_index(drop=True)
        X_train = safe_predictor_frame(train)
        X_validation = safe_predictor_frame(validation)
        y_train = train["count"].to_numpy(dtype=float)
        y_validation = validation["count"].to_numpy(dtype=float)
        prep = build_preparation_pipeline("preserve_with_flags")
        train_matrix = prep.fit_transform(X_train, y_train)
        validation_matrix = prep.transform(X_validation)
        q1, q3 = np.quantile(y_train, [0.25, 0.75])
        upper_fence = float(q3 + 1.5 * (q3 - q1))
        iqr_keep = y_train <= upper_fence
        isolation = IsolationForest(n_estimators=160, contamination=0.01, random_state=SEED, n_jobs=-1)
        isolation_keep = isolation.fit_predict(train_matrix) == 1

        strategies = {
            "keep_raw": (np.ones(len(y_train), dtype=bool), y_train, "raw"),
            "cap_at_training_iqr_fence": (np.ones(len(y_train), dtype=bool), np.minimum(y_train, upper_fence), "raw"),
            "log1p_transform": (np.ones(len(y_train), dtype=bool), np.log1p(y_train), "log"),
            "remove_training_iqr_flags": (iqr_keep, y_train[iqr_keep], "raw"),
            "remove_training_isolation_flags": (isolation_keep, y_train[isolation_keep], "raw"),
        }
        for strategy, (mask, target, scale) in strategies.items():
            started = time.perf_counter()
            model = hgb_model()
            model.fit(train_matrix[mask], target)
            predictions = model.predict(validation_matrix)
            if scale == "log":
                predictions = np.expm1(predictions)
            metrics = regression_metrics(y_validation, predictions)
            fold_rows.append(
                {
                    "strategy": strategy,
                    "fold": fold.name,
                    "training_rows_before": len(y_train),
                    "training_rows_used": int(mask.sum()),
                    "training_upper_iqr_fence": upper_fence,
                    "validation_rows": len(y_validation),
                    "fit_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
    fold_table = pd.DataFrame(fold_rows)
    aggregate = fold_table.groupby("strategy", as_index=False).agg(
        folds=("fold", "count"),
        mean_training_rows_used=("training_rows_used", "mean"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        rmsle_mean=("rmsle", "mean"), rmsle_std=("rmsle", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        total_fit_seconds=("fit_seconds", "sum"),
    ).sort_values("rmsle_mean")
    fold_table.to_csv(tables / "outlier_sensitivity_by_fold.csv", index=False)
    aggregate.to_csv(tables / "outlier_sensitivity_summary.csv", index=False)
    best = aggregate.iloc[0]
    return fold_table, aggregate, {"strategy": str(best["strategy"]), "rmsle": float(best["rmsle_mean"]), "mae": float(best["mae_mean"])}


def outlier_diagnostics(train: pd.DataFrame, tables: Path, figures: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    domain = domain_rule_audit(train)
    domain.to_csv(tables / "domain_rule_audit.csv", index=False)
    y = train["count"].to_numpy(dtype=float)
    q1, q3 = np.quantile(y, [0.25, 0.75])
    upper_fence = float(q3 + 1.5 * (q3 - q1))
    iqr_flag = y > upper_fence

    X_train = safe_predictor_frame(train)
    prep = build_preparation_pipeline("preserve_with_flags")
    matrix = prep.fit_transform(X_train, y)
    isolation = IsolationForest(n_estimators=240, contamination=0.01, random_state=SEED, n_jobs=-1)
    isolation_flag = isolation.fit_predict(matrix) == -1
    scores = isolation.decision_function(matrix)
    timestamp = pd.to_datetime(train["datetime"])
    flagged = train.loc[iqr_flag, ["datetime", "count", "workingday", "weather", "temp", "humidity", "windspeed"]].copy()
    flagged["hour"] = timestamp[iqr_flag].dt.hour.to_numpy()
    flagged["iqr_flag"] = True
    flagged["isolation_forest_flag"] = isolation_flag[iqr_flag]
    flagged.to_csv(tables / "training_iqr_flagged_demand_periods.csv", index=False)

    diagnostics = pd.DataFrame(
        [
            {"method": "IQR target diagnostic", "training_rows": len(train), "flagged_rows": int(iqr_flag.sum()), "flagged_pct": float(iqr_flag.mean() * 100), "parameter": f"upper_fence={upper_fence:.3f}"},
            {"method": "Isolation Forest non-target features", "training_rows": len(train), "flagged_rows": int(isolation_flag.sum()), "flagged_pct": float(isolation_flag.mean() * 100), "parameter": "contamination=0.01"},
            {"method": "Overlap", "training_rows": len(train), "flagged_rows": int((iqr_flag & isolation_flag).sum()), "flagged_pct": float((iqr_flag & isolation_flag).mean() * 100), "parameter": "IQR AND Isolation Forest"},
        ]
    )
    diagnostics.to_csv(tables / "outlier_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].hist(y, bins=35, edgecolor="white")
    axes[0].axvline(upper_fence, color="#DC2626", linestyle="--", label=f"IQR fence {upper_fence:.1f}")
    axes[0].set(title="Training target with IQR diagnostic", xlabel="count", ylabel="Rows")
    axes[0].legend()
    axes[1].scatter(y, scores, s=8, alpha=0.35, c=np.where(isolation_flag, "#DC2626", "#2563EB"))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Isolation Forest score from non-target features", xlabel="count shown post hoc", ylabel="Decision score")
    fig.savefig(figures / "outlier_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top_hour = int(flagged.groupby("hour").size().idxmax()) if len(flagged) else -1
    evidence = {
        "upper_fence": upper_fence,
        "iqr_rows": int(iqr_flag.sum()),
        "iqr_pct": float(iqr_flag.mean() * 100),
        "isolation_rows": int(isolation_flag.sum()),
        "overlap_rows": int((iqr_flag & isolation_flag).sum()),
        "iqr_workingday_pct": float(flagged["workingday"].mean() * 100) if len(flagged) else np.nan,
        "iqr_most_common_hour": top_hour,
        "domain_rule_violations": int(domain["violations"].sum()),
    }
    return diagnostics, evidence


def cluster_matrix_fit(X: pd.DataFrame) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    engineered = CalendarFeatureEngineer().fit_transform(X)
    numeric = ["temp", "atemp", "humidity", "windspeed", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "holiday", "workingday"]
    transformer = ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("weather", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["weather"])],
        remainder="drop",
    )
    pipeline = Pipeline([("calendar", CalendarFeatureEngineer()), ("cluster_features", transformer)])
    matrix = pipeline.fit_transform(X)
    return pipeline, matrix, engineered


def cluster_experiments(train: pd.DataFrame, validation: pd.DataFrame, tables: Path, figures: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X_train = safe_predictor_frame(train)
    X_validation = safe_predictor_frame(validation)
    pipeline, matrix, engineered = cluster_matrix_fit(X_train)
    rows: list[dict[str, object]] = []
    for algorithm in ["K-Means", "Gaussian mixture"]:
        for k in range(2, 7):
            started = time.perf_counter()
            if algorithm == "K-Means":
                model = KMeans(n_clusters=k, n_init=20, random_state=SEED)
                labels = model.fit_predict(matrix)
            else:
                model = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=SEED)
                labels = model.fit_predict(matrix)
            shares = pd.Series(labels).value_counts(normalize=True)
            score = silhouette_score(matrix, labels, sample_size=min(3000, len(matrix)), random_state=SEED)
            rows.append(
                {
                    "algorithm": algorithm,
                    "k": k,
                    "silhouette_score": float(score),
                    "minimum_cluster_share": float(shares.min()),
                    "maximum_cluster_share": float(shares.max()),
                    "operational_size_rule_pass": bool(shares.min() >= 0.05),
                    "fit_and_score_seconds": time.perf_counter() - started,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "clustering_model_comparison.csv", index=False)
    eligible = comparison[comparison["operational_size_rule_pass"]]
    chosen_row = (eligible if len(eligible) else comparison).sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]
    chosen_algorithm, chosen_k = str(chosen_row["algorithm"]), int(chosen_row["k"])
    if chosen_algorithm == "K-Means":
        chosen_model = KMeans(n_clusters=chosen_k, n_init=30, random_state=SEED)
        labels = chosen_model.fit_predict(matrix)
    else:
        chosen_model = GaussianMixture(n_components=chosen_k, covariance_type="diag", n_init=5, random_state=SEED)
        labels = chosen_model.fit_predict(matrix)

    profile = engineered.copy()
    profile["cluster"] = labels
    profile["count_posthoc"] = train["count"].to_numpy()
    persona_rows = []
    for cluster, group in profile.groupby("cluster"):
        dominant_hour = int(group["hour"].mode().iloc[0])
        dominant_weekday = int(group["weekday"].mode().iloc[0])
        weekend_share = float(group["weekend"].mean())
        working_share = float(group["workingday"].mean())
        if dominant_hour <= 5:
            base_name = "Overnight conditions"
        elif 6 <= dominant_hour <= 9 and working_share >= 0.6:
            base_name = "Morning workday conditions"
        elif 16 <= dominant_hour <= 19 and working_share >= 0.6:
            base_name = "Evening workday conditions"
        elif weekend_share >= 0.5:
            base_name = "Weekend daytime conditions"
        else:
            base_name = "Mixed daytime conditions"
        persona_rows.append(
            {
                "cluster": int(cluster),
                "persona": f"{base_name} (cluster {cluster})",
                "rows": len(group),
                "share_pct": len(group) / len(profile) * 100,
                "dominant_hour": dominant_hour,
                "dominant_weekday_monday_0": dominant_weekday,
                "workingday_share_pct": working_share * 100,
                "weekend_share_pct": weekend_share * 100,
                "mean_temp": group["temp"].mean(),
                "mean_humidity": group["humidity"].mean(),
                "mean_windspeed": group["windspeed"].mean(),
                "posthoc_mean_count_not_used_for_fit": group["count_posthoc"].mean(),
                "posthoc_median_count_not_used_for_fit": group["count_posthoc"].median(),
            }
        )
    personas = pd.DataFrame(persona_rows).sort_values("cluster")
    personas.to_csv(tables / "cluster_operational_personas.csv", index=False)

    validation_matrix = pipeline.transform(X_validation)
    validation_labels = chosen_model.predict(validation_matrix)
    distribution = pd.concat(
        [
            pd.Series(labels).value_counts().rename_axis("cluster").reset_index(name="rows").assign(split="train"),
            pd.Series(validation_labels).value_counts().rename_axis("cluster").reset_index(name="rows").assign(split="validation"),
        ],
        ignore_index=True,
    )
    distribution["share_pct"] = distribution.groupby("split")["rows"].transform(lambda x: x / x.sum() * 100)
    distribution.to_csv(tables / "cluster_train_validation_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for algorithm, group in comparison.groupby("algorithm"):
        ax.plot(group["k"], group["silhouette_score"], marker="o", label=algorithm)
    ax.scatter([chosen_k], [chosen_row["silhouette_score"]], s=130, facecolors="none", edgecolors="#DC2626", linewidths=2, label="Selected")
    ax.set(title="Clustering comparison on training-only non-target features", xlabel="Number of clusters", ylabel="Silhouette score")
    ax.legend()
    fig.savefig(figures / "cluster_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    projection = PCA(n_components=2, random_state=SEED).fit_transform(matrix)
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(projection), size=min(3000, len(projection)), replace=False))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    scatter = ax.scatter(projection[sample, 0], projection[sample, 1], c=labels[sample], cmap="tab10", s=8, alpha=0.5)
    ax.set(title=f"PCA projection of selected {chosen_algorithm} solution (k={chosen_k})", xlabel="Principal component 1", ylabel="Principal component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.savefig(figures / "cluster_pca_projection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    evidence = {
        "algorithm": chosen_algorithm,
        "k": chosen_k,
        "silhouette": float(chosen_row["silhouette_score"]),
        "minimum_cluster_share": float(chosen_row["minimum_cluster_share"]),
        "fit_rows": len(train),
        "target_columns_used_for_fit": 0,
    }
    return comparison, personas, evidence


def feature_selection_experiments(train: pd.DataFrame, validation: pd.DataFrame, tables: Path, figures: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    X_train = safe_predictor_frame(train)
    X_validation = safe_predictor_frame(validation)
    y_train = train["count"].to_numpy(dtype=float)
    y_validation = validation["count"].to_numpy(dtype=float)
    prep = build_preparation_pipeline("preserve_with_flags")
    train_matrix = prep.fit_transform(X_train, y_train)
    validation_matrix = prep.transform(X_validation)
    names = np.asarray(prep.named_steps["column_transformer"].get_feature_names_out(), dtype=str)

    mi_scores = mutual_info_regression(train_matrix, y_train, random_state=SEED)
    mi_table = pd.DataFrame({"feature": names, "mutual_information": mi_scores}).sort_values("mutual_information", ascending=False)
    mi_table.to_csv(tables / "mutual_information_ranking.csv", index=False)
    mi_names = set(mi_table.head(30)["feature"])

    timestamps = pd.to_datetime(train["datetime"])
    inner_train_mask = timestamps < pd.Timestamp("2012-04-01")
    inner_validation_mask = ~inner_train_mask
    inner_prep = build_preparation_pipeline("preserve_with_flags")
    inner_X_train = safe_predictor_frame(train.loc[inner_train_mask].reset_index(drop=True))
    inner_X_validation = safe_predictor_frame(train.loc[inner_validation_mask].reset_index(drop=True))
    inner_y_train = train.loc[inner_train_mask, "count"].to_numpy(dtype=float)
    inner_y_validation = train.loc[inner_validation_mask, "count"].to_numpy(dtype=float)
    inner_train_matrix = inner_prep.fit_transform(inner_X_train, inner_y_train)
    inner_validation_matrix = inner_prep.transform(inner_X_validation)
    inner_names = np.asarray(inner_prep.named_steps["column_transformer"].get_feature_names_out(), dtype=str)
    inner_model = hgb_model().fit(inner_train_matrix, inner_y_train)
    permutation = permutation_importance(
        inner_model,
        inner_validation_matrix,
        inner_y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {"feature": inner_names, "importance_mean_mae_increase": permutation.importances_mean, "importance_std": permutation.importances_std}
    ).sort_values("importance_mean_mae_increase", ascending=False)
    permutation_table.to_csv(tables / "permutation_importance_inner_validation.csv", index=False)
    permutation_names = set(permutation_table.head(30)["feature"]).intersection(names)

    domain_names = {
        name for name in names
        if name.startswith("numeric__")
        or any(token in name for token in ["holiday_", "workingday_", "weather_", "year_", "weekend_", "rush_hour_"])
    }
    selections = {
        "all_features": set(names),
        "domain_selected": domain_names,
        "mutual_information_top30": mi_names,
        "permutation_top30": permutation_names,
        "without_weather_inputs": {name for name in names if not any(token in name for token in ["temp", "atemp", "humidity", "windspeed", "weather_"])},
        "without_temporal_inputs": {name for name in names if not any(token in name for token in ["hour", "weekday", "month", "year", "season", "weekend", "rush_hour"])},
    }
    selection_rows = []
    ablation_rows = []
    for selection, selected_names in selections.items():
        indices = np.array([position for position, name in enumerate(names) if name in selected_names], dtype=int)
        if not len(indices):
            raise RuntimeError(f"{selection} selected zero features")
        started = time.perf_counter()
        model = hgb_model().fit(train_matrix[:, indices], y_train)
        predictions = model.predict(validation_matrix[:, indices])
        metrics = regression_metrics(y_validation, predictions)
        ablation_rows.append({"selection": selection, "features": len(indices), "fit_seconds": time.perf_counter() - started, **metrics})
        for rank, index in enumerate(indices):
            selection_rows.append({"selection": selection, "position_in_preprocessor": int(index), "feature": names[index], "selection_order": rank})
    pd.DataFrame(selection_rows).to_csv(tables / "feature_selection_sets.csv", index=False)
    ablation = pd.DataFrame(ablation_rows).sort_values("rmsle")
    ablation.to_csv(tables / "feature_selection_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ordered = ablation.sort_values("rmsle", ascending=True)
    ax.barh(ordered["selection"], ordered["rmsle"])
    ax.set(title="Development validation ablation with a fixed HGB model", xlabel="RMSLE (lower is better)", ylabel="")
    fig.savefig(figures / "feature_selection_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    best = ablation.iloc[0]
    evidence = {"best_selection": str(best["selection"]), "best_rmsle": float(best["rmsle"]), "best_features": int(best["features"]), "locked_test_rows_used": 0}
    return ablation, evidence


def candidate_definitions() -> list[dict[str, object]]:
    return [
        {"candidate": "ridge_raw_a1", "family": "Ridge", "target_scale": "raw", "kind": "ridge", "alpha": 1.0},
        {"candidate": "ridge_raw_a10", "family": "Ridge", "target_scale": "raw", "kind": "ridge", "alpha": 10.0},
        {"candidate": "ridge_log_a1", "family": "Ridge", "target_scale": "log1p", "kind": "ridge", "alpha": 1.0},
        {"candidate": "ridge_log_a10", "family": "Ridge", "target_scale": "log1p", "kind": "ridge", "alpha": 10.0},
        {"candidate": "elastic_log_a0005_l10.2", "family": "Elastic Net", "target_scale": "log1p", "kind": "elastic", "alpha": 0.0005, "l1_ratio": 0.2},
        {"candidate": "elastic_log_a005_l10.5", "family": "Elastic Net", "target_scale": "log1p", "kind": "elastic", "alpha": 0.005, "l1_ratio": 0.5},
        {"candidate": "rf_raw_depth18_leaf1", "family": "Random Forest", "target_scale": "raw", "kind": "rf", "max_depth": 18, "min_samples_leaf": 1},
        {"candidate": "rf_log_depth_none_leaf3", "family": "Random Forest", "target_scale": "log1p", "kind": "rf", "max_depth": None, "min_samples_leaf": 3},
        {"candidate": "hgb_raw_leaf31_lr008", "family": "HistGradientBoosting", "target_scale": "raw", "kind": "hgb", "max_leaf_nodes": 31, "learning_rate": 0.08},
        {"candidate": "hgb_log_leaf15_lr005", "family": "HistGradientBoosting", "target_scale": "log1p", "kind": "hgb", "max_leaf_nodes": 15, "learning_rate": 0.05},
        {"candidate": "hgb_log_leaf31_lr008", "family": "HistGradientBoosting", "target_scale": "log1p", "kind": "hgb", "max_leaf_nodes": 31, "learning_rate": 0.08},
    ]


def make_estimator(definition: dict[str, object]):
    kind = definition["kind"]
    if kind == "ridge":
        return Ridge(alpha=float(definition["alpha"]))
    if kind == "elastic":
        return ElasticNet(alpha=float(definition["alpha"]), l1_ratio=float(definition["l1_ratio"]), max_iter=6000, tol=1e-4, random_state=SEED)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=180,
            max_depth=definition["max_depth"],
            min_samples_leaf=int(definition["min_samples_leaf"]),
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=240,
            max_leaf_nodes=int(definition["max_leaf_nodes"]),
            learning_rate=float(definition["learning_rate"]),
            l2_regularization=0.1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown estimator kind: {kind}")


def model_experiments(raw: pd.DataFrame, tables: Path, figures: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    result_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    definitions = candidate_definitions()
    for fold in expanding_quarter_folds(raw):
        train = raw.iloc[fold.train_indices].reset_index(drop=True)
        validation = raw.iloc[fold.validation_indices].reset_index(drop=True)
        X_train = safe_predictor_frame(train)
        X_validation = safe_predictor_frame(validation)
        y_train = train["count"].to_numpy(dtype=float)
        y_validation = validation["count"].to_numpy(dtype=float)
        prep = build_preparation_pipeline("preserve_with_flags")
        prep_started = time.perf_counter()
        train_matrix = prep.fit_transform(X_train, y_train)
        validation_matrix = prep.transform(X_validation)
        prep_seconds = time.perf_counter() - prep_started

        for statistic in ["mean", "median"]:
            predictions = constant_predictions(train, validation, statistic)
            result_rows.append(
                {
                    "candidate": f"dummy_{statistic}", "family": f"Dummy {statistic}", "target_scale": "raw", "fold": fold.name,
                    "train_rows": len(train), "validation_rows": len(validation), "output_features": 0, "preprocessing_seconds": 0.0,
                    "fit_seconds": 0.0, "parameters": json.dumps({"statistic": statistic}), **regression_metrics(y_validation, predictions),
                }
            )
        seasonal_predictions, tiers = rolling_seasonal_naive(train, validation)
        result_rows.append(
            {
                "candidate": "seasonal_naive", "family": "Seasonal naive", "target_scale": "raw", "fold": fold.name,
                "train_rows": len(train), "validation_rows": len(validation), "output_features": 0, "preprocessing_seconds": 0.0,
                "fit_seconds": 0.0, "parameters": json.dumps({"hierarchy": ["t-168", "t-24", "hour-workingday median", "global median"]}),
                **regression_metrics(y_validation, seasonal_predictions),
            }
        )
        tier_counts = pd.Series(tiers).value_counts()
        for tier, count in tier_counts.items():
            tier_rows.append({"fold": fold.name, "tier": tier, "rows": int(count), "share_pct": count / len(tiers) * 100})

        for definition in definitions:
            estimator = make_estimator(definition)
            target = np.log1p(y_train) if definition["target_scale"] == "log1p" else y_train
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(train_matrix, target)
            predictions = estimator.predict(validation_matrix)
            if definition["target_scale"] == "log1p":
                predictions = np.expm1(predictions)
            parameters = {key: value for key, value in definition.items() if key not in {"candidate", "family", "target_scale"}}
            result_rows.append(
                {
                    "candidate": definition["candidate"], "family": definition["family"], "target_scale": definition["target_scale"], "fold": fold.name,
                    "train_rows": len(train), "validation_rows": len(validation), "output_features": train_matrix.shape[1],
                    "preprocessing_seconds": prep_seconds, "fit_seconds": time.perf_counter() - started,
                    "warning_count": len(caught), "parameters": json.dumps(parameters, sort_keys=True),
                    **regression_metrics(y_validation, predictions),
                }
            )
    fold_results = pd.DataFrame(result_rows)
    aggregate = fold_results.groupby(["candidate", "family", "target_scale"], as_index=False).agg(
        folds=("fold", "count"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        rmsle_mean=("rmsle", "mean"), rmsle_std=("rmsle", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        total_fit_seconds=("fit_seconds", "sum"), warning_count=("warning_count", "sum"),
    ).sort_values("rmsle_mean")
    winner_indices = aggregate.groupby("family")["rmsle_mean"].idxmin()
    family_winners = aggregate.loc[winner_indices].sort_values("rmsle_mean").reset_index(drop=True)
    tier_table = pd.DataFrame(tier_rows)
    fold_results.to_csv(tables / "model_metrics_by_fold.csv", index=False)
    aggregate.to_csv(tables / "model_candidate_summary.csv", index=False)
    family_winners.to_csv(tables / "model_family_winners.csv", index=False)
    tier_table.to_csv(tables / "seasonal_naive_fallback_coverage.csv", index=False)

    learned = aggregate[~aggregate["family"].str.startswith("Dummy") & (aggregate["family"] != "Seasonal naive")]
    winner = learned.iloc[0]
    seasonal = aggregate[aggregate["candidate"] == "seasonal_naive"].iloc[0]
    winner_folds = fold_results[fold_results["candidate"] == winner["candidate"]].set_index("fold")
    seasonal_folds = fold_results[fold_results["candidate"] == "seasonal_naive"].set_index("fold")
    rmsle_improvement = (seasonal["rmsle_mean"] - winner["rmsle_mean"]) / seasonal["rmsle_mean"] * 100
    mae_improvement = (seasonal["mae_mean"] - winner["mae_mean"]) / seasonal["mae_mean"] * 100
    criteria = pd.DataFrame(
        [
            {"criterion": "Mean RMSLE improvement vs seasonal naive", "required": ">= 10%", "measured": rmsle_improvement, "passed": rmsle_improvement >= 10},
            {"criterion": "Mean MAE improvement vs seasonal naive", "required": ">= 5%", "measured": mae_improvement, "passed": mae_improvement >= 5},
            {"criterion": "RMSLE better in majority of folds", "required": ">= 3 of 5", "measured": int((winner_folds["rmsle"] < seasonal_folds["rmsle"]).sum()), "passed": int((winner_folds["rmsle"] < seasonal_folds["rmsle"]).sum()) >= 3},
        ]
    )
    criteria.to_csv(tables / "development_success_criteria.csv", index=False)
    best_config = next(item for item in definitions if item["candidate"] == winner["candidate"])
    (tables / "selected_candidate_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    plot_table = family_winners.sort_values("rmsle_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(plot_table["family"], plot_table["rmsle_mean"], xerr=plot_table["rmsle_std"], capsize=3)
    ax.set(title="Temporal cross-validation: best candidate in each family", xlabel="Mean RMSLE ± fold standard deviation", ylabel="")
    fig.savefig(figures / "model_family_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_candidates = ["seasonal_naive", str(winner["candidate"])]
    if family_winners.iloc[0]["candidate"] not in selected_candidates:
        selected_candidates.append(str(family_winners.iloc[0]["candidate"]))
    trend = fold_results[fold_results["candidate"].isin(selected_candidates)]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for candidate, group in trend.groupby("candidate"):
        order = group.sort_values("fold")
        ax.plot(order["fold"], order["rmsle"], marker="o", label=candidate)
    ax.set(title="RMSLE across expanding temporal folds", xlabel="Fold", ylabel="RMSLE")
    ax.legend()
    fig.savefig(figures / "fold_rmsle_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    evidence = {
        "winner_candidate": str(winner["candidate"]),
        "winner_family": str(winner["family"]),
        "winner_target_scale": str(winner["target_scale"]),
        "winner_rmsle_mean": float(winner["rmsle_mean"]),
        "winner_rmsle_std": float(winner["rmsle_std"]),
        "winner_mae_mean": float(winner["mae_mean"]),
        "winner_rmse_mean": float(winner["rmse_mean"]),
        "winner_r2_mean": float(winner["r2_mean"]),
        "seasonal_rmsle_mean": float(seasonal["rmsle_mean"]),
        "seasonal_mae_mean": float(seasonal["mae_mean"]),
        "rmsle_improvement_pct": float(rmsle_improvement),
        "mae_improvement_pct": float(mae_improvement),
        "rmsle_folds_won": int((winner_folds["rmsle"] < seasonal_folds["rmsle"]).sum()),
        "criteria_all_passed": bool(criteria["passed"].all()),
        "locked_test_rows_used": 0,
    }
    return fold_results, aggregate, family_winners, evidence


def make_reports(
    output: Path,
    outlier: dict[str, object],
    outlier_best: dict[str, float],
    cluster: dict[str, object],
    personas: pd.DataFrame,
    feature: dict[str, object],
    modeling: dict[str, object],
) -> None:
    persona_text = "\n".join(
        f"- **{row.persona}:** {row.share_pct:.1f}% of training periods; dominant hour {int(row.dominant_hour):02d}:00; post-hoc mean count {row.posthoc_mean_count_not_used_for_fit:.1f}."
        for row in personas.itertuples()
    )
    robust_report = f"""# PedalPulse Outliers, Clustering, and Feature Selection

**CRISP-DM phase:** Data Preparation / Modeling iteration
**Locked-test usage:** 0 rows. All decisions use training or development validation only.

## Outlier evidence

The training-only demand IQR upper fence is **{outlier['upper_fence']:.2f}**. It flags **{outlier['iqr_rows']} rows ({outlier['iqr_pct']:.2f}%)**. The most common flagged hour is **{int(outlier['iqr_most_common_hour']):02d}:00**, and **{outlier['iqr_workingday_pct']:.1f}%** of flagged periods are working days. This timing is consistent with legitimate demand peaks, not proof of invalid records. Domain-rule violations were **{outlier['domain_rule_violations']}**.

Isolation Forest, fitted only on non-target features with 1% contamination, flagged **{outlier['isolation_rows']}** training periods; only **{outlier['overlap_rows']}** overlapped the target-IQR flags. Its contamination level is a diagnostic operating point, not an estimated true anomaly rate.

![Outlier diagnostics](figures/outlier_diagnostics.png)

The best mean RMSLE in the fixed-model sensitivity table was produced by **{outlier_best['strategy']}** ({outlier_best['rmsle']:.4f}; MAE {outlier_best['mae']:.2f}). This does not authorize deletion: keeping, capping, log-transforming, and removal strategies change the learning problem, and the untouched test has not been evaluated.

![Outlier sensitivity](figures/outlier_sensitivity.png)

## Clustering

K-Means and diagonal-covariance Gaussian mixtures were compared for k=2–6 using training-only, standardized, non-target features. The operational size rule required every cluster to contain at least 5% of training periods. The selected solution is **{cluster['algorithm']} with k={cluster['k']}**, silhouette **{cluster['silhouette']:.4f}**, and minimum cluster share **{cluster['minimum_cluster_share'] * 100:.1f}%**.

`count` was not used to fit or select clusters. Post-hoc demand summaries are included only to translate segments into operational language.

{persona_text}

![Cluster selection](figures/cluster_selection.png)

![Cluster PCA](figures/cluster_pca_projection.png)

PCA is a two-dimensional projection for visualization; overlap in the plot does not invalidate higher-dimensional separation.

## Feature selection and ablation

Domain-selected features, training-only mutual information, and inner-temporal-validation permutation importance were compared. The best fixed-HGB development-validation subset was **{feature['best_selection']}** with **{feature['best_features']} features** and RMSLE **{feature['best_rmsle']:.4f}**.

![Feature-selection ablation](figures/feature_selection_ablation.png)

This is development evidence, not final generalization evidence. Selection rankings were never computed on the locked test.
"""
    (output / "CHUNK5_REPORT.md").write_text(robust_report, encoding="utf-8")

    criteria = "passed" if modeling["criteria_all_passed"] else "not fully passed"
    model_report = f"""# PedalPulse Baselines and Supervised Modeling

**CRISP-DM phase:** Modeling
**Validation:** Five expanding quarterly folds; no random shuffling.
**Locked-test usage:** 0 rows. Final test evaluation remains deferred to Chunk 7.

## Selected development candidate

The best learned candidate by mean RMSLE is **{modeling['winner_candidate']}** ({modeling['winner_family']}, target scale `{modeling['winner_target_scale']}`).

| Metric | Five-fold mean |
|---|---:|
| MAE | {modeling['winner_mae_mean']:.3f} |
| RMSE | {modeling['winner_rmse_mean']:.3f} |
| RMSLE | {modeling['winner_rmsle_mean']:.4f} |
| R² | {modeling['winner_r2_mean']:.4f} |

RMSLE fold standard deviation is **{modeling['winner_rmsle_std']:.4f}**. Fold-level results, not only averages, are retained in the tables.

## Seasonal-naive comparison

The rolling one-hour-ahead seasonal baseline achieved mean RMSLE **{modeling['seasonal_rmsle_mean']:.4f}** and MAE **{modeling['seasonal_mae_mean']:.3f}**. The selected candidate improved mean RMSLE by **{modeling['rmsle_improvement_pct']:.2f}%** and mean MAE by **{modeling['mae_improvement_pct']:.2f}%**, and beat seasonal-naive RMSLE in **{modeling['rmsle_folds_won']} of 5 folds**. The predeclared development criteria are **{criteria}**.

![Model-family comparison](figures/model_family_comparison.png)

![Fold RMSLE comparison](figures/fold_rmsle_comparison.png)

## Interpretation

- Mean and median dummy baselines, seasonal naive, Ridge, Elastic Net, Random Forest, and HistGradientBoosting were executed.
- Small candidate grids were used; exhaustive tuning was deliberately avoided.
- Raw and `log1p` targets were compared where appropriate, with `expm1` used for log-scale predictions.
- Predictions were clipped to zero before metrics because negative rental counts are operationally invalid.
- XGBoost was skipped because it was not installed; no package was silently added.
- The development winner is a candidate for Chunk 7, not yet a final tested model. Complexity alone was not used as a selection rule.
"""
    (output / "CHUNK6_MODELING_REPORT.md").write_text(model_report, encoding="utf-8")


def create_outlier_sensitivity_figure(aggregate: pd.DataFrame, figures: Path) -> None:
    ordered = aggregate.sort_values("rmsle_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(ordered["strategy"], ordered["rmsle_mean"], xerr=ordered["rmsle_std"], capsize=3)
    axes[0].set(title="Outlier-treatment sensitivity", xlabel="Mean RMSLE ± fold SD")
    axes[1].barh(ordered["strategy"], ordered["mae_mean"], xerr=ordered["mae_std"], capsize=3)
    axes[1].set(title="Same experiments on MAE", xlabel="Mean MAE ± fold SD")
    fig.savefig(figures / "outlier_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def memory_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    before_hash = sha256(args.train)
    raw = pd.read_csv(args.train)
    splits = chronological_split(raw)
    train, validation = splits["train"], splits["validation"]
    write_fold_manifest(raw, tables)

    _, outlier_evidence = outlier_diagnostics(train, tables, figures)
    _, outlier_aggregate, outlier_best = outlier_sensitivity(raw, tables)
    create_outlier_sensitivity_figure(outlier_aggregate, figures)
    _, personas, cluster_evidence = cluster_experiments(train, validation, tables, figures)
    _, feature_evidence = feature_selection_experiments(train, validation, tables, figures)
    _, _, _, modeling_evidence = model_experiments(raw, tables, figures)
    make_reports(args.output, outlier_evidence, outlier_best, cluster_evidence, personas, feature_evidence, modeling_evidence)

    after_hash = sha256(args.train)
    if before_hash != after_hash:
        raise RuntimeError("Raw train.csv changed during experiments")
    runtime = time.perf_counter() - started
    summary = {
        "raw_rows": len(raw),
        "development_rows": len(train) + len(validation),
        "locked_test_rows": len(splits["test"]),
        "locked_test_target_accessed": False,
        "temporal_folds": 5,
        "outlier_strategies": 5,
        "clustering_candidates": 10,
        "feature_selection_variants": 6,
        "supervised_candidates": len(candidate_definitions()),
        "xgboost_status": "skipped: package unavailable",
        "selected_cluster": cluster_evidence,
        "selected_feature_subset": feature_evidence,
        "selected_model": modeling_evidence,
        "raw_sha256_unchanged": before_hash == after_hash,
        "runtime_seconds": runtime,
    }
    (args.output / "chunk56_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "input_sha256": before_hash,
        "random_seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_gib": memory_gib(),
        "versions": {
            "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "xgboost": "not installed; experiment skipped",
        "locked_test_target_accessed": False,
        "runtime_seconds": runtime,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
