"""Bounded, auditable hill-climbing search for clustering configurations."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import (
    ExperimentState,
    FeaturePreprocessor,
    cluster_metrics,
    centroids,
    fit_clusterer,
    stability_score,
)


def load_policy(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    required = {
        "master_seed", "max_trials", "patience", "min_delta", "cluster_counts",
        "algorithms", "feature_views", "transformations", "scalers",
        "outlier_policies", "stability_seeds", "weights",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"AutoResearch policy missing keys: {missing}")
    return policy


def state_key(state: ExperimentState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True)


def state_from_row(row: dict | pd.Series) -> ExperimentState:
    """Convert pandas/NumPy scalar values back to JSON-safe Python types."""
    return ExperimentState(
        feature_view=str(row["feature_view"]),
        transformation=str(row["transformation"]),
        scaler=str(row["scaler"]),
        algorithm=str(row["algorithm"]),
        n_clusters=int(row["n_clusters"]),
        outlier_policy=str(row["outlier_policy"]),
    )


def _interpretability(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    centers = centroids(matrix, labels)
    ranges = np.ptp(centers, axis=0)
    differentiated = int((ranges >= 0.75).sum())
    return float(min(1.0, differentiated / 2.0)), differentiated


def evaluate_state(
    frame: pd.DataFrame,
    state: ExperimentState,
    policy: dict,
    *,
    trial_id: int,
    parent_trial_id: int | None,
    mutation: str,
) -> dict:
    started = time.perf_counter()
    row = {
        "trial_id": trial_id,
        "parent_trial_id": parent_trial_id,
        "mutation": mutation,
        **state.to_dict(),
        "status": "failed",
        "rejection_reason": "",
    }
    try:
        preprocessor = FeaturePreprocessor(state)
        matrix = preprocessor.fit_transform(frame)
        model, labels = fit_clusterer(state, matrix, seed=policy["master_seed"])
        metrics = cluster_metrics(
            matrix, labels,
            sample_size=policy["silhouette_sample"],
            seed=policy["master_seed"],
        )
        stability_mean, stability_sd = stability_score(
            frame, state,
            seeds=policy["stability_seeds"],
            fraction=policy["stability_fraction"],
            master_seed=policy["master_seed"],
        )
        interpretability, differentiated = _interpretability(matrix, labels)
        counts = pd.Series(labels).value_counts()
        reasons = []
        if metrics["minimum_cluster_share"] < policy["minimum_cluster_share"]:
            reasons.append("cluster share below minimum")
        if int(counts.min()) < policy["minimum_cluster_customers"]:
            reasons.append("cluster customer count below minimum")
        if metrics["maximum_cluster_share"] > policy["maximum_cluster_share"]:
            reasons.append("dominant cluster above maximum")

        silhouette_component = np.clip((metrics["silhouette"] + 1.0) / 2.0, 0, 1)
        ch_component = np.clip(np.log1p(metrics["calinski_harabasz"]) / 10.0, 0, 1)
        db_component = 1.0 / (1.0 + metrics["davies_bouldin"])
        balance_component = np.clip(metrics["minimum_cluster_share"] / 0.20, 0, 1)
        complexity_penalty = 0.01 * (state.n_clusters - 2) / 6
        weights = policy["weights"]
        objective = (
            weights["silhouette"] * silhouette_component
            + weights["calinski_harabasz"] * ch_component
            + weights["davies_bouldin"] * db_component
            + weights["stability"] * np.clip(stability_mean, 0, 1)
            + weights["balance"] * balance_component
            + weights["interpretability"] * interpretability
            - complexity_penalty
        )
        row.update(metrics)
        row.update({
            "stability_ari_mean": stability_mean,
            "stability_ari_sd": stability_sd,
            "interpretability_score": interpretability,
            "differentiated_features": differentiated,
            "complexity_penalty": complexity_penalty,
            "objective": float(objective if not reasons else -1.0),
            "status": "rejected" if reasons else "valid",
            "rejection_reason": "; ".join(reasons),
        })
    except Exception as exc:  # preserve failed trials in the audit trail
        row.update({"objective": -1.0, "rejection_reason": f"{type(exc).__name__}: {exc}"})
    row["runtime_seconds"] = time.perf_counter() - started
    return row


def neighbors(state: ExperimentState, policy: dict) -> list[tuple[ExperimentState, str]]:
    proposals: list[tuple[ExperimentState, str]] = []
    dimensions = {
        "feature_view": policy["feature_views"],
        "transformation": policy["transformations"],
        "scaler": policy["scalers"],
        "algorithm": policy["algorithms"],
        "outlier_policy": policy["outlier_policies"],
    }
    for field, values in dimensions.items():
        for value in values:
            if value != getattr(state, field):
                proposals.append((replace(state, **{field: value}), f"{field}: {getattr(state, field)} -> {value}"))
    counts = policy["cluster_counts"]
    index = counts.index(state.n_clusters)
    for offset in (-1, 1):
        other = index + offset
        if 0 <= other < len(counts):
            value = counts[other]
            proposals.append((replace(state, n_clusters=value), f"n_clusters: {state.n_clusters} -> {value}"))
    return sorted(proposals, key=lambda item: state_key(item[0]))


def run_hill_climb(frame: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, ExperimentState]:
    initial = ExperimentState()
    trials: list[dict] = []
    evaluated: set[str] = set()

    def evaluate(state: ExperimentState, parent: int | None, mutation: str) -> dict:
        trial = evaluate_state(
            frame, state, policy,
            trial_id=len(trials) + 1,
            parent_trial_id=parent,
            mutation=mutation,
        )
        trials.append(trial)
        evaluated.add(state_key(state))
        return trial

    current_state = initial
    current = evaluate(initial, None, "initial baseline")
    best_state = initial
    best = current
    no_improvement = 0

    while len(trials) < policy["max_trials"] and no_improvement < policy["patience"]:
        available = [(s, m) for s, m in neighbors(current_state, policy) if state_key(s) not in evaluated]
        if not available:
            no_improvement += 1
            # deterministic restart from a simple unevaluated configuration
            grid = [
                ExperimentState(feature_view=f, transformation=t, scaler=s, algorithm=a, n_clusters=k, outlier_policy=o)
                for f in policy["feature_views"] for t in policy["transformations"]
                for s in policy["scalers"] for a in policy["algorithms"]
                for k in policy["cluster_counts"] for o in policy["outlier_policies"]
            ]
            unseen = [state for state in grid if state_key(state) not in evaluated]
            if not unseen:
                break
            current_state = sorted(unseen, key=state_key)[0]
            current = evaluate(current_state, best["trial_id"], "deterministic restart")
            continue

        batch = available[: min(6, policy["max_trials"] - len(trials))]
        results = [evaluate(state, current["trial_id"], mutation) for state, mutation in batch]
        valid = [result for result in results if result["status"] == "valid"]
        challenger = max(valid, key=lambda row: row["objective"], default=None)
        if challenger and challenger["objective"] >= current["objective"] + policy["min_delta"]:
            current = challenger
            current_state = state_from_row(challenger)
            no_improvement = 0
        else:
            no_improvement += 1

        valid_all = [trial for trial in trials if trial["status"] == "valid"]
        if valid_all:
            best = max(valid_all, key=lambda row: (row["objective"], -row["n_clusters"], -len(row["feature_view"])))
            best_state = state_from_row(best)
            if current["objective"] < best["objective"]:
                current, current_state = best, best_state

    table = pd.DataFrame(trials)
    valid = table[table["status"] == "valid"].copy()
    if valid.empty:
        raise RuntimeError("AutoResearch produced no valid candidate")
    valid = valid.sort_values(["objective", "n_clusters"], ascending=[False, True])
    winner = valid.iloc[0]
    best_state = state_from_row(winner)
    return table, best_state
