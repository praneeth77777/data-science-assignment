#!/usr/bin/env python3
"""Execute the complete SegmentForge CRISP-DM analysis."""

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

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA

from segmentforge.autoresearch import load_policy, run_hill_climb
from segmentforge.clustering import (
    ExperimentState,
    FeaturePreprocessor,
    canonicalize_labels,
    centroids,
    cluster_metrics,
    fit_clusterer,
    predict_labels,
    stability_score,
)
from segmentforge.data import load_raw, prepare_transactions, raw_quality_summary, sha256
from segmentforge.features import build_customer_features, deterministic_customer_split, feature_columns
from segmentforge.paths import CONFIG, FIGURES, MODELS, RAW_WORKBOOK, REPORTS, TABLES, EXPERIMENTS, ensure_artifact_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_WORKBOOK)
    parser.add_argument("--config", type=Path, default=CONFIG)
    return parser.parse_args()


def masked_customer_id(value: int) -> str:
    return hashlib.sha256(f"segmentforge-v1-{int(value)}".encode()).hexdigest()[:12]


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("InvoiceNo", "raw", "Invoice identifier; prefix C denotes cancellation", "identifier"),
        ("StockCode", "raw", "Product/item code", "identifier"),
        ("Description", "raw", "Product description", "text"),
        ("Quantity", "raw", "Item quantity on invoice line", "integer"),
        ("InvoiceDate", "raw", "Invoice timestamp", "datetime"),
        ("UnitPrice", "raw", "Unit price in sterling", "numeric"),
        ("CustomerID", "raw", "Customer identifier; grouping only, never a clustering feature", "identifier"),
        ("Country", "raw", "Customer country", "categorical"),
        ("recency_days", "derived", "Days from last valid purchase to fixed snapshot", "integer"),
        ("frequency", "derived", "Distinct completed purchase invoices", "integer"),
        ("monetary", "derived", "Sum of positive purchase Quantity × UnitPrice", "currency"),
        ("average_order_value", "derived", "Mean completed-invoice value", "currency"),
        ("product_breadth", "derived", "Distinct products purchased", "integer"),
        ("total_units", "derived", "Total positive units purchased", "integer"),
        ("average_basket_units", "derived", "Mean units per completed invoice", "numeric"),
        ("tenure_days", "derived", "Days between first and last valid purchase", "integer"),
        ("cancellation_rate", "derived", "Share of identified deduplicated rows from cancellation invoices", "proportion"),
    ]
    return pd.DataFrame(rows, columns=["field", "stage", "definition", "type"])


def profile_personas(customer_segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["recency_days", "frequency", "monetary", "average_order_value", "product_breadth", "total_units", "tenure_days", "cancellation_rate"]
    profile = customer_segments.groupby("cluster").agg(
        customers=("masked_customer_id", "size"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
        average_order_value=("average_order_value", "median"),
        product_breadth=("product_breadth", "median"),
        total_units=("total_units", "median"),
        tenure_days=("tenure_days", "median"),
        cancellation_rate=("cancellation_rate", "median"),
        revenue=("monetary", "sum"),
    ).reset_index()
    profile["customer_share"] = profile["customers"] / profile["customers"].sum()
    profile["revenue_share"] = profile["revenue"] / profile["revenue"].sum()
    overall = customer_segments[metrics].median()
    highest_value_cluster = int(profile.loc[profile["monetary"].idxmax(), "cluster"])
    names, evidence, actions = [], [], []
    for _, row in profile.iterrows():
        if int(row["cluster"]) == highest_value_cluster:
            name = "High-value frequent customers"
            action = "Prioritize service reliability and retention experiments; do not assume causal lift."
        elif row["monetary"] >= overall["monetary"]:
            name = "Established mid-value customers"
            action = "Test relevant cross-sell hypotheses while monitoring contact pressure."
        elif row["recency_days"] <= overall["recency_days"] and row["frequency"] < overall["frequency"]:
            name = "Recent occasional customers"
            action = "Test onboarding and second-purchase messaging in a controlled experiment."
        elif row["recency_days"] > overall["recency_days"]:
            name = "Long-recency customers"
            action = "Investigate re-engagement eligibility and suppress inappropriate outreach."
        else:
            name = "Developing repeat customers"
            action = "Monitor repeat purchase patterns and category preferences."
        names.append(name)
        evidence.append(
            f"Median R/F/M = {row['recency_days']:.0f} days / {row['frequency']:.0f} invoices / £{row['monetary']:.2f}."
        )
        actions.append(action)
    profile["persona"] = names
    profile["evidence"] = evidence
    profile["suggested_experiment"] = actions
    long = profile.melt(id_vars=["cluster", "persona", "customers"], value_vars=metrics, var_name="feature", value_name="median")
    return profile, long


def save_figures(
    quality: pd.DataFrame,
    exclusions: pd.DataFrame,
    discovery: pd.DataFrame,
    trials: pd.DataFrame,
    segments: pd.DataFrame,
    profiles: pd.DataFrame,
    transformed: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    monthly: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    missing = quality[quality["check"].isin(["missing_customer_id", "missing_description", "exact_duplicate_rows"])]
    axes[0].bar(missing["check"], pd.to_numeric(missing["value"]), color="#2563EB")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(title="Raw-data quality counts", ylabel="Rows")
    axes[1].barh(exclusions["reason"], exclusions["share_of_raw"] * 100, color="#DC2626")
    axes[1].set(title="Declared exclusion diagnostics", xlabel="Share of raw rows (%)")
    fig.savefig(FIGURES / "data_quality_overview.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, feature in zip(axes, ["recency_days", "frequency", "monetary"]):
        ax.hist(discovery[feature], bins=40, color="#2563EB", edgecolor="white")
        ax.set(title=feature.replace("_", " ").title(), xlabel=feature, ylabel="Customers")
        if feature != "recency_days":
            ax.set_yscale("log")
    fig.savefig(FIGURES / "rfm_distributions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    valid = trials[trials["status"] == "valid"].copy().sort_values("objective", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(trials["trial_id"], trials["objective"], marker="o", linewidth=1)
    axes[0].set(title="Bounded hill-climb search trace", xlabel="Trial", ylabel="Composite objective")
    top = valid.head(12).sort_values("objective")
    labels_text = top["algorithm"] + " k=" + top["n_clusters"].astype(str) + " " + top["feature_view"]
    axes[1].barh(labels_text, top["objective"], color="#0F766E")
    axes[1].set(title="Top valid experiments", xlabel="Composite objective")
    fig.savefig(FIGURES / "autoresearch_search.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(transformed)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=0.45, label=f"Cluster {cluster}")
    ax.set(
        title=f"PCA visualization only ({pca.explained_variance_ratio_.sum() * 100:.1f}% variance shown)",
        xlabel="Principal component 1", ylabel="Principal component 2",
    )
    ax.legend()
    fig.savefig(FIGURES / "cluster_pca_projection.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    center_frame = pd.DataFrame(centroids(transformed, labels), columns=feature_names)
    fig, ax = plt.subplots(figsize=(max(9, len(feature_names) * 1.15), 4.5), constrained_layout=True)
    image = ax.imshow(center_frame, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(center_frame)), [f"Cluster {i}" for i in range(len(center_frame))])
    ax.set_title("Cluster centers in transformed feature space")
    fig.colorbar(image, ax=ax, label="Transformed center value")
    fig.savefig(FIGURES / "cluster_profile_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(profiles["cluster"].astype(str), profiles["customer_share"] * 100, color="#2563EB")
    axes[0].set(title="Customer share by segment", xlabel="Cluster", ylabel="Customers (%)")
    axes[1].bar(profiles["cluster"].astype(str), profiles["revenue_share"] * 100, color="#7C3AED")
    axes[1].set(title="Historical revenue share by segment", xlabel="Cluster", ylabel="Revenue (%)")
    fig.savefig(FIGURES / "segment_contribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for cluster, group in monthly.groupby("cluster"):
        ax.plot(pd.to_datetime(group["month"]), group["revenue"], marker="o", label=f"Cluster {cluster}")
    ax.set(title="Monthly valid-purchase revenue by assigned segment", xlabel="Month", ylabel="Revenue (£)")
    ax.legend()
    fig.savefig(FIGURES / "monthly_segment_revenue.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_reports(
    raw_hash: str,
    quality: pd.DataFrame,
    exclusions: pd.DataFrame,
    customer: pd.DataFrame,
    discovery: pd.DataFrame,
    audit: pd.DataFrame,
    trials: pd.DataFrame,
    state: ExperimentState,
    dev_metrics: dict,
    audit_metrics: dict,
    profiles: pd.DataFrame,
    runtime: float,
) -> None:
    winner = trials.loc[trials["status"] == "valid"].sort_values("objective", ascending=False).iloc[0]
    model_card = f"""# SegmentForge Model Card

## Model

- Algorithm: `{state.algorithm}`
- Clusters: {state.n_clusters}
- Feature view: `{state.feature_view}`
- Transformation: `{state.transformation}`
- Scaler: `{state.scaler}`
- Outlier policy: `{state.outlier_policy}`
- Random seed: 42

## Population and validation

- Customer feature rows: {len(customer):,}
- Discovery customers: {len(discovery):,}
- Locked audit customers: {len(audit):,}
- AutoResearch trials: {len(trials)} total; {(trials['status'] == 'valid').sum()} valid; {(trials['status'] == 'rejected').sum()} rejected; {(trials['status'] == 'failed').sum()} failed
- Dataset SHA-256: `{raw_hash}`

| Metric | Discovery | Locked audit |
|---|---:|---:|
| Silhouette | {dev_metrics['silhouette']:.4f} | {audit_metrics['silhouette']:.4f} |
| Calinski-Harabasz | {dev_metrics['calinski_harabasz']:.2f} | {audit_metrics['calinski_harabasz']:.2f} |
| Davies-Bouldin | {dev_metrics['davies_bouldin']:.4f} | {audit_metrics['davies_bouldin']:.4f} |
| Minimum cluster share | {dev_metrics['minimum_cluster_share']:.2%} | {audit_metrics['minimum_cluster_share']:.2%} |
| Maximum cluster share | {dev_metrics['maximum_cluster_share']:.2%} | {audit_metrics['maximum_cluster_share']:.2%} |

Development stability ARI: {winner['stability_ari_mean']:.4f} ± {winner['stability_ari_sd']:.4f} across the predeclared subsamples.

## Intended use

Descriptive customer analysis, campaign hypothesis generation, service planning, and segment monitoring. Human review and controlled experiments are required before customer treatment decisions.

## Limitations

- Clusters are descriptive and are not causal effects, customer lifetime value, or guaranteed future behavior.
- The data describe one UK retailer in 2010–2011.
- Customers without identifiers are excluded from customer-level segmentation.
- Returns and cancellations are represented through diagnostics and a cancellation-rate feature; positive sales define monetary value.
- Cluster membership depends on the observation window, preparation choices, and distance/model assumptions.
- The hill-climb objective is a declared engineering heuristic, not scientific truth.
"""
    (REPORTS / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    data_card = f"""# SegmentForge Data Card

Source: UCI Online Retail, DOI 10.24432/C5BW33, CC BY 4.0.

- Raw shape: {int(quality.loc[quality.check == 'rows', 'value'].iloc[0]):,} rows × {int(quality.loc[quality.check == 'columns', 'value'].iloc[0])} columns.
- Coverage: {quality.loc[quality.check == 'minimum_invoice_date', 'value'].iloc[0]} through {quality.loc[quality.check == 'maximum_invoice_date', 'value'].iloc[0]}.
- Exact duplicates documented: {int(quality.loc[quality.check == 'exact_duplicate_rows', 'value'].iloc[0]):,}.
- Missing CustomerID rows documented: {int(quality.loc[quality.check == 'missing_customer_id', 'value'].iloc[0]):,}.
- Customer-level feature rows: {len(customer):,}.
- Raw workbook is excluded from Git and must be downloaded by each user.

Preparation rules and their overlapping row counts are stored in `artifacts/tables/exclusion_audit.csv`. Exclusion counts must not be summed because a row may meet more than one rule.
"""
    (REPORTS / "DATA_CARD.md").write_text(data_card, encoding="utf-8")

    report = f"""# SegmentForge CRISP-DM Report

## Business understanding

Segment customers from historical purchase behavior to support analysis and controlled campaign hypotheses. The system does not claim causality, profitability lift, or individual intent.

## Data understanding

The raw workbook contained {int(quality.loc[quality.check == 'rows', 'value'].iloc[0]):,} transaction lines. Raw defects and cancellations were documented before preparation. See `raw_quality_summary.csv` and `exclusion_audit.csv`.

## Data preparation

Exact duplicates and rows without CustomerID were excluded from customer aggregation. Monetary behavior uses completed positive-quantity, positive-price invoices. Cancellation behavior is retained as a separate rate. A fixed snapshot one day after the final valid purchase defines recency.

## Modeling and AutoResearch

The bounded hill climber executed {len(trials)} trials and changed one configuration component per mutation. The locked winner was `{state.algorithm}` with k={state.n_clusters}, `{state.feature_view}` features, `{state.transformation}`, `{state.scaler}`, and `{state.outlier_policy}`. The winning composite objective was {winner['objective']:.4f}. All component metrics and rejected/failed trials remain visible.

## Evaluation

Discovery silhouette was {dev_metrics['silhouette']:.4f}; locked-audit silhouette was {audit_metrics['silhouette']:.4f}. Discovery stability ARI was {winner['stability_ari_mean']:.4f}. These are internal clustering diagnostics, not predictive accuracy.

## Deployment

The Streamlit admin dashboard reads frozen aggregate artifacts and a versioned model bundle. It exposes quality, segments, experiments, explainability, and reproducibility without embedding the raw workbook.

## Runtime

Complete analysis runtime: {runtime:.2f} seconds on {platform.platform()} with Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}, and scikit-learn {sklearn.__version__}.
"""
    (REPORTS / "CRISP_DM_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ensure_artifact_dirs()
    raw_hash = sha256(args.data)
    raw = load_raw(args.data)
    quality = raw_quality_summary(raw)
    identified, valid, exclusions = prepare_transactions(raw)
    customer = build_customer_features(identified, valid)
    discovery, audit = deterministic_customer_split(customer)
    policy = load_policy(args.config)

    quality.to_csv(TABLES / "raw_quality_summary.csv", index=False)
    exclusions.to_csv(TABLES / "exclusion_audit.csv", index=False)
    data_dictionary().to_csv(TABLES / "data_dictionary.csv", index=False)
    customer.drop(columns=["CustomerID", "first_purchase", "last_purchase"]).describe(include="all").T.to_csv(TABLES / "customer_feature_summary.csv")
    pd.DataFrame([
        {"partition": "discovery", "customers": len(discovery)},
        {"partition": "locked_audit", "customers": len(audit)},
    ]).to_csv(TABLES / "customer_split_manifest.csv", index=False)

    trials, best_state = run_hill_climb(discovery, policy)
    trials.to_csv(EXPERIMENTS / "autoresearch_trials.csv", index=False)
    trials.loc[trials.status == "valid"].sort_values("objective", ascending=False).to_csv(TABLES / "autoresearch_leaderboard.csv", index=False)
    (EXPERIMENTS / "best_state.json").write_text(json.dumps(best_state.to_dict(), indent=2), encoding="utf-8")

    preprocessor = FeaturePreprocessor(best_state)
    dev_matrix = preprocessor.fit_transform(discovery)
    model, raw_dev_labels = fit_clusterer(best_state, dev_matrix, seed=policy["master_seed"])
    dev_labels, mapping = canonicalize_labels(discovery, raw_dev_labels)
    audit_matrix = preprocessor.transform(audit)
    raw_audit_labels = predict_labels(model, audit_matrix, training_matrix=dev_matrix, training_labels=raw_dev_labels)
    audit_labels = np.array([mapping[int(label)] for label in raw_audit_labels], dtype=int)
    dev_metrics = cluster_metrics(dev_matrix, dev_labels, sample_size=policy["silhouette_sample"], seed=42)
    audit_metrics = cluster_metrics(audit_matrix, audit_labels, sample_size=policy["silhouette_sample"], seed=42)
    stability_mean, stability_sd = stability_score(
        discovery, best_state,
        seeds=policy["stability_seeds"], fraction=policy["stability_fraction"], master_seed=42,
    )
    metric_table = pd.DataFrame([
        {"partition": "discovery", **dev_metrics, "stability_ari_mean": stability_mean, "stability_ari_sd": stability_sd},
        {"partition": "locked_audit", **audit_metrics, "stability_ari_mean": np.nan, "stability_ari_sd": np.nan},
    ])
    metric_table.to_csv(TABLES / "locked_solution_metrics.csv", index=False)

    feature_names = feature_columns(best_state.feature_view)
    raw_centers = centroids(dev_matrix, raw_dev_labels)
    all_customer = pd.concat([discovery.assign(partition="discovery"), audit.assign(partition="locked_audit")], ignore_index=True)
    all_matrix = np.vstack([dev_matrix, audit_matrix])
    all_raw_labels = np.concatenate([raw_dev_labels, raw_audit_labels])
    all_labels = np.concatenate([dev_labels, audit_labels])
    distances = ((all_matrix[:, None, :] - raw_centers[None, :, :]) ** 2).sum(axis=2) ** 0.5
    ordered = np.sort(distances, axis=1)
    segments = all_customer.copy()
    segments["masked_customer_id"] = segments["CustomerID"].map(masked_customer_id)
    segments["cluster"] = all_labels
    segments["distance_to_centroid"] = distances[np.arange(len(distances)), all_raw_labels]
    segments["assignment_margin"] = ordered[:, 1] - ordered[:, 0]
    segments = segments.drop(columns=["CustomerID", "first_purchase", "last_purchase", "snapshot_date"])
    profiles, profile_long = profile_personas(segments)
    persona_map = profiles.set_index("cluster")["persona"]
    segments["persona"] = segments["cluster"].map(persona_map)
    segments.to_csv(TABLES / "customer_segments_masked.csv", index=False)
    profiles.to_csv(TABLES / "cluster_profiles.csv", index=False)
    profile_long.to_csv(TABLES / "cluster_profile_long.csv", index=False)

    assignment = pd.DataFrame({"CustomerID": all_customer["CustomerID"].astype(int), "cluster": all_labels})
    monthly = valid.merge(assignment, on="CustomerID", how="inner")
    monthly["month"] = monthly["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = monthly.groupby(["month", "cluster"], as_index=False).agg(
        revenue=("line_value", "sum"), orders=("InvoiceNo", "nunique"), units=("Quantity", "sum")
    )
    monthly.to_csv(TABLES / "monthly_segment_revenue.csv", index=False)

    bundle = {
        "state": best_state.to_dict(),
        "preprocessor": preprocessor,
        "model": model,
        "training_matrix": dev_matrix,
        "training_labels": raw_dev_labels,
        "canonical_mapping": mapping,
        "feature_names": feature_names,
        "dataset_sha256": raw_hash,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    joblib.dump(bundle, MODELS / "segmentforge_clusterer.joblib", compress=3)
    metadata = {
        "state": best_state.to_dict(), "dataset_sha256": raw_hash,
        "discovery_customers": len(discovery), "locked_audit_customers": len(audit),
        "metrics": {"discovery": dev_metrics, "locked_audit": audit_metrics},
        "stability_ari_mean": stability_mean, "stability_ari_sd": stability_sd,
        "random_seed": 42, "trials": len(trials),
    }
    (MODELS / "segmentforge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_figures(quality, exclusions, discovery, trials, segments, profiles, all_matrix, all_labels, feature_names, monthly)
    runtime = time.perf_counter() - started
    write_reports(raw_hash, quality, exclusions, customer, discovery, audit, trials, best_state, dev_metrics, audit_metrics, profiles, runtime)
    runtime_record = {
        "runtime_seconds": runtime,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "dataset_sha256": raw_hash,
    }
    (EXPERIMENTS / "runtime_environment.json").write_text(json.dumps(runtime_record, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "customers": len(customer), "trials": len(trials),
        "best_state": best_state.to_dict(), "discovery_metrics": dev_metrics,
        "locked_audit_metrics": audit_metrics, "runtime_seconds": runtime,
    }, indent=2))


if __name__ == "__main__":
    main()
