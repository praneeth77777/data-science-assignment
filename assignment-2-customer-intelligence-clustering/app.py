"""SegmentForge data-science administration dashboard."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from segmentforge.paths import EXPERIMENTS, FIGURES, MODELS, REPORTS, TABLES

st.set_page_config(page_title="SegmentForge Admin", page_icon="◉", layout="wide")

PAGES = [
    "Executive Overview", "Data Explorer", "Segment Explorer", "Customer Lookup",
    "Model Lab", "Explainability", "Quality & Drift", "Deployment Admin",
]


@st.cache_data
def table(name: str, directory: Path = TABLES) -> pd.DataFrame:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}. Run scripts/run_pipeline.py.")
    return pd.read_csv(path)


@st.cache_data
def metadata() -> dict:
    path = MODELS / "segmentforge_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def figure(name: str, interpretation: str, caveat: str | None = None) -> None:
    path = FIGURES / name
    if not path.is_file():
        st.error(f"Missing figure: {name}. Run the analysis pipeline.")
        return
    st.image(str(path), width="stretch")
    st.markdown(f"**Measured interpretation:** {interpretation}")
    if caveat:
        st.caption(f"Caveat: {caveat}")


def executive_page() -> None:
    segments = table("customer_segments_masked.csv")
    profiles = table("cluster_profiles.csv")
    quality = table("raw_quality_summary.csv").set_index("check")["value"]
    total_revenue = profiles["revenue"].sum()
    cols = st.columns(5)
    cols[0].metric("Segmented customers", f"{len(segments):,}")
    cols[1].metric("Raw transactions", f"{int(quality['rows']):,}")
    cols[2].metric("Historical revenue", f"£{total_revenue:,.0f}")
    cols[3].metric("Selected clusters", f"{len(profiles)}")
    cols[4].metric("Locked audit customers", f"{(segments.partition == 'locked_audit').sum():,}")
    figure(
        "segment_contribution.png",
        "Cluster 0 contributes most historical valid-purchase revenue, while customer membership is more balanced.",
        "Revenue concentration is descriptive and does not establish future value or causal campaign lift.",
    )
    st.subheader("Operational personas")
    st.dataframe(
        profiles[["cluster", "persona", "customers", "customer_share", "revenue_share", "evidence", "suggested_experiment"]],
        hide_index=True, width="stretch",
        column_config={"customer_share": st.column_config.NumberColumn(format="%.1%%"), "revenue_share": st.column_config.NumberColumn(format="%.1%%")},
    )


def data_explorer_page() -> None:
    quality = table("raw_quality_summary.csv")
    exclusions = table("exclusion_audit.csv")
    dictionary = table("data_dictionary.csv")
    figure(
        "data_quality_overview.png",
        "Missing customer identifiers, cancellations, nonpositive values, and exact duplicates were measured before preparation.",
        "Exclusion counts overlap and must not be summed as mutually exclusive rows.",
    )
    left, right = st.columns(2)
    left.subheader("Raw audit")
    left.dataframe(quality, hide_index=True, width="stretch")
    right.subheader("Preparation diagnostics")
    right.dataframe(exclusions, hide_index=True, width="stretch")
    st.subheader("Data dictionary")
    st.dataframe(dictionary, hide_index=True, width="stretch")
    figure("rfm_distributions.png", "Customer RFM variables are strongly skewed, motivating transformation sensitivity experiments.")


def segment_explorer_page() -> None:
    profiles = table("cluster_profiles.csv")
    selected = st.selectbox("Segment", profiles["cluster"].astype(int).tolist())
    row = profiles.loc[profiles.cluster == selected].iloc[0]
    st.subheader(str(row["persona"]))
    cols = st.columns(5)
    cols[0].metric("Customers", f"{int(row['customers']):,}")
    cols[1].metric("Median recency", f"{row['recency_days']:.0f} days")
    cols[2].metric("Median frequency", f"{row['frequency']:.0f}")
    cols[3].metric("Median monetary", f"£{row['monetary']:,.2f}")
    cols[4].metric("Revenue share", f"{row['revenue_share']:.1%}")
    st.info(str(row["evidence"]))
    figure(
        "cluster_pca_projection.png",
        "The selected labels show structure in the two-component projection.",
        "The model was fitted and evaluated in the full transformed feature space; 2-D separation is not proof of validity.",
    )
    figure("cluster_profile_heatmap.png", "Transformed RFM centers distinguish the four selected clusters.")


def customer_lookup_page() -> None:
    segments = table("customer_segments_masked.csv")
    selected = st.selectbox("Masked customer ID", segments["masked_customer_id"].sort_values())
    row = segments.loc[segments.masked_customer_id == selected].iloc[0]
    st.subheader(f"{row['persona']} — cluster {int(row['cluster'])}")
    cols = st.columns(4)
    cols[0].metric("Recency", f"{row['recency_days']:.0f} days")
    cols[1].metric("Frequency", f"{row['frequency']:.0f} invoices")
    cols[2].metric("Monetary", f"£{row['monetary']:,.2f}")
    cols[3].metric("Product breadth", f"{row['product_breadth']:.0f}")
    st.write({
        "partition": row["partition"],
        "distance_to_centroid": round(float(row["distance_to_centroid"]), 4),
        "assignment_margin": round(float(row["assignment_margin"]), 4),
        "cancellation_rate": round(float(row["cancellation_rate"]), 4),
    })
    st.caption("This is a descriptive assignment, not a prediction of intent or a guarantee of future behavior.")


def model_lab_page() -> None:
    trials = table("autoresearch_trials.csv", EXPERIMENTS)
    leaderboard = table("autoresearch_leaderboard.csv")
    status_counts = trials.status.value_counts()
    cols = st.columns(4)
    cols[0].metric("Trials", len(trials))
    cols[1].metric("Valid", int(status_counts.get("valid", 0)))
    cols[2].metric("Rejected", int(status_counts.get("rejected", 0)))
    cols[3].metric("Failed", int(status_counts.get("failed", 0)))
    figure(
        "autoresearch_search.png",
        "The bounded search retained K-Means with four clusters after comparing component metrics and stability.",
        "The composite objective is a declared engineering heuristic; every component remains visible.",
    )
    columns = ["trial_id", "mutation", "algorithm", "n_clusters", "feature_view", "transformation", "scaler", "outlier_policy", "silhouette", "stability_ari_mean", "minimum_cluster_share", "objective"]
    st.dataframe(leaderboard[columns].head(20), hide_index=True, width="stretch")
    with st.expander("Rejected and failed trials"):
        st.dataframe(trials.loc[trials.status != "valid"], hide_index=True, width="stretch")


def explainability_page() -> None:
    profiles = table("cluster_profiles.csv")
    figure("cluster_profile_heatmap.png", "Cluster profiles differ across transformed Recency, Frequency, and Monetary features.")
    st.dataframe(profiles[["cluster", "persona", "evidence", "suggested_experiment"]], hide_index=True, width="stretch")
    st.warning("Persona labels were created after model selection from measured median profiles. They are descriptive labels, not protected traits or causal explanations.")


def quality_drift_page() -> None:
    monthly = table("monthly_segment_revenue.csv")
    figure(
        "monthly_segment_revenue.png",
        "Historical monthly revenue contributions differ across the frozen customer assignments.",
        "This chart is retrospective. Production drift thresholds require new post-deployment data and remain pending.",
    )
    st.dataframe(monthly.tail(24), hide_index=True, width="stretch")
    st.info("Production drift status: NOT MEASURED. Required future checks include feature drift, centroid movement, segment-share change, and stability degradation.")


def deployment_page() -> None:
    meta = metadata()
    metrics = table("locked_solution_metrics.csv")
    model_path = MODELS / "segmentforge_clusterer.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else "MISSING"
    st.json(meta)
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.write({"model_sha256": digest, "model_exists": model_path.is_file()})
    st.markdown("**Governance documents included in the repository**")
    st.code("\n".join([
        "artifacts/reports/MODEL_CARD.md",
        "artifacts/reports/DATA_CARD.md",
        "artifacts/reports/CRISP_DM_REPORT.md",
    ]))
    st.caption("Raw transaction rows and credentials are not embedded in this application.")


st.title("SegmentForge Customer Intelligence")
st.caption("CRISP-DM clustering, auditable hill climbing, stability analysis, and deployment administration")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption("All displayed metrics are frozen outputs from the documented dataset hash. Assumptions and limitations are labeled.")

try:
    {
        "Executive Overview": executive_page,
        "Data Explorer": data_explorer_page,
        "Segment Explorer": segment_explorer_page,
        "Customer Lookup": customer_lookup_page,
        "Model Lab": model_lab_page,
        "Explainability": explainability_page,
        "Quality & Drift": quality_drift_page,
        "Deployment Admin": deployment_page,
    }[page]()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
