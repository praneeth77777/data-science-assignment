"""PedalPulse Streamlit dashboard for prediction and executed analysis artifacts."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pedalpulse.modeling import empirical_tree_interval, load_bundle, predict_demand  # noqa: E402
from pedalpulse.paths import DEFAULT_MODEL, FIGURES_DIR, REPORTS_DIR, TABLES_DIR  # noqa: E402

st.set_page_config(page_title="PedalPulse", page_icon="🚲", layout="wide")


@st.cache_resource
def cached_model():
    return load_bundle(DEFAULT_MODEL)


@st.cache_data
def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


def show_figure(filename: str, interpretation: str) -> None:
    path = FIGURES_DIR / filename
    if path.is_file():
        st.image(str(path), width="stretch")
        st.caption(f"Interpretation: {interpretation}")
    else:
        st.warning(f"Figure is missing: {filename}")


def prediction_page() -> None:
    st.header("Hourly demand prediction")
    st.write(
        "Enter an operational scenario. The model excludes `casual`, `registered`, and `count`; "
        "it predicts system-wide hourly rentals from calendar and weather conditions."
    )
    if not DEFAULT_MODEL.is_file():
        st.error("The model artifact is missing. Run `python scripts/train_model.py` after downloading train.csv.")
        return

    with st.form("prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            prediction_date = st.date_input("Date", value=date(2012, 12, 20))
            prediction_time = st.time_input("Hour", value=time(8, 0), step=3600)
            season = st.selectbox(
                "Season code",
                options=[1, 2, 3, 4],
                format_func=lambda value: {1: "1 — spring", 2: "2 — summer", 3: "3 — fall", 4: "4 — winter"}[value],
            )
        with middle:
            holiday = int(st.checkbox("Holiday"))
            workingday = int(st.checkbox("Working day", value=True))
            weather = st.selectbox(
                "Weather code",
                options=[1, 2, 3, 4],
                format_func=lambda value: {
                    1: "1 — clear/partly cloudy",
                    2: "2 — mist/cloudy",
                    3: "3 — light precipitation",
                    4: "4 — severe weather",
                }[value],
            )
        with right:
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, step=0.5)
            feels_like = st.number_input("Feels-like temperature (°C)", min_value=-30.0, max_value=70.0, value=21.0, step=0.5)
            humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
            windspeed = st.number_input("Windspeed", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
        submitted = st.form_submit_button("Predict demand", type="primary")

    if submitted:
        timestamp = datetime.combine(prediction_date, prediction_time).replace(minute=0, second=0, microsecond=0)
        frame = pd.DataFrame([{
            "datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "season": season,
            "holiday": holiday,
            "workingday": workingday,
            "weather": weather,
            "temp": temperature,
            "atemp": feels_like,
            "humidity": humidity,
            "windspeed": windspeed,
        }])
        bundle = cached_model()
        prediction = float(predict_demand(bundle, frame)[0])
        lower, upper = empirical_tree_interval(bundle, frame)
        st.metric("Predicted hourly rentals", f"{prediction:,.0f}")
        st.write(f"Heuristic tree-spread range: **{lower[0]:,.0f}–{upper[0]:,.0f} rentals**")
        st.caption(
            "This range is not a calibrated confidence interval. Predictions outside the 2011–2012 operating context require monitoring."
        )


def eda_page() -> None:
    st.header("Data understanding and EDA")
    st.write("The dashboard uses saved aggregate figures; the raw Kaggle dataset is not redistributed.")
    options = {
        "Temporal demand patterns": (
            "temporal_demand_patterns.png",
            "Demand has strong hourly and calendar structure, with commuting periods becoming especially important on working days.",
        ),
        "Hour-by-weekday heatmap": (
            "hour_weekday_heatmap.png",
            "Weekday profiles show pronounced morning and evening peaks, while weekend demand is distributed differently across the day.",
        ),
        "Weather relationships": (
            "weather_relationships.png",
            "Demand is associated with temperature, humidity, and weather severity, but the plots alone do not establish causality.",
        ),
        "Target transformation": (
            "target_scale_comparison.png",
            "The raw count distribution is right-skewed; log1p compresses legitimate demand peaks without deleting observations.",
        ),
        "Correlation analysis": (
            "correlation_analysis.png",
            "Pairwise correlations summarize association only and can miss nonlinear and calendar interactions.",
        ),
        "Group comparisons": (
            "group_comparisons.png",
            "Differences across operational groups are observed associations and may also reflect time, weather, and availability constraints.",
        ),
    }
    selected = st.selectbox("Choose an EDA view", list(options))
    show_figure(*options[selected])


def cluster_page() -> None:
    st.header("Outliers and usage-period clusters")
    first, second = st.columns(2)
    with first:
        show_figure(
            "outlier_diagnostics.png",
            "The training IQR rule flags high commuting demand, so the evidence does not justify treating every flagged peak as invalid.",
        )
        show_figure(
            "outlier_sensitivity.png",
            "Log transformation had the lowest mean sensitivity-study RMSLE; deletion was not selected merely to improve a metric.",
        )
    with second:
        show_figure(
            "cluster_selection.png",
            "K-Means k=2 satisfies the minimum cluster-size rule and provides a balanced operational segmentation.",
        )
        show_figure(
            "cluster_pca_projection.png",
            "PCA is only a two-dimensional visualization; overlap here does not measure the complete standardized feature space.",
        )
    st.subheader("Operational personas")
    st.dataframe(read_table("cluster_operational_personas.csv"), width="stretch", hide_index=True)
    st.caption("Demand values in this table were added after clustering for interpretation and were not cluster inputs.")


def performance_page() -> None:
    st.header("Model performance")
    metrics = read_table("final_locked_test_metrics.csv")
    display = metrics.loc[:, ["candidate", "mae", "rmse", "rmsle", "r2"]].copy()
    st.dataframe(display.round(4), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        show_figure(
            "model_family_comparison.png",
            "Random Forest achieved the lowest five-fold development RMSLE; complexity was not itself the selection criterion.",
        )
        show_figure(
            "fold_rmsle_comparison.png",
            "Fold variation is substantial, so the mean score should not be interpreted without its temporal variability.",
        )
    with right:
        show_figure(
            "performance_slices.png",
            "Errors are largest for high-demand periods; the final quarter contains only season code 4, preventing a between-season test comparison.",
        )
        show_figure(
            "temporal_error_plot.png",
            "Error changes through the locked quarter, which supports ongoing temporal monitoring rather than reliance on one aggregate score.",
        )
    st.info("The locked model was evaluated once. Its RMSLE was 0.3739 versus 0.5255 for seasonal naive.")


def explainability_page() -> None:
    st.header("Explainability and residual audit")
    left, right = st.columns(2)
    with left:
        show_figure(
            "permutation_importance.png",
            "Cyclical hour features dominate permutation importance; correlated predictors can share or redistribute apparent importance.",
        )
        st.dataframe(read_table("locked_test_permutation_importance.csv").head(15), width="stretch", hide_index=True)
    with right:
        show_figure(
            "partial_dependence.png",
            "Partial dependence is associative and may represent unrealistic combinations when predictors are correlated.",
        )
        show_figure(
            "residual_distribution.png",
            "Positive residuals dominate because the model systematically underpredicts this later, higher-demand period.",
        )
    st.warning("SHAP was not run because the package was unavailable; this limitation is preserved in the model card.")


def reproducibility_page() -> None:
    st.header("Reproducibility and audit")
    audit = read_table("final_leakage_reproducibility_audit.csv")
    st.dataframe(audit, width="stretch", hide_index=True)
    if DEFAULT_MODEL.is_file():
        metadata = cached_model()["metadata"]
        st.subheader("Deployment artifact metadata")
        st.json(metadata, expanded=False)
    st.subheader("Documentation")
    for name in ["MODEL_CARD.md", "DATA_CARD.md", "CHUNK7_EVALUATION_REPORT.md"]:
        path = REPORTS_DIR / name
        with st.expander(name):
            st.markdown(path.read_text(encoding="utf-8") if path.is_file() else "Report missing")


st.title("🚲 PedalPulse")
st.caption("CRISP-DM bike-sharing demand prediction, pattern discovery, and explainable machine learning")
page = st.sidebar.radio(
    "Navigate",
    ["Demand prediction", "EDA", "Clusters & outliers", "Model performance", "Explainability", "Reproducibility"],
)
st.sidebar.caption("Raw data are not embedded in this application.")

if page == "Demand prediction":
    prediction_page()
elif page == "EDA":
    eda_page()
elif page == "Clusters & outliers":
    cluster_page()
elif page == "Model performance":
    performance_page()
elif page == "Explainability":
    explainability_page()
else:
    reproducibility_page()
