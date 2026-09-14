from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_artifacts_exist():
    required = [
        "artifacts/models/segmentforge_clusterer.joblib",
        "artifacts/models/segmentforge_metadata.json",
        "artifacts/tables/autoresearch_leaderboard.csv",
        "artifacts/tables/locked_solution_metrics.csv",
        "artifacts/tables/cluster_profiles.csv",
        "artifacts/reports/MODEL_CARD.md",
        "artifacts/reports/DATA_CARD.md",
    ]
    assert all((PROJECT_ROOT / path).is_file() for path in required)


def test_metadata_matches_locked_metrics():
    metadata = json.loads((PROJECT_ROOT / "artifacts/models/segmentforge_metadata.json").read_text())
    metrics = pd.read_csv(PROJECT_ROOT / "artifacts/tables/locked_solution_metrics.csv").set_index("partition")
    assert metadata["state"]["algorithm"] == "kmeans"
    assert metadata["state"]["n_clusters"] == 4
    assert abs(metadata["metrics"]["locked_audit"]["silhouette"] - metrics.loc["locked_audit", "silhouette"]) < 1e-12


def test_masked_export_contains_no_raw_customer_id():
    segments = pd.read_csv(PROJECT_ROOT / "artifacts/tables/customer_segments_masked.csv")
    assert "CustomerID" not in segments.columns
    assert segments["masked_customer_id"].str.fullmatch(r"[0-9a-f]{12}").all()
