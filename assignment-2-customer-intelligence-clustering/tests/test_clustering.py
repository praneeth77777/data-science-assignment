from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from segmentforge.clustering import ExperimentState, FeaturePreprocessor, cluster_metrics, fit_clusterer
from segmentforge.features import build_customer_features
from segmentforge.data import prepare_transactions


def customer_frame(raw_transactions):
    identified, valid, _ = prepare_transactions(raw_transactions)
    return build_customer_features(identified, valid)


def test_preprocessing_is_deterministic(raw_transactions):
    frame = customer_frame(raw_transactions)
    state = ExperimentState(n_clusters=4)
    first = FeaturePreprocessor(state).fit_transform(frame)
    second = FeaturePreprocessor(state).fit_transform(frame)
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()


def test_kmeans_is_reproducible(raw_transactions):
    frame = customer_frame(raw_transactions)
    state = ExperimentState(n_clusters=4)
    matrix = FeaturePreprocessor(state).fit_transform(frame)
    _, first = fit_clusterer(state, matrix, seed=42)
    _, second = fit_clusterer(state, matrix, seed=42)
    assert adjusted_rand_score(first, second) == 1.0


def test_all_algorithm_families_fit(raw_transactions):
    frame = customer_frame(raw_transactions)
    for algorithm in ("kmeans", "gaussian_mixture", "agglomerative"):
        state = ExperimentState(algorithm=algorithm, n_clusters=4)
        matrix = FeaturePreprocessor(state).fit_transform(frame)
        _, labels = fit_clusterer(state, matrix, seed=42)
        assert len(np.unique(labels)) == 4


def test_cluster_metrics_have_expected_directional_domains(raw_transactions):
    frame = customer_frame(raw_transactions)
    state = ExperimentState(n_clusters=4)
    matrix = FeaturePreprocessor(state).fit_transform(frame)
    _, labels = fit_clusterer(state, matrix, seed=42)
    metrics = cluster_metrics(matrix, labels, sample_size=100, seed=42)
    assert -1 <= metrics["silhouette"] <= 1
    assert metrics["calinski_harabasz"] > 0
    assert metrics["davies_bouldin"] >= 0
    assert metrics["minimum_cluster_share"] > 0
