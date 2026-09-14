from __future__ import annotations

from segmentforge.autoresearch import neighbors, state_key
from segmentforge.clustering import ExperimentState


POLICY = {
    "feature_views": ["rfm", "compact"],
    "transformations": ["log1p", "yeo_johnson"],
    "scalers": ["standard", "robust"],
    "algorithms": ["kmeans", "gaussian_mixture"],
    "outlier_policies": ["keep", "winsorize"],
    "cluster_counts": [2, 3, 4, 5],
}


def test_state_key_is_deterministic():
    state = ExperimentState()
    assert state_key(state) == state_key(state)


def test_each_neighbor_changes_exactly_one_component():
    state = ExperimentState()
    for candidate, _ in neighbors(state, POLICY):
        differences = sum(getattr(state, field) != getattr(candidate, field) for field in state.__dataclass_fields__)
        assert differences == 1


def test_neighbor_cluster_counts_stay_in_policy():
    state = ExperimentState(n_clusters=4)
    for candidate, _ in neighbors(state, POLICY):
        assert candidate.n_clusters in POLICY["cluster_counts"]
