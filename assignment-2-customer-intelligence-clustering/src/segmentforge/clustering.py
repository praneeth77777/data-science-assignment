"""Clustering transformations, estimators, metrics, and stability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .features import feature_columns


@dataclass(frozen=True)
class ExperimentState:
    feature_view: str = "rfm"
    transformation: str = "log1p"
    scaler: str = "robust"
    algorithm: str = "kmeans"
    n_clusters: int = 4
    outlier_policy: str = "keep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeaturePreprocessor:
    def __init__(self, state: ExperimentState):
        self.state = state
        self.columns = feature_columns(state.feature_view)
        self.caps_: np.ndarray | None = None
        self.transformer_: PowerTransformer | None = None
        self.scaler_: StandardScaler | RobustScaler | None = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        matrix = frame[self.columns].to_numpy(dtype=float)
        self.caps_ = np.quantile(matrix, 0.99, axis=0)
        matrix = self._cap(matrix)
        if self.state.transformation == "log1p":
            matrix = np.log1p(matrix)
        elif self.state.transformation == "yeo_johnson":
            self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=False).fit(matrix)
            matrix = self.transformer_.transform(matrix)
        else:
            raise ValueError(f"Unknown transformation: {self.state.transformation}")
        if self.state.scaler == "standard":
            self.scaler_ = StandardScaler().fit(matrix)
        elif self.state.scaler == "robust":
            self.scaler_ = RobustScaler().fit(matrix)
        else:
            raise ValueError(f"Unknown scaler: {self.state.scaler}")
        return self

    def _cap(self, matrix: np.ndarray) -> np.ndarray:
        if self.state.outlier_policy == "keep":
            return matrix
        if self.state.outlier_policy != "winsorize":
            raise ValueError(f"Unknown outlier policy: {self.state.outlier_policy}")
        if self.caps_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
        return np.minimum(matrix, self.caps_)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
        matrix = self._cap(frame[self.columns].to_numpy(dtype=float))
        if self.state.transformation == "log1p":
            matrix = np.log1p(matrix)
        else:
            if self.transformer_ is None:
                raise RuntimeError("Power transformer has not been fitted")
            matrix = self.transformer_.transform(matrix)
        matrix = self.scaler_.transform(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("Transformed matrix contains non-finite values")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def fit_clusterer(state: ExperimentState, matrix: np.ndarray, *, seed: int = 42):
    if state.algorithm == "kmeans":
        model = KMeans(n_clusters=state.n_clusters, n_init=20, random_state=seed)
        labels = model.fit_predict(matrix)
    elif state.algorithm == "gaussian_mixture":
        model = GaussianMixture(
            n_components=state.n_clusters,
            covariance_type="diag",
            n_init=3,
            random_state=seed,
            reg_covar=1e-6,
        )
        labels = model.fit_predict(matrix)
    elif state.algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=state.n_clusters, linkage="ward")
        labels = model.fit_predict(matrix)
    else:
        raise ValueError(f"Unknown algorithm: {state.algorithm}")
    return model, labels.astype(int)


def centroids(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([matrix[labels == label].mean(axis=0) for label in sorted(np.unique(labels))])


def predict_labels(model, matrix: np.ndarray, *, training_matrix=None, training_labels=None) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(matrix), dtype=int)
    if training_matrix is None or training_labels is None:
        raise ValueError("Training matrix and labels required for nearest-centroid assignment")
    centers = centroids(training_matrix, training_labels)
    distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype(int)


def cluster_metrics(matrix: np.ndarray, labels: np.ndarray, *, sample_size: int, seed: int) -> dict[str, float]:
    counts = pd.Series(labels).value_counts(normalize=True)
    if len(counts) < 2:
        raise ValueError("At least two clusters are required")
    n = min(sample_size, len(matrix))
    silhouette = silhouette_score(matrix, labels, sample_size=n, random_state=seed)
    return {
        "silhouette": float(silhouette),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "minimum_cluster_share": float(counts.min()),
        "maximum_cluster_share": float(counts.max()),
    }


def stability_score(
    frame: pd.DataFrame,
    state: ExperimentState,
    *,
    seeds: list[int],
    fraction: float,
    master_seed: int,
) -> tuple[float, float]:
    base_pre = FeaturePreprocessor(state)
    base_matrix = base_pre.fit_transform(frame)
    base_model, base_labels = fit_clusterer(state, base_matrix, seed=master_seed)
    scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(frame), size=int(len(frame) * fraction), replace=False))
        subset = frame.iloc[indices]
        sub_pre = FeaturePreprocessor(state)
        sub_matrix = sub_pre.fit_transform(subset)
        sub_model, sub_labels = fit_clusterer(state, sub_matrix, seed=seed)
        base_on_subset = base_labels[indices]
        scores.append(adjusted_rand_score(base_on_subset, sub_labels))
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def canonicalize_labels(frame: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Map labels by descending median monetary value for stable dashboard colors."""
    profile = frame.assign(_label=labels).groupby("_label")["monetary"].median().sort_values(ascending=False)
    mapping = {int(old): int(new) for new, old in enumerate(profile.index)}
    return np.array([mapping[int(label)] for label in labels], dtype=int), mapping
