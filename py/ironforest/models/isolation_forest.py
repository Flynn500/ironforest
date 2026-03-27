"""Isolation Forest for anomaly detection built on tree_engine."""

from typing import Optional
import warnings
from ironforest._core import (
    Array,
    ndutils,
    Ensemble,
    EnsembleConfig,
    Tree,
    TreeConfig,
    TaskType,
    SplitCriterion,
)


class IsolationForest:
    """Isolation Forest for anomaly detection.

    The Isolation Forest algorithm isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum values
    of the selected feature. Anomalies are more susceptible to isolation and have
    shorter average path lengths in the trees.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ):
        """Isolation forest for anomaly detection.

        Args:
            n_estimators: Number of isolation trees in the forest.
            max_samples: Number of samples to draw to train each tree.
                If larger than n_samples, all samples are used.
            contamination: Proportion of outliers in the dataset, in (0, 0.5].
            max_features: Number of features per split. None uses all features.
            random_state: Controls randomness of sampling and tree construction.
        """
        warnings.warn("Isolation Forest will be deprecated in 0.7")
        if contamination <= 0 or contamination > 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.ensemble_ = None
        self.n_features_ = None
        self.threshold_ = None

    def fit(self, X, y=None):
        """Build an isolation forest from the training set X.

        Args:
            X: Array or array-like of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            self
        """
        if not isinstance(X, Array):
            X = ndutils.asarray(X)

        if X.dtype == "float32":
            X = X.astype("float64")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        max_samples = min(self.max_samples, n_samples)

        if self.max_features is not None:
            config = EnsembleConfig(
                n_trees=self.n_estimators,
                tree_config=TreeConfig(
                    task_type=TaskType.anomaly_detection(),
                    n_classes=0,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=self.max_features,
                    criterion=SplitCriterion.random(),
                    seed=self.random_state,
                ),
                bootstrap=False,
                max_samples=max_samples,
                seed=self.random_state,
            )
        else:
            config = EnsembleConfig.isolation_forest(self.n_estimators, max_samples)
            config.seed = self.random_state

        X_flat = X.ravel()
        y_dummy = ndutils.asarray([0.0] * n_samples)

        self.ensemble_ = Ensemble.fit(config, X_flat, y_dummy, n_samples, n_features)

        # score_samples negates the raw Rust output so lower = more anomalous (sklearn convention)
        scores = self.score_samples(X)
        sorted_scores = sorted(scores.tolist())
        threshold_idx = int(self.contamination * len(sorted_scores))
        self.threshold_ = sorted_scores[min(threshold_idx, len(sorted_scores) - 1)]

        return self

    def predict(self, X):
        """Predict outlier labels.

        Args:
            X: Array or array-like of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,): -1 for anomalies, 1 for inliers.
        """
        if self.ensemble_ is None:
            raise ValueError("This IsolationForest instance is not fitted yet")

        scores = self.score_samples(X)
        results = []
        for score in scores:
            results.append(-1.0 if score <= self.threshold_ else 1.0)  # type: ignore

        return ndutils.asarray(results)

    def score_samples(self, X) -> Array:
        """Compute the anomaly score of each sample.

        Lower scores indicate more anomalous samples (sklearn convention).

        Args:
            X: Array or array-like of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,): anomaly scores.
        """
        if self.ensemble_ is None:
            raise ValueError("This IsolationForest instance is not fitted yet")

        if not isinstance(X, Array):
            X = ndutils.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()

        return -self.ensemble_.predict(X_flat, n_samples)  # type: ignore

    def decision_function(self, X):
        """Average anomaly score of X.

        Args:
            X: Array or array-like of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,): anomaly scores.
        """
        return self.score_samples(X)