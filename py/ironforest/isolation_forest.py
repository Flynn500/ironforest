"""Isolation Forest for anomaly detection built on tree_engine."""

from typing import Optional
from ironforest._core import Array, asarray, Tree, TreeConfig, TaskType, SplitCriterion
import random


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
            n_estimators: int, default=100
                The number of isolation trees in the forest.
            max_samples: int, default=256
                The number of samples to draw from X to train each tree.
                If max_samples is larger than the number of samples provided,
                all samples will be used for all trees (no sampling).
            contamination: float, default=0.1
                The proportion of outliers in the dataset. Used to define the
                threshold for the decision function. Must be in (0, 0.5].
            max_features: int or None, default=None
                The number of features to consider when looking for the best split.
                If None, all features are considered.
            random_state: int, default=42
                Controls the randomness of the sampling and tree construction.
        """

        if contamination <= 0 or contamination > 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.n_features_ = None
        self.offset_ = None
        self.threshold_ = None

    def fit(self, X, y=None):
        """Build an isolation forest from the training set X.

        The forest is built by training multiple isolation trees on
        random subsamples of the dataset.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Ignored
                Not used, present for API consistency by convention.

        Returns:
            self : IsolationForest
        """

        if not isinstance(X, Array):
            X = asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        max_samples = min(self.max_samples, n_samples)
        max_features = self.max_features or n_features

        rng = random.Random(self.random_state)

        self.trees_ = []
        for _ in range(self.n_estimators):
            if max_samples < n_samples:
                indices = rng.sample(range(n_samples), max_samples)
            else:
                indices = list(range(n_samples))

            X_sample = asarray([X[idx, col] for idx in indices for col in range(n_features)])

            config = TreeConfig(
                task_type=TaskType.anomaly_detection(),
                n_classes=0,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=max_features,
                criterion=SplitCriterion.random(),
                seed=rng.randint(0, 2**31),
            )

            y_dummy = asarray([0.0] * len(indices))
            tree = Tree.fit(config, X_sample, y_dummy, len(indices), n_features)
            self.trees_.append(tree)

        self.offset_ = self._compute_offset(max_samples)

        scores = self.score_samples(X)
        sorted_scores = sorted(scores.tolist())
        threshold_idx = int(self.contamination * len(sorted_scores))
        self.threshold_ = sorted_scores[threshold_idx]

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                Returns -1 for anomalies/outliers and 1 for inliers.
        """

        if not self.trees_:
            raise ValueError("This IsolationForest instance is not fitted yet")

        if self.threshold_ is None:
            raise ValueError("This IsolationForest instance is not fitted yet")

        scores = self.score_samples(X)
        results = []
        for score in scores:
            results.append(-1.0 if score < self.threshold_ else 1.0)

        return asarray(results)

    def score_samples(self, X):
        """Compute the anomaly score of each sample.

        The anomaly score is computed as the negative average path length
        in the trees, normalized by the expected path length. Lower scores
        indicate anomalies.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                The anomaly score of the input samples. Lower scores indicate
                anomalies, higher scores indicate normal instances.
        """

        if not self.trees_:
            raise ValueError("This IsolationForest instance is not fitted yet")

        if self.offset_ is None:
            raise ValueError("This IsolationForest instance is not fitted yet")

        if not isinstance(X, Array):
            X = asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()

        all_path_lengths = [tree.predict_path_lengths(X_flat, n_samples) for tree in self.trees_]

        results = []
        for i in range(n_samples):
            avg_path_length = sum(paths[i] for paths in all_path_lengths) / len(self.trees_)
            score = -2.0 ** (-avg_path_length / self.offset_)
            results.append(score)

        return asarray(results)

    def decision_function(self, X):
        """Average anomaly score of X.

        The anomaly score is based on the average path length in the trees.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                The anomaly score. The more negative, the more anomalous.
        """
        return self.score_samples(X)

    @staticmethod
    def _compute_offset(n):
        """Compute the offset for normalizing path lengths.

        This is the average path length of an unsuccessful search in a BST,
        which is used to normalize the path lengths.

        Args:
            n: Number of samples

        Returns:
            The normalization offset
        """
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.0
        harmonic = sum(1.0 / i for i in range(1, n))
        return 2.0 * harmonic - 2.0 * (n - 1) / n
