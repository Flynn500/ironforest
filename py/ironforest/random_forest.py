
from typing import Optional, Literal
from ironforest._core import Array, asarray, Tree, TreeConfig, TaskType, SplitCriterion
import random

class RandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["gini", "entropy"] = "gini",
        random_state: int = 42,
    ):
        """Random forest classifier.

        Args:
            n_estimators: int, default=100
                The number of trees in the forest.
            max_depth: int or None, default=None
                The maximum depth of each tree. If None, nodes are expanded
                until all leaves are pure or contain fewer than
                min_samples_split samples.
            min_samples_split: int, default=2
                The minimum number of samples required to split an internal node.
            min_samples_leaf: int, default=1
                The minimum number of samples required to be at a leaf node.
            max_features: int or None, default=None
                The number of features to consider when looking for the best split.
                If None, sqrt(n_features) is used.
            criterion: {"gini", "entropy"}, default="gini"
                The function to measure the quality of a split.
            random_state: int, default=42
                Controls the randomness of the bootstrapping and tree construction.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees_ = []
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """Build a random forest classifier from the training set (X, y).

        The forest is built by training multiple decision trees on
        bootstrap samples of the dataset and aggregating their predictions.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values (class labels).

        Returns:
            self : RandomForestClassifier
        """

        if not isinstance(X, Array):
            X = asarray(X)
        if not isinstance(y, Array):
            y = asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have same first dimension, got {n_samples} and {y.shape[0]}"
            )

        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = n_features

        max_features = self.max_features or int(n_features ** 0.5) or 1
        
        rng = random.Random(self.random_state)

        self.trees_ = []
        for i in range(self.n_estimators):
            indices = [rng.randint(0, n_samples - 1) for _ in range(n_samples)]

            X_boot = asarray([X[idx, col] for idx in indices for col in range(n_features)])
            y_boot = asarray([y[idx] for idx in indices])

            config = TreeConfig(
                task_type=TaskType.classification(),
                n_classes=self.n_classes_,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                criterion={"gini": SplitCriterion.gini(), "entropy": SplitCriterion.entropy()}[self.criterion],
                seed=rng.randint(0, 2**31),
            )

            tree = Tree.fit(config, X_boot, y_boot, n_samples, n_features)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """Build a random forest classifier from the training set (X, y).

        The forest is built by training multiple decision trees on
        bootstrap samples of the dataset and aggregating their predictions.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values (class labels).

        Returns:
            self : RandomForestClassifier
        """

        if not self.trees_:
            raise ValueError("This RandomForestClassifier instance is not fitted yet")

        if not isinstance(X, Array):
            X = asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()

        all_preds = [tree.predict(X_flat, n_samples) for tree in self.trees_]

        results = []
        for i in range(n_samples):
            votes = [0] * self.n_classes_ # type: ignore
            for preds in all_preds:
                votes[int(preds[i])] += 1
            results.append(float(votes.index(max(votes))))

        return asarray(results)


class RandomForestRegressor:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ):
        """Random forest regressor.

        Args:
            n_estimators: int, default=100
                The number of trees in the forest.
            max_depth: int or None, default=None
                The maximum depth of each tree. If None, nodes are expanded
                until all leaves contain fewer than min_samples_split samples.
            min_samples_split: int, default=2
                The minimum number of samples required to split an internal node.
            min_samples_leaf: int, default=1
                The minimum number of samples required to be at a leaf node.
            max_features: int or None, default=None
                The number of features to consider when looking for the best split.
                If None, all features are considered.
            random_state: int, default=42
                Controls the randomness of the bootstrapping and tree construction.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.n_features_ = None

    def fit(self, X, y):
        """Build a random forest regressor from the training set (X, y).

        The forest is built by training multiple decision trees on
        bootstrap samples of the dataset and averaging their predictions.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values.

        Returns:
            self : RandomForestRegressor
        """

        if not isinstance(X, Array):
            X = asarray(X)
        if not isinstance(y, Array):
            y = asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have same first dimension, got {n_samples} and {y.shape[0]}"
            )

        self.n_features_ = n_features

        max_features = self.max_features or n_features

        rng = random.Random(self.random_state)

        self.trees_ = []
        for _ in range(self.n_estimators):
            indices = [rng.randint(0, n_samples - 1) for _ in range(n_samples)]

            X_boot = asarray([X[idx, col] for idx in indices for col in range(n_features)])
            y_boot = asarray([y[idx] for idx in indices])

            config = TreeConfig(
                task_type=TaskType.regression(),
                n_classes=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                criterion=SplitCriterion.mse(),
                seed=rng.randint(0, 2**31),
            )

            tree = Tree.fit(config, X_boot, y_boot, n_samples, n_features)
            self.trees_.append(tree)

        return self
    
    def predict(self, X):
        """Predict regression targets for samples in X.

        The predicted value for each sample is computed as the mean
        prediction of all trees in the forest.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                The predicted values.
        """

        if not self.trees_:
            raise ValueError("This RandomForestRegressor instance is not fitted yet")

        if not isinstance(X, Array):
            X = asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()

        all_preds = [tree.predict(X_flat, n_samples) for tree in self.trees_]

        results = []
        for i in range(n_samples):
            total = sum(preds[i] for preds in all_preds)
            results.append(total / len(self.trees_))

        return asarray(results)