
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.n_features_ = None

    def fit(self, X, y):
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

        import random
        rng = random.Random(self.random_state)

        self.trees_ = []
        for i in range(self.n_estimators):
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