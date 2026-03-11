"""Decision Tree classifier and regressor built on tree_engine."""

from typing import Optional, Literal
from ironforest._core import Array, ndutils, Tree, TreeConfig, TaskType, SplitCriterion


class DecisionTreeClassifier:
    """A decision tree classifier.
    """

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["gini", "entropy", "random_projection"] = "gini",
        projection_type: Literal["gaussian", "sparse"] = "gaussian",
        projection_density: float | None = None,
        random_state: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.projection_type = projection_type
        self.projection_density = projection_density
        self.random_state = random_state
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values (class labels).

        Returns:  
            self : DecisionTreeClassifier
                Fitted estimator.
        """

        if not isinstance(X, Array):
            X = ndutils.asarray(X)
        if not isinstance(y, Array):
            y = ndutils.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have same first dimension, got {n_samples} and {y.shape[0]}"
            )

        # Determine number of classes
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = n_features

        # Create config
        criterion_map = {
            "gini": SplitCriterion.gini(),
            "entropy": SplitCriterion.entropy(),
            "random_projection": SplitCriterion.random_projection(),
        }

        config = TreeConfig(
            task_type=TaskType.classification(),
            n_classes=self.n_classes_,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=criterion_map[self.criterion],
            seed=self.random_state,
            projection_type=self.projection_type,
            projection_density=self.projection_density,
        )

        X_flat = X.ravel()
        self.tree_ = Tree.fit(config, X_flat, y, n_samples, n_features)

        return self

    def predict(self, X):
        """Predict class for X.

        Args
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            y: Array of shape (n_samples,)
                The predicted classes.
        """
        if self.tree_ is None:
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet")

        if not isinstance(X, Array):
            X = ndutils.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()
        return self.tree_.predict(X_flat, n_samples)

    @property
    def n_nodes(self):
        """Number of nodes in the tree."""
        if self.tree_ is None:
            return None
        return self.tree_.n_nodes

    @property
    def max_depth_reached(self):
        """Maximum depth actually reached during training."""
        if self.tree_ is None:
            return None
        return self.tree_.max_depth_reached


class DecisionTreeRegressor:
    """A decision tree regressor.
    """

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["mse", "random_projection"] = "mse",
        projection_type: Literal["gaussian", "sparse"] = "gaussian",
        projection_density: float | None = None,
        random_state: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.projection_type = projection_type
        self.projection_density = projection_density
        self.random_state = random_state
        self.tree_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """Build a decision tree regressor from the training set (X, y).

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values.

        Returns:
            self : DecisionTreeRegressor
                Fitted estimator.
        """

        if not isinstance(X, Array):
            X = ndutils.asarray(X)
        if not isinstance(y, Array):
            y = ndutils.asarray(y)

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

        criterion_map = {
            "mse": SplitCriterion.mse(),
            "random_projection": SplitCriterion.random_projection(),
        }

        config = TreeConfig(
            task_type=TaskType.regression(),
            n_classes=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=criterion_map[self.criterion],
            seed=self.random_state,
            projection_type=self.projection_type,
            projection_density=self.projection_density,
        )
        X_flat = X.ravel()
        self.tree_ = Tree.fit(config, X_flat, y, n_samples, n_features)

        return self

    def predict(self, X):
        """Predict regression target for X.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            y: Array of shape (n_samples,)
                The predicted values.
        """
        if self.tree_ is None:
            raise ValueError("This DecisionTreeRegressor instance is not fitted yet")

        if not isinstance(X, Array):
            X = ndutils.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        X_flat = X.ravel()
        return self.tree_.predict(X_flat, n_samples)

    @property
    def n_nodes(self):
        """Number of nodes in the tree."""
        if self.tree_ is None:
            return None
        return self.tree_.n_nodes

    @property
    def max_depth_reached(self):
        """Maximum depth actually reached during training."""
        if self.tree_ is None:
            return None
        return self.tree_.max_depth_reached


