# """Random Forest classifier and regressor built on tree_engine."""

# from typing import Optional, Literal
# import warnings
# from ironforest._core import (
#     Array,
#     ndutils,
#     Ensemble,
#     EnsembleConfig,
#     TreeConfig,
#     TaskType,
#     SplitCriterion,
#     SplitGeometry,
# )


# class RandomForestClassifier:
#     """Random forest classifier.

#     Builds an ensemble of decision trees on bootstrap samples
#     and aggregates predictions via majority voting.
#     """

#     def __init__(
#         self,
#         *,
#         n_estimators: int = 100,
#         max_depth: Optional[int] = None,
#         min_samples_split: int = 2,
#         min_samples_leaf: int = 1,
#         max_features: Optional[int] = None,
#         criterion: Literal["gini", "entropy"] = "gini",
#         split_geometry: Literal["axis", "random_projection"] = "axis",
#         projection_type: Literal["gaussian", "sparse"] = "gaussian",
#         projection_density: float | None = None,
#         random_state: int = 0,
#     ):
#         """
#         Args:
#             n_estimators: Number of trees in the forest.
#             max_depth: Maximum depth of each tree. None for unlimited.
#             min_samples_split: Minimum samples required to split a node.
#             min_samples_leaf: Minimum samples required at a leaf node.
#             max_features: Features to consider per split. None uses sqrt(n_features).
#             criterion: Split quality measure, "gini" or "entropy".
#             split_geometry: Split direction, "axis" for axis-aligned or "random_projection" for oblique splits.
#             random_state: Controls randomness of bootstrapping and tree construction.
#         """
#         warnings.warn("Random Forest will be deprecated in 0.7")
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.criterion = criterion
#         self.split_geometry = split_geometry
#         self.projection_type = projection_type
#         self.projection_density = projection_density
#         self.random_state = random_state
#         self.ensemble_ = None
#         self.n_classes_ = None
#         self.n_features_ = None

#     def fit(self, X, y):
#         """Build a random forest classifier from the training set (X, y).

#         Args:
#             X: Array or array-like of shape (n_samples, n_features).
#             y: Array or array-like of shape (n_samples,), class labels.

#         Returns:
#             self
#         """
#         if not isinstance(X, Array):
#             X = ndutils.asarray(X)
#         if not isinstance(y, Array):
#             y = ndutils.asarray(y)

#         if X.dtype == "float32":
#             X = X.astype("float64")

#         if X.ndim != 2:
#             raise ValueError(f"X must be 2D array, got {X.ndim}D")
#         if y.ndim != 1:
#             raise ValueError(f"y must be 1D array, got {y.ndim}D")

#         n_samples, n_features = X.shape
#         if y.shape[0] != n_samples:
#             raise ValueError(
#                 f"X and y must have same first dimension, got {n_samples} and {y.shape[0]}"
#             )

#         self.n_classes_ = int(y.max()) + 1
#         self.n_features_ = n_features

#         criterion_map = {
#             "gini": SplitCriterion.gini(),
#             "entropy": SplitCriterion.entropy(),
#         }
#         split_geometry_map = {
#             "axis": SplitGeometry.axis(),
#             "random_projection": SplitGeometry.random_projection(),
#         }

#         tree_config = TreeConfig(
#             task_type=TaskType.classification(),
#             n_classes=self.n_classes_,
#             max_depth=self.max_depth,
#             min_samples_split=self.min_samples_split,
#             min_samples_leaf=self.min_samples_leaf,
#             max_features=self.max_features,
#             criterion=criterion_map[self.criterion],
#             split_geometry=split_geometry_map[self.split_geometry],
#             seed=self.random_state,
#             projection_type=self.projection_type,
#             projection_density=self.projection_density,
#         )

#         config = EnsembleConfig(
#             n_trees=self.n_estimators,
#             tree_config=tree_config,
#             bootstrap=True,
#             max_samples=None,
#             seed=self.random_state,
#         )

#         X_flat = X.ravel()
#         y_flat = y.ravel()

#         self.ensemble_ = Ensemble.fit(config, X_flat, y_flat, n_samples, n_features)

#         return self

#     def predict(self, X):
#         """Predict class labels for samples in X.

#         Args:
#             X: Array or array-like of shape (n_samples, n_features).

#         Returns:
#             Array of shape (n_samples,): predicted class labels.
#         """
#         if self.ensemble_ is None:
#             raise ValueError("This RandomForestClassifier instance is not fitted yet")

#         if not isinstance(X, Array):
#             X = ndutils.asarray(X)

#         if X.ndim != 2:
#             raise ValueError(f"X must be 2D array, got {X.ndim}D")

#         if X.shape[1] != self.n_features_:
#             raise ValueError(
#                 f"X has {X.shape[1]} features, but model was trained with {self.n_features_} features"
#             )

#         n_samples = X.shape[0]
#         X_flat = X.ravel()

#         return self.ensemble_.predict(X_flat, n_samples)


# class RandomForestRegressor:
#     """Random forest regressor.

#     Builds an ensemble of decision trees on bootstrap samples
#     and averages their predictions.
#     """

#     def __init__(
#         self,
#         *,
#         n_estimators: int = 100,
#         max_depth: Optional[int] = None,
#         min_samples_split: int = 2,
#         min_samples_leaf: int = 1,
#         max_features: Optional[int] = None,
#         criterion: Literal["mse"] = "mse",
#         split_geometry: Literal["axis", "random_projection"] = "axis",
#         projection_type: Literal["gaussian", "sparse"] = "gaussian",
#         projection_density: float | None = None,
#         random_state: int = 42,
#     ):
#         """
#         Args:
#             n_estimators: Number of trees in the forest.
#             max_depth: Maximum depth of each tree. None for unlimited.
#             min_samples_split: Minimum samples required to split a node.
#             min_samples_leaf: Minimum samples required at a leaf node.
#             max_features: Features to consider per split. None uses n_features/3.
#             criterion: Split quality measure (currently only "mse").
#             split_geometry: Split direction, "axis" for axis-aligned or "random_projection" for oblique splits.
#             random_state: Controls randomness of bootstrapping and tree construction.
#         """
#         warnings.warn("Random Forest will be deprecated in 0.7")
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.criterion = criterion
#         self.split_geometry = split_geometry
#         self.projection_type = projection_type
#         self.projection_density = projection_density
#         self.random_state = random_state
#         self.ensemble_ = None
#         self.n_features_ = None

#     def fit(self, X, y):
#         """Build a random forest regressor from the training set (X, y).

#         Args:
#             X: Array or array-like of shape (n_samples, n_features).
#             y: Array or array-like of shape (n_samples,), target values.

#         Returns:
#             self
#         """
#         if not isinstance(X, Array):
#             X = ndutils.asarray(X)
#         if not isinstance(y, Array):
#             y = ndutils.asarray(y)

#         if X.dtype == "float32":
#             X = X.astype("float64")

#         if X.ndim != 2:
#             raise ValueError(f"X must be 2D array, got {X.ndim}D")
#         if y.ndim != 1:
#             raise ValueError(f"y must be 1D array, got {y.ndim}D")

#         n_samples, n_features = X.shape
#         if y.shape[0] != n_samples:
#             raise ValueError(
#                 f"X and y must have same first dimension, got {n_samples} and {y.shape[0]}"
#             )

#         self.n_features_ = n_features

#         split_geometry_map = {
#             "axis": SplitGeometry.axis(),
#             "random_projection": SplitGeometry.random_projection(),
#         }

#         tree_config = TreeConfig(
#             task_type=TaskType.regression(),
#             n_classes=0,
#             max_depth=self.max_depth,
#             min_samples_split=self.min_samples_split,
#             min_samples_leaf=self.min_samples_leaf,
#             max_features=self.max_features,
#             criterion=SplitCriterion.mse(),
#             split_geometry=split_geometry_map[self.split_geometry],
#             seed=self.random_state,
#             projection_type=self.projection_type,
#             projection_density=self.projection_density,
#         )

#         config = EnsembleConfig(
#             n_trees=self.n_estimators,
#             tree_config=tree_config,
#             bootstrap=True,
#             max_samples=None,
#             seed=self.random_state,
#         )

#         X_flat = X.ravel()
#         y_flat = y.ravel()

#         self.ensemble_ = Ensemble.fit(config, X_flat, y_flat, n_samples, n_features)

#         return self

#     def predict(self, X):
#         """Predict regression targets for samples in X.

#         Args:
#             X: Array or array-like of shape (n_samples, n_features).

#         Returns:
#             Array of shape (n_samples,): predicted values.
#         """
#         if self.ensemble_ is None:
#             raise ValueError("This RandomForestRegressor instance is not fitted yet")

#         if not isinstance(X, Array):
#             X = ndutils.asarray(X)

#         if X.ndim != 2:
#             raise ValueError(f"X must be 2D array, got {X.ndim}D")

#         if X.shape[1] != self.n_features_:
#             raise ValueError(
#                 f"X has {X.shape[1]} features, but model was trained with {self.n_features_} features"
#             )

#         n_samples = X.shape[0]
#         X_flat = X.ravel()

#         return self.ensemble_.predict(X_flat, n_samples)