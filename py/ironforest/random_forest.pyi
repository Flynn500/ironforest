
from typing import Optional, Literal

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
        ...

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
        ...

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
        ...


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
        ...

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
        ...
    
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
        ...