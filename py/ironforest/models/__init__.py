"""Machine learning models built on top of ironforest."""

from .linear_regression import LinearRegression
from .local_regression import LocalRegression
from .knn import KNNClassifier
from .knn import KNNRegressor

__all__ = [
    "LinearRegression",
    "LocalRegression",
    "KNNClassifier",
    "KNNRegressor"
]
