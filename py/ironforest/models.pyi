"""Type stubs for ironforest.models module."""

from .linear_regression import LinearRegression as LinearRegression
from .decision_tree import DecisionTreeClassifier as DecisionTreeClassifier
from .decision_tree import DecisionTreeRegressor as DecisionTreeRegressor
from .random_forest import RandomForestRegressor as RandomForestRegressor
from .random_forest import RandomForestClassifier as RandomForestClassifier

__all__ = [
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor"
]
