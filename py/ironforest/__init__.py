"""IronForest: A library supporting spatial queries & tree-based machine learning."""

from ironforest._core import (
    Array,
    SpatialIndex,
    TreeType,
    linalg,
    ndutils,
    stats,
    random,
    spatial,
)

from . import models

__all__ = [
    "Array",
    "SpatialIndex",
    "TreeType",
    "ndutils",
    "linalg",
    "stats",
    "random",
    "spatial",
    "models",
]

__version__ = "0.6.0"
