"""Substratum: N-dimensional array library for numerical computation and spatial analysis."""

from ironforest._core import (
    Array,
    zeros,
    ones,
    full,
    asarray,
    eye,
    diag,
    column_stack,
    linalg,
    stats,
    random,
    spatial,
)

from . import models

__all__ = [
    "Array",
    "zeros",
    "ones",
    "full",
    "asarray",
    "eye",
    "diag",
    "column_stack",
    "linalg",
    "stats",
    "random",
    "spatial",
    "models",
]

__version__ = "0.2"
