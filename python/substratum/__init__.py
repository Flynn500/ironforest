"""Substratum: N-dimensional array library for numerical computation and spatial analysis."""

from substratum._core import (
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
]

__version__ = "0.2"
