"""Array utility functions: creation, manipulation, and inspection helpers.

Provides convenience wrappers around :class:`~ironforest._core.Array`
for stacking, splitting, tiling, sorting, and type conversion.

Example::

    from ironforest._core import ndutils

    a = ndutils.asarray([[0, 1, 2], [3, 4, 5]])
    b = ndutils.linspace(0.0, 1.0, 5)
"""

from typing import Any, Sequence
from ironforest._core import Array, ArrayLike, Dtype


def zeros(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
    """Create array filled with zeros."""
    ...

def ones(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
    """Create array filled with ones."""
    ...

def full(shape: Sequence[int], fill_value: float, dtype: Dtype | None = None) -> Array:
    """Create array filled with a specified value."""
    ...

def asarray(data: ArrayLike, shape: Sequence[int] | None = None, dtype: Dtype | None = None) -> Array:
    """Create array from data with optional reshape.

    Args:
        data: Array-like data (Array, NumPy array, list, nested list, or scalar).
        shape: Optional shape. If None, creates a 1D array.
        dtype: Element type. "float"/"float64" (default) or "int"/"int64".
    """
    ...

def eye(n: int, m: int | None = None, k: int | None = None, dtype: Dtype | None = None) -> Array:
    """Create a 2D identity matrix with ones on the k-th diagonal."""
    ...

def diag(v: ArrayLike, k: int | None = None) -> Array:
    """Create a 2D array with v on the k-th diagonal.

    Args:
        v: 1D array-like of diagonal values.
        k: Diagonal offset (0=main, >0=upper, <0=lower).
    """
    ...

def column_stack(arrays: Sequence[Array]) -> Array:
    """Stack 1D or 2D arrays as columns into a 2D array.

    Args:
        arrays: Sequence of 1D or 2D arrays to stack. 1D arrays are treated
            as columns (n, 1). All arrays must have the same number of rows.

    Raises:
        ValueError: If arrays is empty or arrays have mismatched row counts.
    """
    ...

def outer(a: ArrayLike, b: ArrayLike) -> Array:
    """Compute the outer product of two 1D arrays.

    Args:
        a: First 1D array-like.
        b: Second 1D array-like.

    Returns:
        2D array representing the outer product.
    """
    ...

def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
) -> Array:
    """Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples calculated over the interval [`start`, `stop`].
    If `start` and `stop` are arrays, they will be broadcast together, and the output
    shape will be `(*broadcast_shape, num)`.

    Args:
        start: The starting value(s) of the sequence.
        stop: The ending value(s) of the sequence.
        num: Number of samples to generate. Must be >= 2.

    Returns:
        An array of evenly spaced values with shape `(*broadcast_shape, num)`.

    Raises:
        ValueError: If `num` is less than 2.
        ValueError: If `start` and `stop` are not broadcastable.
    """
    ...

def from_numpy(arr: Any) -> Array:
    """Convert a numpy ndarray to an Array.

    Args:
        arr: A numpy ndarray of dtype float64.

    Returns:
        A new Array containing the same data and shape as the numpy array.

    Raises:
        ValueError: If the numpy array cannot be converted (wrong dtype, etc.).
    """
    ...

def to_numpy(arr: Array) -> Any:
    """Convert an Array to a numpy ndarray.

    Args:
        arr: The Array to convert.

    Returns:
        A numpy ndarray of dtype float64 with the same shape and data.
    """
    ...
