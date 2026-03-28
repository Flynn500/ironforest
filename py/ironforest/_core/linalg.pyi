"""Linear algebra routines operating on :class:`~ironforest._core.Array`.

Includes matrix operations and decompositions such as matmul, dot,
transpose, Cholesky, QR, least-squares, and eigendecomposition.

Example::

    from ironforest._core import Array
    from ironforest._core import linalg

    a = Array.asarray([[4.0, 2.0], [2.0, 3.0]])
    vals, vecs = linalg.eig(a)
"""

from ironforest._core import Array, ArrayLike


def matmul(a: ArrayLike, b: ArrayLike) -> Array[float]:
    """Matrix multiplication."""
    ...

def dot(a: ArrayLike, b: ArrayLike) -> Array[float]:
    """Dot/matrix product."""
    ...

def transpose(a: ArrayLike) -> Array[float]:
    """Transpose a 2D matrix."""
    ...

def cholesky(a: ArrayLike) -> Array[float]:
    """Compute Cholesky decomposition.

    Returns lower triangular matrix L where A = L @ L.T.

    Raises:
        ValueError: If matrix is not square or not positive-definite.
    """
    ...

def qr(a: ArrayLike) -> tuple[Array[float], Array[float]]:
    """QR decomposition.

    Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

    Raises:
        ValueError: If array is not 2D.
    """
    ...

def lstsq(a: ArrayLike, b: ArrayLike) -> tuple[Array[float], Array[float]]:
    """Return the least-squares solution to a linear matrix equation.

    Solves the equation ax = b by computing a vector x that minimizes
    the Euclidean 2-norm ||b - ax||^2.

    Args:
        a: Coefficient matrix of shape (M, N).
        b: Ordinate or "dependent variable" values of shape (M,) or (M, K).

    Returns:
        Tuple of (x, residuals) where x is the least squares solution.

    Raises:
        ValueError: If array is not 2D.
    """
    ...

def weighted_lstsq(a: ArrayLike, b: ArrayLike, weights: ArrayLike) -> tuple[Array[float], Array[float]]:
    """Return the weighted least-squares solution to a linear matrix equation.

    Solves the equation ax = b by computing a vector x that minimizes
    the weighted Euclidean norm sum(w_i * (b_i - (ax)_i)^2).

    Args:
        a: Coefficient matrix of shape (M, N).
        b: Ordinate or "dependent variable" values of shape (M,) or (M, K).
        weights: Weight array of shape (M,) with non-negative values.

    Returns:
        Tuple of (x, weighted_residuals) where x is the weighted least squares solution
        and weighted_residuals = sqrt(W) * (b - a @ x).

    Raises:
        ValueError: If arrays have incompatible dimensions or weights are negative.
    """
    ...

def eig(a: ArrayLike) -> tuple[Array[float], Array[float]]:
    """Compute eigenvalues and eigenvectors.

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
        and eigenvectors is a 2D array with eigenvectors as columns.
        Eigenvalues are sorted by absolute value (descending).

    Raises:
        ValueError: If matrix is not square.
    """
    ...

def eig_with_params(a: ArrayLike, max_iter: int = 1000, tol: float = 1e-10) -> tuple[Array[float], Array[float]]:
    """Eigendecomposition with custom iteration parameters.

    Args:
        a: Input square matrix.
        max_iter: Maximum number of QR iterations.
        tol: Convergence tolerance for off-diagonal elements.

    Returns:
        Tuple of (eigenvalues, eigenvectors).

    Raises:
        ValueError: If matrix is not square.
    """
    ...

def eigvals(a: ArrayLike) -> Array[float]:
    """Compute eigenvalues only.

    More efficient than eig() when eigenvectors are not needed.

    Returns:
        1D array of eigenvalues sorted by absolute value (descending).

    Raises:
        ValueError: If matrix is not square.
    """
    ...

def diagonal(a: ArrayLike, k: int | None = None) -> Array[float]:
    """Extract the k-th diagonal from a 2D array."""
    ...

def outer(a: ArrayLike, b: ArrayLike) -> Array[float]:
    """Compute the outer product of two 1D arrays.

    Args:
        a: First 1D array-like.
        b: Second 1D array-like.

    Returns:
        2D array representing the outer product.
    """
    ...
