"""A Rust-based ndarray library. """

from typing import Iterator, List, Sequence, Tuple, overload

class linalg:
    """Linear algebra functions."""

    @staticmethod
    def matmul(a: Array, b: Array) -> Array:
        """Matrix multiplication."""
        ...

    @staticmethod
    def dot(a: Array, b: Array) -> Array:
        """Dot/matrix product."""
        ...

    @staticmethod
    def transpose(a: Array) -> Array:
        """Transpose a 2D matrix."""
        ...

    @staticmethod
    def cholesky(a: Array) -> Array:
        """Compute Cholesky decomposition.

        Returns lower triangular matrix L where A = L @ L.T.

        Raises:
            ValueError: If matrix is not square or not positive-definite.
        """
        ...

    @staticmethod
    def qr(a: Array) -> tuple[Array, Array]:
        """QR decomposition.

        Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

        Raises:
            ValueError: If array is not 2D.
        """
        ...

    @staticmethod
    def eig(a: Array) -> tuple[Array, Array]:
        """Compute eigenvalues and eigenvectors.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
            and eigenvectors is a 2D array with eigenvectors as columns.
            Eigenvalues are sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    @staticmethod
    def eig_with_params(a: Array, max_iter: int = 1000, tol: float = 1e-10) -> tuple[Array, Array]:
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

    @staticmethod
    def eigvals(a: Array) -> Array:
        """Compute eigenvalues only.

        More efficient than eig() when eigenvectors are not needed.

        Returns:
            1D array of eigenvalues sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    @staticmethod
    def diagonal(a: Array, k: int | None = None) -> Array:
        """Extract the k-th diagonal from a 2D array."""
        ...


class stats:
    """Statistical functions."""

    @staticmethod
    def sum(a: Array) -> float:
        """Sum of all elements."""
        ...

    @staticmethod
    def mean(a: Array) -> float:
        """Mean of all elements."""
        ...

    @staticmethod
    def var(a: Array) -> float:
        """Variance of all elements (population variance)."""
        ...

    @staticmethod
    def std(a: Array) -> float:
        """Standard deviation of all elements."""
        ...

    @staticmethod
    def median(a: Array) -> float:
        """Median of all elements."""
        ...

    @staticmethod
    def quantile(a: Array, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...

    @staticmethod
    def any(a: Array) -> bool:
        """True if any element is non-zero."""
        ...

    @staticmethod
    def all(a: Array) -> bool:
        """True if all elements are non-zero."""
        ...

    @staticmethod
    def pearson(a: Array, b: Array) -> float:
        """Compute Pearson correlation coefficient between two arrays.

        Args:
            a: First 1D array.
            b: Second 1D array of the same length.

        Returns:
            Pearson correlation coefficient between -1 and 1.
        """
        ...

    @staticmethod
    def spearman(a: Array, b: Array) -> float:
        """Compute Spearman rank correlation coefficient between two arrays.

        Args:
            a: First 1D array.
            b: Second 1D array of the same length.

        Returns:
            Spearman correlation coefficient between -1 and 1.
        """
        ...


class random:
    """Random number generation."""

    class Generator:
        """Random number generator."""

        def __init__(self) -> None:
            """Create time-seeded generator."""
            ...

        @staticmethod
        def from_seed(seed: int) -> Generator:
            """Create generator with explicit seed."""
            ...

        def uniform(self, low: float, high: float, shape: Sequence[int]) -> Array: ...
        def standard_normal(self, shape: Sequence[int]) -> Array: ...
        def normal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array: ...
        def randint(self, low: int, high: int, shape: Sequence[int]) -> Array: ...
        def gamma(self, shape_param: float, scale: float, shape: Sequence[int]) -> Array:
            """Generate gamma-distributed random samples.

            Args:
                shape_param: Shape parameter (k or alpha), must be positive.
                scale: Scale parameter (theta), must be positive.
                shape: Output array shape.

            Returns:
                Array of gamma-distributed samples.
            """
            ...
        def beta(self, alpha: float, beta: float, shape: Sequence[int]) -> Array:
            """Generate beta-distributed random samples.

            Args:
                alpha: First shape parameter, must be positive.
                beta: Second shape parameter, must be positive.
                shape: Output array shape.

            Returns:
                Array of beta-distributed samples in the interval (0, 1).
            """
            ...

        def lognormal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array:
            """Generate log-normal distributed random samples.

            Args:
                mu: Mean of the underlying normal distribution.
                sigma: Standard deviation of the underlying normal distribution.
                shape: Output array shape.

            Returns:
                Array of log-normal distributed samples.
            """
            ...

    @staticmethod
    def seed(seed: int) -> Generator:
        """Create a seeded random number generator."""
        ...

class Array:
    """N-dimensional array of float64 values."""

    def __init__(self, shape: Sequence[int], data: Sequence[float]) -> None:
        """Create array from shape and flat data list."""
        ...

    @staticmethod
    def zeros(shape: Sequence[int]) -> Array:
        """Create array filled with zeros."""
        ...

    @staticmethod
    def ones(shape: Sequence[int]) -> Array:
        """Create array filled with ones."""
        ...

    @staticmethod
    def full(shape: Sequence[int], fill_value: float) -> Array:
        """Create array filled with a specified value."""
        ...

    @staticmethod
    def asarray(data: Sequence[float], shape: Sequence[int] | None = None) -> Array:
        """Create array from a flat list of data.

        Args:
            data: Flat list of float values.
            shape: Optional shape. If None, creates a 1D array.
        """
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
        """Create a 2D identity matrix with ones on the k-th diagonal."""
        ...

    @staticmethod
    def diag(v: Sequence[float], k: int | None = None) -> Array:
        """Create a 2D array with v on the k-th diagonal."""
        ...

    @staticmethod
    def outer(a: Sequence[float], b: Sequence[float]) -> Array:
        """Compute the outer product of two 1D arrays."""
        ...

    @property
    def shape(self) -> List[int]:
        """Get the shape as a list."""
        ...

    def get(self, indices: Sequence[int]) -> float:
        """Get element at indices."""
        ...

    def tolist(self) -> List[float]:
        """Return data as flat list."""
        ...

    def diagonal(self, k: int | None = None) -> Array:
        """Extract the k-th diagonal from a 2D array."""
        ...
    
    def transpose(self) -> Array:
        """Transpose a 2D matrix."""
        ...

    def t(self) -> Array:
        """Transpose (alias for transpose())."""
        ...

    def matmul(self, other: Array) -> Array:
        """Matrix multiplication."""
        ...

    def dot(self, other: Array) -> Array:
        """Dot/matrix product."""
        ...
    
    def cholesky(self) -> Array:
        """Compute Cholesky decomposition.

        Returns lower triangular matrix L where A = L @ L.T.

        Raises:
            ValueError: If matrix is not square or not positive-definite.
        """
        ...

    def qr(self) -> tuple[Array, Array]:
        """QR decomposition.

        Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

        Raises:
            ValueError: If array is not 2D.
        """
        ...

    def eig(self) -> tuple[Array, Array]:
        """Compute eigenvalues and eigenvectors.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
            and eigenvectors is a 2D array with eigenvectors as columns.
            Eigenvalues are sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def eig_with_params(self, max_iter: int = 1000, tol: float = 1e-10) -> tuple[Array, Array]:
        """Eigendecomposition with custom iteration parameters.

        Args:
            max_iter: Maximum number of QR iterations.
            tol: Convergence tolerance for off-diagonal elements.

        Returns:
            Tuple of (eigenvalues, eigenvectors).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def eigvals(self) -> Array:
        """Compute eigenvalues only.

        More efficient than eig() when eigenvectors are not needed.
        Similar to numpy.linalg.eigvals.

        Returns:
            1D array of eigenvalues sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def __matmul__(self, other: Array) -> Array:
        """Matrix multiplication operator (@)."""
        ...


    def sin(self) -> Array: ...
    def cos(self) -> Array: ...
    def tan(self) -> Array: ...
    def arcsin(self) -> Array: ...
    def arccos(self) -> Array: ...
    def arctan(self) -> Array: ...
    def exp(self) -> Array: ...
    def sqrt(self) -> Array: ...
    def log(self) -> Array:
        """Natural logarithm, element-wise."""
        ...
    def abs(self) -> Array:
        """Absolute value, element-wise."""
        ...
    def sign(self) -> Array:
        """Returns -1, 0, or 1 for each element based on sign."""
        ...
    def clip(self, min: float, max: float) -> Array: ...

    # Statistical reductions
    def sum(self) -> float:
        """Sum of all elements."""
        ...
    def mean(self) -> float:
        """Mean of all elements."""
        ...
    def var(self) -> float:
        """Variance of all elements (population variance)."""
        ...
    def std(self) -> float:
        """Standard deviation of all elements."""
        ...
    def median(self) -> float:
        """Median of all elements."""
        ...
    def quantile(self, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...

    # Logical reductions
    def any(self) -> bool:
        """True if any element is non-zero."""
        ...
    def all(self) -> bool:
        """True if all elements are non-zero."""
        ...

    # Correlation methods
    def pearson(self, other: Array) -> float:
        """Compute Pearson correlation coefficient with another array.

        Args:
            other: Another 1D array of the same length.

        Returns:
            Pearson correlation coefficient between -1 and 1.
        """
        ...
    def spearman(self, other: Array) -> float:
        """Compute Spearman rank correlation coefficient with another array.

        Args:
            other: Another 1D array of the same length.

        Returns:
            Spearman correlation coefficient between -1 and 1.
        """
        ...

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> float | Array:
        """Get element (1D) or row (2D+) at index."""
        ...
    @overload
    def __getitem__(self, index: slice) -> Array: ...
    @overload
    def __getitem__(self, index: Tuple[int, ...]) -> float | Array:
        """Get element at (i, j, ...) or sub-array if fewer indices than dimensions."""
        ...
    @overload
    def __setitem__(self, index: int, value: float) -> None:
        """Set element at index (1D arrays only)."""
        ...
    @overload
    def __setitem__(self, index: Tuple[int, ...], value: float) -> None:
        """Set element at (i, j, ...)."""
        ...
    def __iter__(self) -> Iterator[float]: ...


    @overload
    def __add__(self, other: Array) -> Array: ...
    @overload
    def __add__(self, other: float) -> Array: ...
    def __radd__(self, other: float) -> Array: ...

    @overload
    def __sub__(self, other: Array) -> Array: ...
    @overload
    def __sub__(self, other: float) -> Array: ...
    def __rsub__(self, other: float) -> Array: ...

    @overload
    def __mul__(self, other: Array) -> Array: ...
    @overload
    def __mul__(self, other: float) -> Array: ...
    def __rmul__(self, other: float) -> Array: ...

    @overload
    def __truediv__(self, other: Array) -> Array: ...
    @overload
    def __truediv__(self, other: float) -> Array: ...
    def __rtruediv__(self, other: float) -> Array: ...

    def __neg__(self) -> Array: ...
    def __repr__(self) -> str: ...


class Generator:
    """Random number generator."""

    def __init__(self) -> None:
        """Create time-seeded generator."""
        ...

    @staticmethod
    def from_seed(seed: int) -> Generator:
        """Create generator with explicit seed."""
        ...

    def uniform(self, low: float, high: float, shape: Sequence[int]) -> Array: ...
    def standard_normal(self, shape: Sequence[int]) -> Array: ...
    def normal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array: ...
    def randint(self, low: int, high: int, shape: Sequence[int]) -> Array: ...
    def gamma(self, shape_param: float, scale: float, shape: Sequence[int]) -> Array:
        """Generate gamma-distributed random samples.

        Args:
            shape_param: Shape parameter (k or alpha), must be positive.
            scale: Scale parameter (theta), must be positive.
            shape: Output array shape.

        Returns:
            Array of gamma-distributed samples.
        """
        ...
    def beta(self, alpha: float, beta: float, shape: Sequence[int]) -> Array:
        """Generate beta-distributed random samples.

        Args:
            alpha: First shape parameter, must be positive.
            beta: Second shape parameter, must be positive.
            shape: Output array shape.

        Returns:
            Array of beta-distributed samples in the interval (0, 1).
        """
        ...

    def lognormal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array:
        """Generate log-normal distributed random samples.

        Args:
            mu: Mean of the underlying normal distribution.
            sigma: Standard deviation of the underlying normal distribution.
            shape: Output array shape.

        Returns:
            Array of log-normal distributed samples.
        """
        ...


def zeros(shape: Sequence[int]) -> Array:
    """Create array filled with zeros."""
    ...

def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
    """Create a 2D identity matrix with ones on the k-th diagonal."""
    ...

def diag(v: Sequence[float], k: int | None = None) -> Array:
    """Create a 2D array with v on the k-th diagonal."""
    ...

def outer(a: Sequence[float], b: Sequence[float]) -> Array:
    """Compute the outer product of two 1D arrays."""
    ...

def ones(shape: Sequence[int]) -> Array:
    """Create array filled with ones."""
    ...

def full(shape: Sequence[int], fill_value: float) -> Array:
    """Create array filled with a specified value."""
    ...

def asarray(data: Sequence[float], shape: Sequence[int] | None = None) -> Array:
    """Create array from a flat list of data.

    Args:
        data: Flat list of float values.
        shape: Optional shape. If None, creates a 1D array.
    """
    ...
