"""Type stubs for ironforest._core module.

This file provides type hints for the Rust-based _core module, including
tree_engine classes and other core functionality.
"""

from typing import Optional, TypeVar, Generic, Sequence
T_co = TypeVar("T_co", covariant=True)

class TaskType:
    """Task type for tree algorithms."""

    @staticmethod
    def classification() -> TaskType: ...

    @staticmethod
    def regression() -> TaskType: ...

    @staticmethod
    def anomaly_detection() -> TaskType: ...

    def __repr__(self) -> str: ...


class SplitCriterion:
    """Criterion for evaluating splits."""

    @staticmethod
    def gini() -> SplitCriterion: ...

    @staticmethod
    def entropy() -> SplitCriterion: ...

    @staticmethod
    def mse() -> SplitCriterion: ...

    @staticmethod
    def random() -> SplitCriterion: ...

    @staticmethod
    def random_projection() -> SplitCriterion: ...
    
    def __repr__(self) -> str: ...


class TreeConfig:
    """Configuration for a single decision tree."""

    def __init__(
        self,
        task_type: TaskType,
        n_classes: int = 2,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Optional[SplitCriterion] = None,
        seed: int = 42,
        projection_type = None,
        projection_density = None,
    ) -> None: ...

    @staticmethod
    def classification(n_classes: int) -> TreeConfig: ...

    @staticmethod
    def regression() -> TreeConfig: ...

    @staticmethod
    def isolation(max_samples: int) -> TreeConfig: ...

    @property
    def max_depth(self) -> Optional[int]: ...
    @max_depth.setter
    def max_depth(self, value: Optional[int]) -> None: ...

    @property
    def min_samples_split(self) -> int: ...
    @min_samples_split.setter
    def min_samples_split(self, value: int) -> None: ...

    @property
    def min_samples_leaf(self) -> int: ...
    @min_samples_leaf.setter
    def min_samples_leaf(self, value: int) -> None: ...

    @property
    def max_features(self) -> Optional[int]: ...
    @max_features.setter
    def max_features(self, value: Optional[int]) -> None: ...

    @property
    def seed(self) -> int: ...
    @seed.setter
    def seed(self, value: int) -> None: ...

    def __repr__(self) -> str: ...


class Tree:
    """A single decision tree."""

    @staticmethod
    def fit(
        config: TreeConfig,
        data: object,
        labels: object,
        n_samples: int,
        n_features: int,
    ) -> Tree: ...

    def predict(
        self,
        data: object,
        n_samples: int,
    ) -> object: ...

    def predict_anomaly_scores(
        self,
        data: object,
        n_samples: int,
    ) -> object: ...

    def predict_path_lengths(
        self,
        data: object,
        n_samples: int,
    ) -> object: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_features(self) -> int: ...

    @property
    def max_depth_reached(self) -> int: ...

    @property
    def n_training_samples(self) -> int: ...

    def __repr__(self) -> str: ...


class EnsembleConfig:
    """Configuration for an ensemble of trees."""

    def __init__(
        self,
        n_trees: int,
        tree_config: TreeConfig,
        bootstrap: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None: ...

    @staticmethod
    def random_forest_classifier(n_trees: int, n_classes: int) -> EnsembleConfig: ...

    @staticmethod
    def random_forest_regressor(n_trees: int) -> EnsembleConfig: ...

    @staticmethod
    def isolation_forest(n_trees: int, max_samples: int) -> EnsembleConfig: ...

    @property
    def n_trees(self) -> int: ...
    @n_trees.setter
    def n_trees(self, value: int) -> None: ...

    @property
    def bootstrap(self) -> bool: ...
    @bootstrap.setter
    def bootstrap(self, value: bool) -> None: ...

    @property
    def max_samples(self) -> Optional[int]: ...
    @max_samples.setter
    def max_samples(self, value: Optional[int]) -> None: ...

    @property
    def seed(self) -> int: ...
    @seed.setter
    def seed(self, value: int) -> None: ...

    def __repr__(self) -> str: ...


class Ensemble:
    """An ensemble of decision trees."""

    @staticmethod
    def fit(
        config: EnsembleConfig,
        data: object,
        labels: object,
        n_samples: int,
        n_features: int,
    ) -> Ensemble: ...

    def predict(
        self,
        data: object,
        n_samples: int,
    ) -> object: ...

    @property
    def n_trees(self) -> int: ...

    @property
    def n_training_samples(self) -> int: ...

    def __repr__(self) -> str: ...

"""A library for array-based computation."""

from typing import Iterator, List, Literal, Sequence, Tuple, overload, Any, Union

ArrayLike = Union['Array', Sequence[float], Sequence[Sequence[float]], float, int, Any]
Dtype = Literal["float", "float64", "f64", "int", "int64", "i64"]

class linalg:
    """Linear algebra functions."""

    @staticmethod
    def matmul(a: ArrayLike, b: ArrayLike) -> Array[float]:
        """Matrix multiplication."""
        ...

    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> Array[float]:
        """Dot/matrix product."""
        ...

    @staticmethod
    def transpose(a: ArrayLike) -> Array[float]:
        """Transpose a 2D matrix."""
        ...

    @staticmethod
    def cholesky(a: ArrayLike) -> Array[float]:
        """Compute Cholesky decomposition.

        Returns lower triangular matrix L where A = L @ L.T.

        Raises:
            ValueError: If matrix is not square or not positive-definite.
        """
        ...

    @staticmethod
    def qr(a: ArrayLike) -> tuple[Array[float], Array[float]]:
        """QR decomposition.

        Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

        Raises:
            ValueError: If array is not 2D.
        """
        ...

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def eigvals(a: ArrayLike) -> Array[float]:
        """Compute eigenvalues only.

        More efficient than eig() when eigenvectors are not needed.

        Returns:
            1D array of eigenvalues sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    @staticmethod
    def diagonal(a: ArrayLike, k: int | None = None) -> Array[float]:
        """Extract the k-th diagonal from a 2D array."""
        ...

    @staticmethod
    def outer(a: ArrayLike, b: ArrayLike) -> Array[float]:
        """Compute the outer product of two 1D arrays.

        Args:
            a: First 1D array-like.
            b: Second 1D array-like.

        Returns:
            2D array representing the outer product.
        """
        ...


class stats:
    """Statistical functions."""

    @staticmethod
    def sum(a: ArrayLike) -> float:
        """Sum of all elements."""
        ...

    @staticmethod
    def mean(a: ArrayLike) -> float:
        """Mean of all elements."""
        ...

    @staticmethod
    def var(a: ArrayLike) -> float:
        """Variance of all elements (population variance)."""
        ...

    @staticmethod
    def std(a: ArrayLike) -> float:
        """Standard deviation of all elements."""
        ...

    @staticmethod
    def median(a: ArrayLike) -> float:
        """Median of all elements."""
        ...

    @staticmethod
    def mode(a: ArrayLike) -> float | int:
        """Mode of all elements."""
        ...

    @overload
    @staticmethod
    def quantile(a: ArrayLike, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...

    @overload
    @staticmethod
    def quantile(a: ArrayLike, q: ArrayLike) -> Array[float]:
        """Compute multiple quantiles at once (vectorized).

        Args:
            a: Input array.
            q: Array of quantile values, each in [0, 1].

        Returns:
            1D array of quantile values corresponding to each q.
        """
        ...

    @staticmethod
    def any(a: ArrayLike) -> bool:
        """True if any element is non-zero."""
        ...

    @staticmethod
    def all(a: ArrayLike) -> bool:
        """True if all elements are non-zero."""
        ...

    @staticmethod
    def pearson(a: ArrayLike, b: ArrayLike) -> float:
        """Compute Pearson correlation coefficient between two arrays.

        Args:
            a: First 1D array.
            b: Second 1D array of the same length.

        Returns:
            Pearson correlation coefficient between -1 and 1.
        """
        ...

    @staticmethod
    def spearman(a: ArrayLike, b: ArrayLike) -> float:
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
        def from_seed(seed: int) -> random.Generator:
            """Create generator with explicit seed."""
            ...
        
        @staticmethod
        def new() -> random.Generator:
            """Create a generator."""
            ...

        def uniform(self, low: float, high: float, shape: Sequence[int]) -> Array[float]: ...
        def standard_normal(self, shape: Sequence[int]) -> Array[float]: ...
        def normal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array[float]: ...
        def randint(self, low: int, high: int, shape: Sequence[int]) -> Array: ...
        def gamma(self, shape_param: float, scale: float, shape: Sequence[int]) -> Array[float]:
            """Generate gamma-distributed random samples.

            Args:
                shape_param: Shape parameter (k or alpha), must be positive.
                scale: Scale parameter (theta), must be positive.
                shape: Output array shape.

            Returns:
                Array of gamma-distributed samples.
            """
            ...
        def beta(self, alpha: float, beta: float, shape: Sequence[int]) -> Array[float]:
            """Generate beta-distributed random samples.

            Args:
                alpha: First shape parameter, must be positive.
                beta: Second shape parameter, must be positive.
                shape: Output array shape.

            Returns:
                Array of beta-distributed samples in the interval (0, 1).
            """
            ...

        def lognormal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array[float]:
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

class ndutils:
    """Array utility functions."""

    @staticmethod
    def zeros(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
        """Create array filled with zeros."""
        ...

    @staticmethod
    def ones(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
        """Create array filled with ones."""
        ...

    @staticmethod
    def full(shape: Sequence[int], fill_value: float, dtype: Dtype | None = None) -> Array:
        """Create array filled with a specified value."""
        ...

    @staticmethod
    def asarray(data: ArrayLike, shape: Sequence[int] | None = None, dtype: Dtype | None = None) -> Array:
        """Create array from data with optional reshape.

        Args:
            data: Array-like data (Array, NumPy array, list, nested list, or scalar).
            shape: Optional shape. If None, creates a 1D array.
            dtype: Element type. "float"/"float64" (default) or "int"/"int64".
        """
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None, dtype: Dtype | None = None) -> Array:
        """Create a 2D identity matrix with ones on the k-th diagonal."""
        ...

    @staticmethod
    def diag(v: ArrayLike, k: int | None = None) -> Array:
        """Create a 2D array with v on the k-th diagonal.

        Args:
            v: 1D array-like of diagonal values.
            k: Diagonal offset (0=main, >0=upper, <0=lower).
        """
        ...

    @staticmethod
    def column_stack(arrays: Sequence[Array]) -> Array:
        """Stack 1D or 2D arrays as columns into a 2D array.

        Args:
            arrays: Sequence of 1D or 2D arrays to stack. 1D arrays are treated
                as columns (n, 1). All arrays must have the same number of rows.

        Raises:
            ValueError: If arrays is empty or arrays have mismatched row counts.
        """
        ...

    @staticmethod
    def outer(a: ArrayLike, b: ArrayLike) -> Array:
        """Compute the outer product of two 1D arrays.

        Args:
            a: First 1D array-like.
            b: Second 1D array-like.

        Returns:
            2D array representing the outer product.
        """
        ...
    
    @staticmethod
    def linspace(
        start: ArrayLike,
        stop: ArrayLike,
        num: int,
    ) -> Array:
        """
        Return evenly spaced numbers over a specified interval.

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

        Examples:
            >>> linspace(0.0, 1.0, num=5)
            NdArray([0.0, 0.25, 0.5, 0.75, 1.0])

            >>> linspace([0.0, 10.0], [1.0, 20.0], num=3)
            NdArray([[0.0, 0.5, 1.0], [10.0, 15.0, 20.0]])
        """
        ...

    @staticmethod
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

    @staticmethod
    def to_numpy(arr: Array) -> Any:
        """Convert an Array to a numpy ndarray.

        Args:
            arr: The Array to convert.

        Returns:
            A numpy ndarray of dtype float64 with the same shape and data.
        """
        ...


class Array(Sequence[T_co], Generic[T_co]):
    """N-dimensional array of float64 or int64 values."""

    def __init__(self, shape: Sequence[int], data: Sequence[float] | Sequence[int], dtype: Dtype | None = None) -> None:
        """Create array from shape and flat data list."""
        ...

    @property
    def dtype(self) -> Literal["float64", "int64"]:
        """The element type of this array."""
        ...

    @staticmethod
    def zeros(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
        """Create array filled with zeros."""
        ...

    @staticmethod
    def ones(shape: Sequence[int], dtype: Dtype | None = None) -> Array:
        """Create array filled with ones."""
        ...

    @staticmethod
    def full(shape: Sequence[int], fill_value: float, dtype: Dtype | None = None) -> Array:
        """Create array filled with a specified value."""
        ...

    @staticmethod
    def asarray(data: ArrayLike, shape: Sequence[int] | None = None, dtype: Dtype | None = None) -> Array:
        """Create array from data with optional reshape.

        Args:
            data: Array-like data (Array, NumPy array, list, nested list, or scalar).
            shape: Optional shape. If None, creates a 1D array.
            dtype: Element type. "float"/"float64" (default) or "int"/"int64".
        """
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None, dtype: Dtype | None = None) -> Array:
        """Create a 2D identity matrix with ones on the k-th diagonal."""
        ...

    @staticmethod
    def diag(v: ArrayLike, k: int | None = None) -> Array:
        """Create a 2D array with v on the k-th diagonal.

        Args:
            v: 1D array-like of diagonal values.
            k: Diagonal offset (0=main, >0=upper, <0=lower).
        """
        ...

    @property
    def shape(self) -> List[int]:
        """Get the shape as a list."""
        ...
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""

    def get(self, indices: Sequence[int]) -> float | int:
        """Get element at indices."""
        ...

    def tolist(self) -> list:
        """Return data as nested Python list (floats or ints depending on dtype)."""
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

    def ravel(self) -> Array:
        """Flatten the array to 1D (returns a copy)."""
        ...

    def take(self, indices: Sequence[int]) -> Array:
        """Take elements from the array at specified flat indices.

        Args:
            indices: List of flat (1D) indices to select from the array.
                These indices index into the flattened array.

        Returns:
            A new 1D array containing the elements at the specified indices.
        """
        ...

    def reshape(self, shape: Sequence[int]) -> Array:
        """Return a view of the array with a new shape (same total number of elements).

        Args:
            shape: New shape. The total number of elements must remain unchanged.

        Returns:
            A new Array with the specified shape and the same data.

        Raises:
            ValueError: If the total element count would change.
        """
        ...
    
    def item(self) -> float | int:
        """Convert a single-element array into a Python scalar (float or int)."""
        ...

    def to_numpy(self) -> Any:
        """Convert this Array to a numpy ndarray.

        Returns:
            A numpy ndarray of dtype float64 with the same shape and data.
        """
        ...

    def matmul(self, other: Array) -> Array[float]:
        """Matrix multiplication."""
        ...

    def dot(self, other: Array) -> Array[float]:
        """Dot/matrix product."""
        ...

    def __matmul__(self, other: Array) -> Array[float]:
        """Matrix multiplication operator (@)."""
        ...


    def sin(self) -> Array[float]: ...
    def cos(self) -> Array[float]: ...
    def tan(self) -> Array[float]: ...
    def arcsin(self) -> Array[float]: ...
    def arccos(self) -> Array[float]: ...
    def arctan(self) -> Array[float]: ...
    def exp(self) -> Array[float]: ...
    def sqrt(self) -> Array[float]: ...
    def log(self) -> Array[float]:
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
    def max(self) -> float:
        """Maximum of all elements."""
        ...
    def min(self) -> float:
        """Minimum of all elements."""
    @overload
    def quantile(self, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...
    @overload
    def quantile(self, q: ArrayLike) -> Array[float]:
        """Compute multiple quantiles at once (vectorized).

        Args:
            q: Array-like of quantile values, each in [0, 1].

        Returns:
            1D array of quantile values corresponding to each q.
        """
        ...

    # Logical reductions
    def any(self) -> bool:
        """True if any element is non-zero."""
        ...
    def all(self) -> bool:
        """True if all elements are non-zero."""
        ...

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> float | int | Array:
        """Get element (1D) or row (2D+) at index."""
        ...
    @overload
    def __getitem__(self, index: slice) -> Array:
        """Slice always returns an array."""
        ...
    @overload
    def __getitem__(self, index: List[bool]) -> Array:
        """Boolean mask along the first axis — selects elements (1D) or rows (2D+)."""
        ...
    @overload
    def __getitem__(self, index: Tuple[int, int]) -> float | int:
        """Two integer indices return a scalar."""
        ...
    @overload
    def __getitem__(self, index: Tuple[int, slice]) -> Array:
        """Integer and slice return an array."""
        ...
    @overload
    def __getitem__(self, index: Tuple[slice, int]) -> Array:
        """Slice and integer return an array."""
        ...
    @overload
    def __getitem__(self, index: Tuple[slice, slice]) -> Array:
        """Two slices return an array."""
        ...
    @overload
    def __getitem__(self, index: Tuple[int, ...]) -> float | int | Array:
        """Fallback: multiple indices can return scalar or array."""
        ...
    @overload
    def __setitem__(self, index: int, value: float | int) -> None:
        """Set element at index (1D arrays only)."""
        ...
    @overload
    def __setitem__(self, index: Tuple[int, ...], value: float | int) -> None:
        """Set element at (i, j, ...)."""
        ...
    def __iter__(self) -> Iterator[float | int]: ...
    def __contains__(self, value: float | int) -> bool: ...


    def __add__(self, other: ArrayLike) -> Array: ...
    def __radd__(self, other: float | int) -> Array: ...

    def __sub__(self, other: ArrayLike) -> Array: ...
    def __rsub__(self, other: float | int) -> Array: ...

    def __mul__(self, other: ArrayLike) -> Array: ...
    def __rmul__(self, other: float | int) -> Array: ...

    def __truediv__(self, other: ArrayLike) -> Array:
        """True division. For integer arrays, always returns a float array."""
        ...
    def __rtruediv__(self, other: float | int) -> Array: ...

    def __neg__(self) -> Array: ...

    def __pow__(self, exp: ArrayLike, modulo: None = None) -> Array:
        """Element-wise power (``arr ** exp``). Always returns float64.

        Args:
            exp: Exponent — scalar or same-shape array.
            modulo: Ignored (required by Python's three-argument ``pow()``).

        Returns:
            float64 Array with each element raised to *exp*.
        """
        ...

    def __rpow__(self, base: float | int, modulo: None = None) -> Array:
        """Reverse element-wise power (``base ** arr``). Always returns float64."""
        ...

    def __lt__(self, other: ArrayLike) -> Array:
        """Element-wise ``<``. float input → float64 (1.0/0.0); int → int64 (1/0)."""
        ...

    def __le__(self, other: ArrayLike) -> Array:
        """Element-wise ``<=``."""
        ...

    def __gt__(self, other: ArrayLike) -> Array:
        """Element-wise ``>``."""
        ...

    def __ge__(self, other: ArrayLike) -> Array:
        """Element-wise ``>=``."""
        ...

    def __eq__(self, other: ArrayLike) -> Array:  # type: ignore[override]
        """Element-wise ``==``. Returns an Array of 1.0/0.0 (float) or 1/0 (int),
        *not* a bool. Use ``.all()`` or ``.any()`` to reduce to a scalar."""
        ...

    def __ne__(self, other: ArrayLike) -> Array:  # type: ignore[override]
        """Element-wise ``!=``."""
        ...

    def __repr__(self) -> str: ...




class models:
    """Machine learning models built on top of ironforest."""

    class LinearRegression:
        """Linear regression model using least squares."""

        fit_intercept: bool
        coef_: Array | None
        intercept_: float | None

        def __init__(self, fit_intercept: bool = True) -> None:
            """Initialize linear regression model.

            Args:
                fit_intercept: Whether to calculate the intercept for this model.
            """
            ...

        def fit(self, X: Array, y: Array) -> "models.LinearRegression":
            """Fit linear model.

            Args:
                X: Training data of shape (n_samples, n_features).
                y: Target values of shape (n_samples,) or (n_samples, n_targets).

            Returns:
                self: Fitted estimator.
            """
            ...

        def predict(self, X: Array) -> Array:
            """Predict using the linear model.

            Args:
                X: Samples of shape (n_samples, n_features).

            Returns:
                Predicted values of shape (n_samples,) or (n_samples, n_targets).

            Raises:
                RuntimeError: If model hasn't been fitted yet.
            """
            ...

        def score(self, X: Array, y: Array) -> float:
            """Return the coefficient of determination (R²) of the prediction.

            Args:
                X: Test samples of shape (n_samples, n_features).
                y: True values of shape (n_samples,) or (n_samples, n_targets).

            Returns:
                R² score.

            Raises:
                RuntimeError: If model hasn't been fitted yet.
            """
            ...

        def residuals(self, X: Array, y: Array) -> Array:
            """Calculate residuals (y - y_pred).

            Args:
                X: Samples of shape (n_samples, n_features).
                y: True values of shape (n_samples,) or (n_samples, n_targets).

            Returns:
                Residuals array.

            Raises:
                RuntimeError: If model hasn't been fitted yet.
            """
            ...


class spatial:
    """Spatial data structures and algorithms."""

    class SpatialResult:
        """Result of a spatial query (knn or radius search).

        Attributes:
            indices: Array of indices into the original data.
                Shape (n,) for single queries, (n_queries, k) for batch knn,
                or flat (total,) for batch radius (use ``counts`` to split).
            distances: Array of distances corresponding to each index.
                Same shape as ``indices``.
            counts: Only present for batch radius queries. Shape (n_queries,),
                giving the number of results per query. Use to partition
                ``indices`` and ``distances`` into per-query slices.
        """

        indices: Array[int]
        distances: Array[float]
        counts: Array[int] | None

        def count(self) -> float | Array[int]:
            """Number of results per query.

            Returns a scalar for single queries, or an array of per-query
            counts for batch queries.
            """
            ...
        
        def split(self) -> list["spatial.SpatialResult"]:
            """Splits a batch query into a list of singular spatial results.
            
            Returns a list of spatial results.
            """
            ...
        
        def is_empty(self) -> bool:
            """Check if the query result is empty

            Returns bool, if true the query returned no results
            """
            ...

        def min(self) -> float | Array[float]:
            """Minimum distance across results.

            Returns a scalar for single queries, or an array of per-query
            minimums for batch queries.
            """
            ...

        def max(self) -> float | Array[float]:
            """Maximum distance across results.

            Returns a scalar for single queries, or an array of per-query
            maximums for batch queries.
            """
            ...

        def radius(self) -> float | Array[float]:
            """Maximum distance across results.

            Returns a scalar for single queries, or an array of per-query
            maximums for batch queries.
            """
            ...

        def mean(self) -> float | Array[float]:
            """Mean distance across results.

            Returns a scalar for single queries, or an array of per-query
            means for batch queries.
            """
            ...

        def median(self) -> float | Array[float]:
            """Median distance across results.

            Returns a scalar for single queries, or an array of per-query
            medians for batch queries.
            """
            ...
        
        def var(self) -> float | Array[float]:
            """Variance of distance across results.

            Returns a scalar for single queries, or an array of per-query
            medians for batch queries.
            """
            ...
        
        def std(self) -> float | Array[float]:
            """Standard deviation of distance across results.

            Returns a scalar for single queries, or an array of per-query
            medians for batch queries.
            """
            ...
        
        def quantile(self, q) -> float | Array[float]:
            """Compute a distance quantile for each query's results.
            Args:
                q: Quantile value between 0 and 1 (e.g. 0.5 for median).

            Returns:
                A scalar if single query, or an array of quantiles for batch queries.
                
            Raises:
                ValueError: If q is not in [0, 1].
            """
            ...


        def centroid(self, data: Array) -> Array[float]:
            """Centroid of result points per query.

            Args:
                data: The tree's data in original index order, as returned
                    by ``tree.data()``. Shape (n_points, dim).

            Returns:
                Shape (dim,) for single queries, or (n_queries, dim) for
                batch queries. Returns NaN for queries with no results.
            """
            ...

    class BallTree:
        """Ball tree for efficient nearest neighbor queries.

        A ball tree recursively partitions data into nested hyperspheres (balls).
        Each node in the tree represents a ball that contains a subset of points.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ) -> "spatial.BallTree":
            """Construct a ball tree from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed BallTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...
        
        def __init__(
            self,
            data: ArrayLike,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ):
            """Construct a ball tree from a 2D array of points.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed BallTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> "spatial.SpatialResult":
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                Spatial result object
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                Spatial result object
            """
            ...

        def query_ann(self, query: ArrayLike, k: int, n_candidates: int) -> "spatial.SpatialResult":
            """Find the approximate k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.
                n_candidates: Number of candidates to check before returning the result. 
                Defaults to 2k if n_candidates is None

            Returns:
                Spatial result object.
            """
            ...

        @overload
        def data(self, indices: ArrayLike) -> Array[float]:
            """Return training-data rows at specific original indices.

            Args:
                indices: 1D int array-like of original point indices (as returned
                    by ``query_knn`` or ``query_radius``).

            Returns:
                float64 Array of shape ``(len(indices), n_features)``.

            Raises:
                ValueError: If any index is out of bounds.
            """
            ...

        @overload
        def data(self, indices: None = None) -> Array[float]:
            """Return all training-data points in original index order.

            Returns:
                float64 Array of shape ``(n_points, n_features)``.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the tree to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> tree = BallTree.from_array(data)
                >>> tree.save("my_tree.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.BallTree":
            """Deserialize a tree from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``BallTree`` instance restored from the saved state.

            Example:
                >>> tree = BallTree.load("my_tree.mpack")
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point (scalar, list, or array-like).
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array-like of query points with shape (n_queries, n_features),
                    or 1D array-like representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at all training points (leave-one-out).

            Args:
                queries: If None, computes density at each training point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates at each training point.
            """
            ...

    class KDTree:
        """KD-tree for efficient nearest neighbor queries.

        A KD-tree (k-dimensional tree) recursively partitions data by splitting along
        coordinate axes. Each node represents a hyperrectangular region and splits
        data along the axis with the largest spread.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ) -> "spatial.KDTree":
            """Construct a KD-tree from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed KDTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def __init__(
            self,
            data: ArrayLike,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ):
            """Construct a KD-tree from a 2D array of points.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed KDTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float)-> "spatial.SpatialResult":
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                Spatial result object
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                Spatial result object
            """
            ...

        def query_ann(self, query: ArrayLike, k: int, n_candidates: int) -> "spatial.SpatialResult":
            """Find the approximate k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.
                n_candidates: Number of candidates to check before returning the result. 
                Defaults to 2k if n_candidates is None

            Returns:
                Spatial result object.
            """
            ...

        @overload
        def data(self, indices: ArrayLike) -> Array: ...
        @overload
        def data(self, indices: None = None) -> Array:
            """Return training-data rows at original indices, or all points if omitted.

            Args:
                indices: 1D int array-like of original indices (from ``query_knn`` /
                    ``query_radius``), or ``None`` to return all points.

            Returns:
                float64 Array of shape ``(len(indices), n_features)`` or
                ``(n_points, n_features)`` when called without arguments.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the tree to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> tree = KDTree.from_array(data)
                >>> tree.save("my_tree.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.KDTree":
            """Deserialize a tree from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``KDTree`` instance restored from the saved state.

            Example:
                >>> tree = KDTree.load("my_tree.mpack")
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point (scalar, list, or array-like).
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array-like of query points with shape (n_queries, n_features),
                    or 1D array-like representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at all training points (leave-one-out).

            Args:
                queries: If None, computes density at each training point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates at each training point.
            """
            ...

    class VPTree:
        """Vantage-point tree for efficient nearest neighbor queries.

        A vantage-point tree recursively partitions data by selecting vantage points
        and partitioning based on distances to those points. Each node selects a point
        as a vantage point and splits remaining points by their median distance to it.
        This structure can be more efficient than KD-trees for high-dimensional data.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            selection: Literal["first", "random", "variance"] = "variance"
        ) -> "spatial.VPTree":
            """Construct a vantage-point tree from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                selection: Method for selecting vantage points during tree construction.
                    Options are:
                    - "first": Always select the first point in the partition (default)
                    - "random": Randomly select a point from the partition
                    - "variance: Selects the point with the greatest distance variance from a subset of data

            Returns:
                A constructed VPTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or selection is not one of the valid options.
            """
            ...
        
        def __init__(
            self,
            data: ArrayLike,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            selection: Literal["first", "random", "variance"] = "variance"
        ):
            """Construct a vantage-point tree from a 2D array of points.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                selection: Method for selecting vantage points during tree construction.
                    Options are:
                    - "first": Always select the first point in the partition (default)
                    - "random": Randomly select a point from the partition
                    - "variance: Selects the point with the greatest distance variance from a subset of data

            Returns:
                A constructed VPTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or selection is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> "spatial.SpatialResult":
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                Spatial result object
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                Spatial result object
            """
            ...

        @overload
        def data(self, indices: ArrayLike) -> Array[float]: ...
        @overload
        def data(self, indices: None = None) -> Array[float]:
            """Return training-data rows at original indices, or all points if omitted.

            Args:
                indices: 1D int array-like of original indices (from ``query_knn`` /
                    ``query_radius``), or ``None`` to return all points.

            Returns:
                float64 Array of shape ``(len(indices), n_features)`` or
                ``(n_points, n_features)`` when called without arguments.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the tree to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> tree = VPTree.from_array(data)
                >>> tree.save("my_tree.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.VPTree":
            """Deserialize a tree from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``VPTree`` instance restored from the saved state.

            Example:
                >>> tree = VPTree.load("my_tree.mpack")
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point (scalar, list, or array-like).
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array-like of query points with shape (n_queries, n_features),
                    or 1D array-like representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at all training points (leave-one-out).

            Args:
                queries: If None, computes density at each training point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of density estimates at each training point.
            """
            ...

    # class MTree:
    #     """M-tree for efficient nearest neighbor queries with dynamic insertion.

    #     An M-tree partitions data into nested hyperspheres using routing objects
    #     that must be actual data points. Unlike static trees, the M-tree supports
    #     dynamic insertion without requiring a full rebuild.
    #     """

    #     @staticmethod
    #     def from_array(
    #         array: Array[float],
    #         capacity: int = 50,
    #         metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
    #     ) -> "spatial.MTree":
    #         """Construct an M-tree from a 2D array of points.

    #         Args:
    #             array: 2D array of shape (n_points, n_features) containing the data points.
    #             capacity: Maximum number of entries per node before a split occurs.
    #                 Larger values produce shallower trees but may slow queries.
    #                 Defaults to 50.
    #             metric: Distance metric to use for measuring distances between points.
    #                 Options are:
    #                 - "euclidean": Standard Euclidean (L2) distance (default)
    #                 - "manhattan": Manhattan (L1) distance (taxicab distance)
    #                 - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
    #                 - "cosine": The angular distance between two vectors

    #         Returns:
    #             A constructed MTree instance.

    #         Raises:
    #             AssertionError: If array is not 2-dimensional.
    #             ValueError: If metric is not one of the valid options.
    #         """
    #         ...
        
    #     def __init__(
    #         self,
    #         data: ArrayLike,
    #         capacity: int = 50,
    #         metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
    #     ):
    #         """Construct an M-tree from a 2D array of points.

    #         Args:
    #             data: ArrayLike (n_points, n_features) containing the data points.
    #             capacity: Maximum number of entries per node before a split occurs.
    #                 Larger values produce shallower trees but may slow queries.
    #                 Defaults to 50.
    #             metric: Distance metric to use for measuring distances between points.
    #                 Options are:
    #                 - "euclidean": Standard Euclidean (L2) distance (default)
    #                 - "manhattan": Manhattan (L1) distance (taxicab distance)
    #                 - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
    #                 - "cosine": The angular distance between two vectors

    #         Returns:
    #             A constructed MTree instance.

    #         Raises:
    #             AssertionError: If array is not 2-dimensional.
    #             ValueError: If metric is not one of the valid options.
    #         """
    #         ...

    #     def insert(self, point: ArrayLike) -> None:
    #         """Insert a single point into the tree.

    #         The tree is updated in place. If a node exceeds capacity after insertion,
    #         it is split and the split propagates upward as needed.

    #         Args:
    #             point: 1D array-like of shape (n_features,) representing the point to insert.

    #         Raises:
    #             ValueError: If point dimension does not match the tree's dimension.
    #         """
    #         ...

    #     def query_radius(self, query: ArrayLike, radius: float) -> "spatial.SpatialResult":
    #         """Find all points within a given radius of the query point.

    #         Args:
    #             query: Query point or 2D array of query points.
    #             radius: Search radius. All points with distance <= radius are returned.

    #         Returns:
    #             Spatial result object
    #         """
    #         ...

    #     def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
    #         """Find the k nearest neighbors to the query point.

    #         Args:
    #             query: Query point or 2D array of query points.
    #             k: Number of nearest neighbors to return.

    #         Returns:
    #             Spatial result object
    #         """
    #         ...

    #     @overload
    #     def data(self, indices: ArrayLike) -> Array[float]:
    #         """Return training-data rows at specific original indices.

    #         Args:
    #             indices: 1D int array-like of original point indices (as returned
    #                 by ``query_knn`` or ``query_radius``).

    #         Returns:
    #             float64 Array of shape ``(len(indices), n_features)``.

    #         Raises:
    #             ValueError: If any index is out of bounds.
    #         """
    #         ...

    #     @overload
    #     def data(self, indices: None = None) -> Array[float]:
    #         """Return all training-data points in original insertion order.

    #         Returns:
    #             float64 Array of shape ``(n_points, n_features)``.
    #         """
    #         ...

    #     def save(self, path: str) -> None:
    #         """Serialize the tree to disk in MessagePack format.

    #         Args:
    #             path: File path to write to. Will be created or overwritten.

    #         Example:
    #             >>> tree = MTree.from_array(data)
    #             >>> tree.save("my_tree.mpack")
    #         """
    #         ...

    #     @staticmethod
    #     def load(path: str) -> "spatial.MTree":
    #         """Deserialize a tree from disk.

    #         Args:
    #             path: File path to read from.

    #         Returns:
    #             An MTree instance restored from the saved state.

    #         Example:
    #             >>> tree = MTree.load("my_tree.mpack")
    #         """
    #         ...

    #     @overload
    #     def kernel_density(
    #         self,
    #         queries: ArrayLike,
    #         bandwidth: float = 1.0,
    #         kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
    #         normalize: bool = True
    #     ) -> float:
    #         """Estimate kernel density at a single query point.

    #         Args:
    #             queries: Single query point (scalar, list, or array-like).
    #             bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
    #                 produce smoother estimates. Defaults to 1.0.
    #             kernel: Kernel function to use for density estimation. Options are:
    #                 - "gaussian": Gaussian (normal) kernel (default)
    #                 - "epanechnikov": Epanechnikov kernel
    #                 - "uniform": Uniform (rectangular) kernel
    #                 - "triangular": Triangular kernel
    #             normalize: Bool to control whether normalized values are returned.

    #         Returns:
    #             Density estimate at the query point (float).
    #         """
    #         ...

    #     @overload
    #     def kernel_density(
    #         self,
    #         queries: ArrayLike,
    #         bandwidth: float = 1.0,
    #         kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
    #         normalize: bool = True
    #     ) -> List:
    #         """Estimate kernel density at multiple query points.

    #         Args:
    #             queries: 2D array-like of query points with shape (n_queries, n_features).
    #             bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
    #                 produce smoother estimates. Defaults to 1.0.
    #             kernel: Kernel function to use for density estimation. Options are:
    #                 - "gaussian": Gaussian (normal) kernel (default)
    #                 - "epanechnikov": Epanechnikov kernel
    #                 - "uniform": Uniform (rectangular) kernel
    #                 - "triangular": Triangular kernel
    #             normalize: Bool to control whether normalized values are returned.

    #         Returns:
    #             1D Array of density estimates, one for each query point.
    #         """
    #         ...

    #     @overload
    #     def kernel_density(
    #         self,
    #         queries: None = None,
    #         bandwidth: float = 1.0,
    #         kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
    #         normalize: bool = True
    #     ) -> List:
    #         """Estimate kernel density at all training points.

    #         Args:
    #             queries: If None, computes density at each training point.
    #             bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
    #                 produce smoother estimates. Defaults to 1.0.
    #             kernel: Kernel function to use for density estimation. Options are:
    #                 - "gaussian": Gaussian (normal) kernel (default)
    #                 - "epanechnikov": Epanechnikov kernel
    #                 - "uniform": Uniform (rectangular) kernel
    #                 - "triangular": Triangular kernel
    #             normalize: Bool to control whether normalized values are returned.

    #         Returns:
    #             1D Array of density estimates at each training point.
    #         """
    #         ...

    class RPTree:
        """Random Projection tree for efficient nearest neighbor queries.

        An RP-tree recursively partitions data by projecting points onto random
        directions and splitting at the median. This is more effective than
        axis-aligned splits (KD-tree) in high-dimensional spaces.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            projection: Literal["gaussian", "sparse"] = "gaussian",
            seed: Optional[int] = None,
        ) -> "spatial.RPTree":
            """Construct an RP-tree from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                projection: Random projection type used for splitting. Options: "gaussian", , "sparse".
                    Defaults to "gaussian".
                seed: Optional RNG seed for reproducible tree construction.

            Returns:
                A constructed RPTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or projection is not a valid option.
            """
            ...

        def __init__(
            self,
            data: ArrayLike,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            projection: Literal["gaussian", "sparse"] = "gaussian",
            seed: Optional[int] = None,
        ):
            """Construct an RP-tree from array-like data.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                projection: Random projection type used for splitting. Options: "gaussian", , "sparse".
                    Defaults to "gaussian".
                seed: Optional RNG seed for reproducible tree construction.

            Returns:
                A constructed RPTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or projection is not a valid option.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> "spatial.SpatialResult":
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                Spatial result object.
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                Spatial result object.
            """
            ...
        
        def query_ann(self, query: ArrayLike, k: int, n_candidates: int) -> "spatial.SpatialResult":
            """Find the approximate k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.
                n_candidates: Number of candidates to check before returning the result. 
                Defaults to 2k if n_candidates is None

            Returns:
                Spatial result object.
            """
            ...

        @overload
        def data(self, indices: ArrayLike) -> Array: ...
        @overload
        def data(self, indices: None = None) -> Array:
            """Return training-data rows at original indices, or all points if omitted.

            Args:
                indices: 1D int array-like of original indices, or None to return all points.

            Returns:
                float64 Array of shape (len(indices), n_features) or
                (n_points, n_features) when called without arguments.
            """
            ...


    class AggTree:
        """Aggregation tree for fast approximate kernel density estimation.

        An aggregation tree is a spatial tree structure that enables fast approximate
        kernel density estimation using a Taylor expansion to approximate contributions
        from groups of points. The absolute tolerance (atol) controls how aggressively
        nodes are approximated during queries.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            bandwidth: float = 1.0,
            atol: float = 0.01,
        ) -> "spatial.AggTree":
            """Construct an aggregation tree from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                kernel: Kernel function used to determine node error bounds at build time.
                    Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                bandwidth: Bandwidth used when computing node error bounds at build time.
                    Defaults to 1.0.
                atol: Absolute tolerance for approximation. Nodes whose maximum
                    absolute error is below this threshold are approximated using a
                    Taylor expansion rather than evaluated exactly. Smaller values
                    give more accurate results at the cost of speed. Defaults to 0.01.

            Returns:
                A constructed AggTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or kernel is not one of the valid options.
            """
            ...
        
        def __init__(
            self,
            data: ArrayLike,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            bandwidth: float = 1.0,
            atol: float = 0.01,
        ):
            """Construct an aggregation tree from a 2D array of points.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                leaf_size: Maximum number of points in a leaf node. Smaller values lead to
                    faster queries but slower construction and more memory usage.
                    Defaults to 20.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors
                kernel: Kernel function used to determine node error bounds at build time.
                    Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel
                bandwidth: Bandwidth used when computing node error bounds at build time.
                    Defaults to 1.0.
                atol: Absolute tolerance for approximation. Nodes whose maximum
                    absolute error is below this threshold are approximated using a
                    Taylor expansion rather than evaluated exactly. Smaller values
                    give more accurate results at the cost of speed. Defaults to 0.01.

            Returns:
                A constructed AggTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or kernel is not one of the valid options.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the tree to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> tree = AggTree.from_array(data)
                >>> tree.save("my_tree.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.AggTree":
            """Deserialize a tree from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``AggTree`` instance restored from the saved state.

            Example:
                >>> tree = AggTree.load("my_tree.mpack")
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point (with approximation).

            Args:
                queries: Single query point (scalar, list, or array-like).
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                Approximate density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at multiple query points (with approximation).

            Args:
                queries: 2D array-like of query points with shape (n_queries, n_features),
                    or 1D array-like representing a single point.
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of approximate density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at all training points (with approximation).

            Args:
                queries: If None, computes density at each training point.
                Normalize: Bool to control whether normalized values are returned.

            Returns:
                1D Array of approximate density estimates at each training point.
            """
            ...

    class BruteForce:
        """Brute force nearest neighbor search.

        Computes exact queries by comparing every point in the dataset.
        No tree structure is built — useful as a correctness baseline or
        for very small datasets where tree construction overhead is not worthwhile.
        """

        @staticmethod
        def from_array(
            array: Array[float],
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ) -> "spatial.BruteForce":
            """Construct a BruteForce search structure from a 2D array of points.

            Args:
                array: 2D array of shape (n_points, n_features) containing the data points.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed BruteForce instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def __init__(
            self,
            data: Array[float],
            metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean"
        ):
            """Construct a BruteForce search structure from a 2D array of points.

            Args:
                data: ArrayLike (n_points, n_features) containing the data points.
                metric: Distance metric to use for measuring distances between points.
                    Options are:
                    - "euclidean": Standard Euclidean (L2) distance (default)
                    - "manhattan": Manhattan (L1) distance (taxicab distance)
                    - "chebyshev": Chebyshev (L∞) distance (maximum coordinate difference)
                    - "cosine": The angular distance between two vectors

            Returns:
                A constructed BruteForce instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> "spatial.SpatialResult":
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                Spatial result object
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> "spatial.SpatialResult":
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                Spatial result object
            """
            ...

        @overload
        def data(self, indices: ArrayLike) -> Array[float]: ...
        @overload
        def data(self, indices: None = None) -> Array[float]:
            """Return training-data rows at original indices, or all points if omitted.

            Args:
                indices: 1D int array-like of original indices (from ``query_knn`` /
                    ``query_radius``), or ``None`` to return all points.

            Returns:
                float64 Array of shape ``(len(indices), n_features)`` or
                ``(n_points, n_features)`` when called without arguments.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the tree to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> tree = AggTree.from_array(data)
                >>> tree.save("my_tree.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.BruteForce":
            """Deserialize a tree from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``BruteForce`` instance restored from the saved state.

            Example:
                >>> tree = BruteForce.load("my_tree.mpack")
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point (scalar, list, or array-like).
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Defaults to 1.0.
                kernel: Kernel function to use. Options: "gaussian", "epanechnikov",
                    "uniform", "triangular".
                normalize: Whether to normalize the density estimate. Defaults to False.

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array-like of shape (n_queries, n_features).
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Defaults to 1.0.
                kernel: Kernel function to use. Options: "gaussian", "epanechnikov",
                    "uniform", "triangular".
                normalize: Whether to normalize the density estimate. Defaults to False.

            Returns:
                1D Array of density estimates, one per query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> List:
            """Estimate kernel density at all training points.

            Args:
                queries: If None, computes density at each training point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Defaults to 1.0.
                kernel: Kernel function to use. Options: "gaussian", "epanechnikov",
                    "uniform", "triangular".
                normalize: Whether to normalize the density estimate. Defaults to False.

            Returns:
                1D Array of density estimates at each training point.
            """
            ...

    class ProjectionReducer:
        @property
        def input_dim(self) -> int: ...
        
        @property
        def output_dim(self) -> int: ...

        def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            projection_type: Literal["gaussian", "sparse"] = "gaussian", 
            density: float = 0.1, 
            seed: Optional[int] = None
        ) -> None:
            """
            Initializes a new ProjectionReducer.
            
            :param input_dim: Number of features in the input data.
            :param output_dim: Number of features in the projected space.
            :param projection_type: The method of projection ('gaussian' or 'sparse').
            :param density: The density of the projection matrix (used if type is 'sparse').
            :param seed: Random seed for reproducibility.
            """
            ...

        def transform(self, data: ArrayLike) -> Array[float]:
            """
            Projects the data into the lower-dimensional space.
            
            :param data: Input array of shape (n_samples, input_dim).
            :return: Projected PyArray of shape (n_samples, output_dim).
            """
            ...

        @staticmethod
        def fit_transform(
            data: ArrayLike, 
            output_dim: int, 
            projection_type: Literal["gaussian", "sparse"] = "gaussian", 
            density: float = 0.1, 
            seed: Optional[int] = None
        ) -> Tuple["spatial.ProjectionReducer", Array]:
            """
            Fits a reducer to the data dimensions and returns both the reducer and the transformed data.
            """
            ...
        
        def save(self, path: str) -> None:
            """Serialize the reducer to disk in MessagePack format.

            Args:
                path: File path to write to. Will be created or overwritten.

            Example:
                >>> reducer = ProjectionReducer(data)
                >>> reducer.save("my_reducer.mpack")
            """
            ...

        @staticmethod
        def load(path: str) -> "spatial.BruteForce":
            """Deserialize a reducer from disk.

            Args:
                path: File path to read from.

            Returns:
                A ``ProjectionReducer`` instance restored from the saved state.

            Example:
                >>> reducer = ProjectionReducer.load("my_reducer.mpack")
            """
            ...


