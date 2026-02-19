"""Type stubs for ironforest._core module.

This file provides type hints for the Rust-based _core module, including
tree_engine classes and other core functionality.
"""

from typing import Optional

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

# Type alias for array-like inputs that can be converted to Array
# Accepts: Array, NumPy arrays, Python lists/nested lists, or scalar floats
ArrayLike = Union['Array', Sequence[float], Sequence[Sequence[float]], float, Any]

class linalg:
    """Linear algebra functions."""

    @staticmethod
    def matmul(a: ArrayLike, b: ArrayLike) -> Array:
        """Matrix multiplication."""
        ...

    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> Array:
        """Dot/matrix product."""
        ...

    @staticmethod
    def transpose(a: ArrayLike) -> Array:
        """Transpose a 2D matrix."""
        ...

    @staticmethod
    def cholesky(a: ArrayLike) -> Array:
        """Compute Cholesky decomposition.

        Returns lower triangular matrix L where A = L @ L.T.

        Raises:
            ValueError: If matrix is not square or not positive-definite.
        """
        ...

    @staticmethod
    def qr(a: ArrayLike) -> tuple[Array, Array]:
        """QR decomposition.

        Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

        Raises:
            ValueError: If array is not 2D.
        """
        ...

    @staticmethod
    def lstsq(a: ArrayLike, b: ArrayLike) -> tuple[Array, Array]:
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
    def weighted_lstsq(a: ArrayLike, b: ArrayLike, weights: ArrayLike) -> tuple[Array, Array]:
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
    def eig(a: ArrayLike) -> tuple[Array, Array]:
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
    def eig_with_params(a: ArrayLike, max_iter: int = 1000, tol: float = 1e-10) -> tuple[Array, Array]:
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
    def eigvals(a: ArrayLike) -> Array:
        """Compute eigenvalues only.

        More efficient than eig() when eigenvectors are not needed.

        Returns:
            1D array of eigenvalues sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    @staticmethod
    def diagonal(a: ArrayLike, k: int | None = None) -> Array:
        """Extract the k-th diagonal from a 2D array."""
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

    @overload
    @staticmethod
    def quantile(a: ArrayLike, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...

    @overload
    @staticmethod
    def quantile(a: ArrayLike, q: ArrayLike) -> Array:
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

class Array(Sequence[float]):
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
    def asarray(data: ArrayLike, shape: Sequence[int] | None = None) -> Array:
        """Create array from data with optional reshape.

        Args:
            data: Array-like data (Array, NumPy array, list, nested list, or scalar).
            shape: Optional shape. If None, creates a 1D array.
        """
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
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
    def outer(a: ArrayLike, b: ArrayLike) -> Array:
        """Compute the outer product of two 1D arrays.

        Args:
            a: First 1D array-like.
            b: Second 1D array-like.
        """
        ...

    @property
    def shape(self) -> List[int]:
        """Get the shape as a list."""
        ...
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""

    def get(self, indices: Sequence[int]) -> float:
        """Get element at indices."""
        ...

    def tolist(self) -> List[float]:
        """Return data as nested list."""
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
    
    def item(self) -> float:
        """Convert a single-element array into a float."""
        ...

    @staticmethod
    def from_numpy(arr: Any) -> Array:
        """Convert a numpy array to a substratum Array.

        Args:
            arr: A numpy ndarray of dtype float64.

        Returns:
            A new Array containing the same data and shape as the numpy array.

        Raises:
            ValueError: If the numpy array cannot be converted (wrong dtype, etc.).
        """
        ...

    def to_numpy(self) -> Any:
        """Convert this Array to a numpy ndarray.

        Returns:
            A numpy ndarray of dtype float64 with the same shape and data.
        """
        ...

    def matmul(self, other: Array) -> Array:
        """Matrix multiplication."""
        ...

    def dot(self, other: Array) -> Array:
        """Dot/matrix product."""
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
    def quantile(self, q: ArrayLike) -> Array:
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
    def __getitem__(self, index: int) -> float | Array:
        """Get element (1D) or row (2D+) at index."""
        ...
    @overload
    def __getitem__(self, index: slice) -> Array:
        """Slice always returns an array."""
        ...
    @overload
    def __getitem__(self, index: Tuple[int, int]) -> float:
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
    def __getitem__(self, index: Tuple[int, ...]) -> float | Array:
        """Fallback: multiple indices can return scalar or array."""
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
    def __contains__(self, value: float) -> bool: ...


    def __add__(self, other: ArrayLike) -> Array: ...
    def __radd__(self, other: float) -> Array: ...

    def __sub__(self, other: ArrayLike) -> Array: ...
    def __rsub__(self, other: float) -> Array: ...

    def __mul__(self, other: ArrayLike) -> Array: ...
    def __rmul__(self, other: float) -> Array: ...

    def __truediv__(self, other: ArrayLike) -> Array: ...
    def __rtruediv__(self, other: float) -> Array: ...

    def __neg__(self) -> Array: ...
    def __repr__(self) -> str: ...



def zeros(shape: Sequence[int]) -> Array:
    """Create array filled with zeros."""
    ...

def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
    """Create a 2D identity matrix with ones on the k-th diagonal."""
    ...

def diag(v: ArrayLike, k: int | None = None) -> Array:
    """Create a 2D array with v on the k-th diagonal.

    Args:
        v: 1D array-like of diagonal values.
        k: Diagonal offset (0=main, >0=upper, <0=lower).
    """
    ...

def outer(a: ArrayLike, b: ArrayLike) -> Array:
    """Compute the outer product of two 1D arrays.

    Args:
        a: First 1D array-like.
        b: Second 1D array-like.
    """
    ...

def column_stack(arrays: Sequence[Array]) -> Array:
    """Stack 1D or 2D arrays as columns into a 2D array.

    Args:
        arrays: Sequence of 1D or 2D arrays to stack. 1D arrays are converted
            to columns (n, 1). All arrays must have the same number of rows.

    Returns:
        A 2D array formed by stacking the given arrays column-wise.

    Raises:
        ValueError: If arrays is empty or if arrays have mismatched row counts.
    """
    ...

def ones(shape: Sequence[int]) -> Array:
    """Create array filled with ones."""
    ...

def full(shape: Sequence[int], fill_value: float) -> Array:
    """Create array filled with a specified value."""
    ...

def asarray(data: ArrayLike, shape: Sequence[int] | None = None) -> Array:
    """Create array from data with optional reshape.

    Args:
        data: Array-like data (Array, NumPy array, list, nested list, or scalar).
        shape: Optional shape. If None, creates a 1D array.
    """
    ...

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

    class BallTree:
        """Ball tree for efficient nearest neighbor queries.

        A ball tree recursively partitions data into nested hyperspheres (balls).
        Each node in the tree represents a ball that contains a subset of points.
        """

        @staticmethod
        def from_array(
            array: Array,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean"
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

            Returns:
                A constructed BallTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> List:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> List[Tuple[int, float]]:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                List of tuples (index, distance) for the k nearest neighbors,
                sorted by distance (closest first). The indices can be used to look up
                the actual points in the original data array.
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
            array: Array,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean"
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

            Returns:
                A constructed KDTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> List:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> List[Tuple[int, float]]:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                List of tuples (index, distance) for the k nearest neighbors,
                sorted by distance (closest first). The indices can be used to look up
                the actual points in the original data array.
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
            array: Array,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
            selection: Literal["first", "random"] = "first"
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
                selection: Method for selecting vantage points during tree construction.
                    Options are:
                    - "first": Always select the first point in the partition (default)
                    - "random": Randomly select a point from the partition

            Returns:
                A constructed VPTree instance.

            Raises:
                AssertionError: If array is not 2-dimensional.
                ValueError: If metric or selection is not one of the valid options.
            """
            ...

        def query_radius(self, query: ArrayLike, radius: float) -> Array:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: ArrayLike, k: int) -> List[Tuple[int, float]]:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point (scalar, list, or array-like).
                k: Number of nearest neighbors to return.

            Returns:
                List of tuples (index, distance) for the k nearest neighbors,
                sorted by distance (closest first). The indices can be used to look up
                the actual points in the original data array.
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

    class AggTree:
        """Aggregation tree for fast approximate kernel density estimation.

        An aggregation tree is a spatial tree structure that enables fast approximate
        kernel density estimation using a Taylor expansion to approximate contributions
        from groups of points. The absolute tolerance (atol) controls how aggressively
        nodes are approximated during queries.
        """

        @staticmethod
        def from_array(
            array: Array,
            leaf_size: int = 20,
            metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
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

        @overload
        def kernel_density(
            self,
            queries: ArrayLike,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
            normalize: bool = True
        ) -> float:
            """Estimate kernel density at a single query point (with approximation).

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
                Approximate density estimate at the query point (float).
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
            """Estimate kernel density at multiple query points (with approximation).

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
                1D Array of approximate density estimates, one for each query point.
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
            """Estimate kernel density at all training points (with approximation).

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
                1D Array of approximate density estimates at each training point.
            """
            ...
