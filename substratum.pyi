"""A library for array-based computation."""

from typing import Iterator, List, Literal, Sequence, Tuple, overload

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

    @overload
    @staticmethod
    def quantile(a: Array, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...

    @overload
    @staticmethod
    def quantile(a: Array, q: Array) -> Array:
        """Compute multiple quantiles at once (vectorized).

        Args:
            a: Input array.
            q: Array of quantile values, each in [0, 1].

        Returns:
            1D array of quantile values corresponding to each q.
        """
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
    def asarray(data: Sequence[float] | Array, shape: Sequence[int] | None = None) -> Array:
        """Create array from data with optional reshape.

        Args:
            data: Flat list of float values or existing Array.
            shape: Optional shape. If None, creates a 1D array.
        """
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
        """Create a 2D identity matrix with ones on the k-th diagonal."""
        ...

    @staticmethod
    def diag(v: Sequence[float] | Array, k: int | None = None) -> Array:
        """Create a 2D array with v on the k-th diagonal.

        Args:
            v: 1D array or list of diagonal values.
            k: Diagonal offset (0=main, >0=upper, <0=lower).
        """
        ...

    @staticmethod
    def outer(a: Sequence[float] | Array, b: Sequence[float] | Array) -> Array:
        """Compute the outer product of two 1D arrays.

        Args:
            a: First 1D array or list.
            b: Second 1D array or list.
        """
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

    def take(self, indices: Sequence[int]) -> Array:
        """Take elements from the array at specified flat indices.

        Args:
            indices: List of flat (1D) indices to select from the array.
                These indices index into the flattened array.

        Returns:
            A new 1D array containing the elements at the specified indices.
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
    @overload
    def quantile(self, q: float) -> float:
        """q-th quantile of all elements (q in [0, 1])."""
        ...
    @overload
    def quantile(self, q: Array) -> Array:
        """Compute multiple quantiles at once (vectorized).

        Args:
            q: Array of quantile values, each in [0, 1].

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
    def __contains__(self, value: float) -> bool: ...


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



def zeros(shape: Sequence[int]) -> Array:
    """Create array filled with zeros."""
    ...

def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
    """Create a 2D identity matrix with ones on the k-th diagonal."""
    ...

def diag(v: Sequence[float] | Array, k: int | None = None) -> Array:
    """Create a 2D array with v on the k-th diagonal.

    Args:
        v: 1D array or list of diagonal values.
        k: Diagonal offset (0=main, >0=upper, <0=lower).
    """
    ...

def outer(a: Sequence[float] | Array, b: Sequence[float] | Array) -> Array:
    """Compute the outer product of two 1D arrays.

    Args:
        a: First 1D array or list.
        b: Second 1D array or list.
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

def asarray(data: Sequence[float] | Array, shape: Sequence[int] | None = None) -> Array:
    """Create array from data with optional reshape.

    Args:
        data: Flat list of float values or existing Array.
        shape: Optional shape. If None, creates a 1D array.
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

        def query_radius(self, query: float | Sequence[float] | Array, radius: float) -> List:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: float | Sequence[float] | Array, k: int) -> List:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                k: Number of nearest neighbors to return.

            Returns:
                1D Array of row indices (as floats) for the k nearest neighbors,
                sorted by distance (closest first). These indices can be used to look up
                the actual points in the original data array.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: float | Sequence[float],
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point as a scalar (for 1D data) or list of coordinates.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: Array,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array of query points with shape (n_queries, n_features),
                    or 1D array representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
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

        def query_radius(self, query: float | Sequence[float] | Array, radius: float) -> List:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: float | Sequence[float] | Array, k: int) -> List:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                k: Number of nearest neighbors to return.

            Returns:
                1D Array of row indices (as floats) for the k nearest neighbors,
                sorted by distance (closest first). These indices can be used to look up
                the actual points in the original data array.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: float | Sequence[float],
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point as a scalar (for 1D data) or list of coordinates.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: Array,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array of query points with shape (n_queries, n_features),
                    or 1D array representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
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

        def query_radius(self, query: float | Sequence[float] | Array, radius: float) -> Array:
            """Find all points within a given radius of the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                radius: Search radius. All points with distance <= radius are returned.

            Returns:
                1D Array of row indices (as floats) for all points within the specified
                radius of the query point. These indices can be used to look up the actual
                points in the original data array.
            """
            ...

        def query_knn(self, query: float | Sequence[float] | Array, k: int) -> Array:
            """Find the k nearest neighbors to the query point.

            Args:
                query: Query point as a scalar (for 1D data), list of coordinates, or Array.
                k: Number of nearest neighbors to return.

            Returns:
                1D Array of row indices (as floats) for the k nearest neighbors,
                sorted by distance (closest first). These indices can be used to look up
                the actual points in the original data array.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: float | Sequence[float],
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> float:
            """Estimate kernel density at a single query point.

            Args:
                queries: Single query point as a scalar (for 1D data) or list of coordinates.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                Density estimate at the query point (float).
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: Array,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
        ) -> List:
            """Estimate kernel density at multiple query points.

            Args:
                queries: 2D array of query points with shape (n_queries, n_features),
                    or 1D array representing a single point.
                bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
                    produce smoother estimates. Defaults to 1.0.
                kernel: Kernel function to use for density estimation. Options are:
                    - "gaussian": Gaussian (normal) kernel (default)
                    - "epanechnikov": Epanechnikov kernel
                    - "uniform": Uniform (rectangular) kernel
                    - "triangular": Triangular kernel

            Returns:
                1D Array of density estimates, one for each query point.
            """
            ...

        @overload
        def kernel_density(
            self,
            queries: None = None,
            bandwidth: float = 1.0,
            kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian"
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

            Returns:
                1D Array of density estimates at each training point.
            """
            ...
