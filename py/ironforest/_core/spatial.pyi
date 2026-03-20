"""IronForest spatial: Contains a variety of spatial index trees & a dimension reducer"""

from typing import Optional, Literal, List, Tuple, overload
from ironforest._core import Array, ArrayLike


#Maybe these should carry views of the initial distance array. Will have to actually add views though...
#Would be more performant, instead of creating like 500 ndarrays for a big query, only one.
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

    def split(self) -> list[SpatialResult]:
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
        variances for batch queries.
        """
        ...

    def std(self) -> float | Array[float]:
        """Standard deviation of distance across results.

        Returns a scalar for single queries, or an array of per-query
        standard deviations for batch queries.
        """
        ...

    def quantile(self, q: float) -> float | Array[float]:
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
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        preserve_array: bool = True
    ) -> BallTree:
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
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        copy: bool = True
    ):
        """Construct a ball tree from a 2D array of points."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def query_radius(self, query: ArrayLike, radius: float) -> SpatialResult:
        """Find all points within a given radius of the query point.

        Args:
            query: Query point (scalar, list, or array-like).
            radius: Search radius. All points with distance <= radius are returned.

        Returns:
            Spatial result object
        """
        ...

    def query_knn(self, query: ArrayLike, k: int) -> SpatialResult:
        """Find the k nearest neighbors to the query point.

        Args:
            query: Query point (scalar, list, or array-like).
            k: Number of nearest neighbors to return.

        Returns:
            Spatial result object
        """
        ...

    def query_ann(self, query: ArrayLike, k: int, n_candidates: int,  n_probes: int | None = None) -> SpatialResult:
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
        """Return training-data rows at specific original indices."""
        ...

    @overload
    def data(self, indices: None = None) -> Array[float]:
        """Return all training-data points in original index order."""
        ...

    def save(self, path: str) -> None:
        """Serialize the tree to disk in MessagePack format.

        Args:
            path: File path to write to. Will be created or overwritten.
        """
        ...

    @staticmethod
    def load(path: str) -> BallTree:
        """Deserialize a tree from disk.

        Args:
            path: File path to read from.

        Returns:
            A ``BallTree`` instance restored from the saved state.
        """
        ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: None = None,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...


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
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        preserve_array: bool = True
    ) -> KDTree:
        """Construct a KD-tree from a 2D array of points."""
        ...

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 20,
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        copy: bool = True
    ):
        """Construct a KD-tree from a 2D array of points."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def query_radius(self, query: ArrayLike, radius: float) -> SpatialResult:
        """Find all points within a given radius of the query point."""
        ...

    def query_knn(self, query: ArrayLike, k: int) -> SpatialResult:
        """Find the k nearest neighbors to the query point."""
        ...

    def query_ann(self, query: ArrayLike, k: int, n_candidates: int,  n_probes: int | None = None) -> SpatialResult:
        """Find the approximate k nearest neighbors to the query point."""
        ...

    @overload
    def data(self, indices: ArrayLike) -> Array: ...
    @overload
    def data(self, indices: None = None) -> Array:
        """Return training-data rows at original indices, or all points if omitted."""
        ...

    def save(self, path: str) -> None:
        """Serialize the tree to disk in MessagePack format."""
        ...

    @staticmethod
    def load(path: str) -> KDTree:
        """Deserialize a tree from disk."""
        ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: None = None,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...


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
        selection: Literal["first", "random", "variance"] = "variance",
        preserve_array: bool = True
    ) -> VPTree:
        """Construct a vantage-point tree from a 2D array of points."""
        ...

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 20,
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        selection: Literal["first", "random", "variance"] = "variance",
        copy: bool = True
    ):
        """Construct a vantage-point tree from a 2D array of points."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def query_radius(self, query: ArrayLike, radius: float) -> SpatialResult:
        """Find all points within a given radius of the query point."""
        ...

    def query_knn(self, query: ArrayLike, k: int) -> SpatialResult:
        """Find the k nearest neighbors to the query point."""
        ...

    @overload
    def data(self, indices: ArrayLike) -> Array[float]: ...
    @overload
    def data(self, indices: None = None) -> Array[float]:
        """Return training-data rows at original indices, or all points if omitted."""
        ...

    def save(self, path: str) -> None:
        """Serialize the tree to disk in MessagePack format."""
        ...

    @staticmethod
    def load(path: str) -> VPTree:
        """Deserialize a tree from disk."""
        ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: None = None,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...


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
        preserve_array: bool = True
    ) -> RPTree:
        """Construct an RP-tree from a 2D array of points."""
        ...

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 20,
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        projection: Literal["gaussian", "sparse"] = "gaussian",
        seed: Optional[int] = None,
        copy: bool = True
    ):
        """Construct an RP-tree from array-like data."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def query_radius(self, query: ArrayLike, radius: float) -> SpatialResult:
        """Find all points within a given radius of the query point."""
        ...

    def query_knn(self, query: ArrayLike, k: int) -> SpatialResult:
        """Find the k nearest neighbors to the query point."""
        ...

    def query_ann(self, query: ArrayLike, k: int, n_candidates: int, n_probes: int | None = None) -> SpatialResult:
        """Find the approximate k nearest neighbors to the query point."""
        ...

    @overload
    def data(self, indices: ArrayLike) -> Array: ...
    @overload
    def data(self, indices: None = None) -> Array:
        """Return training-data rows at original indices, or all points if omitted."""
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
        preserve_array: bool = True
    ) -> AggTree:
        """Construct an aggregation tree from a 2D array of points."""
        ...

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 20,
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        bandwidth: float = 1.0,
        atol: float = 0.01,
        copy: bool = True
    ):
        """Construct an aggregation tree from a 2D array of points."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def save(self, path: str) -> None:
        """Serialize the tree to disk in MessagePack format."""
        ...

    @staticmethod
    def load(path: str) -> AggTree:
        """Deserialize a tree from disk."""
        ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: None = None,
        normalize: bool = True
    ) -> float | Array[float]: ...


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
    ) -> BruteForce:
        """Construct a BruteForce search structure from a 2D array of points."""
        ...

    def __init__(
        self,
        data: Array[float],
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        copy: bool = True
    ):
        """Construct a BruteForce search structure from a 2D array of points."""
        ...

    @property
    def dtype(self) -> str:
        """The floating point precision of the tree ('float32' or 'float64')."""
        ...

    def query_radius(self, query: ArrayLike, radius: float) -> SpatialResult:
        """Find all points within a given radius of the query point."""
        ...

    def query_knn(self, query: ArrayLike, k: int) -> SpatialResult:
        """Find the k nearest neighbors to the query point."""
        ...

    @overload
    def data(self, indices: ArrayLike) -> Array[float]: ...
    @overload
    def data(self, indices: None = None) -> Array[float]:
        """Return training-data rows at original indices, or all points if omitted."""
        ...

    def save(self, path: str) -> None:
        """Serialize the tree to disk in MessagePack format."""
        ...

    @staticmethod
    def load(path: str) -> BruteForce:
        """Deserialize a BruteForce instance from disk."""
        ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: ArrayLike,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...

    @overload
    def kernel_density(
        self,
        queries: None = None,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "epanechnikov", "uniform", "triangular"] = "gaussian",
        normalize: bool = True
    ) -> float | Array[float]: ...


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
        seed: Optional[int] = None,
        copy: bool = True
    ) -> None:
        """Initializes a new ProjectionReducer.

        :param input_dim: Number of features in the input data.
        :param output_dim: Number of features in the projected space.
        :param projection_type: The method of projection ('gaussian' or 'sparse').
        :param density: The density of the projection matrix (used if type is 'sparse').
        :param seed: Random seed for reproducibility.
        """
        ...

    def transform(self, data: ArrayLike) -> Array[float]:
        """Projects the data into the lower-dimensional space.

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
    ) -> Tuple[ProjectionReducer, Array]:
        """Fits a reducer to the data dimensions and returns both the reducer and the transformed data."""
        ...

    def save(self, path: str) -> None:
        """Serialize the reducer to disk in MessagePack format."""
        ...

    @staticmethod
    def load(path: str) -> ProjectionReducer:
        """Deserialize a reducer from disk."""
        ...
