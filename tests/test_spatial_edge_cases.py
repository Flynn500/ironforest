"""
Edge case and robustness tests for all spatial tree types.

Covers: construction edge cases, kNN/radius boundary conditions, metric
spot-checks, ANN queries, SpatialResult aggregation, serialization, input
format robustness, and KDE.

The existing tree_correctness_test.py validates index correctness against
brute-force; these tests focus on boundary inputs and undefined-behaviour
scenarios.
"""

import math
import pickle

import numpy as np
import pytest

import ironforest as irn
from ironforest import spatial

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

TREES = {
    "BruteForce": lambda d: spatial.BruteForce.from_array(d),
    "KDTree":     lambda d: spatial.KDTree.from_array(d, leaf_size=20),
    "BallTree":   lambda d: spatial.BallTree.from_array(d, leaf_size=20),
    "VPTree":     lambda d: spatial.VPTree.from_array(d, leaf_size=20, selection="variance"),
    "RPTree":     lambda d: spatial.RPTree.from_array(d, leaf_size=20),
}
TREES_WITH_LEAF = {
    "KDTree":   lambda d, ls: spatial.KDTree.from_array(d, leaf_size=ls),
    "BallTree": lambda d, ls: spatial.BallTree.from_array(d, leaf_size=ls),
    "VPTree":   lambda d, ls: spatial.VPTree.from_array(d, leaf_size=ls, selection="variance"),
    "RPTree":   lambda d, ls: spatial.RPTree.from_array(d, leaf_size=ls),
}
ANN_TREES = {
    "KDTree":   lambda d: spatial.KDTree.from_array(d, leaf_size=20),
    "BallTree": lambda d: spatial.BallTree.from_array(d, leaf_size=20),
    "RPTree":   lambda d: spatial.RPTree.from_array(d, leaf_size=20),
}

ALL_NAMES      = list(TREES.keys())
LEAF_NAMES     = list(TREES_WITH_LEAF.keys())
ANN_NAMES      = list(ANN_TREES.keys())


def make_irn(data_np: np.ndarray) -> irn.Array:
    return irn.ndutils.from_numpy(np.ascontiguousarray(data_np, dtype=np.float64))


def make_tree(name: str, data_np: np.ndarray):
    return TREES[name](make_irn(data_np))


def make_tree_with_leaf(name: str, data_np: np.ndarray, leaf_size: int):
    return TREES_WITH_LEAF[name](make_irn(data_np), leaf_size)


def to_np(arr: irn.Array) -> np.ndarray:
    return np.array(irn.ndutils.to_numpy(arr))


# ---------------------------------------------------------------------------
# Section 1 – Construction edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_single_point_construction(tree_name):
    data = np.array([[1.0, 2.0]])
    tree = make_tree(tree_name, data)
    data_back = to_np(tree.data())
    assert data_back.shape[1] == 2
    assert data_back.shape[0] == 1


@pytest.mark.parametrize("tree_name", LEAF_NAMES)
def test_leaf_size_one(tree_name):
    data = RNG.standard_normal((50, 4))
    tree = make_tree_with_leaf(tree_name, data, leaf_size=1)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 5)
    assert to_np(result.indices).shape[-1] == 5


@pytest.mark.parametrize("tree_name", LEAF_NAMES)
def test_leaf_size_exceeds_n_points(tree_name):
    data = RNG.standard_normal((10, 4))
    tree = make_tree_with_leaf(tree_name, data, leaf_size=100)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 5)
    assert to_np(result.indices).shape[-1] == 5


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_single_dimension(tree_name):
    data = RNG.standard_normal((100, 1))
    tree = make_tree(tree_name, data)
    q = make_irn(np.array([[0.0]]))
    result = tree.query_knn(q, 5)
    indices = to_np(result.indices).flatten()
    assert len(indices) == 5
    assert all(0 <= i < 100 for i in indices)


@pytest.mark.slow
@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_high_dimension_no_crash(tree_name):
    data = RNG.standard_normal((200, 512))
    tree = make_tree(tree_name, data)
    q = make_irn(RNG.standard_normal((1, 512)))
    result = tree.query_knn(q, 5)
    assert to_np(result.distances).min() >= 0.0


# ---------------------------------------------------------------------------
# Section 2 – kNN boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_k_equals_n_points(tree_name):
    n = 10
    data = RNG.standard_normal((n, 3))
    tree = make_tree(tree_name, data)
    q = make_irn(RNG.standard_normal((1, 3)))
    result = tree.query_knn(q, n)
    indices = set(to_np(result.indices).flatten().tolist())
    assert len(indices) == n
    assert indices == set(range(n))


@pytest.mark.xfail(strict=False, reason="documenting: k > n_points clamps or errors — behavior not yet specified")
@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_k_exceeds_n_points(tree_name):
    n = 10
    data = RNG.standard_normal((n, 3))
    tree = make_tree(tree_name, data)
    q = make_irn(RNG.standard_normal((1, 3)))
    result = tree.query_knn(q, n * 2)
    count = len(to_np(result.indices).flatten())
    assert count <= n


@pytest.mark.xfail(strict=False, reason="documenting: k=0 returns empty or raises — behavior not yet specified")
@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_k_equals_zero(tree_name):
    data = RNG.standard_normal((10, 3))
    tree = make_tree(tree_name, data)
    q = make_irn(RNG.standard_normal((1, 3)))
    result = tree.query_knn(q, 0)
    assert result.is_empty()


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_duplicate_points_both_returned(tree_name):
    data = RNG.standard_normal((10, 3))
    data[5] = [0.5, 0.5, 0.5]
    data[7] = [0.5, 0.5, 0.5]
    tree = make_tree(tree_name, data)
    q = make_irn(np.array([[0.5, 0.5, 0.5]]))
    result = tree.query_knn(q, 3)
    indices = set(to_np(result.indices).flatten().tolist())
    assert 5 in indices
    assert 7 in indices
    dists = to_np(result.distances).flatten()
    assert any(math.isclose(d, 0.0) for d in dists[:2])


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_query_dimension_mismatch_raises(tree_name):
    data = RNG.standard_normal((50, 4))
    tree = make_tree(tree_name, data)
    q_wrong = make_irn(RNG.standard_normal((1, 3)))
    with pytest.raises((ValueError, RuntimeError)):
        tree.query_knn(q_wrong, 5)


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_batch_knn_shape(tree_name):
    data = RNG.standard_normal((200, 4))
    queries = RNG.standard_normal((50, 4))
    tree = make_tree(tree_name, data)
    q = make_irn(queries)
    result = tree.query_knn(q, 5)
    idx_np = to_np(result.indices)
    dist_np = to_np(result.distances)
    assert idx_np.shape == (50, 5)
    assert dist_np.shape == (50, 5)
    assert len(result.split()) == 50


# ---------------------------------------------------------------------------
# Section 3 – Radius query boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_radius_zero_exact_match_only(tree_name):
    data = RNG.standard_normal((20, 3))
    exact_point = data[3].copy()
    tree = make_tree(tree_name, data)
    q = make_irn(exact_point.reshape(1, 3))
    result = tree.query_radius(q, 0.0)
    singles = result.split()
    assert len(singles) == 1
    indices = singles[0].indices.tolist()
    assert 3 in indices
    # Only exact matches (distance ~ 0) should be returned
    dists = to_np(singles[0].distances).flatten()
    assert all(math.isclose(d, 0.0, abs_tol=1e-10) for d in dists)


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_radius_very_large_returns_all(tree_name):
    data = RNG.standard_normal((20, 2))
    tree = make_tree(tree_name, data)
    q = make_irn(np.zeros((1, 2)))
    result = tree.query_radius(q, 1e9)
    singles = result.split()
    assert len(singles[0].indices.tolist()) == 20


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_radius_no_results_empty(tree_name):
    data = RNG.standard_normal((10, 3))
    tree = make_tree(tree_name, data)
    # Far-away query with tiny radius
    q = make_irn(np.array([[1000.0, 1000.0, 1000.0]]))
    result = tree.query_radius(q, 0.001)
    singles = result.split()
    assert len(singles) == 1
    assert singles[0].is_empty()
    assert singles[0].count() == 0


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_batch_radius_counts_consistency(tree_name):
    data = RNG.standard_normal((200, 4))
    queries = RNG.standard_normal((20, 4))
    tree = make_tree(tree_name, data)
    q = make_irn(queries)
    result = tree.query_radius(q, 0.5)
    singles = result.split()
    assert len(singles) == 20
    total_from_split = sum(len(s.indices.tolist()) for s in singles)
    total_indices = to_np(result.indices).flatten().shape[0]
    assert total_from_split == total_indices


# ---------------------------------------------------------------------------
# Section 4 – Metric spot-checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ["KDTree", "BallTree"])
def test_cosine_metric_distances_in_range(tree_name):
    # Unit vectors: cosine distance ∈ [0, 2]
    raw = RNG.standard_normal((50, 8))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    data = raw / norms
    irn_data = make_irn(data)
    if tree_name == "KDTree":
        tree = spatial.KDTree.from_array(irn_data, leaf_size=10, metric="cosine")
    else:
        tree = spatial.BallTree.from_array(irn_data, leaf_size=10, metric="cosine")
    q_raw = RNG.standard_normal((1, 8))
    q = make_irn(q_raw / np.linalg.norm(q_raw))
    result = tree.query_knn(q, 5)
    dists = to_np(result.distances).flatten()
    assert all(0.0 <= d <= 2.0 + 1e-9 for d in dists)
    assert not any(math.isnan(d) for d in dists)


@pytest.mark.xfail(strict=False, reason="documenting: cosine with zero-vector may NaN or raise — behavior not yet specified")
@pytest.mark.parametrize("tree_name", ["KDTree", "BallTree"])
def test_cosine_zero_vector_no_crash(tree_name):
    data = RNG.standard_normal((10, 3))
    data[0] = [0.0, 0.0, 0.0]
    irn_data = make_irn(data)
    if tree_name == "KDTree":
        tree = spatial.KDTree.from_array(irn_data, leaf_size=5, metric="cosine")
    else:
        tree = spatial.BallTree.from_array(irn_data, leaf_size=5, metric="cosine")
    q = make_irn(np.array([[0.0, 0.0, 0.0]]))
    result = tree.query_knn(q, 3)
    # Just verify it doesn't panic; distances may be NaN
    assert result is not None


@pytest.mark.parametrize("tree_name", ["KDTree", "BallTree", "BruteForce"])
def test_chebyshev_distance_correct(tree_name):
    data = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
    irn_data = make_irn(data)
    if tree_name == "KDTree":
        tree = spatial.KDTree.from_array(irn_data, leaf_size=5, metric="chebyshev")
    elif tree_name == "BallTree":
        tree = spatial.BallTree.from_array(irn_data, leaf_size=5, metric="chebyshev")
    else:
        tree = spatial.BruteForce.from_array(irn_data, metric="chebyshev")
    q = make_irn(np.array([[0.0, 0.0]]))
    result = tree.query_knn(q, 2)
    dists = sorted(to_np(result.distances).flatten().tolist())
    assert math.isclose(dists[0], 0.0, abs_tol=1e-10)
    assert math.isclose(dists[1], 4.0, rel_tol=1e-9)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_all_metrics_return_positive_distances(metric):
    data = RNG.standard_normal((50, 4))
    irn_data = make_irn(data)
    tree = spatial.KDTree.from_array(irn_data, leaf_size=10, metric=metric)
    q = make_irn(RNG.standard_normal((5, 4)))
    result = tree.query_knn(q, 5)
    dists = to_np(result.distances).flatten()
    assert all(d >= 0.0 for d in dists)
    assert not any(math.isnan(d) for d in dists)


# ---------------------------------------------------------------------------
# Section 5 – ANN query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ANN_NAMES)
def test_ann_returns_k_results(tree_name):
    data = RNG.standard_normal((500, 8))
    tree = ANN_TREES[tree_name](make_irn(data))
    q = make_irn(RNG.standard_normal((1, 8)))
    result = tree.query_ann(q, 10, n_candidates=30)
    indices = to_np(result.indices).flatten()
    dists = to_np(result.distances).flatten()
    assert len(indices) == 10
    assert all(d >= 0.0 for d in dists)


@pytest.mark.parametrize("tree_name", ANN_NAMES)
def test_ann_default_n_candidates(tree_name):
    data = RNG.standard_normal((200, 4))
    tree = ANN_TREES[tree_name](make_irn(data))
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_ann(q, 5)
    assert len(to_np(result.indices).flatten()) == 5


@pytest.mark.parametrize("tree_name", ANN_NAMES)
def test_ann_recall_reasonable(tree_name):
    data = RNG.standard_normal((200, 4))
    queries = RNG.standard_normal((50, 4))
    tree_exact = ANN_TREES[tree_name](make_irn(data))
    tree_ann   = ANN_TREES[tree_name](make_irn(data))
    k = 5

    recall_total = 0
    for i in range(len(queries)):
        q = make_irn(queries[i:i+1])
        exact = set(to_np(tree_exact.query_knn(q, k).indices).flatten().tolist())
        approx = set(to_np(tree_ann.query_ann(q, k, n_candidates=k * 10).indices).flatten().tolist())
        recall_total += len(exact & approx) / k

    mean_recall = recall_total / len(queries)
    assert mean_recall >= 0.6, f"{tree_name} ANN recall {mean_recall:.2f} < 0.6"


# ---------------------------------------------------------------------------
# Section 6 – SpatialResult aggregation
# ---------------------------------------------------------------------------

def test_result_stats_single_query():
    data = RNG.standard_normal((100, 4))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=20)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 10)

    mn  = result.min()
    mx  = result.max()
    avg = result.mean()
    med = result.median()
    std = result.std()
    var = result.var()

    assert isinstance(mn, float)
    assert isinstance(mx, float)
    assert mn <= avg <= mx
    assert mn <= med <= mx
    assert std >= 0.0
    assert var >= 0.0
    assert math.isclose(std ** 2, var, rel_tol=1e-6) # type: ignore

    q0 = result.quantile(0.0)
    q1 = result.quantile(1.0)
    assert math.isclose(q0, mn, rel_tol=1e-6) # type: ignore
    assert math.isclose(q1, mx, rel_tol=1e-6) # type: ignore


@pytest.mark.xfail(strict=False, reason="documenting: empty-result stats return NaN — not yet enforced")
def test_result_stats_empty():
    data = RNG.standard_normal((20, 2))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=20)
    q = make_irn(np.array([[1000.0, 1000.0]]))
    result = tree.query_radius(q, 0.001)
    singles = result.split()
    r = singles[0]
    assert r.is_empty()
    assert r.count() == 0
    assert math.isnan(r.min()) # type: ignore
    assert math.isnan(r.max()) # type: ignore
    assert math.isnan(r.mean()) # type: ignore


def test_result_centroid_single():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tree = spatial.KDTree.from_array(make_irn(square), leaf_size=5)
    q = make_irn(np.array([[0.5, 0.5]]))
    result = tree.query_knn(q, 4)
    centroid = to_np(result.centroid(tree.data())).flatten()
    assert math.isclose(centroid[0], 0.5, abs_tol=1e-9)
    assert math.isclose(centroid[1], 0.5, abs_tol=1e-9)


def test_result_centroid_batch():
    data = RNG.standard_normal((50, 4))
    queries = RNG.standard_normal((5, 4))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=20)
    q = make_irn(queries)
    result = tree.query_knn(q, 4)
    centroid = to_np(result.centroid(tree.data()))
    assert centroid.shape == (5, 4)


def test_result_radius_alias_equals_max():
    data = RNG.standard_normal((100, 4))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=20)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 10)
    assert math.isclose(result.radius(), result.max(), rel_tol=1e-10) # type: ignore


# ---------------------------------------------------------------------------
# Section 7 – Serialization
# ---------------------------------------------------------------------------

SERIAL_TREES = ["KDTree", "BallTree", "VPTree", "BruteForce"]


@pytest.mark.parametrize("tree_name", SERIAL_TREES)
def test_save_load_roundtrip(tree_name, tmp_path_str):
    data = RNG.standard_normal((200, 4))
    irn_data = make_irn(data)
    tree = make_tree(tree_name, data)

    tree.save(tmp_path_str)

    tree2 = make_tree(tree_name, data)
    tree2.load(tmp_path_str)

    queries = RNG.standard_normal((10, 4))
    q = make_irn(queries)

    for i in range(len(queries)):
        qi = make_irn(queries[i:i+1])
        orig_idx  = set(to_np(tree.query_knn(qi, 5).indices).flatten().tolist())
        loaded_idx = set(to_np(tree2.query_knn(qi, 5).indices).flatten().tolist())
        assert orig_idx == loaded_idx, f"{tree_name}: mismatch at query {i}"


@pytest.mark.parametrize("tree_name", SERIAL_TREES)
def test_pickle_roundtrip(tree_name):
    data = RNG.standard_normal((100, 3))
    tree = make_tree(tree_name, data)

    blob = pickle.dumps(tree)
    tree2 = pickle.loads(blob)

    queries = RNG.standard_normal((5, 3))
    for i in range(len(queries)):
        qi = make_irn(queries[i:i+1])
        orig_idx  = set(to_np(tree.query_knn(qi, 5).indices).flatten().tolist())
        loaded_idx = set(to_np(tree2.query_knn(qi, 5).indices).flatten().tolist())
        assert orig_idx == loaded_idx, f"{tree_name}: pickle mismatch at query {i}"


# ---------------------------------------------------------------------------
# Section 8 – Input format robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_float32_numpy_input_no_crash(tree_name):
    data = RNG.standard_normal((50, 4)).astype(np.float32)
    irn_data = irn.ndutils.asarray(data)
    tree = TREES[tree_name](irn_data)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 5)
    assert len(to_np(result.indices).flatten()) == 5


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_fortran_order_numpy_no_crash(tree_name):
    data = np.asfortranarray(RNG.standard_normal((50, 4)))
    irn_data = irn.ndutils.asarray(data)
    tree = TREES[tree_name](irn_data)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 5)
    assert len(to_np(result.indices).flatten()) == 5


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_transposed_numpy_no_crash(tree_name):
    data = RNG.standard_normal((4, 50)).T  # shape (50, 4), Fortran-order
    irn_data = irn.ndutils.asarray(data)
    tree = TREES[tree_name](irn_data)
    q = make_irn(RNG.standard_normal((1, 4)))
    result = tree.query_knn(q, 5)
    assert len(to_np(result.indices).flatten()) == 5


@pytest.mark.parametrize("tree_name", ALL_NAMES)
def test_nested_list_input(tree_name):
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    irn_data = irn.ndutils.asarray(data)
    tree = TREES[tree_name](irn_data)
    q = make_irn(np.array([[3.0, 4.0]]))
    result = tree.query_knn(q, 1)
    idx = to_np(result.indices).flatten().tolist()
    assert len(idx) == 1


def test_tree_data_retrieval_all_points():
    data = RNG.standard_normal((20, 3))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=10)
    data_back = to_np(tree.data())
    assert data_back.shape == (20, 3)
    # Every row from original data should appear in returned data
    orig_rows = {tuple(row) for row in data.round(12)}
    back_rows = {tuple(row) for row in data_back.round(12)}
    assert orig_rows == back_rows


def test_tree_data_retrieval_specific_indices():
    data = RNG.standard_normal((20, 3))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=10)
    q = make_irn(RNG.standard_normal((1, 3)))
    result = tree.query_knn(q, 3)
    subset = to_np(tree.data(result.indices))
    assert subset.shape == (3, 3)


# ---------------------------------------------------------------------------
# Section 9 – KDE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_name", ["KDTree", "BallTree", "BruteForce"])
def test_kde_single_returns_positive(tree_name):
    data = RNG.standard_normal((100, 2))
    irn_data = make_irn(data)
    if tree_name == "KDTree":
        tree = spatial.KDTree.from_array(irn_data, leaf_size=20)
    elif tree_name == "BallTree":
        tree = spatial.BallTree.from_array(irn_data, leaf_size=20)
    else:
        tree = spatial.BruteForce.from_array(irn_data)
    q = make_irn(np.zeros((1, 2)))
    val = tree.kernel_density(q, bandwidth=1.0)
    # For single query, returns a scalar
    if isinstance(val, irn.Array):
        val = float(irn.ndutils.to_numpy(val).flat[0])
    assert isinstance(val, float)
    assert val > 0.0


@pytest.mark.parametrize("tree_name", ["KDTree", "BallTree", "BruteForce"])
def test_kde_batch_returns_array(tree_name):
    data = RNG.standard_normal((100, 2))
    irn_data = make_irn(data)
    if tree_name == "KDTree":
        tree = spatial.KDTree.from_array(irn_data, leaf_size=20)
    elif tree_name == "BallTree":
        tree = spatial.BallTree.from_array(irn_data, leaf_size=20)
    else:
        tree = spatial.BruteForce.from_array(irn_data)
    queries = make_irn(RNG.standard_normal((10, 2)))
    result = tree.kernel_density(queries, bandwidth=1.0)
    vals = to_np(result).flatten() # type: ignore
    assert len(vals) == 10
    assert all(v > 0.0 for v in vals)


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "uniform", "triangular"])
def test_kde_all_kernels_no_crash(kernel):
    data = RNG.standard_normal((50, 2))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=10)
    q = make_irn(np.zeros((1, 2)))
    val = tree.kernel_density(q, bandwidth=1.0, kernel=kernel)
    if isinstance(val, irn.Array):
        val = float(irn.ndutils.to_numpy(val).flat[0])
    assert val >= 0.0


def test_kde_invalid_kernel_raises():
    data = RNG.standard_normal((50, 2))
    tree = spatial.KDTree.from_array(make_irn(data), leaf_size=10)
    q = make_irn(np.zeros((1, 2)))
    with pytest.raises((ValueError, RuntimeError)):
        tree.kernel_density(q, bandwidth=1.0, kernel="rbf") # type: ignore
