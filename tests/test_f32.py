"""
First-class f32 dtype tests for IronForest.

Covers: array creation, numpy interop, astype, linalg dtype preservation,
decomposition upcasting, spatial trees, model wrappers, and arithmetic.
"""

import numpy as np
import pytest

import ironforest as irn
from ironforest import ndutils, linalg, spatial
from ironforest.models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
)

# ---------------------------------------------------------------------------
# 1. Array creation with dtype="float32"
# ---------------------------------------------------------------------------

def test_zeros_f32():
    a = ndutils.zeros([3, 4], dtype="float32")
    assert a.dtype == "float32"

def test_ones_f32():
    a = ndutils.ones([5], dtype="float32")
    assert a.dtype == "float32"

def test_full_f32():
    a = ndutils.full([2, 3], 7.0, dtype="float32")
    assert a.dtype == "float32"

def test_eye_f32():
    a = ndutils.eye(4, dtype="float32")
    assert a.dtype == "float32"

def test_diag_f32():
    v = ndutils.asarray([1.0, 2.0, 3.0], dtype="float32")
    a = ndutils.diag(v)
    assert a.dtype == "float32"

# ---------------------------------------------------------------------------
# 2. From numpy — dtype preservation
# ---------------------------------------------------------------------------

def test_from_numpy_f32():
    np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = ndutils.asarray(np_arr)
    assert a.dtype == "float32"

def test_from_numpy_f64():
    np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    a = ndutils.asarray(np_arr)
    assert a.dtype == "float64"

def test_from_numpy_f32_2d():
    np_arr = np.ones((4, 3), dtype=np.float32)
    a = ndutils.asarray(np_arr)
    assert a.dtype == "float32"

# ---------------------------------------------------------------------------
# 3. to_numpy preserves dtype
# ---------------------------------------------------------------------------

def test_to_numpy_f32_roundtrip():
    np_in = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = ndutils.asarray(np_in)
    np_out = ndutils.to_numpy(a)
    assert np_out.dtype == np.float32
    np.testing.assert_array_almost_equal(np_out, np_in)

def test_to_numpy_f64_unchanged():
    np_in = np.array([1.0, 2.0], dtype=np.float64)
    a = ndutils.asarray(np_in)
    np_out = ndutils.to_numpy(a)
    assert np_out.dtype == np.float64

# ---------------------------------------------------------------------------
# 4. astype conversions
# ---------------------------------------------------------------------------

def test_astype_f32_to_f64():
    a = ndutils.zeros([3], dtype="float32")
    b = a.astype("float64")
    assert b.dtype == "float64"

def test_astype_f64_to_f32():
    a = ndutils.ones([3], dtype="float64")
    b = a.astype("float32")
    assert b.dtype == "float32"

def test_astype_f32_to_int64():
    np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = ndutils.asarray(np_arr)
    b = a.astype("int64")
    assert b.dtype == "int64"

def test_astype_values_preserved():
    np_arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    a = ndutils.asarray(np_arr)
    b = a.astype("float32")
    np.testing.assert_array_almost_equal(ndutils.to_numpy(b), np_arr.astype(np.float32), decimal=5)

# ---------------------------------------------------------------------------
# 5. Linalg dtype preservation (f32 in → f32 out)
# ---------------------------------------------------------------------------

def test_matmul_f32_preserves_dtype():
    a = ndutils.asarray(np.eye(3, dtype=np.float32))
    b = ndutils.asarray(np.ones((3, 2), dtype=np.float32))
    c = linalg.matmul(a, b)
    assert c.dtype == "float32"

def test_dot_f32_preserves_dtype():
    a = ndutils.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = ndutils.asarray(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    c = linalg.dot(a, b)
    assert c.dtype == "float32"

def test_transpose_f32_preserves_dtype():
    a = ndutils.asarray(np.ones((2, 3), dtype=np.float32))
    b = linalg.transpose(a)
    assert b.dtype == "float32"

def test_outer_f32_preserves_dtype():
    a = ndutils.asarray(np.array([1.0, 2.0], dtype=np.float32))
    b = ndutils.asarray(np.array([3.0, 4.0, 5.0], dtype=np.float32))
    c = linalg.outer(a, b)
    assert c.dtype == "float32"

def test_diagonal_f32_preserves_dtype():
    a = ndutils.asarray(np.eye(3, dtype=np.float32))
    d = linalg.diagonal(a)
    assert d.dtype == "float32"

# ---------------------------------------------------------------------------
# 6. Linalg mixed upcasting (f32 + f64 → f64)
# ---------------------------------------------------------------------------

def test_matmul_mixed_upcasts_to_f64():
    a = ndutils.asarray(np.eye(3, dtype=np.float32))
    b = ndutils.asarray(np.ones((3, 2), dtype=np.float64))
    c = linalg.matmul(a, b)
    assert c.dtype == "float64"

def test_dot_mixed_upcasts_to_f64():
    a = ndutils.asarray(np.array([1.0, 2.0], dtype=np.float32))
    b = ndutils.asarray(np.array([3.0, 4.0], dtype=np.float64))
    c = linalg.dot(a, b)
    assert c.dtype == "float64"

# ---------------------------------------------------------------------------
# 7. Decompositions upcast f32 input → return f64
# ---------------------------------------------------------------------------

def test_cholesky_f32_input_returns_f64():
    np_arr = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    a = ndutils.asarray(np_arr)
    l = linalg.cholesky(a)
    assert l.dtype == "float64"

def test_qr_f32_input_returns_f64():
    np_arr = np.ones((3, 2), dtype=np.float32)
    a = ndutils.asarray(np_arr)
    q, r = linalg.qr(a)
    assert q.dtype == "float64"
    assert r.dtype == "float64"

def test_eig_f32_input_returns_f64():
    np_arr = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    a = ndutils.asarray(np_arr)
    vals, vecs = linalg.eig(a)
    assert vals.dtype == "float64"
    assert vecs.dtype == "float64"

# ---------------------------------------------------------------------------
# 8. Spatial trees accept f32 arrays
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
DATA_F32 = RNG.random((50, 4)).astype(np.float32)
QUERY_F32 = RNG.random((5, 4)).astype(np.float32)

TREE_CONSTRUCTORS = [
    ("BruteForce", lambda d: spatial.BruteForce.from_array(d)),
    ("KDTree",     lambda d: spatial.KDTree.from_array(d)),
    ("BallTree",   lambda d: spatial.BallTree.from_array(d)),
    ("VPTree",     lambda d: spatial.VPTree.from_array(d)),
    ("RPTree",     lambda d: spatial.RPTree.from_array(d)),
]

@pytest.mark.parametrize("name,constructor", TREE_CONSTRUCTORS)
def test_spatial_tree_f32_builds(name, constructor):
    arr = ndutils.asarray(DATA_F32)
    tree = constructor(arr)
    assert tree is not None

@pytest.mark.parametrize("name,constructor", TREE_CONSTRUCTORS)
def test_spatial_tree_f32_knn_query(name, constructor):
    arr = ndutils.asarray(DATA_F32)
    tree = constructor(arr)
    q = ndutils.asarray(QUERY_F32)
    result = tree.query_knn(q, k=3)
    assert result.indices.shape[0] == 5
    assert result.indices.shape[1] == 3

# ---------------------------------------------------------------------------
# 9. Decision tree models accept f32 input (via astype guard)
# ---------------------------------------------------------------------------

def _make_clf_data():
    X = RNG.random((100, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float64)
    return X, y

def _make_reg_data():
    X = RNG.random((100, 4)).astype(np.float32)
    y = X[:, 0].astype(np.float64)
    return X, y

def test_decision_tree_classifier_f32_input():
    X, y = _make_clf_data()
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds is not None

def test_decision_tree_regressor_f32_input():
    X, y = _make_reg_data()
    reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds is not None

def test_random_forest_classifier_f32_input():
    X, y = _make_clf_data()
    clf = RandomForestClassifier(n_estimators=5, max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds is not None

def test_random_forest_regressor_f32_input():
    X, y = _make_reg_data()
    reg = RandomForestRegressor(n_estimators=5, max_depth=3)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds is not None

def test_isolation_forest_f32_input():
    X = RNG.random((100, 4)).astype(np.float32)
    iso = IsolationForest(n_estimators=10)
    iso.fit(X)
    scores = iso.score_samples(X)
    assert scores is not None

# ---------------------------------------------------------------------------
# 10. Arithmetic dtype rules
# ---------------------------------------------------------------------------

def test_f32_plus_f32_stays_f32():
    a = ndutils.asarray(np.ones((3,), dtype=np.float32))
    b = ndutils.asarray(np.ones((3,), dtype=np.float32))
    c = a + b
    assert c.dtype == "float32"

def test_f32_plus_f64_upcasts_to_f64():
    a = ndutils.asarray(np.ones((3,), dtype=np.float32))
    b = ndutils.asarray(np.ones((3,), dtype=np.float64))
    c = a + b
    assert c.dtype == "float64"

def test_f32_multiply_f32_stays_f32():
    a = ndutils.asarray(np.array([2.0, 3.0], dtype=np.float32))
    b = ndutils.asarray(np.array([4.0, 5.0], dtype=np.float32))
    c = a * b
    assert c.dtype == "float32"

def test_f32_values_correct():
    a = ndutils.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = ndutils.asarray(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    c = a + b
    np.testing.assert_array_almost_equal(ndutils.to_numpy(c), [5.0, 7.0, 9.0], decimal=5)
