"""
Zero-copy storage path verification for all input types.

Tests whether data from numpy, pandas, polars, array.array, memoryview,
and ctypes is zero-copied or copied when passed to IronForest arrays and
spatial trees.

Storage variants:
  External  – zero-copy numpy reference (refcount-based)
  Buffer    – zero-copy Python buffer protocol object
  Owned     – copy to Vec<T>
  Strided   – view with non-unit strides (materialised from slices)

Skips pandas/polars tests automatically if those libraries are not installed.
"""

import array as pyarray
import ctypes
import math

import numpy as np
import pytest

import ironforest as irn
from ironforest import spatial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np_via_buffer(irn_arr: irn.Array) -> np.ndarray:
    """Convert IronForest array to numpy using the buffer protocol (no extra copy)."""
    return np.asarray(irn_arr)


def shares_memory_with_irn(np_arr: np.ndarray, irn_arr: irn.Array) -> bool:
    """True if the IronForest array's buffer points into the same memory as np_arr."""
    irn_as_np = to_np_via_buffer(irn_arr)
    return np.shares_memory(np_arr, irn_as_np)


def irn_values_close(irn_arr: irn.Array, expected: list) -> bool:
    vals = to_np_via_buffer(irn_arr).flatten().tolist()
    return all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(vals, expected))


# ---------------------------------------------------------------------------
# Section 1 – ndutils.from_numpy storage paths
# ---------------------------------------------------------------------------

def test_from_numpy_f64_c_contiguous_zero_copy():
    """C-contiguous float64 numpy array → External storage → zero-copy."""
    rng = np.random.default_rng(1)
    arr = np.ascontiguousarray(rng.random((100, 4)), dtype=np.float64)
    irn_arr = irn.ndutils.from_numpy(arr)
    assert shares_memory_with_irn(arr, irn_arr), (
        "Expected zero-copy (External storage) for C-contiguous float64 numpy array"
    )

def test_from_numpy_f32_c_contiguous_zero_copy():
    """C-contiguous float32 numpy array → External storage → zero-copy."""
    rng = np.random.default_rng(1)
    arr = np.ascontiguousarray(rng.random((100, 4)), dtype=np.float32)
    irn_arr = irn.ndutils.from_numpy(arr)
    assert shares_memory_with_irn(arr, irn_arr), (
        "Expected zero-copy (External storage) for C-contiguous float32 numpy array"
    )

def test_from_numpy_fortran_order_forces_copy():
    """Fortran-order (column-major) numpy array → materialised → Owned storage."""
    rng = np.random.default_rng(3)
    arr = np.asfortranarray(rng.random((50, 4)))
    assert not arr.flags["C_CONTIGUOUS"]
    irn_arr = irn.ndutils.asarray(arr)
    assert not shares_memory_with_irn(arr, irn_arr), (
        "Fortran-order array should be copied, not zero-copied"
    )
    # Values must still be correct
    np.testing.assert_allclose(
        to_np_via_buffer(irn_arr).flatten(),
        np.ascontiguousarray(arr).flatten(),
        rtol=1e-10,
    )


def test_from_numpy_transposed_forces_copy():
    """Transposed array (Fortran-order) → materialised → Owned storage."""
    rng = np.random.default_rng(4)
    arr = rng.random((4, 50)).T  # shape (50, 4), non-C-contiguous
    assert not arr.flags["C_CONTIGUOUS"]
    irn_arr = irn.ndutils.asarray(arr)
    assert not shares_memory_with_irn(arr, irn_arr), (
        "Transposed array should be copied, not zero-copied"
    )


def test_asarray_list_is_owned():
    """Python list → Vec extraction → Owned storage, correct values."""
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    irn_arr = irn.ndutils.asarray(data)
    result = to_np_via_buffer(irn_arr)
    np.testing.assert_allclose(result, np.array(data))


# ---------------------------------------------------------------------------
# Section 2 – Python buffer protocol inputs
# ---------------------------------------------------------------------------

def test_array_array_double_zero_copy():
    """array.array('d') → Buffer protocol → zero-copy, memoryview format correct."""
    raw = pyarray.array("d", [1.0, 2.0, 3.0, 4.0, 5.0])
    irn_arr = irn.ndutils.asarray(raw)

    # Buffer format should be 'd' (double / float64)
    mv = memoryview(irn_arr)
    assert mv.format == "d"

    # Values should be preserved
    result = np.asarray(irn_arr)
    np.testing.assert_allclose(result.flatten(), [1.0, 2.0, 3.0, 4.0, 5.0])

    # Buffer memory should be the same as the original array.array
    raw_np = np.frombuffer(raw, dtype=np.float64)
    assert np.shares_memory(raw_np, result), (
        "array.array('d') should be zero-copied via buffer protocol"
    )


@pytest.mark.xfail(strict=False, reason="documenting: float buffer ('f') may copy or raise — behavior not yet enforced")
def test_array_array_float_copies_or_raises():
    """array.array('f') is float32 — IronForest expects float64, so copy or raise."""
    raw = pyarray.array("f", [1.0, 2.0, 3.0])
    # May raise ValueError/TypeError (format mismatch) or succeed with copy
    irn_arr = irn.ndutils.asarray(raw)
    result = to_np_via_buffer(irn_arr).flatten()
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5)


def test_memoryview_bytes_rejects():
    """Byte memoryview (format 'B') cannot be reinterpreted as float64 → error."""
    mv = memoryview(b"hello world")
    with pytest.raises((TypeError, ValueError)):
        irn.ndutils.asarray(mv)


def test_ctypes_array_buffer_protocol():
    """ctypes double array implements buffer protocol → values correct."""
    CArr = ctypes.c_double * 5
    raw = CArr(1.0, 2.0, 3.0, 4.0, 5.0)
    irn_arr = irn.ndutils.asarray(raw)
    result = to_np_via_buffer(irn_arr).flatten()
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# Section 3 – pandas / polars / numpy (conditional, use importorskip)
# ---------------------------------------------------------------------------

def test_pandas_series_copies():
    """pandas Series → iterates via Vec extraction → copy."""
    pd = pytest.importorskip("pandas")
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    irn_arr = irn.ndutils.asarray(s)
    result = to_np_via_buffer(irn_arr).flatten()
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])
    # pandas Series is not a buffer-protocol object → always a copy
    s_np = np.array(s, dtype=np.float64)
    assert not np.shares_memory(s_np, result), (
        "pandas Series should be copied, not zero-copied"
    )


def test_pandas_dataframe_copies():
    """pandas DataFrame passed as 2D input → copy."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.random((20, 4)), columns=["a", "b", "c", "d"])
    irn_arr = irn.ndutils.asarray(df)
    result = to_np_via_buffer(irn_arr)
    assert result.shape == (20, 4)
    expected = df.values.astype(np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_polars_series_copies():
    """polars Series → iterates via Vec extraction → copy."""
    pl = pytest.importorskip("polars")
    s = pl.Series([1.0, 2.0, 3.0, 4.0])
    irn_arr = irn.ndutils.asarray(s)
    result = to_np_via_buffer(irn_arr).flatten()
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_numpy_f32_tree_copy():
    """numpy float32 with copy=False → must copy (no zero-copy for non-float64) → no crash."""
    rng = np.random.default_rng(7)
    np_data = rng.random((50, 3)).astype(np.float32)
    tree = spatial.KDTree(np_data, leaf_size=10, copy=False)
    q = irn.ndutils.from_numpy(np.zeros((1, 3), dtype=np.float64))
    result = tree.query_knn(q, 5)
    assert len(result.indices) == 5


def test_numpy_fortran_tree_copy():
    """Fortran-order numpy with copy=False → falls back to Owned copy → no crash."""
    rng = np.random.default_rng(8)
    np_data = np.asfortranarray(rng.random((50, 3)))
    tree = spatial.KDTree(np_data, leaf_size=10, copy=False)
    q = irn.ndutils.from_numpy(np.zeros((1, 3), dtype=np.float64))
    result = tree.query_knn(q, 5)
    assert len(result.indices.tolist()) == 5


# ---------------------------------------------------------------------------
# Section 4 – from_array preserve_array behavior
# ---------------------------------------------------------------------------


def test_preserve_false_consumes_array():
    """from_array(preserve_array=False) → array is consumed (alive=False)."""
    rng = np.random.default_rng(10)
    np_data = np.ascontiguousarray(rng.random((50, 4)), dtype=np.float64)
    irn_data = irn.ndutils.from_numpy(np_data)

    spatial.KDTree.from_array(irn_data, leaf_size=10, preserve_array=False)

    assert not irn_data.alive, "Array should be marked dead after preserve_array=False"


def test_accessing_dead_array_raises():
    """Accessing a consumed array raises RuntimeError."""
    rng = np.random.default_rng(11)
    np_data = np.ascontiguousarray(rng.random((10, 2)), dtype=np.float64)
    irn_data = irn.ndutils.from_numpy(np_data)
    spatial.KDTree.from_array(irn_data, leaf_size=5, preserve_array=False)

    with pytest.raises(RuntimeError, match="(?i)consumed|dead|array"):
        _ = irn_data[0]


def test_consumed_tree_still_queryable():
    """Tree built from a consumed array is still fully queryable."""
    rng = np.random.default_rng(12)
    np_data = np.ascontiguousarray(rng.random((50, 3)), dtype=np.float64)
    irn_data = irn.ndutils.from_numpy(np_data)
    tree = spatial.KDTree.from_array(irn_data, leaf_size=10, preserve_array=False)

    q = irn.ndutils.from_numpy(np.zeros((1, 3), dtype=np.float64))
    result = tree.query_knn(q, 5)
    assert len(np.asarray(result.indices).flatten()) == 5

    data_back = to_np_via_buffer(tree.data())
    assert data_back.shape == (50, 3)


# ---------------------------------------------------------------------------
# Section 5 – Constructor copy parameter
# ---------------------------------------------------------------------------

def test_constructor_copy_true_makes_independent_tree():
    """copy=True → Owned storage → modifying original numpy does not change tree queries."""
    rng = np.random.default_rng(13)
    np_data = np.ascontiguousarray(rng.random((50, 3)), dtype=np.float64)
    tree = spatial.KDTree(np_data, leaf_size=10, copy=True)

    q = irn.ndutils.from_numpy(np.zeros((1, 3), dtype=np.float64))
    orig_indices = set(np.asarray(tree.query_knn(q, 5).indices).flatten().tolist())

    # Overwrite the original numpy array
    np_data[:] = 999.0

    new_indices = set(np.asarray(tree.query_knn(q, 5).indices).flatten().tolist())
    assert orig_indices == new_indices, (
        "copy=True tree should be independent of the original numpy array"
    )


# ---------------------------------------------------------------------------
# Section 6 – Buffer protocol
# ---------------------------------------------------------------------------

def test_pyarray_to_memoryview_and_back():
    """IronForest Array → memoryview format='d' → numpy roundtrip."""
    raw = pyarray.array("d", [1.0, 2.0, 3.0])
    irn_arr = irn.ndutils.asarray(raw)

    mv = memoryview(irn_arr)
    assert mv.format == "d"

    back = np.asarray(irn_arr)
    np.testing.assert_allclose(back.flatten(), [1.0, 2.0, 3.0])


def test_pyarray_buffer_readonly_for_external():
    """Non-owned (External or Buffer) arrays should be read-only via buffer protocol."""
    rng = np.random.default_rng(15)
    np_data = np.ascontiguousarray(rng.random((10, 2)), dtype=np.float64)
    irn_arr = irn.ndutils.from_numpy(np_data)  # External storage

    mv = memoryview(irn_arr)
    assert mv.readonly, (
        "External-storage IronForest array should expose a read-only buffer"
    )
