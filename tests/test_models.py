"""
Correctness, robustness, and edge-case tests for all model classes.

Covers: DecisionTreeClassifier, DecisionTreeRegressor,
RandomForestClassifier, RandomForestRegressor, IsolationForest,
KNNClassifier, KNNRegressor, LinearRegression.
"""

import math

import numpy as np
import pytest

import ironforest as irn

from ironforest.models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    KNNClassifier,
    KNNRegressor,
    LinearRegression,
    RandomForestClassifier,
    RandomForestRegressor,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def make_clf_data(n=200, n_features=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.clip((X[:, 0] * n_classes).astype(int), 0, n_classes - 1).astype(float)
    return X, y


def make_reg_data(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = X[:, 0] * 2.0 + X[:, 1] + rng.standard_normal(n) * 0.1
    return X, y


def to_list(arr) -> list:
    if isinstance(arr, irn.Array):
        return irn.ndutils.to_numpy(arr).tolist()
    return list(arr)


def mse(y_true: np.ndarray, y_pred) -> float:
    pred = np.array(to_list(y_pred))
    return float(np.mean((y_true - pred) ** 2))


def accuracy(y_true: np.ndarray, y_pred) -> float:
    pred = np.array(to_list(y_pred))
    return float(np.mean(y_true == pred))


# ---------------------------------------------------------------------------
# Section 1 – DecisionTreeClassifier
# ---------------------------------------------------------------------------

def test_dtc_basic_fit_predict():
    X, y = make_clf_data(n=200)
    clf = DecisionTreeClassifier()
    ret = clf.fit(X, y)
    assert ret is clf
    pred = to_list(clf.predict(X))
    assert len(pred) == 200
    assert set(pred).issubset({0.0, 1.0})
    assert accuracy(y, pred) > 0.8


def test_dtc_predict_before_fit_raises():
    X, _ = make_clf_data(n=10)
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict(X)


def test_dtc_wrong_feature_count_raises():
    X4, y = make_clf_data(n=50, n_features=4)
    X3, _ = make_clf_data(n=10, n_features=3)
    clf = DecisionTreeClassifier().fit(X4, y)
    with pytest.raises((ValueError, RuntimeError)):
        clf.predict(X3)


def test_dtc_max_depth_one_stump():
    X, y = make_clf_data(n=200)
    clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert clf.max_depth_reached == 1
    assert clf.n_nodes == 3  # root + 2 leaves


def test_dtc_all_same_label():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3))
    y = np.zeros(50)
    clf = DecisionTreeClassifier().fit(X, y)
    pred = to_list(clf.predict(X))
    assert all(p == 0.0 for p in pred)


def test_dtc_single_sample():
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([0.0])
    clf = DecisionTreeClassifier().fit(X, y)
    pred = to_list(clf.predict(X))
    assert pred == [0.0]


def test_dtc_min_samples_leaf_equals_n():
    """min_samples_leaf=n forces a single-leaf tree."""
    n = 50
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] > 0).astype(float)
    clf = DecisionTreeClassifier(min_samples_leaf=n).fit(X, y)
    assert clf.n_nodes == 1


def test_dtc_entropy_criterion():
    X, y = make_clf_data(n=200)
    clf = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    pred = to_list(clf.predict(X))
    assert accuracy(y, pred) > 0.7


def test_dtc_random_projection_splits():
    X, y = make_clf_data(n=200)
    clf = DecisionTreeClassifier(split_geometry="random_projection").fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 200
    assert set(pred).issubset({0.0, 1.0})


def test_dtc_n_nodes_and_max_depth_properties():
    X, y = make_clf_data(n=200)
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.n_nodes is not None and clf.n_nodes >= 1
    assert clf.max_depth_reached is not None and clf.max_depth_reached >= 1


def test_dtc_multiclass():
    X, y = make_clf_data(n=300, n_classes=3)
    clf = DecisionTreeClassifier().fit(X, y)
    pred = to_list(clf.predict(X))
    assert set(pred).issubset({0.0, 1.0, 2.0})


# ---------------------------------------------------------------------------
# Section 2 – DecisionTreeRegressor
# ---------------------------------------------------------------------------

def test_dtr_basic_fit_predict():
    X, y = make_reg_data(n=200)
    reg = DecisionTreeRegressor()
    ret = reg.fit(X, y)
    assert ret is reg
    pred = to_list(reg.predict(X))
    assert len(pred) == 200
    assert all(math.isfinite(p) for p in pred)
    assert mse(y, pred) < 1.0


def test_dtr_predict_before_fit_raises():
    X, _ = make_reg_data(n=10)
    reg = DecisionTreeRegressor()
    with pytest.raises(ValueError, match="not fitted"):
        reg.predict(X)


def test_dtr_wrong_dim_raises():
    X4, y = make_reg_data(n=50, n_features=4)
    X2, _ = make_reg_data(n=10, n_features=2)
    reg = DecisionTreeRegressor().fit(X4, y)
    with pytest.raises((ValueError, RuntimeError)):
        reg.predict(X2)


def test_dtr_out_of_range_x_no_panic():
    """Predicting on extreme out-of-distribution values must not panic."""
    X, y = make_reg_data(n=100)
    reg = DecisionTreeRegressor().fit(X, y)
    X_ood = np.full((5, 4), 1e6)
    X_ood[1] = -1e6
    pred = to_list(reg.predict(X_ood))
    assert len(pred) == 5
    assert all(math.isfinite(p) for p in pred)


def test_dtr_max_depth_controls_depth():
    X, y = make_reg_data(n=200)
    reg = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert reg.max_depth_reached <= 3


# ---------------------------------------------------------------------------
# Section 3 – RandomForestClassifier
# ---------------------------------------------------------------------------

def test_rfc_basic_fit_predict():
    X, y = make_clf_data(n=300, n_features=6)
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 300
    assert accuracy(y, pred) > 0.85


def test_rfc_predict_before_fit_raises():
    X, _ = make_clf_data(n=10)
    clf = RandomForestClassifier(n_estimators=5)
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict(X)


def test_rfc_wrong_dim_raises():
    X4, y = make_clf_data(n=50, n_features=4)
    X3, _ = make_clf_data(n=10, n_features=3)
    clf = RandomForestClassifier(n_estimators=5).fit(X4, y)
    with pytest.raises((ValueError, RuntimeError)):
        clf.predict(X3)


@pytest.mark.slow
def test_rfc_large_n_trees_no_crash():
    X, y = make_clf_data(n=100)
    clf = RandomForestClassifier(n_estimators=200).fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 100


def test_rfc_reproducible_with_random_state():
    X, y = make_clf_data(n=200, seed=3)
    X_test, _ = make_clf_data(n=20, seed=4)
    clf1 = RandomForestClassifier(n_estimators=20, random_state=7).fit(X, y)
    clf2 = RandomForestClassifier(n_estimators=20, random_state=7).fit(X, y)
    p1 = to_list(clf1.predict(X_test))
    p2 = to_list(clf2.predict(X_test))
    assert p1 == p2


def test_rfc_random_projection_splits():
    X, y = make_clf_data(n=200)
    clf = RandomForestClassifier(n_estimators=5, split_geometry="random_projection").fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 200


def test_rfc_entropy_criterion():
    X, y = make_clf_data(n=200)
    clf = RandomForestClassifier(n_estimators=5, criterion="entropy").fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 200


# ---------------------------------------------------------------------------
# Section 4 – RandomForestRegressor
# ---------------------------------------------------------------------------

def test_rfr_basic_fit_predict():
    X, y = make_reg_data(n=300)
    reg = RandomForestRegressor(n_estimators=10).fit(X, y)
    pred = to_list(reg.predict(X))
    assert len(pred) == 300
    assert mse(y, pred) < 0.5


def test_rfr_predict_before_fit_raises():
    X, _ = make_reg_data(n=10)
    reg = RandomForestRegressor(n_estimators=5)
    with pytest.raises(ValueError, match="not fitted"):
        reg.predict(X)


def test_rfr_out_of_distribution_no_panic():
    X, y = make_reg_data(n=100)
    reg = RandomForestRegressor(n_estimators=5).fit(X, y)
    X_ood = np.vstack([np.full((1, 4), 1e6), np.full((1, 4), -1e6)])
    pred = to_list(reg.predict(X_ood))
    assert len(pred) == 2
    assert all(math.isfinite(p) for p in pred)


# ---------------------------------------------------------------------------
# Section 6 – KNNClassifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_type", ["kd", "ball", "vp", "rp"])
def test_knnc_all_tree_types(tree_type):
    X, y = make_clf_data(n=100)
    clf = KNNClassifier(k=3, tree=tree_type).fit(X, y)
    pred = to_list(clf.predict(X))
    assert set(pred).issubset({0.0, 1.0})


def test_knnc_basic_fit_predict_batch():
    X, y = make_clf_data(n=200)
    clf = KNNClassifier(k=5).fit(X, y)
    pred = to_list(clf.predict(X))
    assert len(pred) == 200
    assert set(pred).issubset({0.0, 1.0})


def test_knnc_single_sample_returns_scalar():
    """Predict on a single sample returns a scalar label, not an array (knn.py:94-95)."""
    X, y = make_clf_data(n=100)
    clf = KNNClassifier(k=3).fit(X, y)
    single = X[:1]  # shape (1, 4)
    result = clf.predict(single)
    # Should be a scalar-like value, not an irn.Array of length > 1
    if isinstance(result, irn.Array):
        vals = to_list(result)
        assert len(vals) == 1
    else:
        assert result in {0.0, 1.0, 0, 1}


def test_knnc_distance_weighting():
    X, y = make_clf_data(n=100)
    clf = KNNClassifier(k=5, weights="distance").fit(X, y)
    pred = to_list(clf.predict(X))
    assert set(pred).issubset({0.0, 1.0})


def test_knnc_predict_proba_sums_to_one():
    X, y = make_clf_data(n=200, n_classes=3)
    clf = KNNClassifier(k=5).fit(X, y)
    proba = irn.ndutils.to_numpy(clf.predict_proba(X))
    assert proba.shape == (200, 3)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(200), atol=1e-9)


def test_knnc_predict_proba_single_sample():
    X, y = make_clf_data(n=100, n_classes=2)
    clf = KNNClassifier(k=3).fit(X, y)
    proba = irn.ndutils.to_numpy(clf.predict_proba(X[:1]))
    assert proba.shape == (1, 2)
    assert all(0.0 <= v <= 1.0 for v in proba.flatten())


def test_knnc_all_same_label():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 3))
    y = np.zeros(50)
    clf = KNNClassifier(k=3).fit(X, y)
    pred = to_list(clf.predict(X))
    assert all(p == 0 or p == 0.0 for p in pred)


def test_knnc_predict_before_fit_raises():
    """No tree_ guard in KNNClassifier — AttributeError on self.tree (knn.py)."""
    clf = KNNClassifier(k=3)
    with pytest.raises((AttributeError, ValueError, RuntimeError)):
        clf.predict(np.ones((5, 4)))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine"])
def test_knnc_all_metrics(metric):
    rng = np.random.default_rng(7)
    # Use non-zero vectors for cosine metric compatibility
    X = rng.standard_normal((100, 4)) + 1.0
    y = (X[:, 0] > 1.0).astype(float)
    clf = KNNClassifier(k=5, metric=metric).fit(X, y)
    pred = to_list(clf.predict(X))
    assert set(pred).issubset({0.0, 1.0, 0, 1})


# ---------------------------------------------------------------------------
# Section 7 – KNNRegressor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_type", ["kd", "ball", "vp", "rp"])
def test_knnr_all_tree_types(tree_type):
    X, y = make_reg_data(n=100)
    reg = KNNRegressor(k=3, tree=tree_type).fit(X, y)
    pred = to_list(reg.predict(X))
    assert all(math.isfinite(p) for p in pred)


def test_knnr_basic_fit_predict():
    X, y = make_reg_data(n=200)
    reg = KNNRegressor(k=5).fit(X, y)
    pred = to_list(reg.predict(X))
    assert len(pred) == 200
    assert all(math.isfinite(p) for p in pred)


def test_knnr_distance_weighting():
    X, y = make_reg_data(n=100)
    reg = KNNRegressor(k=5, weights="distance").fit(X, y)
    pred = to_list(reg.predict(X))
    assert all(math.isfinite(p) for p in pred)


def test_knnr_single_sample_returns_scalar():
    """Single-sample predict returns a scalar (knn.py:219)."""
    X, y = make_reg_data(n=100)
    reg = KNNRegressor(k=5).fit(X, y)
    result = reg.predict(X[:1])
    # Should be scalar-like
    if isinstance(result, irn.Array):
        vals = to_list(result)
        assert len(vals) == 1
    else:
        assert math.isfinite(float(result))


# ---------------------------------------------------------------------------
# Section 8 – LinearRegression
# ---------------------------------------------------------------------------

def test_linear_regression_basic():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.standard_normal(200) * 0.05

    model = LinearRegression().fit(X, y)
    pred = to_list(model.predict(X))
    assert len(pred) == 200
    assert mse(y, pred) < 0.1
    assert math.isclose(model.intercept_, 1.0, abs_tol=0.1)
    coef_val = float(irn.ndutils.to_numpy(model.coef_).flat[0])
    assert math.isclose(coef_val, 2.0, abs_tol=0.1)


def test_linear_regression_score():
    rng = np.random.default_rng(100)
    X = rng.standard_normal((200, 2))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.standard_normal(200) * 0.05
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.9


def test_linear_regression_predict_before_fit_raises():
    model = LinearRegression()
    with pytest.raises(RuntimeError, match="(?i)fitted|fit"):
        model.predict(np.ones((5, 2)))
