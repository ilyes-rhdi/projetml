"""Fast checks of LocalStackingRegressor. Run from the project root: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# The modules under test live one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_on_dataset import make_regional_data  # noqa: E402
from local_stacking import LocalStackingRegressor  # noqa: E402


def fast_experts():
    """Cheap experts so the tests run in seconds. One linear, two local/nonlinear."""
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("tree", DecisionTreeRegressor(max_depth=6, random_state=0)),
        ("knn", KNeighborsRegressor(n_neighbors=10)),
    ]


def split(X, y, n_train):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@pytest.fixture(scope="module")
def regional():
    """Model fitted on data where different experts win in different regions.

    1200 training points: with the weak test experts, smaller sets sometimes give too
    little evidence for the strict rule to go local."""
    X_train, y_train, X_test, y_test = split(*make_regional_data(n_samples=1600, seed=1), n_train=1200)
    model = LocalStackingRegressor(fast_experts(), random_state=0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def linear():
    """Model fitted on a linear target plus noise: no regional structure at all."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 5))
    y = X @ np.array([1.5, -2.0, 0.5, 1.0, 0.0]) + rng.normal(scale=0.5, size=800)
    X_train, y_train, X_test, y_test = split(X, y, n_train=600)
    model = LocalStackingRegressor(fast_experts(), random_state=0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


def plain_stacking(model, y_train, X_test):
    """Positive linear stacking refitted from scratch on the model's own OOF predictions."""
    stack = LinearRegression(positive=True).fit(model.oof_predictions_, y_train)
    return stack.predict(model.predict_experts(X_test))


def test_forced_global_equals_plain_stacking(regional):
    model, _, y_train, X_test, _ = regional
    np.testing.assert_allclose(model.predict(X_test, shrink=np.inf),
                               plain_stacking(model, y_train, X_test), rtol=1e-12, atol=1e-12)


def test_local_weights_are_non_negative(regional):
    model, _, _, X_test, _ = regional
    weights = model.local_weights(X_test, shrink=1e-4)
    assert weights.shape == (len(X_test), 3)
    assert np.all(weights >= 0)


def test_very_large_shrink_gives_global_weights(regional):
    model, _, _, X_test, _ = regional
    weights = model.local_weights(X_test[:20], shrink=1e8)
    np.testing.assert_allclose(weights, np.tile(model.global_weights_, (20, 1)), atol=1e-4)


def test_regional_data_goes_local_and_beats_global(regional):
    model, _, _, X_test, y_test = regional
    assert model.is_local_
    assert np.isfinite(model.shrink_)
    assert rmse(y_test, model.predict(X_test)) < rmse(y_test, model.predict(X_test, shrink=np.inf))


def test_linear_data_stays_global_and_equals_stacking(linear):
    model, _, y_train, X_test, _ = linear
    assert not model.is_local_
    assert model.shrink_ == np.inf
    np.testing.assert_allclose(model.predict(X_test), plain_stacking(model, y_train, X_test),
                               rtol=1e-12, atol=1e-12)


def test_selection_errors_are_paired_over_all_candidates(regional):
    model = regional[0]
    assert list(model.selection_errors_) == list(model.shrink_grid) + [np.inf]
    sizes = {len(errors) for errors in model.selection_errors_.values()}
    assert sizes == {min(model.selection_sample, len(model.y_train_))}


def test_fixed_random_state_is_deterministic(regional):
    _, X_train, y_train, X_test, _ = regional
    first = LocalStackingRegressor(fast_experts(), random_state=3).fit(X_train, y_train)
    second = LocalStackingRegressor(fast_experts(), random_state=3).fit(X_train, y_train)
    assert first.shrink_ == second.shrink_
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))


def test_clone_gives_an_unfitted_copy_with_same_params(regional):
    model = regional[0]
    copy = clone(model)
    assert copy.get_params()["n_neighbors"] == model.n_neighbors
    assert copy.get_params()["rule"] == model.rule
    assert not hasattr(copy, "shrink_")


def test_unknown_rule_is_rejected():
    X, y = make_regional_data(n_samples=100, seed=0)
    with pytest.raises(ValueError):
        LocalStackingRegressor(fast_experts(), rule="best").fit(X, y)
