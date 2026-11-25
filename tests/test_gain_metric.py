"""Fast checks for the gain-oriented metric prototype (V3)."""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competence_metric_local_stacking import CompetenceMetricLocalStack  # noqa: E402
from gain_metric_local_stacking import learn_gain_metric  # noqa: E402
from stability_gated_local_stacking import _oof_predictions  # noqa: E402


def cheap_experts():
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("tree", DecisionTreeRegressor(max_depth=4, random_state=0)),
    ]


def make_data(n=80, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=n)
    return X, y


def test_gain_metric_returns_mean_one_weights():
    X, y = make_data()
    oof = _oof_predictions(X, y, cheap_experts(), 3, 0)
    weights, evidence, scores = learn_gain_metric(
        X, y, oof, seed=0, metric_strength=0.5, n_neighbors=8,
        n_splits=1, max_passes=1, factors=(0.5, 2.0), max_rows=60,
    )
    assert weights.shape == (X.shape[1],)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    assert abs(float(weights.mean()) - 1.0) < 1e-9
    assert evidence.shape == weights.shape
    assert np.all(scores >= 0)


def test_zero_strength_is_uniform():
    X, y = make_data()
    oof = _oof_predictions(X, y, cheap_experts(), 3, 0)
    weights, evidence, _ = learn_gain_metric(
        X, y, oof, metric_strength=0.0, n_neighbors=8,
    )
    assert np.allclose(weights, 1.0)
    assert np.allclose(evidence, 0.0)


def test_gain_learner_hook_in_local_stack():
    X, y = make_data()
    model = CompetenceMetricLocalStack(
        estimators=cheap_experts(), cv=3, n_neighbors=8,
        metric_strength=0.5, permutations=4, seed=0,
        metric_learner=learn_gain_metric,
        metric_kwargs={
            "n_splits": 1, "max_passes": 1,
            "factors": (0.5, 2.0), "max_rows": 60,
        },
    ).fit(X, y)
    assert model.metric_weights_.shape == (X.shape[1],)
    predictions = model.predict_global(X[:10])
    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))
