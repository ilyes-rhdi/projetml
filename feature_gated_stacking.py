"""Feature-conditioned expert-error gating with a nested global fallback.

This research implementation trains experts, obtains out-of-fold residuals,
predicts each expert's log squared error from X, and converts those error
predictions into instance-specific ensemble weights. The gate/global mixture is
selected by an inner CV that keeps the validation rows out of expert, gate, and
stack fitting. It is an experimental baseline, not a claim of novelty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_experts(seed: int = 0):
    """A small, diverse pool matching the reference project's expert families."""
    return [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=1.0, random_state=seed,
        )),
        ("rf", RandomForestRegressor(
            n_estimators=60, min_samples_leaf=2, n_jobs=1, random_state=seed,
        )),
    ]


def out_of_fold_predictions(X, y, estimators, n_splits=3, seed=0):
    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        raise ValueError("At least two training rows are required.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.empty((len(y), len(estimators)), dtype=float)
    for fit_idx, held_idx in splitter.split(X):
        for j, (_, estimator) in enumerate(estimators):
            fitted = clone(estimator).fit(X[fit_idx], y[fit_idx])
            predictions[held_idx, j] = fitted.predict(X[held_idx])
    return predictions


@dataclass
class FittedFeatureGate:
    names: tuple[str, ...]
    estimators: list
    gate_models: list
    global_stack: LinearRegression
    oof_predictions: np.ndarray
    gate_cv_rmse: np.ndarray

    def _expert_predictions(self, X):
        return np.column_stack([model.predict(X) for model in self.estimators])

    def predict_components(self, X, beta=1.0):
        """Return global and feature-gated predictions plus gated weights."""
        expert_predictions = self._expert_predictions(X)
        global_prediction = self.global_stack.predict(expert_predictions)
        predicted_log_mse = np.column_stack([model.predict(X) for model in self.gate_models])
        logits = -float(beta) * predicted_log_mse
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(np.clip(logits, -60.0, 0.0))
        weights /= weights.sum(axis=1, keepdims=True)
        gated_prediction = self.global_stack.intercept_ + np.sum(
            expert_predictions * weights, axis=1
        )
        return global_prediction, gated_prediction, weights, expert_predictions

    def predict(self, X, beta=1.0, gate_mix=1.0):
        """Blend global stacking with the learned gate; gate_mix=0 is global."""
        global_prediction, gated_prediction, _, _ = self.predict_components(X, beta)
        return (1.0 - gate_mix) * global_prediction + gate_mix * gated_prediction


def fit_feature_gate(X, y, estimators=None, cv=3, seed=0):
    """Fit base experts and error gates on one training partition only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    estimators = make_experts(seed) if estimators is None else estimators
    names = tuple(name for name, _ in estimators)
    oof = out_of_fold_predictions(X, y, estimators, n_splits=cv, seed=seed)
    stack = LinearRegression(positive=True).fit(oof, y)
    full_estimators = [clone(estimator).fit(X, y) for _, estimator in estimators]

    residual2 = (y[:, None] - oof) ** 2
    # A scale-relative floor keeps exact-fit rows finite without affecting ranking.
    floor = max(float(np.var(y)) * 1e-8, 1e-12)
    log_residual2 = np.log(np.maximum(residual2, floor))
    min_leaf = max(10, min(25, len(y) // 20))
    gates = []
    for j in range(oof.shape[1]):
        gate = HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=min_leaf, l2_regularization=2.0,
            early_stopping=False, random_state=seed + j,
        )
        gate.fit(X, log_residual2[:, j])
        gates.append(gate)
    return FittedFeatureGate(
        names=names,
        estimators=full_estimators,
        gate_models=gates,
        global_stack=stack,
        oof_predictions=oof,
        gate_cv_rmse=np.sqrt(np.mean(residual2, axis=0)),
    )


def select_gate_mix(X, y, seed=0, inner_folds=3, oof_folds=3, estimators=None):
    """Select error-gate temperature/mix using only nested inner-CV predictions.

    The one-standard-error fallback is a selection heuristic, not a guarantee.
    Outer CV is used by the caller to evaluate this complete selection procedure.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    splitter = KFold(n_splits=min(inner_folds, len(y)), shuffle=True, random_state=seed)
    betas = (0.25, 0.5, 1.0, 2.0)
    mixes = (0.25, 0.5, 0.75, 1.0)
    candidate_predictions = {(beta, mix): np.empty(len(y)) for beta in betas for mix in mixes}
    global_predictions = np.empty(len(y))

    for inner_id, (fit_idx, valid_idx) in enumerate(splitter.split(X)):
        model = fit_feature_gate(
            X[fit_idx], y[fit_idx], estimators=estimators, cv=oof_folds,
            seed=seed + 101 * (inner_id + 1),
        )
        global_pred, _, _, _ = model.predict_components(X[valid_idx], beta=1.0)
        global_predictions[valid_idx] = global_pred
        for beta in betas:
            _, gate_pred, _, _ = model.predict_components(X[valid_idx], beta=beta)
            for mix in mixes:
                candidate_predictions[(beta, mix)][valid_idx] = (
                    (1.0 - mix) * global_pred + mix * gate_pred
                )

    global_errors = (y - global_predictions) ** 2
    errors = {key: (y - pred) ** 2 for key, pred in candidate_predictions.items()}
    best = min(errors, key=lambda key: errors[key].mean())
    paired_gain = global_errors - errors[best]
    standard_error = paired_gain.std(ddof=1) / np.sqrt(len(paired_gain)) if len(y) > 1 else np.inf
    if paired_gain.mean() > standard_error:
        return {"beta": best[0], "gate_mix": best[1], "selected_local": True,
                "inner_global_mse": float(global_errors.mean()),
                "inner_gate_mse": float(errors[best].mean()),
                "inner_gain": float(paired_gain.mean()),
                "inner_gain_se": float(standard_error)}
    return {"beta": 1.0, "gate_mix": 0.0, "selected_local": False,
            "inner_global_mse": float(global_errors.mean()),
            "inner_gate_mse": float(errors[best].mean()),
            "inner_gain": float(paired_gain.mean()),
            "inner_gain_se": float(standard_error)}
