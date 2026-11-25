"""Bootstrap-stability-gated local stacking (experimental research variant).

For each query, fit the usual shrunk local stack on its k nearest training
cases. Bootstrap the neighborhood and measure the spread of the resulting
query predictions. A dimensionless stability score controls how far the final
prediction moves from global stacking toward local stacking. Hyperparameters
are chosen using nested, fold-isolated CV by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from local_stacking import solve_local_weights


def make_experts(seed=0):
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=1.0, random_state=seed,
        )),
        ("rf", RandomForestRegressor(
            n_estimators=60, min_samples_leaf=2, n_jobs=1, random_state=seed,
        )),
    ]


def _oof_predictions(X, y, estimators, n_splits, seed):
    splitter = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    oof = np.empty((len(y), len(estimators)), dtype=float)
    for fit_idx, held_idx in splitter.split(X):
        for j, (_, estimator) in enumerate(estimators):
            fitted = clone(estimator).fit(X[fit_idx], y[fit_idx])
            oof[held_idx, j] = fitted.predict(X[held_idx])
    return oof


@dataclass
class FittedLocalStack:
    scaler: StandardScaler
    experts: list
    oof_predictions: np.ndarray
    y: np.ndarray
    stack: LinearRegression
    neighbors: NearestNeighbors
    y_scale: float

    def predict_components(self, X, shrink=0.1, bootstrap=12, seed=0):
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        query_predictions = np.column_stack([m.predict(X_scaled) for m in self.experts])
        global_prediction = self.stack.intercept_ + query_predictions @ self.stack.coef_
        neighbor_idx = self.neighbors.kneighbors(X_scaled, return_distance=False)
        centered_targets = self.y - self.stack.intercept_
        rng = np.random.default_rng(seed)
        local_prediction = np.empty(len(X), dtype=float)
        local_sd = np.empty(len(X), dtype=float)
        for i, idx in enumerate(neighbor_idx):
            local_weights = solve_local_weights(
                self.oof_predictions[idx], centered_targets[idx], self.stack.coef_, shrink
            )
            local_prediction[i] = self.stack.intercept_ + query_predictions[i] @ local_weights
            bootstrap_predictions = np.empty(bootstrap, dtype=float)
            for b in range(bootstrap):
                sampled = rng.choice(idx, size=len(idx), replace=True)
                boot_weights = solve_local_weights(
                    self.oof_predictions[sampled], centered_targets[sampled],
                    self.stack.coef_, shrink,
                )
                bootstrap_predictions[b] = (
                    self.stack.intercept_ + query_predictions[i] @ boot_weights
                )
            local_sd[i] = bootstrap_predictions.std(ddof=1) if bootstrap > 1 else 0.0
        return global_prediction, local_prediction, local_sd

    def predict(self, X, shrink=0.1, stability_strength=1.0, bootstrap=12, seed=0):
        if np.isinf(shrink):
            X = self.scaler.transform(np.asarray(X, dtype=float))
            query_predictions = np.column_stack([m.predict(X) for m in self.experts])
            return self.stack.intercept_ + query_predictions @ self.stack.coef_
        global_pred, local_pred, local_sd = self.predict_components(
            X, shrink=shrink, bootstrap=bootstrap, seed=seed
        )
        trust = np.exp(-float(stability_strength) * local_sd / self.y_scale)
        return global_pred + trust * (local_pred - global_pred)


def fit_local_stack(X, y, estimators=None, cv=3, n_neighbors=50, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 3:
        raise ValueError("At least three rows are required.")
    estimators = make_experts(seed) if estimators is None else estimators
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    oof = _oof_predictions(X_scaled, y, estimators, cv, seed)
    stack = LinearRegression(positive=True).fit(oof, y)
    fitted_experts = [clone(est).fit(X_scaled, y) for _, est in estimators]
    k = min(n_neighbors, len(y) - 1)
    neighbors = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    y_scale = max(float(np.std(y)), 1e-8)
    return FittedLocalStack(
        scaler=scaler, experts=fitted_experts, oof_predictions=oof, y=y,
        stack=stack, neighbors=neighbors, y_scale=y_scale,
    )


def select_stability_rule(X, y, seed=0, inner_folds=3, oof_folds=3,
                          n_neighbors=50, bootstrap=8,
                          selection_sample=1000,
                          shrink_grid=(1e-3, 1e-2, 1e-1, 1.0),
                          strength_grid=(0.25, 0.5, 1.0, 2.0)):
    """Select local shrinkage/stability strength with fold-isolated inner CV."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed + 17)
    sample = rng.choice(len(y), size=min(selection_sample, len(y)), replace=False)
    position = np.full(len(y), -1, dtype=int)
    position[sample] = np.arange(len(sample))
    splitter = KFold(n_splits=min(inner_folds, len(y)), shuffle=True, random_state=seed)
    predictions = {(lam, strength): np.empty(len(sample))
                   for lam in shrink_grid for strength in strength_grid}
    global_predictions = np.empty(len(sample))
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X)):
        valid_sample = valid_idx[position[valid_idx] >= 0]
        if not len(valid_sample):
            continue
        valid_pos = position[valid_sample]
        model = fit_local_stack(
            X[fit_idx], y[fit_idx], cv=oof_folds, n_neighbors=n_neighbors,
            seed=seed + 97 * (fold + 1),
        )
        # Bootstrap once per lambda; the same uncertainty estimate supports all
        # strengths in the grid and makes comparisons paired on identical rows.
        for lam in shrink_grid:
            gp, lp, sd = model.predict_components(
                X[valid_sample], shrink=lam, bootstrap=bootstrap,
                seed=seed + 1009 * (fold + 1) + int(abs(np.log10(lam)) * 10),
            )
            global_predictions[valid_pos] = gp
            for strength in strength_grid:
                trust = np.exp(-strength * sd / model.y_scale)
                predictions[(lam, strength)][valid_pos] = gp + trust * (lp - gp)

    y_sample = y[sample]
    global_errors = (y_sample - global_predictions) ** 2
    candidate_errors = {key: (y_sample - pred) ** 2 for key, pred in predictions.items()}
    best = min(candidate_errors, key=lambda key: candidate_errors[key].mean())
    gain = global_errors - candidate_errors[best]
    se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(y) > 1 else np.inf
    if gain.mean() > se:
        return {"shrink": best[0], "strength": best[1], "selected_local": True,
                "inner_global_mse": float(global_errors.mean()),
                "inner_candidate_mse": float(candidate_errors[best].mean()),
                "inner_gain": float(gain.mean()), "inner_gain_se": float(se)}
    return {"shrink": np.inf, "strength": 0.0, "selected_local": False,
            "inner_global_mse": float(global_errors.mean()),
            "inner_candidate_mse": float(candidate_errors[best].mean()),
            "inner_gain": float(gain.mean()), "inner_gain_se": float(se)}
