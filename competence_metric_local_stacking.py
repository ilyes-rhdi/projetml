"""Competence-aligned neighborhood metric for local stacking (research prototype).

The feature metric is learned from cross-fitted expert errors. Features are
weighted only when their association with the identity of the locally best
expert exceeds a permutation-based null threshold. This is intended to test
whether competence-relevant neighborhoods improve local stacking when ordinary
Euclidean neighborhoods are diluted by irrelevant predictors.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from local_stacking import solve_local_weights
from stability_gated_local_stacking import _oof_predictions, make_experts


def learn_competence_metric(X, y, oof_predictions, *, seed=0, permutations=24,
                            metric_strength=0.75, **kwargs):
    """Return diagonal metric weights from cross-fitted expert competence.

    Relative absolute errors identify the best expert per training case. A
    feature receives weight only if its mutual information with that identity
    exceeds the 95th percentile obtained after permuting the competence labels.
    The result is shrunk toward uniform Euclidean distance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    oof = np.asarray(oof_predictions, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or len(oof) != len(y):
        raise ValueError("X, y, and OOF predictions must have matching rows.")
    if not 0.0 <= metric_strength <= 1.0:
        raise ValueError("metric_strength must be between 0 and 1.")

    # Normalize each expert's error so that a globally weaker model does not
    # lose every local region by scale alone.
    losses = np.abs(y[:, None] - oof)
    scales = np.maximum(np.mean(losses, axis=0), 1e-12)
    competence = np.argmin(losses / scales, axis=1)
    if metric_strength == 0.0 or len(np.unique(competence)) < 2:
        return np.ones(X.shape[1]), np.zeros(X.shape[1]), np.zeros(X.shape[1])
    discrete = np.array([len(np.unique(X[:, j])) <= 2 for j in range(X.shape[1])])
    mi_X = X.copy()
    # sklearn's discrete-feature MI path expects class-like integer values;
    # standardized binary columns are floats, so restore their two categories.
    for j in np.flatnonzero(discrete):
        mi_X[:, j] = (mi_X[:, j] > np.median(mi_X[:, j])).astype(int)
    n_neighbors = min(3, max(1, len(y) - 1))
    observed = mutual_info_classif(
        mi_X, competence, discrete_features=discrete,
        n_neighbors=n_neighbors, random_state=seed,
    )

    rng = np.random.default_rng(seed + 701)
    null = np.empty((permutations, X.shape[1]), dtype=float)
    for b in range(permutations):
        null[b] = mutual_info_classif(
            mi_X, rng.permutation(competence), discrete_features=discrete,
            n_neighbors=n_neighbors, random_state=seed + b + 1,
        )
    # Use the permutation distribution of the maximum feature score. This
    # controls false discoveries across the whole metric rather than testing
    # each dimension independently.
    threshold = np.quantile(null.max(axis=1), 0.95)
    evidence = np.maximum(observed - threshold, 0.0)
    if evidence.sum() <= 1e-12:
        return np.ones(X.shape[1]), evidence, observed

    relative = evidence * (X.shape[1] / evidence.sum())
    weights = (1.0 - metric_strength) + metric_strength * relative
    # Keep a floor so finite-sample MI noise cannot completely erase a feature.
    weights = np.maximum(weights, 0.1)
    weights /= weights.mean()
    return weights, evidence, observed


class CompetenceMetricLocalStack:
    """Local stacking using a cross-fitted, competence-aligned feature metric."""

    def __init__(self, estimators=None, *, cv=3, n_neighbors=50,
                 metric_strength=0.5, permutations=24, seed=0,
                 metric_learner=None, metric_kwargs=None):
        self.estimators = make_experts(seed) if estimators is None else estimators
        self.cv = cv
        self.n_neighbors = n_neighbors
        self.metric_strength = metric_strength
        self.permutations = permutations
        self.seed = seed
        self.metric_learner = metric_learner
        self.metric_kwargs = {} if metric_kwargs is None else dict(metric_kwargs)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.scaler_ = StandardScaler().fit(X)
        self.X_scaled_ = self.scaler_.transform(X)
        self.oof_predictions_ = _oof_predictions(
            self.X_scaled_, y, self.estimators, self.cv, self.seed,
        )
        self.global_stack_ = LinearRegression(positive=True).fit(self.oof_predictions_, y)
        self.global_weights_ = self.global_stack_.coef_
        self.intercept_ = float(self.global_stack_.intercept_)
        learner = self.metric_learner or learn_competence_metric
        (self.metric_weights_, self.metric_evidence_,
         self.metric_mi_) = learner(
            self.X_scaled_, y, self.oof_predictions_, seed=self.seed,
            permutations=self.permutations, metric_strength=self.metric_strength,
            n_neighbors=self.n_neighbors, **self.metric_kwargs,
        )
        transformed = self.X_scaled_ * np.sqrt(self.metric_weights_)
        # Build the search index on first local prediction. If validation
        # selects the global stack, inference does not need a neighbor index.
        self.transformed_train_ = transformed
        self.neighbors_ = None
        self.y_ = y
        self.estimators_ = [clone(est).fit(self.X_scaled_, y) for _, est in self.estimators]
        self.n_features_in_ = X.shape[1]
        return self

    def predict_components(self, X, shrink=0.1):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        query_predictions = np.column_stack([est.predict(X_scaled) for est in self.estimators_])
        global_pred = self.intercept_ + query_predictions @ self.global_weights_
        transformed = X_scaled * np.sqrt(self.metric_weights_)
        if self.neighbors_ is None:
            self.neighbors_ = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(self.y_) - 1),
            ).fit(self.transformed_train_)
        idx = self.neighbors_.kneighbors(transformed, return_distance=False)
        centered_y = self.y_ - self.intercept_
        weights = np.array([
            solve_local_weights(self.oof_predictions_[rows], centered_y[rows],
                                self.global_weights_, shrink)
            for rows in idx
        ])
        local_pred = self.intercept_ + np.sum(query_predictions * weights, axis=1)
        return global_pred, local_pred

    def predict_global(self, X):
        """Predict with the global stack without searching for neighbors."""
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        query_predictions = np.column_stack([est.predict(X_scaled) for est in self.estimators_])
        return self.intercept_ + query_predictions @ self.global_weights_

    def predict(self, X, shrink=0.1):
        if np.isinf(shrink):
            return self.predict_global(X)
        return self.predict_components(X, shrink=shrink)[1]


def select_shrinkage(X, y, *, estimators=None, cv=3, n_neighbors=50,
                     metric_strength=0.5, permutations=24, seed=0,
                     shrink_grid=(0.01, 0.1, 1.0, np.inf),
                     metric_learner=None, metric_kwargs=None):
    """Choose local shrinkage, including global stacking, by training-only CV."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    split = KFold(n_splits=min(cv, len(y)), shuffle=True, random_state=seed)
    squared_errors = {lam: [] for lam in shrink_grid}
    for fold, (fit_idx, valid_idx) in enumerate(split.split(X)):
        model = CompetenceMetricLocalStack(
            estimators=estimators, cv=cv, n_neighbors=n_neighbors,
            metric_strength=metric_strength, permutations=permutations,
            seed=seed + 31 * (fold + 1),
            metric_learner=metric_learner, metric_kwargs=metric_kwargs,
        ).fit(X[fit_idx], y[fit_idx])
        for lam in shrink_grid:
            global_pred, local_pred = model.predict_components(
                X[valid_idx], shrink=0.1 if np.isinf(lam) else lam,
            )
            pred = global_pred if np.isinf(lam) else local_pred
            squared_errors[lam].extend((y[valid_idx] - pred) ** 2)
    global_key = next(lam for lam in shrink_grid if np.isinf(lam))
    local_grid = [lam for lam in shrink_grid if np.isfinite(lam)]
    best_local = min(local_grid, key=lambda lam: float(np.mean(squared_errors[lam])))
    gain = np.asarray(squared_errors[global_key]) - np.asarray(squared_errors[best_local])
    gain_se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else np.inf
    if gain.mean() > gain_se:
        return best_local
    return global_key
