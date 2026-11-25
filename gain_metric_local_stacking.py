"""Gain-oriented diagonal metric for local stacking (V3 prototype).

The current prototype learns the neighbourhood metric from a proxy target: the
identity of the locally best expert. This module instead chooses the diagonal
feature weights by minimizing a cross-validated local-stacking RMSE on the
training fold. The downstream objective is therefore the actual prediction
loss, not a classification proxy.

The search is a small coordinate descent on log feature weights. Each split
uses 70 % of a capped training subsample to fit the local weights and 30 % to
score a candidate metric, so the metric is not selected on the same rows that
fit the local stack. The final metric is the average of the splits.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from local_stacking import solve_local_weights


def _weights_from_theta(theta, metric_strength):
    """Map unconstrained coordinates to positive, mean-one feature weights."""
    raw = np.exp(np.clip(metric_strength * np.asarray(theta, dtype=float), -2.5, 2.5))
    raw = np.clip(raw, 0.1, 10.0)
    return raw / raw.mean()


def _local_mse(weights, X, y, oof, global_weights, intercept,
               index_idx, query_idx, n_neighbors):
    """Local-stacking MSE on query rows, with neighbours drawn from index rows."""
    index_idx = np.asarray(index_idx, dtype=int)
    query_idx = np.asarray(query_idx, dtype=int)
    if len(index_idx) < 3 or len(query_idx) == 0:
        return np.inf

    transformed = X * np.sqrt(weights)
    k = min(n_neighbors + 1, len(index_idx))
    neighbours = NearestNeighbors(n_neighbors=k).fit(transformed[index_idx])
    candidate_rows = neighbours.kneighbors(
        transformed[query_idx], return_distance=False,
    )

    squared = []
    for query, local_rows in zip(query_idx, candidate_rows):
        rows = index_idx[local_rows]
        rows = rows[rows != query][:n_neighbors]
        if len(rows) < 2:
            continue
        local_weights = solve_local_weights(
            oof[rows], y[rows] - intercept, global_weights, 1.0,
        )
        prediction = intercept + oof[query] @ local_weights
        squared.append((y[query] - prediction) ** 2)
    return float(np.mean(squared)) if squared else np.inf


def _coordinate_descent(X, y, oof, global_weights, intercept,
                        fit_idx, valid_idx, *, n_neighbors, metric_strength,
                        max_passes, factors, seed):
    """Choose diagonal weights by coordinate descent on validation MSE."""
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    best_weights = _weights_from_theta(theta, metric_strength)
    best_mse = _local_mse(
        best_weights, X, y, oof, global_weights, intercept,
        fit_idx, valid_idx, n_neighbors,
    )
    if not np.isfinite(best_mse):
        return best_weights, best_mse

    log_factors = np.log(np.asarray(factors, dtype=float))
    for _ in range(max_passes):
        improved = False
        for feature in range(n_features):
            current_theta = theta.copy()
            current_weights = _weights_from_theta(current_theta, metric_strength)
            current_mse = _local_mse(
                current_weights, X, y, oof, global_weights, intercept,
                fit_idx, valid_idx, n_neighbors,
            )
            local_theta = current_theta
            local_mse = current_mse
            for step in log_factors:
                candidate_theta = current_theta.copy()
                candidate_theta[feature] += step
                candidate_weights = _weights_from_theta(
                    candidate_theta, metric_strength,
                )
                mse = _local_mse(
                    candidate_weights, X, y, oof, global_weights, intercept,
                    fit_idx, valid_idx, n_neighbors,
                )
                if mse < local_mse - 1e-12:
                    local_mse = mse
                    local_theta = candidate_theta
            if local_mse < best_mse - 1e-12:
                theta = local_theta
                best_mse = local_mse
                improved = True
        if not improved:
            break
    return _weights_from_theta(theta, metric_strength), best_mse


def learn_gain_metric(X, y, oof_predictions, *, seed=0, permutations=24,
                      metric_strength=0.75, n_neighbors=50,
                      n_splits=1, max_passes=2, factors=(0.5, 2.0),
                      max_rows=600, min_rows=40, **kwargs):
    """Return diagonal weights tuned for the local-stacking RMSE.

    Parameters mirror ``learn_competence_metric`` so the learner can be passed
    to ``CompetenceMetricLocalStack``. ``permutations`` is accepted for API
    compatibility but is unused.

    Returns
    -------
    weights : ndarray of shape (n_features,)
        Positive, mean-one feature weights.
    evidence : ndarray of shape (n_features,)
        Positive deviations from uniform, for diagnostics.
    scores : ndarray
        Validation MSE of each split, for diagnostics.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    oof = np.asarray(oof_predictions, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or len(oof) != len(y):
        raise ValueError("X, y, and OOF predictions must have matching rows.")
    if not 0.0 <= metric_strength <= 1.0:
        raise ValueError("metric_strength must be between 0 and 1.")

    n_features = X.shape[1]
    if metric_strength == 0.0 or len(y) < min_rows:
        return np.ones(n_features), np.zeros(n_features), np.zeros(1)

    stack = LinearRegression(positive=True).fit(oof, y)
    global_weights = np.asarray(stack.coef_, dtype=float)
    intercept = float(stack.intercept_)

    rng = np.random.default_rng(seed + 1301)
    n_rows = len(y)
    subset = (
        np.arange(n_rows) if n_rows <= max_rows
        else rng.choice(n_rows, max_rows, replace=False)
    )

    weight_samples = []
    validation_scores = []
    for _ in range(max(1, n_splits)):
        order = rng.permutation(len(subset))
        cut = max(2, int(0.7 * len(order)))
        fit_idx = subset[order[:cut]]
        valid_idx = subset[order[cut:]]
        if len(fit_idx) < 3 or len(valid_idx) < 2:
            continue
        weights, mse = _coordinate_descent(
            X, y, oof, global_weights, intercept,
            fit_idx, valid_idx,
            n_neighbors=n_neighbors, metric_strength=metric_strength,
            max_passes=max_passes, factors=factors, seed=seed,
        )
        if np.isfinite(mse):
            weight_samples.append(weights)
            validation_scores.append(mse)

    if not weight_samples:
        return np.ones(n_features), np.zeros(n_features), np.zeros(1)

    weights = np.mean(weight_samples, axis=0)
    weights = np.clip(weights, 0.1, 10.0)
    weights /= weights.mean()
    evidence = np.maximum(weights - 1.0, 0.0)
    return weights, evidence, np.asarray(validation_scores, dtype=float)
