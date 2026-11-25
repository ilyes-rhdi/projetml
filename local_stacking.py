"""Local stacking with a data-chosen fallback to global stacking.

The idea in plain words
-----------------------
Stacking combines several regression models ("experts") with one set of weights,
learned on out-of-fold predictions. Those weights are the same everywhere in the
feature space. But one expert can be better in one region and another expert in
another region. Local stacking re-learns the weights for every query point, using
only the training points that are close to it. Because 50 neighbours is little data,
the local weights are pulled toward the global ones. How hard they are pulled is
chosen on the training data, and "do not go local at all" (plain global stacking)
is one of the candidates, so the method falls back to stacking when local weights
do not help.

The maths
---------
Let M be the number of experts and n the number of training points.

1. Z (n x M): out-of-fold (OOF) predictions of the experts on the training set.
   Row i holds what each expert predicts for point i when it was trained without i.
2. Global stacking: non-negative linear regression of y on Z gives weights w_g >= 0
   and an intercept b.
3. For a query x, let N be the indices of its k nearest training points (Euclidean
   distance in standardised feature space), Z_N their OOF predictions and
   r_N = y_N - b their targets minus the global intercept. The local weights solve

       w(x) = argmin_{w >= 0}  ||r_N - Z_N w||^2  +  lambda * s * ||w - w_g||^2,
       with s = ||Z_N||_F^2 / M.

   The ridge term pulls w toward w_g. Scaling it by s makes lambda unitless: the
   same lambda means the same pull whatever the scale of y. The problem is a
   non-negative least squares (NNLS) on an augmented system: stack the rows
   sqrt(lambda * s) * I under Z_N and sqrt(lambda * s) * w_g under r_N.
4. Prediction: b + p(x) . w(x), where p(x) holds the experts' predictions at x.
   lambda -> 0 gives purely local weights; lambda -> infinity gives w(x) = w_g,
   which is plain global stacking.

Choosing lambda (and the fallback)
----------------------------------
Candidates: the values in `shrink_grid` plus "global" (lambda = infinity). Each
candidate's squared error is measured on at most `selection_sample` training points.
- Local candidates never use the query point itself: the neighbours of a training
  point are searched among the *other* training points.
- The global stack would have seen the point it is judged on, which would favour it
  unfairly. So it is cross-fitted: the training set is split in 5 folds, the global
  stack (w_g, b) is refitted on 4 folds and every candidate is judged on the 5th.
Rules:
- "simple": the candidate with the lowest mean squared error wins.
- "strict" (default): take the best local lambda; go local only if its mean gain
  over global is larger than one standard error of the paired per-point gains.
  Otherwise stay global. This protects against noise-driven wins.

What is fitted on what (leakage safeguards)
-------------------------------------------
- The scaler is fitted on the training data passed to `fit` only.
- OOF predictions come from cross-validation inside the training data, so Z holds
  honest, out-of-sample predictions. The experts are then refitted on all training
  data to predict new points.
- The neighbour index holds training points only; new queries look up training
  points, never other test points.
- lambda is chosen with training data only (see above).
- Calling `predict` on the training points themselves is NOT honest: each point is
  then among its own neighbours. Evaluate on held-out data.

Cost
----
- Fit: (cv + 1) fits of every expert, plus about 5 * len(shrink_grid) *
  min(n, selection_sample) small NNLS problems for the selection.
- Predict: one small NNLS (k + M rows, M columns) per query point when the model is
  local; a single matrix product when it stays global.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

GLOBAL = np.inf
"""Shrink value meaning "no local weights": lambda = infinity, plain global stacking."""

SELECTION_FOLDS = 5
"""Number of folds used to cross-fit the global stack during the choice of lambda."""

RULES = ("strict", "simple")


# ------------------------------------------------------------ core maths

def solve_local_weights(neighbour_preds, neighbour_targets, global_weights, shrink):
    """Local stacking weights for one query point.

    Solves min_{w >= 0} ||t - Z w||^2 + shrink * s * ||w - w_g||^2 with
    s = ||Z||_F^2 / M, as an NNLS on an augmented system.

    Parameters
    ----------
    neighbour_preds : ndarray of shape (k, M)
        OOF predictions of the M experts on the k neighbours (Z_N).
    neighbour_targets : ndarray of shape (k,)
        Targets of the neighbours minus the global intercept (r_N).
    global_weights : ndarray of shape (M,)
        Global stacking weights w_g, the point the weights are pulled toward.
    shrink : float
        Pull strength lambda (finite, > 0).

    Returns
    -------
    ndarray of shape (M,)
        Non-negative local weights.
    """
    n_experts = neighbour_preds.shape[1]
    # Scaling by the size of Z makes lambda unitless, so one grid fits every dataset.
    penalty = np.sqrt(shrink * (neighbour_preds ** 2).sum() / n_experts)
    design = np.vstack([neighbour_preds, penalty * np.eye(n_experts)])
    rhs = np.concatenate([neighbour_targets, penalty * global_weights])
    return nnls(design, rhs)[0]


def local_weight_matrix(neighbour_idx, oof_preds, targets, global_weights, shrink):
    """Local weights for many query points at once.

    Parameters
    ----------
    neighbour_idx : ndarray of shape (n_queries, k)
        Row q holds the indices (into the training set) of query q's neighbours.
    oof_preds : ndarray of shape (n_train, M)
        OOF predictions of the experts on the training set (Z).
    targets : ndarray of shape (n_train,)
        Training targets minus the global intercept (y - b).
    global_weights : ndarray of shape (M,)
        Global stacking weights w_g.
    shrink : float
        Pull strength lambda; GLOBAL (infinity) returns w_g for every query.

    Returns
    -------
    ndarray of shape (n_queries, M)
        One row of non-negative weights per query.
    """
    if shrink == GLOBAL:
        return np.tile(global_weights, (len(neighbour_idx), 1))
    return np.array([
        solve_local_weights(oof_preds[idx], targets[idx], global_weights, shrink)
        for idx in neighbour_idx
    ])


def combine(query_preds, neighbour_idx, oof_preds, targets, global_weights, shrink):
    """Weighted combination p(x) . w(x) for each query, without the intercept.

    Parameters
    ----------
    query_preds : ndarray of shape (n_queries, M)
        Experts' predictions at the query points, p(x).
    neighbour_idx, oof_preds, targets, global_weights, shrink
        As in `local_weight_matrix`.

    Returns
    -------
    ndarray of shape (n_queries,)
    """
    if shrink == GLOBAL:
        # A single matrix product: no need to build one weight row per query.
        return query_preds @ global_weights
    weights = local_weight_matrix(neighbour_idx, oof_preds, targets, global_weights, shrink)
    return (query_preds * weights).sum(axis=1)


# ------------------------------------------------------ choice of lambda

def held_out_squared_errors(oof_preds, y, train_neighbours, shrink_grid, selection_sample, rng):
    """Squared error of every candidate on training points it did not see.

    Parameters
    ----------
    oof_preds : ndarray of shape (n_train, M)
        OOF predictions of the experts on the training set.
    y : ndarray of shape (n_train,)
        Training targets.
    train_neighbours : ndarray of shape (n_train, k)
        Neighbours of every training point among the OTHER training points.
    shrink_grid : sequence of float
        Local candidates for lambda.
    selection_sample : int
        Maximum number of training points the errors are measured on.
    rng : numpy.random.Generator
        Source of randomness for the sample and the cross-fitting folds.

    Returns
    -------
    dict
        Maps each candidate (the values of `shrink_grid`, then GLOBAL) to an array of
        per-point squared errors, aligned across candidates (same points, same order),
        so that differences between candidates are paired.
    """
    sample = rng.choice(len(oof_preds), size=min(selection_sample, len(oof_preds)), replace=False)
    # position[i] = where training point i sits in the sample, or -1 if not sampled.
    position = np.full(len(oof_preds), -1)
    position[sample] = np.arange(len(sample))
    errors = {shrink: np.empty(len(sample)) for shrink in list(shrink_grid) + [GLOBAL]}

    # Cross-fit the global stack so that it is judged on points it did not see,
    # exactly like the local variants, whose neighbours never include the point itself.
    splitter = KFold(SELECTION_FOLDS, shuffle=True, random_state=int(rng.integers(1 << 30)))
    for fit_idx, held_idx in splitter.split(oof_preds):
        held = held_idx[position[held_idx] >= 0]
        stack = LinearRegression(positive=True).fit(oof_preds[fit_idx], y[fit_idx])
        targets = y - stack.intercept_
        for shrink in errors:
            pred = combine(oof_preds[held], train_neighbours[held], oof_preds, targets,
                           stack.coef_, shrink)
            errors[shrink][position[held]] = (targets[held] - pred) ** 2
    return errors


def choose_shrink(errors, shrink_grid, rule):
    """Pick lambda from the held-out errors.

    Parameters
    ----------
    errors : dict
        Output of `held_out_squared_errors`.
    shrink_grid : sequence of float
        Local candidates (the GLOBAL key of `errors` is the fallback).
    rule : {"strict", "simple"}
        "simple": lowest mean squared error among all candidates, global included.
        "strict": best local lambda only if its mean gain over global exceeds one
        standard error of the paired gains, otherwise GLOBAL.

    Returns
    -------
    float
        The chosen lambda, GLOBAL (infinity) when the model stays global.
    """
    if rule == "simple":
        return min(errors, key=lambda shrink: errors[shrink].mean())
    best_local = min(shrink_grid, key=lambda shrink: errors[shrink].mean())
    gain = errors[GLOBAL] - errors[best_local]
    standard_error = gain.std(ddof=1) / np.sqrt(len(gain))
    return best_local if gain.mean() > standard_error else GLOBAL


# ------------------------------------------------------------ estimator

class LocalStackingRegressor(RegressorMixin, BaseEstimator):
    """Stacking whose weights adapt to the neighbourhood of each query point.

    Parameters
    ----------
    estimators : list of (str, estimator)
        The experts. Each is cloned, so the objects passed here stay unfitted.
        Give them a fixed random_state if you need reproducible results.
    n_neighbors : int, default=50
        Number of neighbours k used to fit the local weights. Capped at
        n_train - 1.
    shrink_grid : tuple of float, default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        Local candidates for the pull strength lambda. "Global" (infinity) is
        always added as a further candidate.
    rule : {"strict", "simple"}, default="strict"
        How lambda is chosen; see `choose_shrink`.
    cv : int, default=5
        Number of folds for the experts' out-of-fold predictions.
    selection_sample : int, default=1000
        Maximum number of training points used to choose lambda.
    random_state : int or None, default=None
        Seeds the OOF folds, the selection sample and the cross-fitting folds.
        It does not seed the experts; set their own random_state.

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted on the training features. Experts and neighbour search see
        standardised features.
    estimators_ : list of estimators
        The experts refitted on all training data, in the order of `estimators`.
    oof_predictions_ : ndarray of shape (n_train, M)
        Out-of-fold predictions of the experts on the training set (Z).
    y_train_ : ndarray of shape (n_train,)
        Training targets, needed at prediction time to fit the local weights.
    global_weights_ : ndarray of shape (M,)
        Global stacking weights w_g (non-negative).
    intercept_ : float
        Global stacking intercept b, shared by global and local predictions.
    n_neighbors_ : int
        Number of neighbours actually used.
    neighbors_ : NearestNeighbors
        Index of the standardised training points.
    selection_errors_ : dict
        Candidate lambda (the grid values, then np.inf) -> per-point held-out squared
        errors used for the choice. Same points in every array, so they can be
        compared pairwise.
    shrink_ : float
        The chosen lambda; np.inf when the model stays global.
    is_local_ : bool
        True when the rule chose a finite lambda.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, estimators, n_neighbors=50, shrink_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                 rule="strict", cv=5, selection_sample=1000, random_state=None):
        self.estimators = estimators
        self.n_neighbors = n_neighbors
        self.shrink_grid = shrink_grid
        self.rule = rule
        self.cv = cv
        self.selection_sample = selection_sample
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the experts, the global stack, the neighbour index and choose lambda.

        Parameters
        ----------
        X : array-like of shape (n_train, n_features)
            Numeric training features.
        y : array-like of shape (n_train,)
            Training targets.

        Returns
        -------
        self
        """
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        X, y = validate_data(self, X, y, y_numeric=True)
        y = np.asarray(y, dtype=float)
        # One generator for the whole fit, so a fixed random_state gives a fixed model.
        rng = np.random.default_rng(self.random_state)

        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)
        self.oof_predictions_ = self._out_of_fold_predictions(X_scaled, y)
        self.estimators_ = [clone(est).fit(X_scaled, y) for _, est in self.estimators]
        self.y_train_ = y

        stack = LinearRegression(positive=True).fit(self.oof_predictions_, y)
        self.global_weights_ = stack.coef_
        self.intercept_ = float(stack.intercept_)

        # A training point must not be its own neighbour, hence at most n - 1.
        self.n_neighbors_ = min(self.n_neighbors, len(X_scaled) - 1)
        self.neighbors_ = NearestNeighbors(n_neighbors=self.n_neighbors_).fit(X_scaled)
        # Called without X, kneighbors excludes each training point from its own list.
        train_neighbours = self.neighbors_.kneighbors(return_distance=False)

        self.selection_errors_ = held_out_squared_errors(
            self.oof_predictions_, y, train_neighbours, self.shrink_grid,
            self.selection_sample, rng)
        self.shrink_ = choose_shrink(self.selection_errors_, self.shrink_grid, self.rule)
        self.is_local_ = bool(self.shrink_ != GLOBAL)
        return self

    def _out_of_fold_predictions(self, X_scaled, y):
        """OOF predictions of every expert: each column j, row i comes from expert j
        trained on the folds that do not contain point i."""
        oof = np.zeros((len(X_scaled), len(self.estimators)))
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        for fit_idx, held_idx in folds.split(X_scaled):
            for j, (_, est) in enumerate(self.estimators):
                fitted = clone(est).fit(X_scaled[fit_idx], y[fit_idx])
                oof[held_idx, j] = fitted.predict(X_scaled[held_idx])
        return oof

    def predict_experts(self, X):
        """Predictions of every refitted expert.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, M)
            Column j holds the predictions of expert j, p(x).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_scaled = self.scaler_.transform(X)
        return np.column_stack([est.predict(X_scaled) for est in self.estimators_])

    def _neighbours_of(self, X):
        """Indices of the training neighbours of each query (queries are new points)."""
        X = validate_data(self, X, reset=False)
        return self.neighbors_.kneighbors(self.scaler_.transform(X), return_distance=False)

    def predict(self, X, shrink=None):
        """Predict with local (or global) stacking weights.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New points. Do not pass training points: they would be their own
            neighbours.
        shrink : float or None, default=None
            Pull strength to use. None uses the chosen `shrink_`. Pass np.inf for
            plain global stacking or a grid value to force a local variant, e.g. to
            compare variants without refitting.

        Returns
        -------
        ndarray of shape (n_queries,)
        """
        check_is_fitted(self)
        shrink = self.shrink_ if shrink is None else shrink
        query_preds = self.predict_experts(X)
        neighbour_idx = None if shrink == GLOBAL else self._neighbours_of(X)
        return self.intercept_ + combine(query_preds, neighbour_idx, self.oof_predictions_,
                                         self.y_train_ - self.intercept_,
                                         self.global_weights_, shrink)

    def local_weights(self, X, shrink=None):
        """The weights w(x) given to each expert at each query point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
        shrink : float or None, default=None
            As in `predict`.

        Returns
        -------
        ndarray of shape (n_queries, M)
            Non-negative weights, one row per query, columns in the order of
            `estimators`. Every row equals `global_weights_` when shrink is np.inf.
        """
        check_is_fitted(self)
        shrink = self.shrink_ if shrink is None else shrink
        neighbour_idx = self._neighbours_of(X)
        return local_weight_matrix(neighbour_idx, self.oof_predictions_,
                                   self.y_train_ - self.intercept_,
                                   self.global_weights_, shrink)
