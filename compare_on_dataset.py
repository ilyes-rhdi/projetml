"""Compare local stacking with simpler ways of combining the same experts.

Methods, all fitted inside the training fold only:
- best_single_expert : the expert with the lowest RMSE on the training OOF predictions
- simple_average     : plain mean of the experts' predictions
- global_stacking    : non-negative linear stacking (one set of weights everywhere)
- local_raw          : local stacking with the smallest lambda, no fallback
- local_strict       : local stacking with lambda chosen by the strict rule (the method)

Evaluation: 5-fold cross-validation repeated 3 times (15 folds, seeded).

Usage:
    python compare_on_dataset.py --demo
    python compare_on_dataset.py --csv data.csv --target price
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

from local_stacking import LocalStackingRegressor

METHODS = ["best_single_expert", "simple_average", "global_stacking", "local_raw", "local_strict"]
REFERENCE = "global_stacking"


def make_experts(seed):
    """The expert pool. Change it here to try other experts.

    Mirrors the "similaire" pool of the research code: three strong but different
    models (a linear one and two tree ensembles).
    """
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("hgb", HistGradientBoostingRegressor(random_state=seed)),
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    ]


# ------------------------------------------------------------------ data

def make_regional_data(n_samples=1000, seed=0):
    """Synthetic data where different experts are best in different regions.

    Five uniform features in [-1, 1]. The target has a steep linear trend everywhere.
    Where x0 < 0 that is all (plus noise): Ridge is by far the best expert there,
    because trees approximate a smooth slope with steps. Where x0 > 0 a checkerboard
    step and a wave are added: Ridge cannot follow them and the tree models win.
    The added part has mean zero and is uncorrelated with every feature, so Ridge's
    single linear fit stays accurate on the linear side. One global set of weights
    cannot be right on both sides; local weights can.

    Returns
    -------
    X : ndarray of shape (n_samples, 5)
    y : ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 5))
    linear = 4.0 * X[:, 1] + 3.0 * X[:, 2] + 2.0 * X[:, 3] - 3.0 * X[:, 4]
    nonlinear = 3.0 * np.sign(X[:, 2] * X[:, 3]) + 2.0 * np.cos(3.0 * np.pi * X[:, 3])
    y = linear + (X[:, 0] > 0) * nonlinear + rng.normal(scale=0.3, size=n_samples)
    return X, y


def load_csv(path, target):
    """Read a CSV, one-hot encode the non-numeric columns, drop rows with missing values.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    df = pd.read_csv(path).dropna()
    features = pd.get_dummies(df.drop(columns=target), dtype=float)
    return features.to_numpy(float), df[target].to_numpy(float)


# ------------------------------------------------------------ evaluation

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_fold(X_train, y_train, X_test, y_test, seed):
    """Fit everything on the training fold, return RMSE per method and the rule's choice."""
    model = LocalStackingRegressor(make_experts(seed), random_state=seed).fit(X_train, y_train)
    expert_preds = model.predict_experts(X_test)
    # The best single expert is chosen on training OOF predictions, never on the test fold.
    oof_rmse = np.sqrt(((y_train[:, None] - model.oof_predictions_) ** 2).mean(axis=0))

    preds = {
        "best_single_expert": expert_preds[:, oof_rmse.argmin()],
        "simple_average": expert_preds.mean(axis=1),
        "global_stacking": model.predict(X_test, shrink=np.inf),
        "local_raw": model.predict(X_test, shrink=min(model.shrink_grid)),
        "local_strict": model.predict(X_test),
    }
    return {name: rmse(y_test, p) for name, p in preds.items()}, model.is_local_


def cross_validate(X, y, seed=0):
    """Repeated 5-fold CV x 3. Returns a (15 x methods) RMSE table and the rule's choices."""
    rows, went_local = [], []
    splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(X)):
        scores, is_local = evaluate_fold(X[train], y[train], X[test], y[test], seed=fold)
        rows.append(scores)
        went_local.append(is_local)
        print(f"fold {fold + 1:2d}/15 done (strict rule went {'local' if is_local else 'global'})",
              flush=True)
    return pd.DataFrame(rows, columns=METHODS), went_local


def report(scores, went_local):
    """Print mean RMSE, gap to global stacking and a paired Wilcoxon test per method."""
    reference = scores[REFERENCE]
    print(f"\n{'method':20s} {'mean RMSE':>10s} {'vs stacking':>12s} {'wins/losses':>12s} {'Wilcoxon p':>11s}")
    for method in METHODS:
        diff = scores[method] - reference
        gap = (scores[method].mean() / reference.mean() - 1) * 100
        # Wilcoxon is undefined when every difference is zero (e.g. the reference itself).
        p_value = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        record = f"{int((diff < 0).sum())}/{int((diff > 0).sum())}"
        print(f"{method:20s} {scores[method].mean():10.4f} {gap:+11.2f}% {record:>12s} {p_value:11.3f}")
    print(f"\nThe strict rule went local in {sum(went_local)} of {len(went_local)} folds.")
    print("Negative gap = better than global stacking. wins/losses are counted over the 15 folds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use the built-in synthetic regional dataset")
    source.add_argument("--csv", help="path to a CSV file")
    parser.add_argument("--target", help="name of the target column (with --csv)")
    args = parser.parse_args()

    if args.demo:
        X, y = make_regional_data()
    elif args.target is None:
        parser.error("--csv needs --target")
    else:
        X, y = load_csv(args.csv, args.target)
    print(f"{X.shape[0]} rows, {X.shape[1]} features")
    report(*cross_validate(X, y))


if __name__ == "__main__":
    main()
