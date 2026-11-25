"""Nested-CV pilot for a competence-aligned local-stacking metric."""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from run_stability_gating_experiments import load_uci_data, make_synthetic
from stability_gated_local_stacking import make_experts
from competence_metric_local_stacking import (
    CompetenceMetricLocalStack,
    select_shrinkage,
)


def datasets_for_run(include_uci=True):
    data = make_synthetic(seed=19, n=900)
    regional_X, regional_y = data["Synthetic regional specialization"]
    rng = np.random.default_rng(2201)
    noise = rng.normal(size=(len(regional_X), 25))
    data["Synthetic regional + 25 noise features"] = (
        np.column_stack([regional_X, noise]), regional_y,
    )
    hom_X, hom_y = data["Synthetic homogeneous competence"]
    noise = rng.normal(size=(len(hom_X), 25))
    data["Synthetic homogeneous + 25 noise features"] = (
        np.column_stack([hom_X, noise]), hom_y,
    )
    if include_uci:
        data.update(load_uci_data(appliances_rows=3000))
    return data


def evaluate(name, X, y, *, outer_folds=3, inner_folds=2, repeats=1,
             seed=411, metric_strength=0.5, permutations=12):
    rows = []
    for repeat in range(repeats):
      outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed + repeat * 1009)
      for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        fold_seed = seed + repeat * 1009 + fold * 113
        predictions = {}
        fitted = {}
        for variant, strength in (("euclidean_local", 0.0),
                                  ("competence_metric_local", metric_strength)):
            lambda_choice = select_shrinkage(
                X_train, y_train, estimators=make_experts(fold_seed),
                cv=inner_folds, seed=fold_seed + 7,
                metric_strength=strength, permutations=permutations,
            )
            model = CompetenceMetricLocalStack(
                estimators=make_experts(fold_seed), cv=inner_folds,
                metric_strength=strength, permutations=permutations,
                seed=fold_seed + 19,
            ).fit(X_train, y_train)
            global_pred, local_pred = model.predict_components(
                X_test, shrink=0.1 if np.isinf(lambda_choice) else lambda_choice,
            )
            pred = global_pred if np.isinf(lambda_choice) else local_pred
            predictions[variant] = pred
            fitted[variant] = (model, lambda_choice)
        for variant, pred in predictions.items():
            rows.append({
                "dataset": name,
                "repeat": repeat + 1,
                "fold": fold,
                "method": variant,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "shrink": float(fitted[variant][1]),
                "active_metric_features": int(
                    np.sum(fitted[variant][0].metric_evidence_ > 0)
                ),
                "top_metric_features": ",".join(map(str, np.argsort(
                    fitted[variant][0].metric_weights_
                )[-5:][::-1])),
            })
        rows.append({
            "dataset": name,
            "repeat": repeat + 1,
            "fold": fold,
            "method": "global_stacking",
            "rmse": float(np.sqrt(mean_squared_error(
                y_test, fitted["euclidean_local"][0].predict_components(
                    X_test, shrink=0.1
                )[0],
            ))),
            "shrink": np.inf,
            "active_metric_features": 0,
            "top_metric_features": "",
        })
        print(f"{name}: fold {fold}/{outer_folds} done", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-dataset")
    parser.add_argument("--skip-uci", action="store_true")
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=12)
    parser.add_argument("--metric-strength", type=float, default=0.5)
    args = parser.parse_args()

    datasets = datasets_for_run(include_uci=not args.skip_uci)
    if args.only_dataset:
        if args.only_dataset not in datasets:
            raise SystemExit(f"Unknown dataset {args.only_dataset!r}; available: {list(datasets)}")
        datasets = {args.only_dataset: datasets[args.only_dataset]}

    results = []
    for index, (name, (X, y)) in enumerate(datasets.items()):
        if args.only_dataset is None and name not in {
            "Synthetic regional + 25 noise features",
            "Synthetic homogeneous + 25 noise features",
            "UCI Bike Sharing (daily)",
            "UCI Appliances Energy (3000 sample)",
            "UCI Concrete Slump Test",
        }:
            continue
        print(f"Running {name}: n={len(y)}, p={X.shape[1]}", flush=True)
        results.append(evaluate(
            name, X, y, outer_folds=args.outer_folds,
            inner_folds=args.inner_folds, repeats=args.repeats,
            seed=411 + 1009 * index,
            permutations=args.permutations, metric_strength=args.metric_strength,
        ))
    folds = pd.concat(results, ignore_index=True)
    suffix = ""
    if args.only_dataset:
        slug = re.sub(r"[^a-z0-9]+", "_", args.only_dataset.lower()).strip("_")
        suffix = f"_{slug}"
    suffix += f"_strength_{args.metric_strength:g}_repeats_{args.repeats}"
    folds.to_csv(f"competence_metric_folds{suffix}.csv", index=False)
    summary = folds.groupby(["dataset", "method"], as_index=False).agg(
        mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"),
        mean_active_metric_features=("active_metric_features", "mean"),
    )
    summary.to_csv(f"competence_metric_summary{suffix}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
