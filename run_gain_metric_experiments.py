"""Screening runner for the gain-oriented metric (V3).

Compares, on identical outer folds:

- global stacking;
- local stacking with Euclidean neighbourhoods;
- local stacking with the current MI/argmin competence metric;
- local stacking with the gain-oriented metric.

All metric, shrinkage and fallback choices are made inside the outer training
fold. The outer test fold is only used for the final RMSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from competence_metric_local_stacking import (
    CompetenceMetricLocalStack,
    select_shrinkage,
)
from gain_metric_local_stacking import learn_gain_metric
from run_competence_metric_experiments import datasets_for_run
from stability_gated_local_stacking import make_experts

SCREENING_DATASETS = [
    "Synthetic regional + 25 noise features",
    "Synthetic homogeneous + 25 noise features",
    "UCI CCPP",
    "UCI Bike Sharing (daily)",
]

CONFIRMATION_DATASETS = [
    "Synthetic regional + 25 noise features",
    "Synthetic homogeneous + 25 noise features",
    "UCI CCPP",
    "UCI Student Performance (Math)",
    "UCI Bike Sharing (daily)",
    "UCI Appliances Energy (3000 sample)",
    "UCI Concrete Slump Test",
]


def evaluate(name, X, y, *, outer_folds=5, inner_folds=2, repeats=1, seed=411,
             metric_strength=0.5, permutations=12, gain_kwargs=None,
             force_shrink=None):
    rows = []
    gain_kwargs = {} if gain_kwargs is None else dict(gain_kwargs)

    for repeat in range(repeats):
        outer = KFold(
            n_splits=outer_folds, shuffle=True,
            random_state=seed + repeat * 1009,
        )
        for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            fold_seed = seed + repeat * 1009 + fold * 113

            variants = [
                ("euclidean_local", 0.0, None, {}),
                ("current_mi", metric_strength, None, {}),
                ("gain_oriented", metric_strength, learn_gain_metric, gain_kwargs),
            ]
            fitted = {}
            for method, strength, learner, kwargs in variants:
                lambda_choice = (
                    force_shrink if force_shrink is not None
                    else select_shrinkage(
                        X_train, y_train,
                        estimators=make_experts(fold_seed),
                        cv=inner_folds, seed=fold_seed + 7,
                        metric_strength=strength, permutations=permutations,
                        metric_learner=learner, metric_kwargs=kwargs,
                    )
                )
                model = CompetenceMetricLocalStack(
                    estimators=make_experts(fold_seed), cv=inner_folds,
                    metric_strength=strength, permutations=permutations,
                    seed=fold_seed + 19,
                    metric_learner=learner, metric_kwargs=kwargs,
                ).fit(X_train, y_train)
                fitted[method] = (model, lambda_choice)

                if np.isinf(lambda_choice):
                    prediction = model.predict_global(X_test)
                else:
                    prediction = model.predict(X_test, shrink=lambda_choice)

                rows.append({
                    "dataset": name,
                    "repeat": repeat + 1,
                    "fold": fold,
                    "method": method,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                    "shrink": float(lambda_choice),
                    "active_metric_features": int(
                        np.sum(model.metric_evidence_ > 0)
                    ),
                    "top_metric_features": ",".join(map(str, np.argsort(
                        model.metric_weights_
                    )[-5:][::-1])),
                })

            global_prediction = fitted["euclidean_local"][0].predict_global(X_test)
            rows.append({
                "dataset": name,
                "repeat": repeat + 1,
                "fold": fold,
                "method": "global_stacking",
                "rmse": float(np.sqrt(mean_squared_error(y_test, global_prediction))),
                "shrink": np.inf,
                "active_metric_features": 0,
                "top_metric_features": "",
            })
            print(
                f"{name}: repeat {repeat + 1}, fold {fold}/{outer_folds} done",
                flush=True,
            )
    return pd.DataFrame(rows)


def comparison_table(folds):
    pivot = folds.pivot_table(
        index=["dataset", "repeat", "fold"],
        columns="method", values="rmse",
    )
    rows = []
    for dataset, group in pivot.groupby(level=0):
        row = {"dataset": dataset, "folds": len(group)}
        for candidate, reference in [
            ("current_mi", "global_stacking"),
            ("gain_oriented", "global_stacking"),
            ("gain_oriented", "current_mi"),
            ("gain_oriented", "euclidean_local"),
        ]:
            if candidate not in group or reference not in group:
                continue
            delta = group[candidate] - group[reference]
            row[f"{candidate}_vs_{reference}_wins"] = int((delta < -1e-12).sum())
            row[f"{candidate}_vs_{reference}_ties"] = int(
                np.isclose(delta, 0.0, atol=1e-12).sum()
            )
            row[f"{candidate}_vs_{reference}_losses"] = int((delta > 1e-12).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="*", default=SCREENING_DATASETS,
        help="Dataset names to run; defaults to the screening subset.",
    )
    parser.add_argument("--all", action="store_true", help="Run the confirmation subset.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--metric-strength", type=float, default=0.5)
    parser.add_argument("--permutations", type=int, default=12)
    parser.add_argument("--gain-splits", type=int, default=1)
    parser.add_argument("--gain-passes", type=int, default=2)
    parser.add_argument("--gain-max-rows", type=int, default=10000)
    parser.add_argument("--force-shrink", type=float, default=None,
                        help="Diagnostic mode: force one local lambda instead of selecting it.")
    parser.add_argument("--seed", type=int, default=411)
    args = parser.parse_args()

    wanted = CONFIRMATION_DATASETS if args.all else args.datasets
    all_datasets = datasets_for_run(include_uci=True)
    missing = [name for name in wanted if name not in all_datasets]
    if missing:
        raise SystemExit(f"Unknown datasets: {missing}. Available: {list(all_datasets)}")

    gain_kwargs = {
        "n_splits": args.gain_splits,
        "max_passes": args.gain_passes,
        "max_rows": args.gain_max_rows,
        "factors": (0.5, 2.0),
    }

    results = []
    for index, name in enumerate(wanted):
        X, y = all_datasets[name]
        print(f"Running {name}: n={len(y)}, p={X.shape[1]}", flush=True)
        results.append(evaluate(
            name, X, y,
            outer_folds=args.outer_folds, inner_folds=args.inner_folds,
            repeats=args.repeats, seed=args.seed + 1009 * index,
            metric_strength=args.metric_strength,
            permutations=args.permutations, gain_kwargs=gain_kwargs,
            force_shrink=args.force_shrink,
        ))

    folds = pd.concat(results, ignore_index=True)
    summary = folds.groupby(["dataset", "method"], as_index=False).agg(
        mean_rmse=("rmse", "mean"),
        sd_rmse=("rmse", "std"),
        mean_active_metric_features=("active_metric_features", "mean"),
    )
    comparison = comparison_table(folds)

    output_dir = Path("results/gain_metric")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"_strength_{args.metric_strength:g}"
        f"_repeats_{args.repeats}"
        f"_splits_{args.gain_splits}"
        f"_passes_{args.gain_passes}"
    )
    if args.force_shrink is not None:
        suffix += f"_forced_{args.force_shrink:g}"
    folds.to_csv(output_dir / f"gain_metric_folds{suffix}.csv", index=False)
    summary.to_csv(output_dir / f"gain_metric_summary{suffix}.csv", index=False)
    comparison.to_csv(output_dir / f"gain_metric_comparison{suffix}.csv", index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nComparisons")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
