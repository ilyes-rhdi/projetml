"""Nested comparison of stability-gated local stacking and ensemble baselines.

Downloads numeric UCI datasets from the UCI repository into the OS temp cache.
The outer repeated 5-fold CV estimates performance; all gate/shrinkage choices
are made in a separate, fold-isolated inner CV. Fold p-values are intentionally
not reported because repeated CV folds are dependent.
"""

from __future__ import annotations

import argparse
import io
import re
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from feature_gated_stacking import fit_feature_gate, select_gate_mix
from local_stacking import LocalStackingRegressor
from stability_gated_local_stacking import (
    fit_local_stack,
    make_experts as make_local_experts,
    select_stability_rule,
)


UCI_URLS = {
    "ccpp": "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    "superconductivity": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "student": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "bike": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "appliances": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
    "slump": "https://archive.ics.uci.edu/static/public/182/concrete+slump+test.zip",
}


def _download_cached(name, url):
    cache = Path(tempfile.gettempdir()) / "feature_gating_uci_cache"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{name}.zip"
    if destination.exists() and destination.stat().st_size:
        return destination.read_bytes()
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "research-experiment/1.0"})
            data = urllib.request.urlopen(request, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)):
                pass
            destination.write_bytes(data)
            return data
        except Exception as exc:  # retry transient UCI archive timeouts
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download UCI {name}: {last_error}")


def _xlsx_to_frame(xlsx_bytes):
    """Read a simple numeric XLSX worksheet using Python's standard library."""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for item in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.findall(".//m:t", ns)))
        sheet_name = next(n for n in archive.namelist()
                          if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        root = ElementTree.fromstring(archive.read(sheet_name))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def column_number(reference):
        letters = re.match(r"[A-Z]+", reference).group(0)
        number = 0
        for letter in letters:
            number = number * 26 + ord(letter) - ord("A") + 1
        return number - 1

    rows = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        values = {}
        for cell in row.findall("m:c", ns):
            index = column_number(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
            else:
                raw = cell.find("m:v", ns)
                value = raw.text if raw is not None else ""
                if cell_type == "s" and value:
                    value = shared[int(value)]
            values[index] = value
        if values:
            rows.append([values.get(i, "") for i in range(max(values) + 1)])
    headers = rows[0]
    frame = pd.DataFrame(rows[1:], columns=headers)
    return frame.apply(pd.to_numeric, errors="raise")


def load_uci_data(include_superconductivity=False, appliances_rows=6000):
    datasets = {}

    ccpp_archive = zipfile.ZipFile(io.BytesIO(_download_cached("ccpp", UCI_URLS["ccpp"])))
    xlsx_path = next(n for n in ccpp_archive.namelist() if n.lower().endswith(".xlsx"))
    ccpp = _xlsx_to_frame(ccpp_archive.read(xlsx_path))
    datasets["UCI CCPP"] = (ccpp.drop(columns="PE").to_numpy(float), ccpp["PE"].to_numpy(float))

    student_outer = zipfile.ZipFile(io.BytesIO(_download_cached("student", UCI_URLS["student"])))
    nested_bytes = next(student_outer.read(n) for n in student_outer.namelist()
                        if n.lower().endswith("student.zip"))
    student_inner = zipfile.ZipFile(io.BytesIO(nested_bytes))
    student_path = next(n for n in student_inner.namelist() if n.lower().endswith("student-mat.csv"))
    student = pd.read_csv(student_inner.open(student_path), sep=";")
    y_student = student.pop("G3").to_numpy(float)
    X_student = pd.get_dummies(student, dtype=float).to_numpy(float)
    datasets["UCI Student Performance (Math)"] = (X_student, y_student)

    bike_archive = zipfile.ZipFile(io.BytesIO(_download_cached("bike", UCI_URLS["bike"])))
    bike_path = next(n for n in bike_archive.namelist() if n.lower().endswith("day.csv"))
    bike = pd.read_csv(bike_archive.open(bike_path))
    y_bike = bike["cnt"].to_numpy(float)
    # casual + registered sum to cnt, so both are deliberately excluded.
    X_bike_frame = bike.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    X_bike = pd.get_dummies(X_bike_frame, columns=[
        c for c in ("season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit")
        if c in X_bike_frame.columns
    ], dtype=float).to_numpy(float)
    datasets["UCI Bike Sharing (daily)"] = (X_bike, y_bike)

    appliances_archive = zipfile.ZipFile(io.BytesIO(
        _download_cached("appliances", UCI_URLS["appliances"])
    ))
    appliances_path = next(n for n in appliances_archive.namelist()
                           if n.lower().endswith("energydata_complete.csv"))
    appliances = pd.read_csv(appliances_archive.open(appliances_path))
    y_appliances = appliances.pop("Appliances").to_numpy(float)
    # Date is removed; rv1 and rv2 are deliberately retained as irrelevant-feature stressors.
    X_appliances = appliances.drop(columns=["date"]).to_numpy(float)
    if appliances_rows and len(y_appliances) > appliances_rows:
        rng = np.random.default_rng(2027)
        selected = np.sort(rng.choice(len(y_appliances), size=appliances_rows, replace=False))
        X_appliances, y_appliances = X_appliances[selected], y_appliances[selected]
        appliances_name = f"UCI Appliances Energy ({appliances_rows} sample)"
    else:
        appliances_name = "UCI Appliances Energy"
    datasets[appliances_name] = (X_appliances, y_appliances)

    slump_archive = zipfile.ZipFile(io.BytesIO(_download_cached("slump", UCI_URLS["slump"])))
    slump_path = next(n for n in slump_archive.namelist()
                      if n.lower().endswith("slump_test.data"))
    slump = pd.read_csv(slump_archive.open(slump_path))
    y_slump = slump["SLUMP(cm)"].to_numpy(float)
    X_slump = slump.drop(columns=[
        "No", "SLUMP(cm)", "FLOW(cm)", "Compressive Strength (28-day)(Mpa)",
    ]).to_numpy(float)
    datasets["UCI Concrete Slump Test"] = (X_slump, y_slump)

    if include_superconductivity:
        super_archive = zipfile.ZipFile(io.BytesIO(
            _download_cached("superconductivity", UCI_URLS["superconductivity"])
        ))
        superconductivity = pd.read_csv(super_archive.open("train.csv"))
        target = "critical_temp" if "critical_temp" in superconductivity.columns else superconductivity.columns[-1]
        X_super = superconductivity.drop(columns=target).to_numpy(float)
        y_super = superconductivity[target].to_numpy(float)
        rng = np.random.default_rng(2026)
        if len(y_super) > 8000:
            selected = np.sort(rng.choice(len(y_super), size=8000, replace=False))
            X_super, y_super = X_super[selected], y_super[selected]
        datasets["UCI Superconductivity (8k sample)"] = (X_super, y_super)
    return datasets


def make_synthetic(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 5))
    linear = 4.0 * X[:, 1] + 3.0 * X[:, 2] + 2.0 * X[:, 3] - 3.0 * X[:, 4]
    nonlinear = 3.0 * np.sign(X[:, 2] * X[:, 3]) + 2.0 * np.cos(3.0 * np.pi * X[:, 3])
    y_regional = linear + (X[:, 0] > 0) * nonlinear + rng.normal(scale=0.3, size=n)
    X_hom = rng.normal(size=(n, 5))
    y_homogeneous = (
        X_hom @ np.array([1.5, -2.0, 0.5, 1.0, 0.0])
        + rng.normal(scale=0.5, size=n)
    )
    return {
        "Synthetic regional specialization": (X, y_regional),
        "Synthetic homogeneous competence": (X_hom, y_homogeneous),
    }


def evaluate_dataset(name, X, y, outer_folds=5, repeats=2, seed=2026,
                     inner_folds=3, bootstrap_inner=6, bootstrap_outer=12,
                     selection_sample=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    fold_rows = []
    for repeat in range(repeats):
        splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
            fold_seed = seed + repeat * 1000 + fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            stability_choice = select_stability_rule(
                X_train_scaled, y_train, seed=fold_seed + 11,
                inner_folds=inner_folds, oof_folds=3, bootstrap=bootstrap_inner,
                selection_sample=selection_sample,
            )
            stability_model = fit_local_stack(
                X_train_scaled, y_train, cv=3, seed=fold_seed + 17,
            )
            raw_local_lambda = stability_choice["shrink"] if stability_choice["selected_local"] else 0.1
            global_stab, local_stab, local_sd = stability_model.predict_components(
                X_test_scaled, shrink=raw_local_lambda, bootstrap=bootstrap_outer, seed=fold_seed + 19,
            )
            if stability_choice["selected_local"]:
                trust = np.exp(-stability_choice["strength"] * local_sd / stability_model.y_scale)
                selected_stab = global_stab + trust * (local_stab - global_stab)
            else:
                selected_stab = global_stab

            shared_experts = make_local_experts(fold_seed + 17)
            gate_choice = select_gate_mix(
                X_train_scaled, y_train, seed=fold_seed + 29,
                inner_folds=inner_folds, oof_folds=3, estimators=shared_experts,
            )
            gate_model = fit_feature_gate(
                X_train_scaled, y_train, estimators=shared_experts,
                cv=3, seed=fold_seed + 17,
            )
            global_gate, raw_gate, _, expert_predictions = gate_model.predict_components(
                X_test_scaled, beta=gate_choice["beta"],
            )
            selected_gate = (1.0 - gate_choice["gate_mix"]) * global_gate + \
                gate_choice["gate_mix"] * raw_gate
            if not gate_choice["selected_local"]:
                selected_gate = global_stab

            oof_rmse = gate_model.gate_cv_rmse
            local_reference = LocalStackingRegressor(
                estimators=make_local_experts(fold_seed + 17),
                n_neighbors=50, cv=3, selection_sample=min(300, len(y_train)),
                random_state=fold_seed + 37,
            ).fit(X_train, y_train)
            predictions = {
                "best_single_expert": expert_predictions[:, int(np.argmin(oof_rmse))],
                "simple_average": expert_predictions.mean(axis=1),
                "global_stacking": global_stab,
                "feature_error_gate_raw": raw_gate,
                "feature_error_gate_selected": selected_gate,
                "local_stacking_strict_reference": local_reference.predict(X_test),
                "stability_gated_local_selected": selected_stab,
                "stability_gated_local_raw": local_stab,
            }
            for method, pred in predictions.items():
                fold_rows.append({
                    "dataset": name,
                    "repeat": repeat + 1,
                    "fold": fold + 1,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "method": method,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "r2": float(r2_score(y_test, pred)),
                    "stability_gate_local_selected": stability_choice["selected_local"],
                    "stability_lambda": stability_choice["shrink"],
                    "stability_strength": stability_choice["strength"],
                    "stability_inner_gain": stability_choice["inner_gain"],
                    "stability_inner_gain_se": stability_choice["inner_gain_se"],
                    "feature_gate_local_selected": gate_choice["selected_local"],
                    "feature_gate_beta": gate_choice["beta"],
                    "feature_gate_mix": gate_choice["gate_mix"],
                    "reference_local_selected": local_reference.is_local_,
                    "reference_local_lambda": local_reference.shrink_,
                })
            print(f"{name}: repeat {repeat + 1}/{repeats}, fold {fold + 1}/{outer_folds} done",
                  flush=True)
    return pd.DataFrame(fold_rows)


def summarize(folds):
    rows = []
    for (dataset, method), group in folds.groupby(["dataset", "method"], sort=False):
        global_rmse = folds[(folds.dataset == dataset) &
                            (folds.method == "global_stacking")].set_index(["repeat", "fold"])["rmse"]
        aligned = group.set_index(["repeat", "fold"])["rmse"]
        differences = aligned - global_rmse
        rows.append({
            "dataset": dataset,
            "method": method,
            "mean_fold_rmse": group.rmse.mean(),
            "sd_fold_rmse": group.rmse.std(ddof=1),
            "mean_fold_mae": group.mae.mean(),
            "mean_fold_r2": group.r2.mean(),
            "wins_vs_global": int((differences < -1e-12).sum()),
            "ties_vs_global": int((np.abs(differences) <= 1e-12).sum()),
            "losses_vs_global": int((differences > 1e-12).sum()),
            "selected_local_folds": int(group.stability_gate_local_selected.sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-uci", action="store_true", help="only run synthetic controls and diabetes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--selection-sample", type=int, default=1000)
    parser.add_argument("--superconductivity-rows", type=int, default=8000)
    parser.add_argument("--appliances-rows", type=int, default=6000)
    parser.add_argument("--include-superconductivity", action="store_true",
                        help="download and test the large (21k x 81) UCI dataset")
    parser.add_argument("--only-dataset", help="evaluate one exact dataset name from the loaded suite")
    args = parser.parse_args()

    datasets = make_synthetic(seed=3, n=1000)
    diabetes = load_diabetes()
    datasets["scikit-learn Diabetes"] = (diabetes.data.astype(float), diabetes.target.astype(float))
    if not args.skip_uci:
        try:
            uci = load_uci_data(
                include_superconductivity=args.include_superconductivity,
                appliances_rows=args.appliances_rows,
            )
            for name, (X, y) in uci.items():
                if name.startswith("UCI Superconductivity") and len(y) > args.superconductivity_rows:
                    rng = np.random.default_rng(2026)
                    selected = np.sort(rng.choice(len(y), size=args.superconductivity_rows,
                                                  replace=False))
                    X, y = X[selected], y[selected]
                    name = f"UCI Superconductivity ({args.superconductivity_rows} sample)"
                datasets[name] = (X, y)
        except Exception as exc:
            print(f"UCI dataset loading stopped: {exc}", flush=True)
            raise

    if args.only_dataset:
        if args.only_dataset not in datasets:
            raise ValueError(f"Unknown dataset {args.only_dataset!r}; available: {list(datasets)}")
        datasets = {args.only_dataset: datasets[args.only_dataset]}

    all_folds = []
    for index, (name, (X, y)) in enumerate(datasets.items()):
        print(f"Starting {name}: {len(y)} rows, {X.shape[1]} features", flush=True)
        all_folds.append(evaluate_dataset(
            name, X, y, outer_folds=args.outer_folds, repeats=args.repeats,
            seed=2026 + index * 10_000, selection_sample=args.selection_sample,
        ))
    folds = pd.concat(all_folds, ignore_index=True)
    summary = summarize(folds)
    suffix = ""
    if args.only_dataset:
        suffix = "_" + re.sub(r"[^A-Za-z0-9]+", "_", args.only_dataset).strip("_").lower()
    folds.to_csv(f"feature_gating_folds{suffix}.csv", index=False)
    summary.to_csv(f"feature_gating_summary{suffix}.csv", index=False)
    report = [
        "# Stability-gated local stacking: experiment report",
        "",
        f"Nested repeated {args.outer_folds}-fold CV, {args.repeats} repeat(s); all selection is inside outer training folds.",
        f"Inner local/global tuning evaluates a fixed sample of at most {args.selection_sample} outer-training cases.",
        "The fold scores are descriptive; repeated folds are dependent, so this report gives no fold-level p-values.",
        "The stability gate bootstraps each query's neighbor set and exponentially reduces the local deviation according to the bootstrap prediction SD.",
        "A one-standard-error inner-CV rule may select the global stack exactly.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Dataset notes",
        "",
        "- UCI Bike Sharing uses the daily table; `casual` and `registered` are excluded because they sum to target `cnt`. Random CV tests interpolation, not forecasting.",
        "- UCI Student Performance uses the math table and one-hot encodes categorical input columns; `G3` is the target.",
        "- UCI Appliances Energy is randomly subsampled to the configured row limit; its two random noise features are retained as an irrelevant-feature stress test.",
        "- UCI Superconductivity is randomly subsampled to the configured row limit for compute budget.",
        "- Synthetic regional specialization is a positive control; homogeneous competence is a negative control.",
        "- The existing feature-error gate and the reference local-stacking estimator are included as comparisons, not claimed as new methods.",
    ]
    Path(f"feature_gating_report{suffix}.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"Saved feature_gating_folds{suffix}.csv, feature_gating_summary{suffix}.csv, "
          f"and feature_gating_report{suffix}.md", flush=True)


if __name__ == "__main__":
    main()
