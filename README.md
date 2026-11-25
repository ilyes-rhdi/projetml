# Local stacking with a data-chosen fallback

> **Projet de groupe :** le plan de travail, le tableau des variantes et la répartition des tâches sont dans [README_variantes.md](./README_variantes.md).

> **Read this first.** The course requires the group to implement the novel component
> itself. This folder is a reference to learn from and to check your own results
> against, not code to submit as is. Re-implement it, and make sure every member can
> explain every line.

## The method in five lines

1. Several regression models ("experts") are trained; their out-of-fold (OOF) predictions on the training set are kept.
2. Global stacking learns one set of non-negative weights for the experts from those OOF predictions.
3. Local stacking re-learns the weights for each query point from its k = 50 nearest training points, pulled toward the global weights by a ridge penalty of strength lambda.
4. Lambda is chosen on the training data among a small grid plus "global" (lambda = infinity, plain stacking).
5. The default "strict" rule goes local only when the gain over global is larger than its standard error; otherwise it falls back to plain stacking.

The full maths, the leakage safeguards and the cost are in the module docstring of
`local_stacking.py`.

## Files

| File | What it holds |
|---|---|
| `local_stacking.py` | The method: `LocalStackingRegressor`, a scikit-learn estimator, and the small functions it is built from. |
| `compare_on_dataset.py` | Compares the method with simpler baselines under repeated cross-validation, on a CSV or a built-in demo dataset. |
| `tests/test_local_stacking.py` | Fast pytest checks of the properties the method must have. |
| `requirements.txt` | Package versions the code was run with. |

## Install

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```

## Run

Demo, on synthetic data where different experts are best in different regions (about one minute):

```
python compare_on_dataset.py --demo
```

Your own CSV (numeric features; text columns are one-hot encoded, rows with missing values are dropped):

```
python compare_on_dataset.py --csv path/to/data.csv --target price
```

Tests (a few seconds), from this folder:

```
python -m pytest tests
```

The comparison prints, for each method, the mean RMSE over 15 folds (5-fold CV repeated
3 times), the gap in % to global stacking (negative = better), wins/losses over the
folds, a paired Wilcoxon p-value against global stacking, and in how many folds the
strict rule went local. The expert pool is defined in `make_experts` at the top of
`compare_on_dataset.py`.

## Using the estimator

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from local_stacking import LocalStackingRegressor

model = LocalStackingRegressor(
    estimators=[("ridge", Ridge()), ("rf", RandomForestRegressor(random_state=0))],
    random_state=0,
).fit(X_train, y_train)

model.predict(X_test)                    # the method (lambda chosen by the rule)
model.predict(X_test, shrink=np.inf)     # plain global stacking, same fitted experts
model.local_weights(X_test)              # weight of each expert at each test point
model.is_local_, model.shrink_           # what the rule decided
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `estimators` | required | List of `(name, estimator)`: the experts. They are cloned. Set their own `random_state` for reproducibility. |
| `n_neighbors` | `50` | k, the number of training neighbours used to fit the local weights (capped at n_train - 1). |
| `shrink_grid` | `(1e-4, 1e-3, 1e-2, 1e-1, 1.0)` | Local candidates for lambda. "Global" (infinity) is always added. Small lambda = more local. |
| `rule` | `"strict"` | `"strict"`: local only if the gain over global exceeds one standard error. `"simple"`: lowest held-out error wins. |
| `cv` | `5` | Folds for the experts' OOF predictions. |
| `selection_sample` | `1000` | Maximum number of training points used to choose lambda. |
| `random_state` | `None` | Seeds the OOF folds, the selection sample and the selection folds (not the experts). |

### Public methods

| Method | Returns |
|---|---|
| `fit(X, y)` | `self` |
| `predict(X, shrink=None)` | Predictions. `shrink=None` uses the chosen lambda; pass `np.inf` for global stacking or any lambda to force it. |
| `local_weights(X, shrink=None)` | Array `(n_queries, n_experts)` of non-negative weights, for figures. |
| `predict_experts(X)` | Array `(n_queries, n_experts)` of each refitted expert's predictions. |

### Learned attributes

| Attribute | Meaning |
|---|---|
| `global_weights_` | Global stacking weights w_g (non-negative). |
| `intercept_` | Global stacking intercept b (also used by the local predictions). |
| `shrink_` | Chosen lambda; `np.inf` when the model stays global. |
| `is_local_` | `True` when the rule chose a finite lambda. |
| `selection_errors_` | Dict: candidate lambda (grid values, then `np.inf`) -> per-point held-out squared errors. Same points in every array, so you can compare candidates pairwise. |
| `oof_predictions_` | OOF predictions of the experts on the training set (Z). |
| `estimators_` | Experts refitted on all training data. |
| `scaler_`, `neighbors_`, `n_neighbors_` | Feature scaler, neighbour index and the k actually used. |
| `y_train_`, `n_features_in_` | Training targets (needed at prediction time) and number of features. |

## Known limits

- **Cost.** Prediction solves one small non-negative least squares problem per query
  point when the model goes local; choosing lambda solves several thousand at fit time.
  Global stacking is a single matrix product.
- **k is fixed** at 50 and not tuned. Very small datasets give neighbourhoods that cover
  a large part of the data; very large ones may want a larger k.
- **Euclidean neighbours** on standardised features: every feature counts the same,
  including irrelevant ones and one-hot columns. Neighbourhoods degrade in high
  dimension.
- **Gains only on some datasets.** On data where one combination is right everywhere,
  the strict rule stays global and the method equals plain stacking (no gain, no loss).
  Gains appear only when different experts are better in different regions.
- **Do not predict on training points**: each would be among its own neighbours, so
  in-sample errors look better than they are.
