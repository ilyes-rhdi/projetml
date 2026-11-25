# Stability-gated local stacking: experiment report

Nested repeated 5-fold CV, 1 repeat(s); all selection is inside outer training folds.
Inner local/global tuning evaluates a fixed sample of at most 500 outer-training cases.
The fold scores are descriptive; repeated folds are dependent, so this report gives no fold-level p-values.
The stability gate bootstraps each query's neighbor set and exponentially reduces the local deviation according to the bootstrap prediction SD.
A one-standard-error inner-CV rule may select the global stack exactly.

## Summary

| dataset                             | method                          |   mean_fold_rmse |   sd_fold_rmse |   mean_fold_mae |   mean_fold_r2 |   wins_vs_global |   ties_vs_global |   losses_vs_global |   selected_local_folds |
|:------------------------------------|:--------------------------------|-----------------:|---------------:|----------------:|---------------:|-----------------:|-----------------:|-------------------:|-----------------------:|
| UCI Appliances Energy (6000 sample) | best_single_expert              |          86.0202 |         4.2287 |         44.8664 |         0.3264 |                0 |                0 |                  5 |                      1 |
| UCI Appliances Energy (6000 sample) | simple_average                  |          86.0934 |         4.6341 |         46.1581 |         0.3254 |                0 |                0 |                  5 |                      1 |
| UCI Appliances Energy (6000 sample) | global_stacking                 |          84.6301 |         4.4600 |         44.2113 |         0.3480 |                0 |                5 |                  0 |                      1 |
| UCI Appliances Energy (6000 sample) | feature_error_gate_raw          |          86.7143 |         4.5616 |         41.5268 |         0.3155 |                0 |                0 |                  5 |                      1 |
| UCI Appliances Energy (6000 sample) | feature_error_gate_selected     |          84.6463 |         4.4553 |         43.9467 |         0.3477 |                0 |                4 |                  1 |                      1 |
| UCI Appliances Energy (6000 sample) | local_stacking_strict_reference |          84.6521 |         4.4774 |         44.1696 |         0.3477 |                1 |                0 |                  4 |                      1 |
| UCI Appliances Energy (6000 sample) | stability_gated_local_selected  |          84.9144 |         4.7786 |         44.4741 |         0.3438 |                0 |                4 |                  1 |                      1 |
| UCI Appliances Energy (6000 sample) | stability_gated_local_raw       |          86.1488 |         4.5765 |         46.2220 |         0.3245 |                0 |                0 |                  5 |                      1 |

## Dataset notes

- UCI Bike Sharing uses the daily table; `casual` and `registered` are excluded because they sum to target `cnt`. Random CV tests interpolation, not forecasting.
- UCI Student Performance uses the math table and one-hot encodes categorical input columns; `G3` is the target.
- UCI Appliances Energy is randomly subsampled to the configured row limit; its two random noise features are retained as an irrelevant-feature stress test.
- UCI Superconductivity is randomly subsampled to the configured row limit for compute budget.
- Synthetic regional specialization is a positive control; homogeneous competence is a negative control.
- The existing feature-error gate and the reference local-stacking estimator are included as comparisons, not claimed as new methods.