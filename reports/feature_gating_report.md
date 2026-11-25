# Stability-gated local stacking: experiment report

Nested repeated 5-fold CV, 2 repeat(s); all selection is inside outer training folds.
The fold scores are descriptive; repeated folds are dependent, so this report gives no fold-level p-values.
The stability gate bootstraps each query's neighbor set and exponentially reduces the local deviation according to the bootstrap prediction SD.
A one-standard-error inner-CV rule may select the global stack exactly.

## Summary

| dataset                           | method                          |   mean_fold_rmse |   sd_fold_rmse |   mean_fold_mae |   mean_fold_r2 |   wins_vs_global |   ties_vs_global |   losses_vs_global |   selected_local_folds |
|:----------------------------------|:--------------------------------|-----------------:|---------------:|----------------:|---------------:|-----------------:|-----------------:|-------------------:|-----------------------:|
| Synthetic regional specialization | best_single_expert              |           1.2671 |         0.0905 |          0.9588 |         0.9131 |                2 |                0 |                  8 |                     10 |
| Synthetic regional specialization | simple_average                  |           1.6591 |         0.0970 |          1.2024 |         0.8513 |                0 |                0 |                 10 |                     10 |
| Synthetic regional specialization | global_stacking                 |           1.2485 |         0.0908 |          0.9401 |         0.9156 |                0 |               10 |                  0 |                     10 |
| Synthetic regional specialization | feature_error_gate_raw          |           1.2321 |         0.1178 |          0.8100 |         0.9176 |                6 |                0 |                  4 |                     10 |
| Synthetic regional specialization | feature_error_gate_selected     |           1.1941 |         0.1068 |          0.8453 |         0.9226 |               10 |                0 |                  0 |                     10 |
| Synthetic regional specialization | local_stacking_strict_reference |           1.1842 |         0.0980 |          0.8445 |         0.9239 |               10 |                0 |                  0 |                     10 |
| Synthetic regional specialization | stability_gated_local_selected  |           1.1785 |         0.0996 |          0.8399 |         0.9246 |               10 |                0 |                  0 |                     10 |
| Synthetic regional specialization | stability_gated_local_raw       |           1.1785 |         0.1005 |          0.8391 |         0.9246 |               10 |                0 |                  0 |                     10 |
| Synthetic homogeneous competence  | best_single_expert              |           0.5012 |         0.0186 |          0.4035 |         0.9702 |                4 |                0 |                  6 |                      2 |
| Synthetic homogeneous competence  | simple_average                  |           0.6114 |         0.0310 |          0.4856 |         0.9558 |                0 |                0 |                 10 |                      2 |
| Synthetic homogeneous competence  | global_stacking                 |           0.5015 |         0.0179 |          0.4037 |         0.9702 |                0 |               10 |                  0 |                      2 |
| Synthetic homogeneous competence  | feature_error_gate_raw          |           0.5565 |         0.0245 |          0.4456 |         0.9634 |                0 |                0 |                 10 |                      2 |
| Synthetic homogeneous competence  | feature_error_gate_selected     |           0.5015 |         0.0179 |          0.4037 |         0.9702 |                0 |               10 |                  0 |                      2 |
| Synthetic homogeneous competence  | local_stacking_strict_reference |           0.5015 |         0.0183 |          0.4037 |         0.9702 |                3 |                0 |                  7 |                      2 |
| Synthetic homogeneous competence  | stability_gated_local_selected  |           0.5023 |         0.0174 |          0.4043 |         0.9701 |                0 |                8 |                  2 |                      2 |
| Synthetic homogeneous competence  | stability_gated_local_raw       |           0.5051 |         0.0166 |          0.4057 |         0.9698 |                3 |                0 |                  7 |                      2 |
| scikit-learn Diabetes             | best_single_expert              |          55.3480 |         4.0195 |         44.8057 |         0.4754 |                5 |                0 |                  5 |                      3 |
| scikit-learn Diabetes             | simple_average                  |          56.4860 |         4.5450 |         45.6149 |         0.4539 |                2 |                0 |                  8 |                      3 |
| scikit-learn Diabetes             | global_stacking                 |          55.4470 |         4.0481 |         44.8535 |         0.4737 |                0 |               10 |                  0 |                      3 |
| scikit-learn Diabetes             | feature_error_gate_raw          |          56.7464 |         4.4663 |         45.8052 |         0.4492 |                2 |                0 |                  8 |                      3 |
| scikit-learn Diabetes             | feature_error_gate_selected     |          55.4470 |         4.0481 |         44.8535 |         0.4737 |                0 |               10 |                  0 |                      3 |
| scikit-learn Diabetes             | local_stacking_strict_reference |          55.5246 |         4.0922 |         45.0830 |         0.4719 |                6 |                0 |                  4 |                      3 |
| scikit-learn Diabetes             | stability_gated_local_selected  |          55.2943 |         4.3137 |         44.6613 |         0.4761 |                2 |                7 |                  1 |                      3 |
| scikit-learn Diabetes             | stability_gated_local_raw       |          55.2610 |         4.6200 |         44.4339 |         0.4760 |                6 |                0 |                  4 |                      3 |
| UCI CCPP                          | best_single_expert              |           3.3577 |         0.1586 |          2.3896 |         0.9612 |                4 |                0 |                  6 |                      3 |
| UCI CCPP                          | simple_average                  |           3.6020 |         0.1505 |          2.7157 |         0.9554 |                0 |                0 |                 10 |                      3 |
| UCI CCPP                          | global_stacking                 |           3.3580 |         0.1628 |          2.4208 |         0.9612 |                0 |               10 |                  0 |                      3 |
| UCI CCPP                          | feature_error_gate_raw          |           3.8714 |         0.1817 |          2.9514 |         0.9484 |                0 |                0 |                 10 |                      3 |
| UCI CCPP                          | feature_error_gate_selected     |           3.3580 |         0.1628 |          2.4208 |         0.9612 |                0 |               10 |                  0 |                      3 |
| UCI CCPP                          | local_stacking_strict_reference |           3.3582 |         0.1641 |          2.4208 |         0.9612 |                6 |                0 |                  4 |                      3 |
| UCI CCPP                          | stability_gated_local_selected  |           3.3588 |         0.1645 |          2.4176 |         0.9612 |                1 |                7 |                  2 |                      3 |
| UCI CCPP                          | stability_gated_local_raw       |           3.3561 |         0.1659 |          2.4139 |         0.9612 |                6 |                0 |                  4 |                      3 |
| UCI Student Performance (Math)    | best_single_expert              |           1.5961 |         0.3610 |          1.0383 |         0.8719 |                0 |                0 |                 10 |                      1 |
| UCI Student Performance (Math)    | simple_average                  |           1.5735 |         0.3573 |          1.0384 |         0.8753 |                2 |                0 |                  8 |                      1 |
| UCI Student Performance (Math)    | global_stacking                 |           1.5512 |         0.3546 |          1.0086 |         0.8786 |                0 |               10 |                  0 |                      1 |
| UCI Student Performance (Math)    | feature_error_gate_raw          |           1.5655 |         0.3691 |          1.0163 |         0.8769 |                4 |                0 |                  6 |                      1 |
| UCI Student Performance (Math)    | feature_error_gate_selected     |           1.5495 |         0.3539 |          1.0069 |         0.8791 |                3 |                4 |                  3 |                      1 |
| UCI Student Performance (Math)    | local_stacking_strict_reference |           1.5572 |         0.3657 |          1.0230 |         0.8772 |                5 |                0 |                  5 |                      1 |
| UCI Student Performance (Math)    | stability_gated_local_selected  |           1.5605 |         0.3417 |          1.0111 |         0.8774 |                0 |                9 |                  1 |                      1 |
| UCI Student Performance (Math)    | stability_gated_local_raw       |           1.5694 |         0.3362 |          1.0284 |         0.8761 |                2 |                0 |                  8 |                      1 |
| UCI Bike Sharing (daily)          | best_single_expert              |         655.2893 |        91.8893 |        463.9710 |         0.8826 |                3 |                0 |                  7 |                      9 |
| UCI Bike Sharing (daily)          | simple_average                  |         652.2131 |        81.4013 |        457.7095 |         0.8843 |                3 |                0 |                  7 |                      9 |
| UCI Bike Sharing (daily)          | global_stacking                 |         648.5161 |        87.4956 |        453.9030 |         0.8854 |                0 |               10 |                  0 |                      9 |
| UCI Bike Sharing (daily)          | feature_error_gate_raw          |         656.5191 |        78.8581 |        468.7388 |         0.8828 |                3 |                0 |                  7 |                      9 |
| UCI Bike Sharing (daily)          | feature_error_gate_selected     |         645.1656 |        82.0516 |        453.0505 |         0.8867 |                5 |                1 |                  4 |                      9 |
| UCI Bike Sharing (daily)          | local_stacking_strict_reference |         641.1833 |        83.3059 |        445.2296 |         0.8880 |                8 |                0 |                  2 |                      9 |
| UCI Bike Sharing (daily)          | stability_gated_local_selected  |         633.8166 |        86.0114 |        439.6270 |         0.8906 |                8 |                1 |                  1 |                      9 |
| UCI Bike Sharing (daily)          | stability_gated_local_raw       |         633.0537 |        87.2506 |        438.8746 |         0.8908 |                9 |                0 |                  1 |                      9 |

## Dataset notes

- UCI Bike Sharing uses the daily table; `casual` and `registered` are excluded because they sum to target `cnt`. Random CV tests interpolation, not forecasting.
- UCI Student Performance uses the math table and one-hot encodes categorical input columns; `G3` is the target.
- UCI Superconductivity is randomly subsampled to the configured row limit for compute budget.
- Synthetic regional specialization is a positive control; homogeneous competence is a negative control.
- The existing feature-error gate and the reference local-stacking estimator are included as comparisons, not claimed as new methods.