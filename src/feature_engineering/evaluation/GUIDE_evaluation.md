# GUIDE - evaluation/

## Part 1 - Conceptual Explanation

`evaluation/` answers the question that follows feature computation: does a
feature contain evidence about its target? It runs after
`engineering/compute.py` and uses the feature frame produced by the pipeline:
one row per symbol and timestamp, with feature and target columns.

It provides three kinds of evidence, from simple association to more careful
checks:

1. Association. The information coefficient (IC) is the correlation between
   the feature now and the target later. `ic.py` computes it per symbol through
   time (time-series IC), per timestamp across symbols (cross-sectional IC,
   only meaningful for wide universes), and over trailing windows (rolling IC,
   the stability view).
2. Inference. Forward targets computed every bar overlap, so nearby errors are
   serially correlated. Symbols also move together, so pooled rows are not
   independent. `regression.py` runs the regression with Driscoll-Kraay
   standard errors: scores are summed within each timestamp (handling
   cross-symbol correlation), then a Newey-West kernel over timestamps handles
   serial correlation. Its lag rule always covers the target's mechanical
   overlap. With one symbol this reduces exactly to classic Newey-West.
3. Shape. One correlation can hide a relationship that appears only at the
   extremes. `quantiles.py` places the feature into per-symbol quantiles and
   summarizes the target in each bucket.

`summary.py` runs all three checks for many features and returns one ranked
table. `plots.py` shows the same evidence with a violin plot for each feature
quantile, a state chart with mean, middle-half, and 10th-to-90th ranges, and
stacked rolling-IC stability panels.

Statistical honesty rules baked in:

- Descriptive IC numbers do not claim significance; inference goes through the
  panel-robust regression only.
- The rolling Spearman recomputes ranks inside every window instead of rolling
  a Pearson correlation over full-sample ranks, because those are different
  statistics.
- The summary table warns that screening many features creates a multiple-testing
  problem: one t-statistic near 2 in 20 features is expected by chance.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `ic.py` | Time-series, cross-sectional, and rolling information coefficients plus `ic_summary`. |
| `regression.py` | `newey_west_regression` (pooled ordinary least squares with Driscoll-Kraay standard errors) and the `default_kernel_lags` rule (maximum of the size rule and horizon minus 1). |
| `quantiles.py` | Target summary statistics and raw values per feature quantile bucket. |
| `summary.py` | `evaluate_features`: one row per feature, ranked by absolute t-statistic. |
| `plots.py` | `violin_by_quantile`, `spread_rows_by_state`, `rolling_ic_panels`; Okabe-Ito colour convention, figures returned, optional `save_path`. |

Start with `summary.py` to see the one-call workflow, then read `ic.py` and
`regression.py` for the statistics. Tests live in `tests/test_evaluation.py`
and `tests/test_evaluation_plots.py`.

## Part 3 - Short Journal

- 2026-08-10: Created the subpackage with IC, regression, quantile,
  summary-table, and plotting modules, alongside the new
  `next_n_bar_realized_volatility` volatility target in `features/returns.py`.
- 2026-08-10: Audit-driven corrections in the same session: pooled Newey-West
  replaced with Driscoll-Kraay standard errors (cross-symbol dependence had
  inflated t-statistics), rolling Spearman now uses average-tie ranks and
  leaves tied/constant windows NaN, quantile bucketing excludes (with a
  warning) symbols whose ties collapse buckets, the tercile plot helper
  degrades instead of crashing on heavy ties, and infinities are masked before
  every statistic.
- 2026-08-10: Spelled out abbreviated result fields: `mean_ts_ic` -> `mean_time_series_ic`, `t_stat` -> `t_statistic`, `beta_se` -> `beta_standard_error`, `std_ic` -> `ic_standard_deviation`, `icir` -> `ic_information_ratio`, `hac_lags` -> `kernel_lags`, `q10`/`q90` -> `percentile_10`/`percentile_90`, `std` -> `standard_deviation`, and `n` -> `observations`.
