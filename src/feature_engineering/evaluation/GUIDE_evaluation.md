# GUIDE - evaluation/

## Part 1 - Conceptual Explanation

This folder asks the question that follows feature computation: does a feature
contain useful information about its target? It works on the feature frame
produced by the engineering stage: one row per symbol and timestamp, with
feature and target columns.

It provides three views of the evidence:

1. **Association.** The information coefficient (IC) is the correlation between
   a feature now and a target later. The package calculates it through time for
   each symbol, across symbols at each timestamp, and in trailing windows to
   show stability.
2. **Inference.** Targets calculated at every bar share future bars, so nearby
   observations are related. Symbols can also move together. The regression
   accounts for both problems: Driscoll-Kraay standard errors handle dependence
   across symbols, while a Newey-West time-series adjustment handles dependence
   across timestamps. Its lag rule always covers the target’s built-in overlap.
   With one symbol, this is exactly the classic Newey-West adjustment.
3. **Shape.** A single correlation can miss a relationship that appears only at
   high or low feature values. Quantile analysis sorts each symbol’s feature
   values into buckets and summarizes the target in each bucket.

The summary step runs these checks for many features and returns one ranked
table. The plotting step shows the same evidence with quantile violin plots,
state charts showing the mean and ranges, and rolling-IC stability panels.

The package also enforces a few safeguards:

- IC values are descriptive. Claims about statistical significance should use
  the panel-robust regression.
- Rolling Spearman correlation recalculates ranks inside each window. Rolling a
  Pearson correlation over ranks computed from the full sample is a different
  statistic.
- Screening many features creates a multiple-testing problem: a t-statistic
  near 2 in one of 20 features can occur by chance. The summary table warns
  about this.
- A frame may contain more than one forward target. Automatic feature selection
  cannot identify targets from the frame alone, so tell it which columns are
  targets. Otherwise another target can be scored as a feature and rank first
  simply because it overlaps the target being tested.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `ic.py` | Time-series, cross-sectional, and rolling information coefficients, plus `ic_summary`. |
| `regression.py` | `newey_west_regression`, a pooled ordinary least squares regression with Driscoll-Kraay standard errors, and `default_kernel_lags`, the lag rule using the larger of the size-based rule and `horizon - 1`. |
| `quantiles.py` | Target summaries and raw values for each feature quantile bucket. |
| `summary.py` | `evaluate_features`: one row per feature, ranked by absolute t-statistic. `target_columns` keeps other forward targets out of the candidate set. |
| `plots.py` | `violin_by_quantile`, `spread_rows_by_state`, and `rolling_ic_panels`. Uses the Okabe-Ito colour convention, returns figures, and accepts an optional `save_path`. |

Start with `summary.py` for the one-call workflow. Then read `ic.py` and
`regression.py` for the statistics. Tests are in `tests/test_evaluation.py`
and `tests/test_evaluation_plots.py`.

## Part 3 - Short Journal

- 2026-08-10: Created the subpackage with IC, regression, quantile, summary-table, and plotting modules, alongside the new `next_n_bar_realized_volatility` volatility target in `engineering/features/targets.py`.
- 2026-08-10: Audit-driven corrections in the same session: pooled Newey-West was replaced with Driscoll-Kraay standard errors because cross-symbol dependence had inflated t-statistics; rolling Spearman now uses average-tie ranks and leaves tied or constant windows as NaN; quantile bucketing excludes symbols whose ties collapse buckets and issues a warning; the tercile plot helper degrades instead of crashing on heavy ties; and infinities are masked before every statistic.
- 2026-08-10: Spelled out abbreviated result fields: `mean_ts_ic` -> `mean_time_series_ic`, `t_stat` -> `t_statistic`, `beta_se` -> `beta_standard_error`, `std_ic` -> `ic_standard_deviation`, `icir` -> `ic_information_ratio`, `hac_lags` -> `kernel_lags`, `q10`/`q90` -> `percentile_10`/`percentile_90`, `std` -> `standard_deviation`, and `n` -> `observations`.
- 2026-08-19: Regression now rejects invalid lag and target-horizon inputs; quantile plots reject empty finite samples and consistently exclude infinities.
- 2026-08-24: `evaluate_features` gained `target_columns` so automatic feature selection cannot pick up another forward target. The “all rows” legend entry in `violin_by_quantile` and `spread_rows_by_state` now follows the `percent` flag instead of always printing a percent.
