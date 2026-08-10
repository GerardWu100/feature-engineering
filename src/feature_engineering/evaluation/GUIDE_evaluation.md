# GUIDE - evaluation/

## Part 1 - Conceptual Explanation

`evaluation/` answers the question that follows feature computation: does a
feature actually predict its target? It sits after `pipeline/engineer.py` in
the research workflow and consumes the long feature frame the pipeline
produces (one row per symbol and timestamp, feature and target columns).

Three kinds of evidence, in increasing strictness:

1. Association. The information coefficient (IC) is the correlation between
   the feature now and the target later. `ic.py` computes it per symbol
   through time (time-series IC), per timestamp across symbols
   (cross-sectional IC, only meaningful for wide universes), and over trailing
   windows (rolling IC, the stability view).
2. Inference. Forward targets computed every bar overlap, so consecutive
   errors are serially correlated and ordinary least squares standard errors
   are too small. `regression.py` runs the regression with Newey-West
   (heteroskedasticity- and autocorrelation-consistent) standard errors and a
   lag rule that always covers the target's mechanical overlap.
3. Shape. A single correlation can hide a relationship that lives only in the
   extremes. `quantiles.py` buckets the feature into per-symbol quantiles and
   summarizes the target inside each bucket.

`summary.py` runs all three for many features and returns one ranked table.
`plots.py` draws the same evidence: a violin per feature quantile, the
spread-row state chart (mean dot, middle-half bar, 10th-90th line per
low/neutral/high state — the grammar from the Sun Life assessment submission),
and stacked rolling-IC stability panels.

Statistical honesty rules baked in:

- Descriptive IC numbers never claim significance; inference goes through the
  Newey-West regression only.
- The rolling Spearman recomputes ranks inside every window instead of rolling
  a Pearson correlation over full-sample ranks, because those are different
  statistics.
- The summary table docstring warns that screening many features is multiple
  testing: one t-statistic near 2 in 20 features is expected by luck.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `ic.py` | Time-series, cross-sectional, and rolling information coefficients plus `ic_summary`. |
| `regression.py` | `newey_west_regression` and the `default_hac_lags` rule (max of the size rule and horizon - 1). |
| `quantiles.py` | Target summary statistics and raw values per feature quantile bucket. |
| `summary.py` | `evaluate_features`: one row per feature, ranked by absolute t-statistic. |
| `plots.py` | `violin_by_quantile`, `spread_rows_by_state`, `rolling_ic_panels`; Okabe-Ito colour convention, figures returned, optional `save_path`. |

Start with `summary.py` to see the one-call workflow, then read `ic.py` and
`regression.py` for the statistics. Tests live in `tests/test_evaluation.py`
and `tests/test_evaluation_plots.py`.

## Part 3 - Short Journal

- 2026-08-10: Created the subpackage with IC, Newey-West regression, quantile,
  summary-table, and plotting modules, alongside the new
  `next_n_bar_realized_vol` volatility target in `features/returns.py`.
