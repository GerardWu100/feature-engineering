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
   errors are serially correlated; symbols also move together, so pooled rows
   are not independent. `regression.py` runs the regression with
   Driscoll-Kraay standard errors: scores are summed within each timestamp
   (handling cross-symbol correlation), then a Newey-West kernel over
   timestamps handles serial correlation, with a lag rule that always covers
   the target's mechanical overlap. With one symbol this reduces exactly to
   classic Newey-West.
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
| `regression.py` | `newey_west_regression` (pooled OLS with Driscoll-Kraay standard errors) and the `default_hac_lags` rule (max of the size rule and horizon - 1). |
| `quantiles.py` | Target summary statistics and raw values per feature quantile bucket. |
| `summary.py` | `evaluate_features`: one row per feature, ranked by absolute t-statistic. |
| `plots.py` | `violin_by_quantile`, `spread_rows_by_state`, `rolling_ic_panels`; Okabe-Ito colour convention, figures returned, optional `save_path`. |

Start with `summary.py` to see the one-call workflow, then read `ic.py` and
`regression.py` for the statistics. Tests live in `tests/test_evaluation.py`
and `tests/test_evaluation_plots.py`.

## Part 3 - Short Journal

- 2026-08-10: Created the subpackage with IC, regression, quantile,
  summary-table, and plotting modules, alongside the new
  `next_n_bar_realized_vol` volatility target in `features/returns.py`.
- 2026-08-10: Audit-driven corrections in the same session: pooled Newey-West
  replaced with Driscoll-Kraay standard errors (cross-symbol dependence had
  inflated t-statistics), rolling Spearman now uses average-tie ranks and
  leaves tied/constant windows NaN, quantile bucketing excludes (with a
  warning) symbols whose ties collapse buckets, the tercile plot helper
  degrades instead of crashing on heavy ties, and infinities are masked before
  every statistic.
