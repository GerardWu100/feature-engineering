"""Feature evaluation: does a computed feature contain evidence about its target?

This subpackage sits after feature computation in the research workflow:

    load -> clean -> compute_features -> evaluate (this package) -> model

It answers three questions about each (feature, target) pair:

1. Association: information coefficient (IC), the correlation between the
   feature now and the target later (``ic.py``).
2. Inference: is the association statistically distinguishable from zero once
   overlapping forward windows and cross-symbol co-movement are accounted for?
   Pooled ordinary least squares with Driscoll-Kraay standard errors
   (``regression.py``).
3. Shape: how does the target behave across feature quantile buckets, beyond a
   single correlation number (``quantiles.py``).

``summary.py`` runs all three for many features at once and returns one tidy
table. ``plots.py`` draws the same evidence: distribution (violin), quantile
means, spread rows, and rolling IC stability.

All functions take the long feature frame produced by the pipeline: one row per
``(symbol, timestamp)`` with feature and target columns. They never mutate the
caller's frame.
"""

from feature_engineering.evaluation.ic import (
    cross_sectional_ic,
    ic_summary,
    rolling_ic,
    time_series_ic,
)
from feature_engineering.evaluation.plots import (
    rolling_ic_panels,
    spread_rows_by_state,
    violin_by_quantile,
)
from feature_engineering.evaluation.quantiles import target_by_feature_quantile
from feature_engineering.evaluation.regression import (
    RegressionResult,
    newey_west_regression,
)
from feature_engineering.evaluation.summary import evaluate_features

__all__ = [
    "RegressionResult",
    "cross_sectional_ic",
    "evaluate_features",
    "ic_summary",
    "newey_west_regression",
    "rolling_ic",
    "rolling_ic_panels",
    "spread_rows_by_state",
    "target_by_feature_quantile",
    "time_series_ic",
    "violin_by_quantile",
]
