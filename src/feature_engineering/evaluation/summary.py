"""One-call feature evaluation: a table over many feature-target pairs.

This is the main entry point for research sessions. It runs the information
coefficient, panel-robust regression, and quantile-spread checks from the
sibling modules and returns one row per feature, sorted by absolute
t-statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.evaluation.ic import time_series_ic
from feature_engineering.evaluation.quantiles import target_by_feature_quantile
from feature_engineering.evaluation.regression import newey_west_regression

# Columns that are identifiers or targets, never candidate features.
NON_FEATURE_COLUMNS = {"symbol", "timestamp"}


def evaluate_features(
    frame: pd.DataFrame,
    target: str,
    *,
    features: list[str] | None = None,
    target_horizon_bars: int | None = None,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Evaluate every feature against one forward target.

    Parameters
    ----------
    frame
        Long feature frame from the pipeline: ``symbol``, ``timestamp``, feature
        columns, and at least one forward target column.
    target
        Target column name to evaluate against.
    features
        Feature column names to test. ``None`` (default) tests every numeric
        column except identifiers and the target itself.
    target_horizon_bars
        Forward horizon of the target in bars (e.g. 20 for a 20-bar target).
        Passed to the Newey-West lag rule so overlapping windows are covered.
        ``None`` falls back to the sample-size rule.
    quantiles
        Bucket count for the quantile spread diagnostic.

    Returns
    -------
    pandas.DataFrame
        One row per feature, sorted by ``abs(t_statistic)`` descending, columns:

        - ``feature``: feature name.
        - ``mean_time_series_ic``: time-series Spearman IC averaged across symbols.
          Descriptive strength of the rank relationship, in [-1, 1].
        - ``beta``: Newey-West regression slope, target units per one standard
          deviation of the feature.
        - ``t_statistic`` / ``p_value``: Newey-West inference on the slope. This is
          the significance column; the IC columns are descriptive.
        - ``r_squared``: in-sample explained variance (usually near zero for
          single features; that is normal).
        - ``quantile_spread``: mean target in the top feature bucket minus the
          bottom bucket. A monotonicity-free measure of economic size.
        - ``observations``: pooled rows used in the regression.

    Notes
    -----
    Screening many features is multiple testing: with 20 features, one
    t-statistic near 2 is expected by luck alone. Treat this table as a
    ranking device, not as proof, and confirm survivors out of sample.
    """
    if features is None:
        features = [
            column
            for column in frame.columns
            if column not in NON_FEATURE_COLUMNS
            and column != target
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    if not features:
        raise ValueError("No feature columns to evaluate.")

    rows: list[dict[str, object]] = []
    for feature in features:
        row: dict[str, object] = {"feature": feature}

        per_symbol_ic = time_series_ic(frame, feature, target)
        row["mean_time_series_ic"] = float(per_symbol_ic.mean())

        try:
            regression = newey_west_regression(
                frame,
                feature,
                target,
                target_horizon_bars=target_horizon_bars,
            )
            row.update(
                beta=regression.beta,
                t_statistic=regression.t_statistic,
                p_value=regression.p_value,
                r_squared=regression.r_squared,
                observations=regression.observations,
            )
        except ValueError:
            # Constant features (or too few rows) have no identified slope;
            # report them as untestable instead of failing the whole table.
            row.update(
                beta=np.nan,
                t_statistic=np.nan,
                p_value=np.nan,
                r_squared=np.nan,
                observations=0,
            )

        try:
            buckets = target_by_feature_quantile(
                frame, feature, target, quantiles=quantiles
            )
            top = buckets.loc[buckets["quantile"].idxmax(), "mean"]
            bottom = buckets.loc[buckets["quantile"].idxmin(), "mean"]
            row["quantile_spread"] = float(top - bottom)
        except ValueError:
            row["quantile_spread"] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)
    return table.reindex(
        table["t_statistic"]
        .abs()
        .sort_values(ascending=False, na_position="last")
        .index
    ).reset_index(drop=True)
