"""Trend features for stock OHLCV data.

Trend features measure direction or persistence. They answer questions such as:
is price above its recent average, and has price moved up over a lookback?
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.features.registry import as_feature_column, register


@register(
    category="trend",
    lookback=lambda params: params["window"],
    description="Simple moving average of close prices over a rolling window.",
    calculation="mean(close) over trailing window",
)
def moving_average(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute a simple moving average of close prices.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with ``close`` values sorted by time.
    params
        Requires ``window``, the number of rows in the rolling average.

    Returns
    -------
    pandas.Series
        Moving-average values aligned to ``df.index``. The first ``window - 1``
        rows are ``NaN`` because a full window is required.
    """
    window = int(params["window"])
    if window < 1:
        raise ValueError("moving_average requires window >= 1.")

    # Row-count windows are easier to reason about in the simplified project.
    # The input is already sorted per symbol by the pipeline.
    values = df["close"].rolling(window=window, min_periods=window).mean()
    return as_feature_column(values)


@register(
    category="trend",
    lookback=lambda params: params.get("window", 20),
    description="Close price divided by its moving average minus one.",
    calculation="close_t / moving_average_t - 1",
)
def price_vs_sma(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute normalized distance between close and its moving average.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column.
    params
        Supports ``window`` as the rolling average length. Default is 20 rows.

    Returns
    -------
    pandas.Series
        Dimensionless distance from the moving average, aligned to ``df.index``.
    """
    window = int(params.get("window", 20))
    if window < 1:
        raise ValueError("price_vs_sma requires window >= 1.")

    average = df["close"].rolling(window=window, min_periods=window).mean()

    # Dividing by the average makes the feature comparable across symbols with
    # different price levels.
    values = df["close"] / average - 1.0
    return as_feature_column(values)


@register(
    category="trend",
    lookback=lambda params: params.get("periods", 20),
    description="Rate of change over a fixed number of rows.",
    calculation="close_t / close_{t-periods} - 1",
)
def rate_of_change(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute lagged percentage price change.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column.
    params
        Supports ``periods`` as the lag length. Default is 20 rows.

    Returns
    -------
    pandas.Series
        Percentage price change over ``periods`` rows, aligned to ``df.index``.
    """
    periods = int(params.get("periods", 20))
    if periods < 1:
        raise ValueError("rate_of_change requires periods >= 1.")

    return as_feature_column(df["close"].pct_change(periods=periods))
