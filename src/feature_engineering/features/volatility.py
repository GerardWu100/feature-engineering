"""Volatility features for stock OHLCV data.

Volatility features measure the size and instability of price movement. They do
not try to predict direction; they describe how noisy or risky the price path is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.features.registry import as_feature_column, register


@register(
    category="volatility",
    lookback=lambda params: params["window"],
    description="Rolling standard deviation of log returns.",
    calculation="std(ln(close_t / close_{t-1})) over trailing window",
)
def rolling_std(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute rolling standard deviation of log returns.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column.
    params
        Requires ``window``, the number of price rows in the rolling standard
        deviation context.

    Returns
    -------
    pandas.Series
        Rolling sample standard deviation aligned to ``df.index``. For
        ``window=3``, the third price can produce a value from the two returns
        formed by those three prices.
    """
    window = int(params["window"])
    if window < 2:
        raise ValueError("rolling_std requires window >= 2.")

    # The first log return is NaN because one price alone cannot form a return.
    # A window of N prices contains N - 1 adjacent returns. The rolling return
    # window therefore uses window - 1 values, not window values.
    log_returns = np.log(df["close"] / df["close"].shift(1))
    return_window = window - 1
    minimum_returns = max(2, return_window)
    values = log_returns.rolling(
        window=return_window,
        min_periods=minimum_returns,
    ).std()
    return as_feature_column(values)


@register(
    category="volatility",
    lookback=0,
    description="High-low bar range as a fraction of close.",
    calculation="(high_t - low_t) / close_t",
)
def bar_range_pct(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute intrabar high-low range as a percent of close.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with ``high``, ``low``, and ``close`` columns.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        Range percentages aligned to ``df.index``.
    """
    values = (df["high"] - df["low"]) / df["close"]
    return as_feature_column(values)
