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


# Default Average True Range smoothing length. Named for easy discovery and
# config override.
DEFAULT_ATR_WINDOW = 14


@register(
    category="volatility",
    lookback=lambda params: int(params.get("window", DEFAULT_ATR_WINDOW)),
    description="Wilder's Average True Range: smoothed bar-to-bar price range.",
    calculation="Wilder_avg(true_range, window), TR = max(h-l, |h-prev_c|, |l-prev_c|)",
)
def average_true_range(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute Wilder's Average True Range (ATR).

    True Range (TR) for a bar is the largest of three distances:

        TR = max(high - low, |high - prev_close|, |low - prev_close|)

    It captures gaps that a plain high-low range misses. ATR is Wilder's
    smoothed average of TR and is reported in price units (dollars), so it is
    not directly comparable across symbols with different price levels.

    The first bar has no previous close, so its TR is just ``high - low``. ATR
    becomes valid at the ``window``-th bar (index ``window - 1``).

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with ``high``, ``low``, and ``close`` columns.
    params
        Supports ``window`` (default 14), the Wilder smoothing length in bars.

    Returns
    -------
    pandas.Series
        ATR in price units aligned to ``df.index``.

    Raises
    ------
    ValueError
        If ``window`` is less than two.
    """
    window = int(params.get("window", DEFAULT_ATR_WINDOW))
    if window < 2:
        raise ValueError("average_true_range requires window >= 2.")

    high = df["high"]
    low = df["low"]
    previous_close = df["close"].shift(1)

    # Three candidate ranges. On the first bar previous_close is NaN, so the two
    # gap terms are NaN and the row-wise max falls back to (high - low).
    high_low_range = high - low
    high_close_gap = (high - previous_close).abs()
    low_close_gap = (low - previous_close).abs()
    true_range = pd.concat(
        [high_low_range, high_close_gap, low_close_gap], axis=1
    ).max(axis=1)

    # Wilder smoothing (EMA with alpha = 1 / window). true_range has no NaN, so
    # the recursion seeds on the first bar and matches the online accumulator.
    average_range = true_range.ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()
    return as_feature_column(average_range)
