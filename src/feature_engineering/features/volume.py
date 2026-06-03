"""Volume and liquidity features for stock OHLCV data.

Volume features measure trading activity. They help distinguish quiet price
moves from moves supported by unusually high participation.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.features.registry import as_feature_column, register


@register(
    category="volume",
    lookback=lambda params: params["window"],
    description="Current volume divided by rolling average volume.",
    calculation="volume_t / mean(volume) over trailing window",
)
def volume_ratio(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute current volume relative to recent average volume.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``volume`` column.
    params
        Requires ``window``, the number of rows in the rolling average.

    Returns
    -------
    pandas.Series
        Relative volume aligned to ``df.index``.
    """
    window = int(params["window"])
    if window < 1:
        raise ValueError("volume_ratio requires window >= 1.")

    average_volume = df["volume"].rolling(window=window, min_periods=window).mean()
    values = df["volume"] / average_volume

    # If a valid input has zero average volume, avoid infinite values in model
    # input data by using NaN for undefined relative volume.
    values = values.replace([float("inf"), float("-inf")], float("nan"))
    return as_feature_column(values)


@register(
    category="volume",
    lookback=0,
    description="Dollar value traded in each bar.",
    calculation="close_t * volume_t",
)
def dollar_volume(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute dollar volume as close price times share volume.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with ``close`` and ``volume`` columns.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        Dollar volume aligned to ``df.index``.
    """
    return as_feature_column(df["close"] * df["volume"])


@register(
    category="volume",
    lookback=1,
    description="One-period percentage change in volume.",
    calculation="volume_t / volume_{t-1} - 1",
)
def volume_change(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute one-period percentage change in volume.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``volume`` column.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        Volume percentage change aligned to ``df.index``.
    """
    return as_feature_column(df["volume"].pct_change())
