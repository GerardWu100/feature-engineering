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


def _vwap_series(df: pd.DataFrame) -> pd.Series:
    """Return the cumulative Volume Weighted Average Price within the frame.

    VWAP accumulates from the first row of ``df`` to each row:

        typical_price = (high + low + close) / 3
        vwap_t = sum(typical_price * volume)[0..t] / sum(volume)[0..t]

    Because features are computed per group, VWAP resets at each group boundary.
    With ``reset_by_session = true`` (see ``pipeline/engineer.py``) each group is
    one symbol-day, which is the standard intraday "session VWAP". Without a
    session reset on intraday data, VWAP accumulates across days, which is rarely
    what you want.

    Parameters
    ----------
    df
        Single-group OHLCV frame with ``high``, ``low``, ``close``, ``volume``.

    Returns
    -------
    pandas.Series
        VWAP in price units aligned to ``df.index``. ``NaN`` where cumulative
        volume is still zero.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_price_volume = (typical_price * df["volume"]).cumsum()
    cumulative_volume = df["volume"].cumsum()

    # Guard the zero-volume start so undefined VWAP is NaN, not infinity.
    vwap = cumulative_price_volume / cumulative_volume.replace(0.0, float("nan"))
    return vwap


@register(
    category="volume",
    lookback=0,
    description="Cumulative Volume Weighted Average Price within the group/session.",
    calculation="cumsum((h+l+c)/3 * volume) / cumsum(volume)",
)
def vwap(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute cumulative VWAP within each computation group.

    Parameters
    ----------
    df
        Single-group OHLCV frame with ``high``, ``low``, ``close``, ``volume``.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        VWAP in price units aligned to ``df.index``.
    """
    return as_feature_column(_vwap_series(df))


@register(
    category="volume",
    lookback=0,
    description="Close price distance from cumulative VWAP, as a fraction.",
    calculation="close_t / vwap_t - 1",
)
def price_vs_vwap(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute the close-to-VWAP distance as a dimensionless fraction.

    Dividing by VWAP makes this comparable across symbols and price levels,
    unlike the raw VWAP price level. Positive means trading above the session's
    volume-weighted average; negative means below.

    Parameters
    ----------
    df
        Single-group OHLCV frame with ``high``, ``low``, ``close``, ``volume``.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        ``close / vwap - 1`` aligned to ``df.index``.
    """
    vwap_values = _vwap_series(df)
    return as_feature_column(df["close"] / vwap_values - 1.0)
