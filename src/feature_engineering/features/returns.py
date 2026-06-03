"""Return and target features for stock OHLCV data.

OHLCV means open, high, low, close, and volume bar data. Return features measure
price change through time. Target features are forward-looking labels for model
training, so they must not be used as live input signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.features.registry import as_feature_column, register


@register(
    category="returns",
    lookback=1,
    description="Natural-log return from one close to the next.",
    calculation="ln(close_t / close_{t-1})",
)
def log_return(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute one-period log returns from close prices.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        Log returns aligned to ``df.index``. The first row is ``NaN`` because
        there is no previous close.
    """
    close = df["close"]

    # Log return is additive through time, which makes it convenient for many
    # statistical models and for tracing multi-period returns.
    return as_feature_column(np.log(close / close.shift(1)))


@register(
    category="returns",
    lookback=1,
    description="Simple percentage return from one close to the next.",
    calculation="close_t / close_{t-1} - 1",
)
def simple_return(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute one-period simple returns from close prices.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.
    params
        Unused parameter dict, accepted for the shared feature signature.

    Returns
    -------
    pandas.Series
        Simple returns aligned to ``df.index``. The first row is ``NaN``.
    """
    # pct_change expresses one-step simple return directly:
    # close_t / close_{t-1} - 1.
    return as_feature_column(df["close"].pct_change())


@register(
    category="target",
    lookback=0,
    description="Forward N-trading-day return target from the current bar close.",
    calculation="EOD_close_{day+n} / close_t - 1",
)
def next_n_day_return(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute a forward day-level return target from each current bar.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with ``ts`` and ``close`` columns.
    params
        Supports ``days`` as the positive integer forecast horizon.

    Returns
    -------
    pandas.Series
        Forward return target aligned to ``df.index``. The numerator is the
        future day-level close, and the denominator is the current row's close.
        The final ``days`` sessions are ``NaN`` because the future close is
        unavailable.

    Raises
    ------
    ValueError
        If ``days`` is less than one.
    """
    days = int(params.get("days", 1))
    if days < 1:
        raise ValueError("next_n_day_return requires days >= 1.")

    # Normalize timestamps to calendar dates so each row can look up the close
    # of the future session without using the current session's end close as an
    # intraday denominator.
    session_date = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    day_close = df.groupby(session_date, sort=True)["close"].last()
    future_day_close = day_close.shift(-days)

    # Map each intraday row to its future session close, then divide by the
    # current row close to produce the forward return label.
    future_close = session_date.map(future_day_close)
    values = future_close / df["close"] - 1.0
    return as_feature_column(values)
