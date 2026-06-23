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
    description="Forward N-bar simple return target from the current bar close.",
    calculation="close_{t+bars} / close_t - 1",
)
def next_n_bar_return(df: pd.DataFrame, params: dict) -> pd.Series:
    """Compute a forward N-bar return target from each current bar.

    The horizon is measured in bars (rows), not calendar days. A bar is one row
    of the input frame: a daily bar on daily data, or a one-minute bar on
    one-minute data. The caller controls the bar size by choosing the source
    data and, for intraday runs, the ``reset_by_session`` option in
    ``compute_features`` (see ``pipeline/engineer.py``), which prevents the
    forward shift from crossing the overnight gap.

    Parameters
    ----------
    df
        Single-symbol OHLCV frame with a ``close`` column, sorted by time. The
        pipeline guarantees this ordering and per-symbol isolation, so the
        forward shift below never reaches into another ticker's rows.
    params
        Supports ``bars`` as the positive integer forecast horizon in rows.
        Default is 1.

    Returns
    -------
    pandas.Series
        Forward return target aligned to ``df.index``. The numerator is the
        close ``bars`` rows ahead and the denominator is the current row close.
        The final ``bars`` rows are ``NaN`` because the future close is
        unavailable.

    Raises
    ------
    ValueError
        If ``bars`` is less than one.
    """
    bars = int(params.get("bars", 1))
    if bars < 1:
        raise ValueError("next_n_bar_return requires bars >= 1.")

    close = df["close"]

    # Forward simple return over a fixed number of bars. shift(-bars) brings the
    # future close back to the current row; the last ``bars`` rows become NaN
    # because their future close does not exist in the frame.
    future_close = close.shift(-bars)
    values = future_close / close - 1.0
    return as_feature_column(values)
