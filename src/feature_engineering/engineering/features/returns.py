"""Return features for stock OHLCV data.

OHLCV means open, high, low, close, and volume bar data. Return features measure
backward-looking price change through time using close prices. Forward-looking
labels live in ``targets.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.engineering.features.registry import as_feature_column, register


@register(
    category="returns",
    lookback=1,
    description="Natural-log return from one close to the next.",
    calculation="ln(close_t / close_{t-1})",
)
def log_return(frame: pd.DataFrame) -> pd.Series:
    """Compute one-period log returns from close prices.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.

    Returns
    -------
    pandas.Series
        Log returns aligned to ``frame.index``. The first row is ``NaN`` because
        there is no previous close.
    """
    close = frame["close"]

    # Log return is additive through time, which makes it convenient for many
    # statistical models and for tracing multi-period returns.
    return as_feature_column(np.log(close / close.shift(1)))


@register(
    category="returns",
    lookback=1,
    description="Simple percentage return from one close to the next.",
    calculation="close_t / close_{t-1} - 1",
)
def simple_return(frame: pd.DataFrame) -> pd.Series:
    """Compute one-period simple returns from close prices.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.

    Returns
    -------
    pandas.Series
        Simple returns aligned to ``frame.index``. The first row is ``NaN``.
    """
    # pct_change expresses one-step simple return directly:
    # close_t / close_{t-1} - 1.
    return as_feature_column(frame["close"].pct_change())
