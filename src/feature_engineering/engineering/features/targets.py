"""Target features: forward-looking labels for supervised learning.

A target answers "what happens after this bar?" so it is computed from future
rows. Targets must never be used as live input signals; the config excludes
the ``target`` category from live feature sets by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.engineering.features.registry import (
    as_feature_column,
    register,
)


@register(
    category="target",
    lookback=0,
    description="Forward N-bar simple return target from the current bar close.",
    calculation="close_{t+bars} / close_t - 1",
)
def next_n_bar_return(frame: pd.DataFrame, parameters: dict) -> pd.Series:
    """Compute a forward N-bar return target from each current bar.

    The horizon is measured in bars (rows), not calendar days. A bar is one row
    of the input frame: a daily bar on daily data, or a one-minute bar on
    one-minute data. The caller controls the bar size by choosing the source
    data and, for intraday runs, the ``reset_by_session`` option in
    ``compute_features`` (see ``engineering/compute.py``), which prevents the
    forward shift from crossing the overnight gap.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time. The
        pipeline guarantees this ordering and per-symbol isolation, so the
        forward shift below never reaches into another ticker's rows.
    parameters
        Supports ``bars`` as the positive integer forecast horizon in rows.
        Default is 1.

    Returns
    -------
    pandas.Series
        Forward return target aligned to ``frame.index``. The numerator is the
        close ``bars`` rows ahead and the denominator is the current row close.
        The final ``bars`` rows are ``NaN`` because the future close is
        unavailable.

    Raises
    ------
    ValueError
        If ``bars`` is less than one.
    """
    bars = int(parameters.get("bars", 1))
    if bars < 1:
        raise ValueError("next_n_bar_return requires bars >= 1.")

    close = frame["close"]

    # Forward simple return over a fixed number of bars. shift(-bars) brings the
    # future close back to the current row; the last ``bars`` rows become NaN
    # because their future close does not exist in the frame.
    future_close = close.shift(-bars)
    values = future_close / close - 1.0
    return as_feature_column(values)


# Default forward horizon for the realized-volatility target, in bars. Named so
# callers can discover and override it instead of relying on a hidden number.
DEFAULT_REALIZED_VOLATILITY_BARS = 20


@register(
    category="target",
    lookback=0,
    description="Forward realized volatility: std of the next N one-bar log returns.",
    calculation="std(ln(close_{t+k} / close_{t+k-1}) for k = 1..bars)",
)
def next_n_bar_realized_volatility(frame: pd.DataFrame, parameters: dict) -> pd.Series:
    """Compute a forward realized-volatility target from each current bar.

    This is the volatility analog of ``next_n_bar_return``: instead of asking
    "how much will price move (direction)?", it asks "how unstable will price
    be (magnitude)?" over the next ``bars`` rows. Realized volatility here is
    the sample standard deviation of the next ``bars`` one-bar log returns:

        target_t = std(r_{t+1}, ..., r_{t+bars})
        where r_{t+k} = ln(close_{t+k} / close_{t+k-1})

    The sample standard deviation (ddof = 1) is used so the target matches the
    backward-looking ``rolling_standard_deviation`` feature; a model can then be read as
    "predict the next window of the same statistic the feature measures over
    the previous window". The value is per-bar volatility in decimal return
    units, not annualized. The horizon is measured in bars (rows), not calendar
    days, exactly as in ``next_n_bar_return``.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time. The
        pipeline guarantees per-symbol isolation, so the forward window never
        reaches into another ticker's rows.
    parameters
        Supports ``bars``, the positive integer forecast horizon in rows.
        Default is ``DEFAULT_REALIZED_VOLATILITY_BARS``. Must be at least 2 because a
        standard deviation of a single return is undefined.

    Returns
    -------
    pandas.Series
        Forward realized volatility aligned to ``frame.index``. The final ``bars``
        rows are ``NaN`` because their future returns are incomplete.

    Raises
    ------
    ValueError
        If ``bars`` is less than two.
    """
    bars = int(parameters.get("bars", DEFAULT_REALIZED_VOLATILITY_BARS))
    if bars < 2:
        raise ValueError(
            "next_n_bar_realized_volatility requires bars >= 2 because the standard "
            "deviation of a single return is undefined."
        )

    # One-bar log returns; log returns are used because they add through time.
    log_returns = np.log(frame["close"] / frame["close"].shift(1))

    # Backward rolling std first, then shift the finished statistic back:
    # backward_std[t] = std(r_{t-bars+1} .. r_t), so backward_std[t + bars]
    # = std(r_{t+1} .. r_{t+bars}), which is exactly the forward window row t
    # needs. min_periods = bars forbids partially filled windows: a target
    # computed from fewer future returns than promised would silently change
    # the label definition. The final ``bars`` rows become NaN via the shift.
    backward_std = log_returns.rolling(window=bars, min_periods=bars).std()
    values = backward_std.shift(-bars)
    return as_feature_column(values)
