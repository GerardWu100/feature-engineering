"""Target features: forward-looking labels for supervised learning.

A target answers "what happens after this bar?" so it is computed from future
rows. Targets must never be used as live input signals; the config excludes
the ``target`` category from live feature sets by default.

Two families live here. Fixed-horizon targets (``next_n_bar_return``,
``next_n_bar_realized_volatility``) assume the position is held for exactly
``bars`` rows. Triple-barrier targets (``triple_barrier_label``,
``triple_barrier_bars_to_exit``, ``triple_barrier_exit_return``) let the data
decide the holding period: the position closes at a volatility-scaled
take-profit, a stop-loss, or a time limit, whichever comes first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering.engineering.features.registry import (
    as_feature_column,
    register,
)
from feature_engineering.engineering.features.volatility import (
    rolling_standard_deviation,
)


@register(
    category="target",
    lookback=0,
    description="Forward N-bar simple return target from the current bar close.",
    calculation="close_{t+bars} / close_t - 1",
)
def next_n_bar_return(frame: pd.DataFrame, *, bars: int = 1) -> pd.Series:
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
    bars
        Positive integer forecast horizon in rows. Default is 1.

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
    bars = int(bars)
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
def next_n_bar_realized_volatility(
    frame: pd.DataFrame,
    *,
    bars: int = DEFAULT_REALIZED_VOLATILITY_BARS,
) -> pd.Series:
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
    bars
        Positive integer forecast horizon in rows. Default is
        ``DEFAULT_REALIZED_VOLATILITY_BARS``. Must be at least 2 because a
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
    bars = int(bars)
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


# Triple-barrier defaults (Lopez de Prado, Advances in Financial Machine
# Learning, chapter 3). The time barrier counts bars; the price barriers are
# multiples of trailing volatility; the volatility window counts prices, the
# same convention as ``rolling_standard_deviation``.
DEFAULT_TRIPLE_BARRIER_MAX_BARS = 20
DEFAULT_TRIPLE_BARRIER_MULTIPLE = 2.0
DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW = 20

# Value written to ``triple_barrier_label`` for each outcome.
UPPER_BARRIER_LABEL = 1.0
LOWER_BARRIER_LABEL = -1.0
TIME_BARRIER_LABEL = 0.0


@dataclass(frozen=True)
class TripleBarrierOutcome:
    """Per-row result of one triple-barrier scan, aligned to the input frame.

    Attributes
    ----------
    label
        ``+1`` if the upper barrier was reached first, ``-1`` if the lower
        barrier was reached first, ``0`` if neither within ``max_bars``, and
        ``NaN`` where the label is undefined.
    bars_to_exit
        Number of bars from entry to exit: the first-touch bar, or ``max_bars``
        when the time barrier ends the trade. ``NaN`` where the label is.
    exit_return
        ``close_exit / close_t - 1`` at the exit bar. ``NaN`` where the label is.
    """

    label: pd.Series
    bars_to_exit: pd.Series
    exit_return: pd.Series


def _triple_barrier_lookback(parameters: dict) -> int:
    return int(
        parameters.get("volatility_window", DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW)
    )


def scan_triple_barrier(
    frame: pd.DataFrame,
    *,
    max_bars: int = DEFAULT_TRIPLE_BARRIER_MAX_BARS,
    upper_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    lower_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    volatility_window: int = DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW,
) -> TripleBarrierOutcome:
    """Run the triple-barrier scan once and return every derived target.

    The method simulates a position opened at the close of bar ``t`` and
    closed by whichever of three exits comes first:

    - the upper barrier, a take-profit at ``+upper_multiple * sigma_t``;
    - the lower barrier, a stop-loss at ``-lower_multiple * sigma_t``;
    - the time barrier, forced exit after ``max_bars`` bars.

    ``sigma_t`` is the trailing sample standard deviation of one-bar log
    returns over ``volatility_window`` prices ending at bar ``t`` (the same
    number the ``rolling_standard_deviation`` feature reports), so the barrier
    width is known when the position is opened and cannot leak the future.
    Barriers are tested against closes only. Using highs and lows would catch
    intrabar touches, but a bar that spans both barriers would then need an
    ordering rule; closes keep the label unambiguous and consistent with the
    other close-based targets. The price test uses simple forward returns:

        r_{t,k} = close_{t+k} / close_t - 1,  k = 1..max_bars
        k* = first k with r_{t,k} >= upper_multiple * sigma_t
                       or r_{t,k} <= -lower_multiple * sigma_t

    Rows are undefined (``NaN`` in every output) when ``sigma_t`` is not yet
    available, when ``sigma_t`` is zero (the barriers collapse onto the entry
    price), or when fewer than ``max_bars`` future bars exist in the group.
    The last rule is deliberate: a row near the end of the data could be
    labelled only if it happened to hit a barrier early, which would bias the
    tail of every dataset toward ``+1``/``-1`` outcomes. Dropping the whole
    tail keeps one definition for every labelled row.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.
    max_bars
        Time barrier in bars (rows). Must be at least 1.
    upper_multiple, lower_multiple
        Take-profit and stop-loss widths as positive multiples of ``sigma_t``.
    volatility_window
        Number of prices in the trailing volatility estimate. Must be at least
        3 (three prices form the two returns a sample standard deviation needs).

    Returns
    -------
    TripleBarrierOutcome
        Label, bars to exit, and exit return, each aligned to ``frame.index``.

    Raises
    ------
    ValueError
        If ``max_bars`` is below 1, a multiple is not positive, or
        ``volatility_window`` is below 3.
    """
    max_bars = int(max_bars)
    if max_bars < 1:
        raise ValueError("triple barrier targets require max_bars >= 1.")
    if not upper_multiple > 0 or not lower_multiple > 0:
        raise ValueError("triple barrier multiples must be positive numbers.")
    volatility_window = int(volatility_window)
    if volatility_window < 3:
        raise ValueError("triple barrier targets require volatility_window >= 3.")

    close = frame["close"].to_numpy(dtype="float64")
    row_count = len(close)
    sigma = rolling_standard_deviation(frame, window=volatility_window).to_numpy(
        dtype="float64"
    )

    upper_threshold = float(upper_multiple) * sigma
    lower_threshold = -float(lower_multiple) * sigma

    label = np.full(row_count, np.nan)
    bars_to_exit = np.full(row_count, np.nan)
    exit_return = np.full(row_count, np.nan)

    # A row can be labelled only when its barrier width is defined and its full
    # time window fits inside the group. Everything else stays NaN.
    has_full_window = np.arange(row_count) + max_bars < row_count
    unresolved = has_full_window & np.isfinite(sigma) & (sigma > 0)

    # Walk the horizon one step at a time. At step k every still-open position
    # is tested against its barriers; the first touch fixes the label, and the
    # position leaves the unresolved set so later bars cannot overwrite it.
    for step in range(1, max_bars + 1):
        forward_return = np.full(row_count, np.nan)
        forward_return[: row_count - step] = (
            close[step:] / close[: row_count - step] - 1.0
        )

        touched_upper = unresolved & (forward_return >= upper_threshold)
        touched_lower = unresolved & (forward_return <= lower_threshold)

        label[touched_upper] = UPPER_BARRIER_LABEL
        label[touched_lower] = LOWER_BARRIER_LABEL
        touched = touched_upper | touched_lower
        bars_to_exit[touched] = step
        exit_return[touched] = forward_return[touched]
        unresolved &= ~touched

    # Positions still open after the last step exit at the time barrier. The
    # final forward_return from the loop is the max_bars-ahead return.
    label[unresolved] = TIME_BARRIER_LABEL
    bars_to_exit[unresolved] = max_bars
    exit_return[unresolved] = forward_return[unresolved]

    index = frame.index
    return TripleBarrierOutcome(
        label=as_feature_column(pd.Series(label, index=index)),
        bars_to_exit=as_feature_column(pd.Series(bars_to_exit, index=index)),
        exit_return=as_feature_column(pd.Series(exit_return, index=index)),
    )


@register(
    category="target",
    lookback=_triple_barrier_lookback,
    description=(
        "Triple-barrier label: +1 if the upper volatility-scaled barrier is hit "
        "first, -1 if the lower barrier is hit first, 0 if neither within max_bars."
    ),
    calculation=(
        "first k<=max_bars with close_{t+k}/close_t-1 >= upper_multiple*sigma_t "
        "(+1) or <= -lower_multiple*sigma_t (-1); else 0; sigma_t = trailing "
        "std of log returns over volatility_window prices"
    ),
)
def triple_barrier_label(
    frame: pd.DataFrame,
    *,
    max_bars: int = DEFAULT_TRIPLE_BARRIER_MAX_BARS,
    upper_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    lower_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    volatility_window: int = DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW,
) -> pd.Series:
    """Label which barrier a position opened at bar ``t`` reaches first.

    See :func:`scan_triple_barrier` for the full definition and parameters.
    Values are ``+1`` (take-profit first), ``-1`` (stop-loss first), ``0``
    (time barrier, neither price barrier reached), or ``NaN`` (undefined).
    When screening with ``evaluate_features``, pass
    ``target_horizon_bars=max_bars``: a label can depend on closes up to
    ``max_bars`` ahead, so neighbouring rows overlap for up to that many bars.
    """
    return scan_triple_barrier(
        frame,
        max_bars=max_bars,
        upper_multiple=upper_multiple,
        lower_multiple=lower_multiple,
        volatility_window=volatility_window,
    ).label


@register(
    category="target",
    lookback=_triple_barrier_lookback,
    description=(
        "Bars from entry until the first triple-barrier exit; equals max_bars "
        "when the time barrier ends the trade."
    ),
    calculation="k* from triple_barrier_label, or max_bars when no price barrier is hit",
)
def triple_barrier_bars_to_exit(
    frame: pd.DataFrame,
    *,
    max_bars: int = DEFAULT_TRIPLE_BARRIER_MAX_BARS,
    upper_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    lower_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    volatility_window: int = DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW,
) -> pd.Series:
    """Holding time in bars implied by the triple-barrier rule.

    This is a time-to-event target: how long until the move plays out. Values
    at the time barrier equal ``max_bars`` and are censored, meaning the true
    time to a price barrier is at least that long but unknown. Treat them as a
    lower bound, not as an observed exit time. See :func:`scan_triple_barrier`.
    """
    return scan_triple_barrier(
        frame,
        max_bars=max_bars,
        upper_multiple=upper_multiple,
        lower_multiple=lower_multiple,
        volatility_window=volatility_window,
    ).bars_to_exit


@register(
    category="target",
    lookback=_triple_barrier_lookback,
    description=(
        "Simple return from the entry close to the close of the triple-barrier "
        "exit bar."
    ),
    calculation="close_{t+k*} / close_t - 1 with k* from triple_barrier_label",
)
def triple_barrier_exit_return(
    frame: pd.DataFrame,
    *,
    max_bars: int = DEFAULT_TRIPLE_BARRIER_MAX_BARS,
    upper_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    lower_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    volatility_window: int = DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW,
) -> pd.Series:
    """Return a trader would book under the triple-barrier exit rule.

    Unlike ``next_n_bar_return``, the holding period varies per row, so this is
    the continuous companion to ``triple_barrier_label``: it keeps the size of
    the move, not only which barrier ended it. The exit price is the close of
    the exit bar, which can overshoot the barrier level. See
    :func:`scan_triple_barrier`.
    """
    return scan_triple_barrier(
        frame,
        max_bars=max_bars,
        upper_multiple=upper_multiple,
        lower_multiple=lower_multiple,
        volatility_window=volatility_window,
    ).exit_return
