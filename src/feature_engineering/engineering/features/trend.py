"""Trend features for stock OHLCV data.

Trend features measure direction or persistence. They answer questions such as:
is price above its recent average, and has price moved up over a lookback?
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.engineering.features.registry import as_feature_column, register

# Named defaults so an omitted parameter means the same thing whether the
# feature is called directly or configured in config.toml.
DEFAULT_MOVING_AVERAGE_WINDOW = 20
DEFAULT_RATE_OF_CHANGE_PERIODS = 20
DEFAULT_RSI_WINDOW = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9


@register(
    category="trend",
    lookback=lambda parameters: int(
        parameters.get("window", DEFAULT_MOVING_AVERAGE_WINDOW)
    ),
    description="Simple moving average of close prices over a rolling window.",
    calculation="mean(close) over trailing window",
)
def moving_average(
    frame: pd.DataFrame,
    *,
    window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> pd.Series:
    """Compute a simple moving average of close prices.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with ``close`` values sorted by time.
    window
        Number of rows in the rolling average. Default is 20.

    Returns
    -------
    pandas.Series
        Moving-average values aligned to ``frame.index``. The first ``window - 1``
        rows are ``NaN`` because a full window is required.
    """
    window = int(window)
    if window < 1:
        raise ValueError("moving_average requires window >= 1.")

    # Row-count windows are easier to reason about in the simplified project.
    # The input is already sorted per symbol by the pipeline.
    values = frame["close"].rolling(window=window, min_periods=window).mean()
    return as_feature_column(values)


@register(
    category="trend",
    lookback=lambda parameters: int(
        parameters.get("window", DEFAULT_MOVING_AVERAGE_WINDOW)
    ),
    description="Close price divided by its moving average minus one.",
    calculation="close_t / moving_average_t - 1",
)
def price_vs_moving_average(
    frame: pd.DataFrame,
    *,
    window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> pd.Series:
    """Compute normalized distance between close and its moving average.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column.
    window
        Rolling average length. Default is 20 rows.

    Returns
    -------
    pandas.Series
        Dimensionless distance from the moving average, aligned to ``frame.index``.
    """
    window = int(window)
    if window < 1:
        raise ValueError("price_vs_moving_average requires window >= 1.")

    average = frame["close"].rolling(window=window, min_periods=window).mean()

    # Dividing by the average makes the feature comparable across symbols with
    # different price levels.
    values = frame["close"] / average - 1.0
    return as_feature_column(values)


@register(
    category="trend",
    lookback=lambda parameters: int(
        parameters.get("periods", DEFAULT_RATE_OF_CHANGE_PERIODS)
    ),
    description="Rate of change over a fixed number of rows.",
    calculation="close_t / close_{t-periods} - 1",
)
def rate_of_change(
    frame: pd.DataFrame,
    *,
    periods: int = DEFAULT_RATE_OF_CHANGE_PERIODS,
) -> pd.Series:
    """Compute lagged percentage price change.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column.
    periods
        Lag length. Default is 20 rows.

    Returns
    -------
    pandas.Series
        Percentage price change over ``periods`` rows, aligned to ``frame.index``.
    """
    periods = int(periods)
    if periods < 1:
        raise ValueError("rate_of_change requires periods >= 1.")

    return as_feature_column(frame["close"].pct_change(periods=periods))


def _ema(values: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``adjust=False`` recursion.

    ``adjust=False`` gives the standard recursive EMA used by trading platforms:

        ema_t = ema_{t-1} + alpha * (x_t - ema_{t-1}),  alpha = 2 / (span + 1)

    The first ``span - 1`` rows are ``NaN`` (``min_periods=span``).

    Parameters
    ----------
    values
        Input series with no internal gaps (leading NaNs must be dropped first).
    span
        EMA span; ``alpha = 2 / (span + 1)``.

    Returns
    -------
    pandas.Series
        EMA aligned to ``values.index``.
    """
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_average(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothed moving average with ``adjust=False``.

    Wilder smoothing is an EMA with ``alpha = 1 / window``:

        rma_t = rma_{t-1} + (1 / window) * (x_t - rma_{t-1})

    Used by RSI and ATR. Like :func:`_ema` it expects a gap-free input so the
    recursion seeds unambiguously and matches the online accumulator.
    """
    return values.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


@register(
    category="trend",
    lookback=lambda parameters: int(parameters.get("window", DEFAULT_RSI_WINDOW)),
    description="Wilder's Relative Strength Index momentum oscillator (0-100).",
    calculation="100 - 100 / (1 + avg_gain / avg_loss), Wilder-smoothed over window",
)
def relative_strength_index(
    frame: pd.DataFrame,
    *,
    window: int = DEFAULT_RSI_WINDOW,
) -> pd.Series:
    """Compute Wilder's Relative Strength Index.

    The Relative Strength Index measures the ratio of average up-moves to
    average down-moves over a window and maps it to a 0-100 oscillator. Values above 70 are conventionally
    "overbought" and below 30 "oversold".

    The leading ``NaN`` price-difference row is dropped before smoothing so the
    Wilder recursion seeds on the first real gain/loss. The result is then
    reindexed back, leaving the first ``window`` rows as ``NaN``.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column, sorted by time.
    window
        Smoothing length in bars. Default is 14.

    Returns
    -------
    pandas.Series
        RSI in [0, 100] aligned to ``frame.index``. A window with no down-moves
        yields 100; a perfectly flat window yields ``NaN`` (0/0 strength ratio).

    Raises
    ------
    ValueError
        If ``window`` is less than two.
    """
    window = int(window)
    if window < 2:
        raise ValueError("relative_strength_index requires window >= 2.")

    close = frame["close"]

    # One-bar price change. The first row is NaN (no previous close); drop it so
    # the smoothing recursion has a clean, gap-free start.
    price_change = close.diff().iloc[1:]
    gain = price_change.clip(lower=0.0)
    loss = (-price_change).clip(lower=0.0)

    # Wilder-smoothed average gain and loss over the window.
    average_gain = _wilder_average(gain, window)
    average_loss = _wilder_average(loss, window)

    # Relative strength = average gain / average loss; RSI compresses it into
    # 0-100. When average loss is zero, RSI saturates at 100. When both are
    # zero (a flat window), 0/0 is NaN and RSI is undefined.
    relative_strength = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)

    # Reindex to the original rows so the dropped first bar becomes NaN again.
    return as_feature_column(rsi.reindex(close.index))


def _macd_line_series(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Return the MACD line: fast EMA minus slow EMA of close.

    The MACD line is valid only where both EMAs are valid, so its first valid
    row is at index ``slow - 1``.
    """
    return _ema(close, fast) - _ema(close, slow)


def _macd_signal_from_line(macd: pd.Series, signal: int) -> pd.Series:
    """Return the MACD signal line: an EMA of a precomputed MACD line.

    The MACD line's leading warmup ``NaN`` rows are dropped before smoothing so
    the signal EMA seeds on the first real MACD value, then the result is
    reindexed back so those early rows stay ``NaN``. Both ``macd_signal`` and
    ``macd_histogram`` go through this one helper so their signal-line math
    cannot drift apart. Taking the line as input lets callers that also need
    the line itself compute it only once.
    """
    return _ema(macd.dropna(), signal).reindex(macd.index)


@register(
    category="trend",
    lookback=lambda parameters: int(parameters.get("slow", DEFAULT_MACD_SLOW)),
    description="MACD line: difference between fast and slow EMAs of close.",
    calculation="EMA(close, fast) - EMA(close, slow)",
)
def macd_line(
    frame: pd.DataFrame,
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
) -> pd.Series:
    """Compute the MACD (Moving Average Convergence Divergence) line.

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column.
    fast, slow
        EMA spans. Defaults are 12 and 26; ``fast`` must be less than ``slow``.

    Returns
    -------
    pandas.Series
        MACD line aligned to ``frame.index``; ``NaN`` until the slow EMA is valid.

    Raises
    ------
    ValueError
        If ``fast`` is not less than ``slow``.
    """
    fast, slow = int(fast), int(slow)
    if fast >= slow:
        raise ValueError("macd_line requires fast < slow.")

    return as_feature_column(_macd_line_series(frame["close"], fast, slow))


@register(
    category="trend",
    lookback=lambda parameters: (
        int(parameters.get("slow", DEFAULT_MACD_SLOW))
        + int(parameters.get("signal", DEFAULT_MACD_SIGNAL))
    ),
    description="MACD signal line: EMA of the MACD line.",
    calculation="EMA(MACD_line, signal)",
)
def macd_signal(
    frame: pd.DataFrame,
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> pd.Series:
    """Compute the MACD signal line (EMA of the MACD line).

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column.
    fast, slow, signal
        EMA spans. Defaults are 12, 26, and 9; ``fast`` must be less than ``slow``.

    Returns
    -------
    pandas.Series
        Signal line aligned to ``frame.index``. The MACD line's leading ``NaN``
        rows are dropped before the signal EMA so it seeds cleanly, then the
        result is reindexed back.
    """
    fast, slow, signal = int(fast), int(slow), int(signal)
    if fast >= slow:
        raise ValueError("macd_signal requires fast < slow.")

    # Delegate the warmup-drop + reindex to the shared helper so this stays in
    # lockstep with the histogram's signal line.
    macd = _macd_line_series(frame["close"], fast, slow)
    return as_feature_column(_macd_signal_from_line(macd, signal))


@register(
    category="trend",
    lookback=lambda parameters: (
        int(parameters.get("slow", DEFAULT_MACD_SLOW))
        + int(parameters.get("signal", DEFAULT_MACD_SIGNAL))
    ),
    description="MACD histogram: MACD line minus signal line.",
    calculation="MACD_line - EMA(MACD_line, signal)",
)
def macd_histogram(
    frame: pd.DataFrame,
    *,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> pd.Series:
    """Compute the MACD histogram (MACD line minus signal line).

    Parameters
    ----------
    frame
        Single-symbol OHLCV frame with a ``close`` column.
    fast, slow, signal
        EMA spans. Defaults are 12, 26, and 9; ``fast`` must be less than ``slow``.

    Returns
    -------
    pandas.Series
        Histogram aligned to ``frame.index``; valid once the signal line is valid.
    """
    fast, slow, signal = int(fast), int(slow), int(signal)
    if fast >= slow:
        raise ValueError("macd_histogram requires fast < slow.")

    # Compute the line once and feed it to the signal helper so the two EMAs of
    # close are not evaluated twice.
    macd = _macd_line_series(frame["close"], fast, slow)
    signal_line = _macd_signal_from_line(macd, signal)

    # The histogram is the gap between momentum (MACD line) and its own EMA.
    return as_feature_column(macd - signal_line)
