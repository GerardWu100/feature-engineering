"""True incremental (online) feature computation for live, bar-by-bar use.

Why this exists
---------------
The batch features in ``feature_engineering.features`` recompute over the whole
history every call. In a live loop that is O(n) per new bar and O(n^2) over a
session. This module keeps a small amount of state per symbol and updates each
feature in O(1) (or O(window) bounded) work per bar, independent of how long the
session has run.

Design
------
Each feature has a small accumulator class with one method, ``update(bar)``,
returning the feature value for that bar (``NaN`` during warmup). The math is
written to match the batch feature exactly; ``tests/test_engines.py`` feeds the
same bars through both paths and asserts equality, which is the contract that
keeps the two implementations from drifting.

Targets (the ``target`` category, e.g. ``next_n_bar_return``) are forward
looking and cannot be produced online; ``OnlineFeatureEngine`` rejects them.

A "bar" is any mapping (a dict or a pandas ``Series``) with the keys ``symbol``,
``ts``, ``open``, ``high``, ``low``, ``close``, and ``volume``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Mapping, Protocol

import pandas as pd

from feature_engineering.features.trend import (
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_RSI_WINDOW,
)
from feature_engineering.features.volatility import DEFAULT_ATR_WINDOW
from feature_engineering.pipeline.engineer import (
    _resolve_feature,
    _selected_feature_configs,
)

NAN = float("nan")
DEFAULT_SMA_WINDOW = 20
DEFAULT_ROC_PERIODS = 20


def _is_nan(value: float) -> bool:
    """Return True when ``value`` is NaN (NaN is the only value not equal to itself)."""
    return value != value


class OnlineFeature(Protocol):
    """One stateful feature updated bar by bar."""

    def update(self, bar: Mapping[str, Any]) -> float:
        """Consume one bar and return this feature's value for it."""
        ...


class _Ema:
    """Recursive exponential moving average matching pandas ``adjust=False``.

    Recurrence: ``ema_t = ema_{t-1} + alpha * (x_t - ema_{t-1})`` seeded at the
    first input. Output is ``NaN`` until ``min_count`` inputs have been seen,
    which reproduces ``ewm(..., adjust=False, min_periods=min_count)``.
    """

    def __init__(self, alpha: float, min_count: int) -> None:
        self._alpha = alpha
        self._min_count = min_count
        self._value: float | None = None
        self._count = 0

    def update(self, x: float) -> float:
        self._count += 1
        if self._value is None:
            self._value = x
        else:
            self._value += self._alpha * (x - self._value)
        return self._value if self._count >= self._min_count else NAN


class _RollingSum:
    """Fixed-window running sum over a deque; O(1) per update.

    Returns the current sum only once the window is full, else ``NaN`` count is
    signalled by ``is_full``. Callers read ``.is_full`` and ``.sum`` directly.
    """

    def __init__(self, window: int) -> None:
        self._window = window
        self._buffer: deque[float] = deque()
        self.sum = 0.0

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._window

    def update(self, x: float) -> None:
        self._buffer.append(x)
        self.sum += x
        if len(self._buffer) > self._window:
            self.sum -= self._buffer.popleft()


class _LogReturn:
    """ln(close_t / close_{t-1}); NaN on the first bar."""

    def __init__(self) -> None:
        self._prev: float | None = None

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        result = math.log(close / self._prev) if self._prev is not None else NAN
        self._prev = close
        return result


class _SimpleReturn:
    """close_t / close_{t-1} - 1; NaN on the first bar."""

    def __init__(self) -> None:
        self._prev: float | None = None

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        result = close / self._prev - 1.0 if self._prev is not None else NAN
        self._prev = close
        return result


class _VolumeChange:
    """volume_t / volume_{t-1} - 1; NaN on the first bar or zero prior volume."""

    def __init__(self) -> None:
        self._prev: float | None = None

    def update(self, bar: Mapping[str, Any]) -> float:
        volume = bar["volume"]
        if self._prev is None or self._prev == 0.0:
            result = NAN
        else:
            result = volume / self._prev - 1.0
        self._prev = volume
        return result


class _BarRangePct:
    """(high - low) / close; stateless."""

    def update(self, bar: Mapping[str, Any]) -> float:
        return (bar["high"] - bar["low"]) / bar["close"]


class _DollarVolume:
    """close * volume; stateless."""

    def update(self, bar: Mapping[str, Any]) -> float:
        return bar["close"] * bar["volume"]


class _MovingAverage:
    """Simple moving average of close over ``window`` bars."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._sum = _RollingSum(window)

    def update(self, bar: Mapping[str, Any]) -> float:
        self._sum.update(bar["close"])
        return self._sum.sum / self._window if self._sum.is_full else NAN


class _PriceVsSma:
    """close / SMA(close, window) - 1."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._sum = _RollingSum(window)

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        self._sum.update(close)
        if not self._sum.is_full:
            return NAN
        return close / (self._sum.sum / self._window) - 1.0


class _RateOfChange:
    """close_t / close_{t-periods} - 1 using a bounded buffer of closes."""

    def __init__(self, periods: int) -> None:
        self._periods = periods
        self._closes: deque[float] = deque(maxlen=periods + 1)

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        self._closes.append(close)
        if len(self._closes) == self._periods + 1:
            return close / self._closes[0] - 1.0
        return NAN


class _VolumeRatio:
    """volume_t / mean(volume, window); NaN when the window mean is zero."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._sum = _RollingSum(window)

    def update(self, bar: Mapping[str, Any]) -> float:
        volume = bar["volume"]
        self._sum.update(volume)
        if not self._sum.is_full:
            return NAN
        mean_volume = self._sum.sum / self._window
        if mean_volume == 0.0:
            return NAN
        return volume / mean_volume


class _RollingStd:
    """Sample std (ddof=1) of log returns over the window's ``window - 1`` returns.

    Maintains running sum and sum of squares of the buffered log returns so each
    update is O(1). Matches ``volatility.rolling_std``: a window of N prices uses
    the N-1 adjacent returns.
    """

    def __init__(self, window: int) -> None:
        self._return_window = window - 1
        self._prev: float | None = None
        self._returns: deque[float] = deque()
        self._sum = 0.0
        self._sum_squares = 0.0

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        if self._prev is None:
            self._prev = close
            return NAN

        log_return = math.log(close / self._prev)
        self._prev = close

        self._returns.append(log_return)
        self._sum += log_return
        self._sum_squares += log_return * log_return
        if len(self._returns) > self._return_window:
            oldest = self._returns.popleft()
            self._sum -= oldest
            self._sum_squares -= oldest * oldest

        count = len(self._returns)
        if count != self._return_window or self._return_window < 2:
            return NAN

        mean = self._sum / count
        # Sample variance with Bessel's correction (ddof=1), clamped at zero to
        # absorb tiny negative values from floating-point cancellation.
        variance = (self._sum_squares - count * mean * mean) / (count - 1)
        return math.sqrt(variance) if variance > 0.0 else 0.0


class _Rsi:
    """Wilder's RSI via two Wilder-smoothed averages (alpha = 1 / window)."""

    def __init__(self, window: int) -> None:
        self._prev: float | None = None
        self._avg_gain = _Ema(1.0 / window, window)
        self._avg_loss = _Ema(1.0 / window, window)

    def update(self, bar: Mapping[str, Any]) -> float:
        close = bar["close"]
        if self._prev is None:
            self._prev = close
            return NAN

        change = close - self._prev
        self._prev = close
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0

        average_gain = self._avg_gain.update(gain)
        average_loss = self._avg_loss.update(loss)
        if _is_nan(average_gain) or _is_nan(average_loss):
            return NAN
        if average_loss == 0.0 and average_gain == 0.0:
            return NAN  # flat window: 0/0 strength ratio is undefined
        if average_loss == 0.0:
            return 100.0  # only up-moves: RSI saturates
        relative_strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + relative_strength)


class _Atr:
    """Wilder's Average True Range (alpha = 1 / window)."""

    def __init__(self, window: int) -> None:
        self._prev_close: float | None = None
        self._avg_true_range = _Ema(1.0 / window, window)

    def update(self, bar: Mapping[str, Any]) -> float:
        high = bar["high"]
        low = bar["low"]
        if self._prev_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = bar["close"]
        return self._avg_true_range.update(true_range)


class _MacdLine:
    """Fast EMA minus slow EMA of close; NaN until both EMAs are valid."""

    def __init__(self, fast: int, slow: int) -> None:
        self._fast = _Ema(2.0 / (fast + 1), fast)
        self._slow = _Ema(2.0 / (slow + 1), slow)

    def value(self, close: float) -> float:
        fast_ema = self._fast.update(close)
        slow_ema = self._slow.update(close)
        if _is_nan(fast_ema) or _is_nan(slow_ema):
            return NAN
        return fast_ema - slow_ema

    def update(self, bar: Mapping[str, Any]) -> float:
        return self.value(bar["close"])


class _MacdSignal:
    """EMA of the MACD line. Returns the (macd_line, signal) pair internally."""

    def __init__(self, fast: int, slow: int, signal: int) -> None:
        self._line = _MacdLine(fast, slow)
        self._signal = _Ema(2.0 / (signal + 1), signal)

    def value(self, close: float) -> tuple[float, float]:
        macd_line = self._line.value(close)
        if _is_nan(macd_line):
            return (NAN, NAN)
        # The signal EMA only advances on real MACD values, matching the batch
        # path that drops the MACD line's warmup NaNs before smoothing.
        signal = self._signal.update(macd_line)
        return (macd_line, signal)

    def update(self, bar: Mapping[str, Any]) -> float:
        _macd_line, signal = self.value(bar["close"])
        return signal


class _MacdHistogram:
    """MACD line minus its signal line."""

    def __init__(self, fast: int, slow: int, signal: int) -> None:
        self._signal = _MacdSignal(fast, slow, signal)

    def update(self, bar: Mapping[str, Any]) -> float:
        macd_line, signal = self._signal.value(bar["close"])
        if _is_nan(macd_line) or _is_nan(signal):
            return NAN
        return macd_line - signal


class _Vwap:
    """Cumulative VWAP within the engine's current session for one symbol."""

    def __init__(self) -> None:
        self._cumulative_price_volume = 0.0
        self._cumulative_volume = 0.0

    def value(self, bar: Mapping[str, Any]) -> float:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        self._cumulative_price_volume += typical_price * bar["volume"]
        self._cumulative_volume += bar["volume"]
        if self._cumulative_volume == 0.0:
            return NAN
        return self._cumulative_price_volume / self._cumulative_volume

    def update(self, bar: Mapping[str, Any]) -> float:
        return self.value(bar)


class _PriceVsVwap:
    """close / VWAP - 1."""

    def __init__(self) -> None:
        self._vwap = _Vwap()

    def update(self, bar: Mapping[str, Any]) -> float:
        vwap_value = self._vwap.value(bar)
        if _is_nan(vwap_value):
            return NAN
        return bar["close"] / vwap_value - 1.0


# Map each registry feature function name to a factory that builds its online
# accumulator from the feature's params. Targets are intentionally absent.
ONLINE_FEATURE_FACTORIES: dict[str, Callable[[dict[str, Any]], OnlineFeature]] = {
    "log_return": lambda params: _LogReturn(),
    "simple_return": lambda params: _SimpleReturn(),
    "volume_change": lambda params: _VolumeChange(),
    "bar_range_pct": lambda params: _BarRangePct(),
    "dollar_volume": lambda params: _DollarVolume(),
    "moving_average": lambda params: _MovingAverage(int(params["window"])),
    "price_vs_sma": lambda params: _PriceVsSma(
        int(params.get("window", DEFAULT_SMA_WINDOW))
    ),
    "rate_of_change": lambda params: _RateOfChange(
        int(params.get("periods", DEFAULT_ROC_PERIODS))
    ),
    "rolling_std": lambda params: _RollingStd(int(params["window"])),
    "volume_ratio": lambda params: _VolumeRatio(int(params["window"])),
    "relative_strength_index": lambda params: _Rsi(
        int(params.get("window", DEFAULT_RSI_WINDOW))
    ),
    "average_true_range": lambda params: _Atr(
        int(params.get("window", DEFAULT_ATR_WINDOW))
    ),
    "macd_line": lambda params: _MacdLine(
        int(params.get("fast", DEFAULT_MACD_FAST)),
        int(params.get("slow", DEFAULT_MACD_SLOW)),
    ),
    "macd_signal": lambda params: _MacdSignal(
        int(params.get("fast", DEFAULT_MACD_FAST)),
        int(params.get("slow", DEFAULT_MACD_SLOW)),
        int(params.get("signal", DEFAULT_MACD_SIGNAL)),
    ),
    "macd_histogram": lambda params: _MacdHistogram(
        int(params.get("fast", DEFAULT_MACD_FAST)),
        int(params.get("slow", DEFAULT_MACD_SLOW)),
        int(params.get("signal", DEFAULT_MACD_SIGNAL)),
    ),
    "vwap": lambda params: _Vwap(),
    "price_vs_vwap": lambda params: _PriceVsVwap(),
}


class OnlineFeatureEngine:
    """Compute features incrementally, one bar at a time, per symbol.

    Parameters
    ----------
    config
        Config dict with a ``features`` section (same shape as ``config.toml``).
        Category filters and ``reset_by_session`` are honored. The ``run``
        section is not required.

    Raises
    ------
    ValueError
        At construction, if a selected feature is a forward-looking ``target``
        or has no online implementation.

    Notes
    -----
    State is kept per symbol. When ``reset_by_session`` is true, a symbol's
    accumulators are rebuilt whenever the bar's calendar date changes, mirroring
    the batch ``reset_by_session`` behavior (this is also what makes intraday
    VWAP a per-session figure).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        features_config = config.get("features", {})
        self._reset_by_session = bool(features_config.get("reset_by_session", False))

        # Blueprint: (column_name, factory, params). Built once; used to spin up
        # fresh accumulators per symbol (and per session when resetting).
        self._blueprints: list[
            tuple[str, Callable[[dict[str, Any]], OnlineFeature], dict[str, Any]]
        ] = []
        for feature_item in _selected_feature_configs(config):
            column_name, spec, params = _resolve_feature(feature_item)
            function_name = feature_item["fn"]
            if spec.category == "target":
                raise ValueError(
                    f"Feature {column_name!r} is a forward-looking target and "
                    "cannot be computed online."
                )
            if function_name not in ONLINE_FEATURE_FACTORIES:
                raise ValueError(
                    f"Feature fn {function_name!r} has no online implementation."
                )
            self._blueprints.append(
                (column_name, ONLINE_FEATURE_FACTORIES[function_name], params)
            )

        # Per-symbol state: {symbol: {"accumulators": {col: acc}, "date": date}}.
        self._state: dict[str, dict[str, Any]] = {}

    @property
    def feature_names(self) -> list[str]:
        """Return the output feature column names in config order."""
        return [column_name for column_name, _factory, _params in self._blueprints]

    def reset(self) -> None:
        """Forget all per-symbol state (start fresh)."""
        self._state.clear()

    def update(self, bar: Mapping[str, Any]) -> dict[str, float]:
        """Consume one bar and return ``{feature_name: value}`` for it.

        Parameters
        ----------
        bar
            Mapping with ``symbol``, ``ts`` (when ``reset_by_session``), and
            ``open``/``high``/``low``/``close``/``volume``.

        Returns
        -------
        dict
            Feature values for this bar. Warmup values are ``NaN``.
        """
        symbol = bar["symbol"]
        session_date = self._session_date(bar) if self._reset_by_session else None
        state = self._state.get(symbol)

        # Build fresh accumulators for a new symbol, or when the session rolls
        # over and resets are enabled.
        if state is None or (self._reset_by_session and state["date"] != session_date):
            state = {
                "accumulators": {
                    column_name: factory(params)
                    for column_name, factory, params in self._blueprints
                },
                "date": session_date,
            }
            self._state[symbol] = state

        return {
            column_name: accumulator.update(bar)
            for column_name, accumulator in state["accumulators"].items()
        }

    def stream_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replay a historical frame bar by bar through the online engine.

        This is a convenience for backtests, demos, and the batch-equivalence
        tests. It sorts by ``symbol`` then ``ts`` (so each symbol's bars arrive
        in time order) and returns a feature frame in the same row order as
        ``compute_features``.

        Parameters
        ----------
        frame
            OHLCV frame with ``symbol``, ``ts``, and OHLCV columns.

        Returns
        -------
        pandas.DataFrame
            ``symbol``, ``ts``, and one column per configured feature.
        """
        self.reset()
        sorted_frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        for bar in sorted_frame.to_dict("records"):
            feature_values = self.update(bar)
            rows.append({"symbol": bar["symbol"], "ts": bar["ts"], **feature_values})
        return pd.DataFrame(rows, columns=["symbol", "ts", *self.feature_names])

    def _session_date(self, bar: Mapping[str, Any]) -> Any:
        """Return the calendar date used as the session-reset key for a bar."""
        return pd.Timestamp(bar["ts"]).normalize()
