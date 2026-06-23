"""Formula tests for the simplified categorized feature library."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from feature_engineering.features.returns import (
    log_return,
    next_n_bar_return,
    simple_return,
)
from feature_engineering.features.trend import (
    moving_average,
    price_vs_sma,
    rate_of_change,
)
from feature_engineering.features.volatility import bar_range_pct, rolling_std
from feature_engineering.features.volume import (
    dollar_volume,
    volume_change,
    volume_ratio,
)


def _sample_ohlcv_frame() -> pd.DataFrame:
    """Build a deterministic single-symbol OHLCV sample for feature tests."""
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
        ]
    ).tz_localize("America/New_York")

    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "ts": timestamps,
            "open": [100.0, 101.0, 102.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 103.0, 104.0],
            "close": [100.0, 101.0, 103.0, 104.0, 106.0],
            "volume": [1000.0, 1200.0, 1800.0, 1600.0, 2400.0],
        }
    )


def test_return_features_match_manual_formulas() -> None:
    """Return features should match textbook one-period return formulas."""
    frame = _sample_ohlcv_frame()

    log_values = log_return(frame, {})
    simple_values = simple_return(frame, {})

    assert pd.isna(log_values.iloc[0])
    assert pd.isna(simple_values.iloc[0])
    assert math.isclose(log_values.iloc[1], math.log(101.0 / 100.0))
    assert math.isclose(simple_values.iloc[2], 103.0 / 101.0 - 1.0)


def test_next_n_bar_return_is_forward_simple_return_over_bars() -> None:
    """Forward target should be close[t+bars]/close[t] - 1, NaN in the final bars."""
    frame = _sample_ohlcv_frame()
    # Closes are [100, 101, 103, 104, 106].

    one_bar = next_n_bar_return(frame, {"bars": 1})
    two_bar = next_n_bar_return(frame, {"bars": 2})

    # One bar ahead: each row divided by the next close.
    assert math.isclose(one_bar.iloc[0], 101.0 / 100.0 - 1.0)
    assert math.isclose(one_bar.iloc[1], 103.0 / 101.0 - 1.0)
    assert math.isclose(one_bar.iloc[3], 106.0 / 104.0 - 1.0)
    assert pd.isna(one_bar.iloc[4])

    # Two bars ahead: the final two rows have no future close.
    assert math.isclose(two_bar.iloc[0], 103.0 / 100.0 - 1.0)
    assert math.isclose(two_bar.iloc[2], 106.0 / 103.0 - 1.0)
    assert pd.isna(two_bar.iloc[3])
    assert pd.isna(two_bar.iloc[4])


def test_trend_features_match_manual_formulas() -> None:
    """Trend features should expose simple moving-average and lagged-return math."""
    frame = _sample_ohlcv_frame()

    moving_average_values = moving_average(frame, {"window": 3})
    price_vs_sma_values = price_vs_sma(frame, {"window": 3})
    rate_of_change_values = rate_of_change(frame, {"periods": 2})

    expected_sma = (100.0 + 101.0 + 103.0) / 3.0
    assert pd.isna(moving_average_values.iloc[0])
    assert pd.isna(moving_average_values.iloc[1])
    assert math.isclose(moving_average_values.iloc[2], expected_sma)
    assert math.isclose(price_vs_sma_values.iloc[2], 103.0 / expected_sma - 1.0)
    assert math.isclose(rate_of_change_values.iloc[2], 103.0 / 100.0 - 1.0)


def test_volatility_features_match_manual_formulas() -> None:
    """Volatility features should measure return dispersion and bar range size."""
    frame = _sample_ohlcv_frame()

    rolling_values = rolling_std(frame, {"window": 3})
    range_values = bar_range_pct(frame, {})

    log_returns = np.log(
        pd.Series([100.0, 101.0, 103.0]) / pd.Series([np.nan, 100.0, 101.0])
    )
    expected_std = float(log_returns.std())

    assert pd.isna(rolling_values.iloc[0])
    assert pd.isna(rolling_values.iloc[1])
    assert math.isclose(rolling_values.iloc[2], expected_std)
    assert math.isclose(range_values.iloc[0], (101.0 - 99.0) / 100.0)


def test_rolling_std_window_counts_prices_not_returns() -> None:
    """A volatility window of N prices should use the N - 1 returns inside that window."""
    frame = _sample_ohlcv_frame()

    rolling_values = rolling_std(frame, {"window": 3})

    recent_prices = pd.Series([103.0, 104.0, 106.0])
    recent_log_returns = np.log(recent_prices / recent_prices.shift(1))
    expected_std = float(recent_log_returns.std())

    assert math.isclose(rolling_values.iloc[4], expected_std)


def test_volume_features_match_manual_formulas() -> None:
    """Volume features should expose relative, dollar, and percent-change volume."""
    frame = _sample_ohlcv_frame()

    ratio_values = volume_ratio(frame, {"window": 3})
    dollar_values = dollar_volume(frame, {})
    change_values = volume_change(frame, {})

    expected_mean_volume = (1000.0 + 1200.0 + 1800.0) / 3.0

    assert pd.isna(ratio_values.iloc[0])
    assert pd.isna(ratio_values.iloc[1])
    assert math.isclose(ratio_values.iloc[2], 1800.0 / expected_mean_volume)
    assert math.isclose(dollar_values.iloc[2], 103.0 * 1800.0)
    assert math.isclose(change_values.iloc[1], 1200.0 / 1000.0 - 1.0)
