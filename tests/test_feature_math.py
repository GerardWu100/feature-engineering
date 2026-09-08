"""Formula tests for the simplified categorized feature library."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from feature_engineering.engineering.features.returns import (
    log_return,
    simple_return,
)
from feature_engineering.engineering.features.targets import (
    next_n_bar_realized_volatility,
    next_n_bar_return,
    scan_triple_barrier,
    triple_barrier_bars_to_exit,
    triple_barrier_exit_return,
    triple_barrier_label,
)
from feature_engineering.engineering.features.trend import (
    moving_average,
    price_vs_moving_average,
    rate_of_change,
)
from feature_engineering.engineering.features.volatility import (
    bar_range_percent,
    rolling_standard_deviation,
)
from feature_engineering.engineering.features.volume import (
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
            "timestamp": timestamps,
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

    log_values = log_return(frame)
    simple_values = simple_return(frame)

    assert pd.isna(log_values.iloc[0])
    assert pd.isna(simple_values.iloc[0])
    assert math.isclose(log_values.iloc[1], math.log(101.0 / 100.0))
    assert math.isclose(simple_values.iloc[2], 103.0 / 101.0 - 1.0)


def test_next_n_bar_return_is_forward_simple_return_over_bars() -> None:
    """Forward target should be close[t+bars]/close[t] - 1, NaN in the final bars."""
    frame = _sample_ohlcv_frame()
    # Closes are [100, 101, 103, 104, 106].

    one_bar = next_n_bar_return(frame, bars=1)
    two_bar = next_n_bar_return(frame, bars=2)

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


def test_next_n_bar_realized_vol_is_forward_std_of_log_returns() -> None:
    """Vol target should be std of the next `bars` log returns, NaN at the tail."""
    frame = _sample_ohlcv_frame()
    # Closes are [100, 101, 103, 104, 106], so the one-bar log returns are
    # r_1 = ln(101/100), r_2 = ln(103/101), r_3 = ln(104/103), r_4 = ln(106/104).

    two_bar_vol = next_n_bar_realized_volatility(frame, bars=2)

    # Row 0 looks forward at (r_1, r_2); row 2 looks forward at (r_3, r_4).
    expected_row0 = float(
        pd.Series([math.log(101.0 / 100.0), math.log(103.0 / 101.0)]).std()
    )
    expected_row2 = float(
        pd.Series([math.log(104.0 / 103.0), math.log(106.0 / 104.0)]).std()
    )
    assert math.isclose(two_bar_vol.iloc[0], expected_row0)
    assert math.isclose(two_bar_vol.iloc[2], expected_row2)

    # The final `bars` rows have incomplete future windows and must be NaN.
    assert pd.isna(two_bar_vol.iloc[3])
    assert pd.isna(two_bar_vol.iloc[4])

    # bars = 1 is rejected: the std of a single return is undefined.
    with pytest.raises(ValueError):
        next_n_bar_realized_volatility(frame, bars=1)


def test_trend_features_match_manual_formulas() -> None:
    """Trend features should expose simple moving-average and lagged-return math."""
    frame = _sample_ohlcv_frame()

    moving_average_values = moving_average(frame, window=3)
    price_vs_sma_values = price_vs_moving_average(frame, window=3)
    rate_of_change_values = rate_of_change(frame, periods=2)

    expected_sma = (100.0 + 101.0 + 103.0) / 3.0
    assert pd.isna(moving_average_values.iloc[0])
    assert pd.isna(moving_average_values.iloc[1])
    assert math.isclose(moving_average_values.iloc[2], expected_sma)
    assert math.isclose(price_vs_sma_values.iloc[2], 103.0 / expected_sma - 1.0)
    assert math.isclose(rate_of_change_values.iloc[2], 103.0 / 100.0 - 1.0)


def test_volatility_features_match_manual_formulas() -> None:
    """Volatility features should measure return dispersion and bar range size."""
    frame = _sample_ohlcv_frame()

    rolling_values = rolling_standard_deviation(frame, window=3)
    range_values = bar_range_percent(frame)

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

    rolling_values = rolling_standard_deviation(frame, window=3)

    recent_prices = pd.Series([103.0, 104.0, 106.0])
    recent_log_returns = np.log(recent_prices / recent_prices.shift(1))
    expected_std = float(recent_log_returns.std())

    assert math.isclose(rolling_values.iloc[4], expected_std)


def test_rolling_standard_deviation_rejects_two_price_window() -> None:
    """Two prices contain one return, which cannot have a sample deviation."""
    frame = _sample_ohlcv_frame()

    with pytest.raises(ValueError, match="window >= 3"):
        rolling_standard_deviation(frame, window=2)


def test_volume_features_match_manual_formulas() -> None:
    """Volume features should expose relative, dollar, and percent-change volume."""
    frame = _sample_ohlcv_frame()

    ratio_values = volume_ratio(frame, window=3)
    dollar_values = dollar_volume(frame)
    change_values = volume_change(frame)

    expected_mean_volume = (1000.0 + 1200.0 + 1800.0) / 3.0

    assert pd.isna(ratio_values.iloc[0])
    assert pd.isna(ratio_values.iloc[1])
    assert math.isclose(ratio_values.iloc[2], 1800.0 / expected_mean_volume)
    assert math.isclose(dollar_values.iloc[2], 103.0 * 1800.0)
    assert math.isclose(change_values.iloc[1], 1200.0 / 1000.0 - 1.0)


def _triple_barrier_frame(closes: list[float]) -> pd.DataFrame:
    """Build a single-symbol frame whose only meaningful column is close."""
    timestamps = pd.date_range("2024-01-02", periods=len(closes), freq="D")
    close = pd.Series(closes, dtype="float64")
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(closes),
            "timestamp": timestamps,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000.0] * len(closes),
        }
    )


def test_triple_barrier_label_first_touch_within_time_barrier() -> None:
    """Label, bars-to-exit, and exit return must agree on the first barrier hit."""
    # Three warm-up prices give sigma at row 2 from returns ln(101/100) and
    # ln(100/101). From row 2 (close 100) the path is 101, 103, 104, 97, 100, 100.
    closes = [100.0, 101.0, 100.0, 101.0, 103.0, 104.0, 97.0, 100.0, 100.0]
    frame = _triple_barrier_frame(closes)
    sigma_row2 = float(
        pd.Series([math.log(101.0 / 100.0), math.log(100.0 / 101.0)]).std()
    )
    # Barrier width 2 * sigma is about 2.8%: 101 (+1.0%) misses, 103 (+3.0%)
    # hits the upper barrier at step 2.
    assert 0.02 < 2.0 * sigma_row2 < 0.03

    outcome = scan_triple_barrier(
        frame, max_bars=3, upper_multiple=2.0, lower_multiple=2.0, volatility_window=3
    )

    assert pd.isna(outcome.label.iloc[0]) and pd.isna(outcome.label.iloc[1])
    assert outcome.label.iloc[2] == 1.0
    assert outcome.bars_to_exit.iloc[2] == 2.0
    assert math.isclose(outcome.exit_return.iloc[2], 103.0 / 100.0 - 1.0)

    # Row 5 (close 104): next closes 97, 100, 100. 97 is -6.7%, a stop hit at
    # step 1 regardless of the later recovery.
    assert outcome.label.iloc[5] == -1.0
    assert outcome.bars_to_exit.iloc[5] == 1.0
    assert math.isclose(outcome.exit_return.iloc[5], 97.0 / 104.0 - 1.0)

    # The last max_bars rows have no full window and stay NaN in every output.
    for position in (6, 7, 8):
        assert pd.isna(outcome.label.iloc[position])
        assert pd.isna(outcome.bars_to_exit.iloc[position])
        assert pd.isna(outcome.exit_return.iloc[position])


def test_triple_barrier_label_is_zero_when_no_barrier_is_reached() -> None:
    """A quiet path inside both barriers exits at the time barrier with label 0."""
    # Sigma at row 2 comes from +5% then -4.8% returns, so 2 * sigma is far
    # wider than the later 0.1% moves.
    closes = [100.0, 105.0, 100.0, 100.1, 100.2, 100.1, 100.3, 100.2]
    frame = _triple_barrier_frame(closes)

    outcome = scan_triple_barrier(
        frame, max_bars=3, upper_multiple=2.0, lower_multiple=2.0, volatility_window=3
    )

    assert outcome.label.iloc[2] == 0.0
    assert outcome.bars_to_exit.iloc[2] == 3.0
    assert math.isclose(outcome.exit_return.iloc[2], 100.1 / 100.0 - 1.0)

    # Registered wrappers return the matching column of the same scan.
    label = triple_barrier_label(frame, max_bars=3, volatility_window=3)
    bars = triple_barrier_bars_to_exit(frame, max_bars=3, volatility_window=3)
    exit_return = triple_barrier_exit_return(frame, max_bars=3, volatility_window=3)
    pd.testing.assert_series_equal(label, outcome.label)
    pd.testing.assert_series_equal(bars, outcome.bars_to_exit)
    pd.testing.assert_series_equal(exit_return, outcome.exit_return)


def test_triple_barrier_label_ignores_bars_after_the_first_touch() -> None:
    """Changing closes beyond the first touch must not change the label."""
    base = [100.0, 101.0, 100.0, 101.0, 103.0, 104.0, 97.0, 100.0, 100.0]
    perturbed = base.copy()
    perturbed[5] = 50.0  # after row 2's touch at step 2 (index 4)

    base_label = triple_barrier_label(
        _triple_barrier_frame(base), max_bars=3, volatility_window=3
    )
    perturbed_label = triple_barrier_label(
        _triple_barrier_frame(perturbed), max_bars=3, volatility_window=3
    )
    assert base_label.iloc[2] == perturbed_label.iloc[2] == 1.0


def test_triple_barrier_rejects_invalid_parameters_and_flat_volatility() -> None:
    """Parameter guards fire, and a zero-volatility row is undefined (NaN)."""
    frame = _triple_barrier_frame([100.0] * 6)
    with pytest.raises(ValueError):
        triple_barrier_label(frame, max_bars=0)
    with pytest.raises(ValueError):
        triple_barrier_label(frame, upper_multiple=0.0)
    with pytest.raises(ValueError):
        triple_barrier_label(frame, volatility_window=2)

    flat = triple_barrier_label(frame, max_bars=1, volatility_window=3)
    assert flat.isna().all()
