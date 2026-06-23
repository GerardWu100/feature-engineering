"""Tests for the batch FeatureEngine and the incremental OnlineFeatureEngine.

The central test is equivalence: feeding the same bars through the batch feature
functions and through the online accumulators must produce the same numbers.
That contract is what lets the two implementations evolve without drifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.engine import FeatureEngine, OnlineFeatureEngine
from feature_engineering.pipeline.engineer import compute_features

# Every non-target feature with small windows so warmup finishes inside the
# sample. These params are shared by the batch and online paths under test.
ALL_ONLINE_FEATURES = {
    "features": {
        "params": [
            {"name": "log_return", "fn": "log_return"},
            {"name": "simple_return", "fn": "simple_return"},
            {"name": "volume_change", "fn": "volume_change"},
            {"name": "bar_range_pct", "fn": "bar_range_pct"},
            {"name": "dollar_volume", "fn": "dollar_volume"},
            {"name": "ma_5", "fn": "moving_average", "window": 5},
            {"name": "price_vs_sma_10", "fn": "price_vs_sma", "window": 10},
            {"name": "roc_4", "fn": "rate_of_change", "periods": 4},
            {"name": "rolling_std_6", "fn": "rolling_std", "window": 6},
            {"name": "volume_ratio_5", "fn": "volume_ratio", "window": 5},
            {"name": "rsi_14", "fn": "relative_strength_index", "window": 14},
            {"name": "atr_10", "fn": "average_true_range", "window": 10},
            {"name": "macd", "fn": "macd_line", "fast": 4, "slow": 8},
            {
                "name": "macd_sig",
                "fn": "macd_signal",
                "fast": 4,
                "slow": 8,
                "signal": 3,
            },
            {
                "name": "macd_hist",
                "fn": "macd_histogram",
                "fast": 4,
                "slow": 8,
                "signal": 3,
            },
            {"name": "vwap", "fn": "vwap"},
            {"name": "price_vs_vwap", "fn": "price_vs_vwap"},
        ]
    }
}


def _make_symbol_frame(
    symbol: str, n_bars: int, start_price: float, seed: int
) -> pd.DataFrame:
    """Build a deterministic single-symbol intraday OHLCV frame for one day."""
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(0.0, 0.01, n_bars)
    close = start_price * np.exp(np.cumsum(log_steps))
    high = close * (1.0 + rng.uniform(0.0, 0.01, n_bars))
    low = close * (1.0 - rng.uniform(0.0, 0.01, n_bars))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n_bars))
    volume = rng.integers(1_000, 5_000, n_bars).astype(float)
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=n_bars, freq="min")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _multi_symbol_frame() -> pd.DataFrame:
    """Two symbols, single continuous session, enough bars to clear all warmups."""
    aapl = _make_symbol_frame("AAPL", n_bars=80, start_price=100.0, seed=1)
    msft = _make_symbol_frame("MSFT", n_bars=80, start_price=300.0, seed=2)
    return pd.concat([aapl, msft], ignore_index=True)


def test_online_engine_matches_batch_for_every_feature() -> None:
    """Online accumulators must reproduce the batch feature values bar for bar."""
    frame = _multi_symbol_frame()

    batch = compute_features(frame, ALL_ONLINE_FEATURES)
    online = OnlineFeatureEngine(ALL_ONLINE_FEATURES).stream_frame(frame)

    # Both paths return rows in (symbol, ts) order, so columns line up directly.
    assert list(online.columns) == list(batch.columns)
    for column in batch.columns:
        if column in ("symbol", "ts"):
            assert online[column].tolist() == batch[column].tolist()
            continue
        np.testing.assert_allclose(
            online[column].to_numpy(dtype=float),
            batch[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"online/batch mismatch in column {column}",
        )


def test_feature_engine_transform_matches_compute_features() -> None:
    """The cached batch wrapper must equal the one-shot compute_features."""
    frame = _multi_symbol_frame()

    wrapper = FeatureEngine(ALL_ONLINE_FEATURES).transform(frame)
    one_shot = compute_features(frame, ALL_ONLINE_FEATURES)

    pd.testing.assert_frame_equal(wrapper, one_shot)


def test_feature_engine_resolves_and_caches_feature_names() -> None:
    """The engine should expose its output columns and reject unknown features."""
    engine = FeatureEngine(ALL_ONLINE_FEATURES)
    assert "rsi_14" in engine.feature_names
    assert "vwap" in engine.feature_names

    with pytest.raises(KeyError):
        FeatureEngine({"features": {"params": [{"name": "x", "fn": "does_not_exist"}]}})


def test_online_engine_rejects_forward_looking_targets() -> None:
    """A target feature has no honest online value and must be refused up front."""
    config = {
        "features": {
            "params": [{"name": "y", "fn": "next_n_bar_return", "bars": 1}],
        }
    }
    with pytest.raises(ValueError, match="forward-looking target"):
        OnlineFeatureEngine(config)


def test_rsi_stays_within_bounds_and_atr_is_positive() -> None:
    """RSI must live in [0, 100]; ATR is a non-negative price range."""
    frame = _make_symbol_frame("AAPL", n_bars=80, start_price=100.0, seed=3)
    config = {
        "features": {
            "params": [
                {"name": "rsi", "fn": "relative_strength_index", "window": 14},
                {"name": "atr", "fn": "average_true_range", "window": 14},
            ]
        }
    }

    featured = compute_features(frame, config)
    rsi = featured["rsi"].dropna()
    atr = featured["atr"].dropna()

    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()
    assert (atr > 0.0).all()


def test_macd_histogram_equals_line_minus_signal() -> None:
    """The histogram is defined as MACD line minus signal line."""
    frame = _make_symbol_frame("AAPL", n_bars=80, start_price=100.0, seed=4)
    config = {
        "features": {
            "params": [
                {"name": "line", "fn": "macd_line", "fast": 4, "slow": 8},
                {"name": "sig", "fn": "macd_signal", "fast": 4, "slow": 8, "signal": 3},
                {
                    "name": "hist",
                    "fn": "macd_histogram",
                    "fast": 4,
                    "slow": 8,
                    "signal": 3,
                },
            ]
        }
    }

    featured = compute_features(frame, config)
    expected = (featured["line"] - featured["sig"]).to_numpy(dtype=float)
    np.testing.assert_allclose(
        featured["hist"].to_numpy(dtype=float), expected, equal_nan=True
    )


def test_session_reset_resets_vwap_and_matches_online() -> None:
    """With reset_by_session, VWAP restarts each day in both batch and online."""
    day_one = _make_symbol_frame("AAPL", n_bars=10, start_price=100.0, seed=5)
    day_two = _make_symbol_frame("AAPL", n_bars=10, start_price=120.0, seed=6)
    # Shift day two onto the next calendar date but keep the same clock times.
    day_two = day_two.assign(ts=day_two["ts"] + pd.Timedelta(days=1))
    frame = pd.concat([day_one, day_two], ignore_index=True)

    config = {
        "features": {
            "reset_by_session": True,
            "params": [
                {"name": "vwap", "fn": "vwap"},
                {"name": "ma_3", "fn": "moving_average", "window": 3},
            ],
        }
    }

    batch = compute_features(frame, config)
    online = OnlineFeatureEngine(config).stream_frame(frame)

    # The first bar of day two (row 10) restarts VWAP at its own typical price,
    # so VWAP equals (high+low+close)/3 of that single bar, not a value carried
    # over from day one.
    first_day_two_bar = frame.iloc[10]
    expected_vwap = (
        first_day_two_bar["high"]
        + first_day_two_bar["low"]
        + first_day_two_bar["close"]
    ) / 3.0
    assert np.isclose(batch.loc[10, "vwap"], expected_vwap)
    # The 3-bar moving average also resets: row 10 has no same-day history.
    assert pd.isna(batch.loc[10, "ma_3"])

    np.testing.assert_allclose(
        online["vwap"].to_numpy(dtype=float),
        batch["vwap"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )
