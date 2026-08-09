"""OHLCV schema and market-data validation tests for the load stage."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.pipeline.clean import clean_ohlcv
from feature_engineering.pipeline.load import _finalize_ohlcv_frame

EXCHANGE_TIMEZONE = "America/New_York"


def _raw_frame() -> pd.DataFrame:
    """Build an unsorted OHLCV frame with parseable string values."""
    return pd.DataFrame(
        {
            "symbol": [" MSFT ", "AAPL"],
            "ts": ["2024-01-02 09:31:00", "2024-01-02 09:30:00"],
            "open": ["201.0", "100.0"],
            "high": ["202.0", "101.0"],
            "low": ["200.0", "99.0"],
            "close": ["201.5", "100.5"],
            "volume": ["1200", "1000"],
        }
    )


def test_finalize_ohlcv_frame_normalizes_schema_and_sorting() -> None:
    """Finalized OHLCV data should be typed, stripped, sorted, and reindexed."""
    finalized = _finalize_ohlcv_frame(_raw_frame(), exchange_timezone=EXCHANGE_TIMEZONE)

    assert finalized["symbol"].tolist() == ["AAPL", "MSFT"]
    assert finalized.index.tolist() == [0, 1]
    assert pd.api.types.is_datetime64_any_dtype(finalized["ts"])
    assert pd.api.types.is_float_dtype(finalized["close"])


def test_missing_symbol_values_are_rejected() -> None:
    """Empty or whitespace-only symbols would create phantom feature groups."""
    frame = _raw_frame()
    frame.loc[0, "symbol"] = "   "

    with pytest.raises(ValueError, match="symbol values must be present and non-empty"):
        _finalize_ohlcv_frame(frame, exchange_timezone=EXCHANGE_TIMEZONE)


def test_duplicate_symbol_timestamp_bars_are_rejected() -> None:
    """Duplicate bars should not enter feature computation."""
    frame = _raw_frame()
    frame.loc[1, "symbol"] = "MSFT"
    frame.loc[1, "ts"] = "2024-01-02 09:31:00"

    with pytest.raises(
        ValueError,
        match="Duplicate OHLCV bars found for symbol/timestamp pairs",
    ):
        _finalize_ohlcv_frame(frame, exchange_timezone=EXCHANGE_TIMEZONE)


def test_negative_volume_is_dropped_by_cleaning() -> None:
    """Negative volume is invalid market data and should be removed."""
    frame = _finalize_ohlcv_frame(_raw_frame(), exchange_timezone=EXCHANGE_TIMEZONE)
    frame.loc[frame["symbol"] == "AAPL", "volume"] = -1.0

    cleaned, report = clean_ohlcv(frame)

    assert cleaned["symbol"].tolist() == ["MSFT"]
    assert report["rules"]["drop_negative_volume"]["dropped"] == 1


def test_zero_volume_is_kept_by_cleaning() -> None:
    """Zero volume is a valid no-trades bar and must survive cleaning."""
    frame = _finalize_ohlcv_frame(_raw_frame(), exchange_timezone=EXCHANGE_TIMEZONE)
    frame.loc[frame["symbol"] == "AAPL", "volume"] = 0.0

    cleaned, _report = clean_ohlcv(frame)

    assert cleaned["symbol"].tolist() == ["AAPL", "MSFT"]
