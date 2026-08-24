"""Timezone and session semantics tests for CSV data loading.

These tests pin two loading contracts:

1. Naive input timestamps are exchange-local wall-clock time; timezone-aware
   inputs are converted to the exchange timezone and stored naive.
2. The ``run.session`` filter applies to CSV loads. CSV runs default to
   ``full`` so daily bars stamped at midnight survive, but an explicit
   session is always honored.
3. ClickHouse loads re-check the session window after converting timestamps to
   exchange-local time, so a source column stored in another timezone fails at
   load time instead of silently returning the wrong part of the day. Checked
   here against synthetic frames; no database connection is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from feature_engineering.engineering.load import (
    _verify_session_contract,
    load_ohlcv,
)


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    """Write one OHLCV CSV with the standard header plus the given rows."""
    csv_path = tmp_path / "prices.csv"
    header = "symbol,timestamp,open,high,low,close,volume"
    csv_path.write_text("\n".join([header, *rows]), encoding="utf-8")
    return csv_path


def _csv_config(csv_path: Path, **run_overrides: Any) -> dict[str, Any]:
    """Return a minimal CSV run config with optional extra run keys."""
    run_config: dict[str, Any] = {
        "source": "csv",
        "input_path": str(csv_path),
        "symbols": ["AAPL"],
        "start_date": "2024-01-02",
        "end_date": "2024-01-02",
        "exchange_timezone": "America/New_York",
    }
    run_config.update(run_overrides)
    return {"run": run_config}


def test_regular_session_filter_applies_to_csv_loads(tmp_path: Path) -> None:
    """An explicit regular session must drop pre-market bars from CSV data."""
    csv_path = _write_csv(
        tmp_path,
        [
            # 08:00 New York: pre-market, outside regular trading hours.
            "AAPL,2024-01-02 08:00:00,99,100,98,99,900",
            "AAPL,2024-01-02 09:30:00,100,101,99,100,1000",
            "AAPL,2024-01-02 09:31:00,101,102,100,101,1100",
        ],
    )

    loaded = load_ohlcv(_csv_config(csv_path, session="regular"))

    assert loaded["close"].tolist() == [100.0, 101.0]


def test_csv_session_defaults_to_full_so_daily_bars_survive(
    tmp_path: Path,
) -> None:
    """Without run.session, midnight-stamped daily bars must not be dropped."""
    csv_path = _write_csv(
        tmp_path,
        ["AAPL,2024-01-02 00:00:00,100,101,99,100,1000"],
    )

    loaded = load_ohlcv(_csv_config(csv_path))

    assert len(loaded) == 1


def test_regular_session_uses_exchange_timezone_for_utc_timestamps(
    tmp_path: Path,
) -> None:
    """UTC timestamps are included or excluded by New York clock time."""
    csv_path = _write_csv(
        tmp_path,
        [
            # 13:00 UTC is 08:00 New York: pre-market, excluded from regular.
            "AAPL,2024-01-02 13:00:00+00:00,99,100,98,99,900",
            # 14:30 UTC is 09:30 New York: the first regular-session bar.
            "AAPL,2024-01-02 14:30:00+00:00,100,101,99,100,1000",
            # Naive timestamps are already exchange-local.
            "AAPL,2024-01-02 09:31:00,101,102,100,101,1100",
        ],
    )

    loaded = load_ohlcv(_csv_config(csv_path, session="regular"))

    assert loaded["close"].tolist() == [100.0, 101.0]


def test_loaded_timestamps_are_naive_exchange_local(tmp_path: Path) -> None:
    """Aware inputs are converted to the exchange timezone and stored naive."""
    csv_path = _write_csv(
        tmp_path,
        ["AAPL,2024-01-02 14:30:00+00:00,100,101,99,100,1000"],
    )

    loaded = load_ohlcv(_csv_config(csv_path))

    timestamp = loaded["timestamp"].iloc[0]
    assert timestamp.tzinfo is None
    assert str(timestamp) == "2024-01-02 09:30:00"


def test_duplicate_bars_after_timezone_normalization_are_rejected(
    tmp_path: Path,
) -> None:
    """Equivalent UTC and local timestamps are the same bar, so loading fails."""
    csv_path = _write_csv(
        tmp_path,
        [
            # Both rows are 09:30 New York once normalized.
            "AAPL,2024-01-02 14:30:00+00:00,100,101,99,100,1000",
            "AAPL,2024-01-02 09:30:00,101,102,100,101,1100",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate OHLCV bars found for symbol/timestamp pairs",
    ):
        load_ohlcv(_csv_config(csv_path))


def _bars_at(local_times: list[str]) -> pd.DataFrame:
    """Return a minimal loader-shaped frame at the given exchange-local times."""
    return pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": pd.to_datetime(local_times),
        }
    )


def test_session_contract_accepts_bars_inside_the_window() -> None:
    """Exchange-local regular-hours bars satisfy the contract silently."""
    inside = _bars_at(["2024-01-02 09:30:00", "2024-01-02 15:59:00"])

    _verify_session_contract(inside, session="regular")


def test_session_contract_rejects_bars_shifted_out_of_the_window() -> None:
    """A UTC-typed source column lands outside regular hours once converted.

    14:30 UTC is 09:30 New York. A column typed ``DateTime('UTC')`` passes the
    database-side ``toHour`` filter on its UTC hour, but the loader converts it
    to exchange-local time, and 09:30 UTC (04:30 New York) is nowhere near the
    regular session the query asked for.
    """
    shifted = _bars_at(["2024-01-02 04:30:00", "2024-01-02 10:59:00"])

    with pytest.raises(ValueError, match="fall outside the 'regular' session"):
        _verify_session_contract(shifted, session="regular")


def test_session_contract_is_a_no_op_for_the_full_session() -> None:
    """``full`` applies no time-of-day filter, so there is nothing to verify."""
    any_hour = _bars_at(["2024-01-02 02:15:00", "2024-01-02 23:45:00"])

    _verify_session_contract(any_hour, session="full")
