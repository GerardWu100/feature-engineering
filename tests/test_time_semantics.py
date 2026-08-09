"""Timezone and session semantics tests for CSV data loading.

These tests pin two loading contracts:

1. Naive input timestamps are exchange-local wall-clock time; timezone-aware
   inputs are converted to the exchange timezone and stored naive.
2. The ``run.session`` filter applies to CSV loads. CSV runs default to
   ``full`` so daily bars stamped at midnight survive, but an explicit
   session is always honored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from feature_engineering.pipeline.load import load_ohlcv


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    """Write one OHLCV CSV with the standard header plus the given rows."""
    csv_path = tmp_path / "prices.csv"
    header = "symbol,ts,open,high,low,close,volume"
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


def test_rth_session_filter_applies_to_csv_loads(tmp_path: Path) -> None:
    """An explicit rth session must drop pre-market bars from CSV data."""
    csv_path = _write_csv(
        tmp_path,
        [
            # 08:00 New York: pre-market, outside regular trading hours.
            "AAPL,2024-01-02 08:00:00,99,100,98,99,900",
            "AAPL,2024-01-02 09:30:00,100,101,99,100,1000",
            "AAPL,2024-01-02 09:31:00,101,102,100,101,1100",
        ],
    )

    loaded = load_ohlcv(_csv_config(csv_path, session="rth"))

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
            # 13:00 UTC is 08:00 New York: pre-market, excluded from rth.
            "AAPL,2024-01-02 13:00:00+00:00,99,100,98,99,900",
            # 14:30 UTC is 09:30 New York: the first regular-session bar.
            "AAPL,2024-01-02 14:30:00+00:00,100,101,99,100,1000",
            # Naive timestamps are already exchange-local.
            "AAPL,2024-01-02 09:31:00,101,102,100,101,1100",
        ],
    )

    loaded = load_ohlcv(_csv_config(csv_path, session="rth"))

    assert loaded["close"].tolist() == [100.0, 101.0]


def test_loaded_timestamps_are_naive_exchange_local(tmp_path: Path) -> None:
    """Aware inputs are converted to the exchange timezone and stored naive."""
    csv_path = _write_csv(
        tmp_path,
        ["AAPL,2024-01-02 14:30:00+00:00,100,101,99,100,1000"],
    )

    loaded = load_ohlcv(_csv_config(csv_path))

    timestamp = loaded["ts"].iloc[0]
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
