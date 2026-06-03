"""Tests for the simplified load-clean-engineer-export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from feature_engineering.pipeline.clean import clean_ohlcv
from feature_engineering.pipeline.engineer import compute_features
from feature_engineering.pipeline.export import export_features
from feature_engineering.pipeline.load import load_ohlcv


def _raw_frame() -> pd.DataFrame:
    """Build toy raw OHLCV data with one invalid row for cleaning tests."""
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
        ]
    ).tz_localize("America/New_York")

    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "ts": timestamps,
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [101.0, 102.0, 199.0, 202.0],
            "low": [99.0, 100.0, 201.0, 200.0],
            "close": [100.0, 102.0, 200.0, 202.0],
            "volume": [1000.0, 1200.0, 1500.0, 1800.0],
        }
    )


def test_load_ohlcv_reads_local_csv_and_filters_symbols(tmp_path: Path) -> None:
    """Local CSV loading should support deterministic small runs without ClickHouse."""
    csv_path = tmp_path / "prices.csv"
    _raw_frame().to_csv(csv_path, index=False)
    config = {
        "run": {
            "source": "csv",
            "input_path": str(csv_path),
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-02",
        }
    }

    loaded = load_ohlcv(config)

    assert loaded["symbol"].tolist() == ["AAPL", "AAPL"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["ts"])


def test_clickhouse_loader_rejects_unsafe_table_identifier_before_querying() -> None:
    """ClickHouse table names should be validated before any SQL query is built."""
    config = {
        "run": {
            "source": "clickhouse",
            "table": "stocks;DROP TABLE stocks",
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        }
    }

    with pytest.raises(ValueError, match="Invalid ClickHouse table identifier"):
        load_ohlcv(config)


def test_clean_ohlcv_drops_invalid_price_rows() -> None:
    """Cleaning should drop only rows that violate simple OHLCV invariants."""
    cleaned, report = clean_ohlcv(_raw_frame())

    assert cleaned["symbol"].tolist() == ["AAPL", "AAPL", "MSFT"]
    assert report["initial_rows"] == 4
    assert report["final_rows"] == 3
    assert report["rules"]["drop_high_lt_low"]["dropped"] == 1


def test_clean_ohlcv_drops_missing_numeric_rows() -> None:
    """Missing numeric OHLCV values should not survive into feature calculations."""
    frame = _raw_frame()
    frame.loc[0, "close"] = float("nan")

    cleaned, report = clean_ohlcv(frame)

    assert cleaned["symbol"].tolist() == ["AAPL", "MSFT"]
    assert report["rules"]["drop_missing_numeric_values"]["dropped"] == 1


def test_compute_features_respects_category_filters() -> None:
    """Feature engineering should compute only included non-excluded categories."""
    cleaned, _report = clean_ohlcv(_raw_frame())
    config = {
        "features": {
            "include_categories": ["returns", "trend"],
            "exclude_categories": ["target"],
            "params": [
                {"name": "log_return", "fn": "log_return", "enabled": True},
                {"name": "ma_2", "fn": "moving_average", "window": 2, "enabled": True},
                {"name": "range", "fn": "bar_range_pct", "enabled": True},
                {
                    "name": "next_day",
                    "fn": "next_n_day_return",
                    "days": 1,
                    "enabled": True,
                },
            ],
        }
    }

    featured = compute_features(cleaned, config)

    assert "log_return" in featured.columns
    assert "ma_2" in featured.columns
    assert "range" not in featured.columns
    assert "next_day" not in featured.columns


def test_compute_features_keeps_symbol_histories_separate() -> None:
    """Rolling features should reset when the symbol changes."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "ts": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:31:00",
                ]
            ),
            "open": [100.0, 102.0, 200.0, 204.0],
            "high": [101.0, 103.0, 201.0, 205.0],
            "low": [99.0, 101.0, 199.0, 203.0],
            "close": [100.0, 102.0, 200.0, 204.0],
            "volume": [1000.0, 1200.0, 1500.0, 1800.0],
        }
    )
    config = {
        "features": {
            "params": [
                {"name": "ma_2", "fn": "moving_average", "window": 2},
            ],
        }
    }

    featured = compute_features(frame, config)

    # The first row for each symbol has no two-row history, so a cross-symbol
    # leak would show up as a non-null value on the first MSFT row.
    assert pd.isna(featured.loc[0, "ma_2"])
    assert featured.loc[1, "ma_2"] == 101.0
    assert pd.isna(featured.loc[2, "ma_2"])
    assert featured.loc[3, "ma_2"] == 202.0


def test_export_features_writes_dataset_and_catalog(tmp_path: Path) -> None:
    """Export should write feature data plus a small readable catalog."""
    featured = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": pd.to_datetime(["2024-01-02 09:30:00"]),
            "log_return": [0.01],
        }
    )
    config = {
        "run": {
            "output_dir": str(tmp_path),
            "output_formats": ["csv", "parquet"],
            "version": "test",
        },
        "features": {
            "params": [
                {"name": "log_return", "fn": "log_return", "enabled": True},
            ]
        },
    }

    paths = export_features(featured, config)

    assert paths["csv"].exists()
    assert paths["parquet"].exists()
    assert paths["csv"].stem.startswith("features_vtest_")
    timestamp_text = paths["csv"].stem.removeprefix("features_vtest_")
    assert len(timestamp_text) == len("20240102_093000_123456")
    assert paths["catalog_csv"].exists()

    summary = json.loads(paths["summary_json"].read_text())
    assert summary["rows"] == 1
    assert summary["features"] == ["log_return"]
