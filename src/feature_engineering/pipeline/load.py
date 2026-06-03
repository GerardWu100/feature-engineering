"""Load stock OHLCV data for the simple feature pipeline.

The loader supports two sources:

``csv``
    Local file loading for tests, examples, and small research runs.
``clickhouse``
    Database loading for the FirstRate stock table used by the original project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from feature_engineering.pipeline.constants import NUMERIC_OHLCV_COLUMNS, OHLCV_COLUMNS, SQL_IDENTIFIER_PATTERN

EXTENDED_SESSION_START_MINUTE = 4 * 60
EXTENDED_SESSION_END_MINUTE = 19 * 60 + 59
RTH_SESSION_START_MINUTE = 9 * 60 + 30
RTH_SESSION_END_MINUTE = 15 * 60 + 59


def load_ohlcv(config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data according to the ``[run]`` config section.

    Parameters
    ----------
    config
        Project config. ``config["run"]["source"]`` may be ``csv`` or
        ``clickhouse``. If omitted, ``clickhouse`` is used.

    Returns
    -------
    pandas.DataFrame
        Sorted OHLCV data with columns ``symbol``, ``ts``, ``open``, ``high``,
        ``low``, ``close``, and ``volume``.
    """
    run_config = config["run"]
    source = run_config.get("source", "clickhouse")

    if source == "csv":
        return _load_csv(run_config)

    if source == "clickhouse":
        return _load_clickhouse(run_config)

    raise ValueError(f"Unsupported data source: {source}")


def _load_csv(run_config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file and apply basic run filters."""
    input_path = Path(run_config["input_path"])
    frame = pd.read_csv(input_path)

    # Parse timestamps immediately so sorting, date filtering, and feature
    # windows all operate on real datetime values.
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame = _filter_frame(frame, run_config)
    return _finalize_ohlcv_frame(frame)


def _load_clickhouse(run_config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from ClickHouse using environment variables."""
    symbols = _validated_symbols(run_config["symbols"])
    table = _validated_sql_identifier(str(run_config.get("table", "stocks")), "table")
    start_date = pd.Timestamp(run_config["start_date"]).date()
    end_date = pd.Timestamp(run_config["end_date"]).date()
    session_filter = _session_filter_sql(run_config.get("session", "rth"))
    client = _build_clickhouse_client_from_env()

    # Regular trading hours are the default because most intraday feature
    # experiments should avoid thin pre-market and after-hours bars.
    query = f"""
        SELECT symbol, ts, open, high, low, close, volume
        FROM firstrate.{table}
        WHERE symbol IN %(symbols)s
          AND toDate(ts) >= toDate(%(start_date)s)
          AND toDate(ts) <= toDate(%(end_date)s)
          {session_filter}
        ORDER BY symbol, ts
    """

    query_parameters = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
    }
    result = client.query_df(query, parameters=query_parameters)
    return _finalize_ohlcv_frame(result)


def _validated_symbols(symbols: list[Any]) -> list[str]:
    """Return non-empty symbol strings safe for parameterized ClickHouse queries."""
    validated = [str(symbol).strip() for symbol in symbols]
    if not validated:
        raise ValueError("ClickHouse loading requires at least one symbol.")

    for symbol in validated:
        if not symbol:
            raise ValueError("ClickHouse symbols cannot be empty strings.")

    return validated


def _validated_sql_identifier(value: str, label: str) -> str:
    """Return a simple ClickHouse identifier after rejecting unsafe characters."""
    if not SQL_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid ClickHouse {label} identifier: {value}")

    return value


def _session_filter_sql(session: str) -> str:
    """Return a ClickHouse SQL filter for the requested trading session."""
    if session == "full":
        return ""

    if session == "extended":
        # 04:00 through 19:59, useful for pre-market and after-hours studies.
        return (
            "AND (toHour(ts) * 60 + toMinute(ts)) "
            f"BETWEEN {EXTENDED_SESSION_START_MINUTE} AND {EXTENDED_SESSION_END_MINUTE}"
        )

    if session == "rth":
        # 09:30 through 15:59 regular trading hours.
        return (
            "AND (toHour(ts) * 60 + toMinute(ts)) "
            f"BETWEEN {RTH_SESSION_START_MINUTE} AND {RTH_SESSION_END_MINUTE}"
        )

    raise ValueError(f"Unsupported session filter: {session}")


def _filter_frame(frame: pd.DataFrame, run_config: dict[str, Any]) -> pd.DataFrame:
    """Apply symbol and date filters to a local OHLCV frame."""
    filtered = frame.copy()

    symbols = run_config.get("symbols")
    if symbols:
        # Symbol filtering runs first so date comparisons touch fewer rows.
        filtered = filtered[filtered["symbol"].isin(symbols)]

    if "start_date" in run_config:
        start = pd.Timestamp(run_config["start_date"]).date()
        filtered = filtered[filtered["ts"].dt.date >= start]

    if "end_date" in run_config:
        end = pd.Timestamp(run_config["end_date"]).date()
        filtered = filtered[filtered["ts"].dt.date <= end]

    return filtered


def _build_clickhouse_client_from_env() -> Any:
    """Build a ClickHouse client using .env overrides with sensible defaults."""
    import clickhouse_connect

    # Load project-level environment values so local research runs can switch
    # hosts and credentials without editing code.
    load_dotenv()

    client_options = {
        "host": os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "8123")),
        "username": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        "secure": os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
        "verify": os.getenv("CLICKHOUSE_VERIFY", "false").lower() == "true",
    }
    return clickhouse_connect.get_client(**client_options)


def _finalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select standard columns, coerce types, and sort by symbol and time."""
    missing_columns = [
        column for column in OHLCV_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing OHLCV columns: {missing_columns}")

    finalized = frame.loc[:, OHLCV_COLUMNS].copy()
    finalized["ts"] = pd.to_datetime(finalized["ts"])

    numeric_columns = NUMERIC_OHLCV_COLUMNS
    for column in numeric_columns:
        finalized[column] = pd.to_numeric(finalized[column], errors="coerce")

    # Stable ordering is a contract for all feature functions.
    finalized = finalized.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return finalized
