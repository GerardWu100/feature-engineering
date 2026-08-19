"""Load stock OHLCV data for the simple feature pipeline.

The loader supports two sources:

``csv``
    Local file loading for tests, examples, and small research runs.
``clickhouse``
    Database loading for the FirstRate stock table used by the original project.

Data contract (assumptions every caller must satisfy)
-----------------------------------------------------
1. Adjusted prices. ``open``, ``high``, ``low``, and ``close`` must already be
   split- and dividend-adjusted. This pipeline does not apply corporate-action
   adjustments. Unadjusted prices make a split look like a large return and
   silently corrupt every return, trend, and volatility feature.
2. Exchange-local timestamps. All loaded ``timestamp`` values are normalized to naive
   exchange-local wall-clock time (``run.exchange_timezone``, default
   US/Eastern). Naive input timestamps are trusted to already be exchange-local;
   timezone-aware inputs are converted. The ClickHouse session filter below
   selects regular trading hours with ``toHour(ts)``/``toMinute(ts)``, so a
   database column stored in UTC would select the wrong bars. The intraday
   ``reset_by_session`` option in ``compute.py`` likewise groups by the local
   calendar date.
3. One row per symbol per bar, with consistent bar size across the run.
   Duplicate ``(symbol, timestamp)`` bars are rejected with an error because they
   would double-count rows inside every rolling window.

Known limitation: naive local timestamps cannot distinguish the repeated
01:00-01:59 hour on the autumn daylight-saving fall-back day. Timezone-aware
overnight data crossing that hour collapses onto the same wall-clock times and
is rejected as duplicate bars. The ``regular`` and ``extended`` sessions never
include that hour, so this only affects ``full``-session overnight data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from feature_engineering.engineering.constants import (
    DEFAULT_CLICKHOUSE_TABLE,
    DEFAULT_CSV_SESSION,
    DEFAULT_EXCHANGE_TIMEZONE,
    DEFAULT_SESSION,
    NUMERIC_OHLCV_COLUMNS,
    OHLCV_COLUMNS,
    SQL_IDENTIFIER_PATTERN,
    sort_by_symbol_and_time,
)

EXTENDED_SESSION_START_MINUTE = 4 * 60
EXTENDED_SESSION_END_MINUTE = 19 * 60 + 59
REGULAR_SESSION_START_MINUTE = 9 * 60 + 30
REGULAR_SESSION_END_MINUTE = 15 * 60 + 59

# Inclusive (start, end) minute-of-day bounds per supported trading session, or
# None for no time-of-day filter. This is the single source of truth for
# sessions: the SQL fragments below and the pandas mask for CSV loads are both
# derived from it, and config validation derives its allowed-session set from
# the SQL mapping, so adding a session here supports it end to end.
SESSION_MINUTE_RANGES: dict[str, tuple[int, int] | None] = {
    # All bars, no time-of-day filter.
    "full": None,
    # 04:00 through 19:59, useful for pre-market and after-hours studies.
    "extended": (EXTENDED_SESSION_START_MINUTE, EXTENDED_SESSION_END_MINUTE),
    # 09:30 through 15:59 regular trading hours.
    "regular": (REGULAR_SESSION_START_MINUTE, REGULAR_SESSION_END_MINUTE),
}

# Name of the timestamp column in the ClickHouse source table. It is aliased to
# ``timestamp`` on the way out (see the query below), so this abbreviation stays
# confined to the SQL that talks to the database.
CLICKHOUSE_TIMESTAMP_COLUMN = "ts"

SESSION_FILTER_SQL: dict[str, str] = {
    session: (
        ""
        if minute_range is None
        else (
            f"AND (toHour({CLICKHOUSE_TIMESTAMP_COLUMN}) * 60 "
            f"+ toMinute({CLICKHOUSE_TIMESTAMP_COLUMN})) "
            f"BETWEEN {minute_range[0]} AND {minute_range[1]}"
        )
    )
    for session, minute_range in SESSION_MINUTE_RANGES.items()
}


def load_ohlcv(config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data according to the ``[run]`` config section.

    Parameters
    ----------
    config
        Project config. ``config["run"]["source"]`` must be ``csv`` or
        ``clickhouse``.

    Returns
    -------
    pandas.DataFrame
        Sorted OHLCV data with columns ``symbol``, ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, and ``volume``.
    """
    run_config = config["run"]
    source = run_config["source"]

    if source == "csv":
        return _load_csv(run_config)

    if source == "clickhouse":
        return _load_clickhouse(run_config)

    raise ValueError(f"Unsupported data source: {source}")


def _load_csv(run_config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file and apply basic run filters."""
    input_path = Path(run_config["input_path"])
    frame = pd.read_csv(input_path)

    # Normalize schema and timestamps first so symbol, date, and session
    # filters all operate on validated exchange-local datetime values.
    exchange_timezone = run_config.get("exchange_timezone", DEFAULT_EXCHANGE_TIMEZONE)
    frame = _finalize_ohlcv_frame(frame, exchange_timezone=exchange_timezone)
    return _filter_frame(frame, run_config)


def _load_clickhouse(run_config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from ClickHouse using environment variables."""
    symbols = _validated_symbols(run_config["symbols"])
    table = _validated_sql_identifier(
        str(run_config.get("table", DEFAULT_CLICKHOUSE_TABLE)), "table"
    )
    start_date = pd.Timestamp(run_config["start_date"]).date()
    end_date = pd.Timestamp(run_config["end_date"]).date()

    session = run_config.get("session", DEFAULT_SESSION)
    if session not in SESSION_FILTER_SQL:
        raise ValueError(f"Unsupported session filter: {session}")
    session_filter = SESSION_FILTER_SQL[session]

    client = _build_clickhouse_client_from_env()

    # Regular trading hours are the default because most intraday feature
    # experiments should avoid thin pre-market and after-hours bars.
    # The database column is named ``ts``; it is aliased to ``timestamp`` here so
    # every frame inside this package uses the spelled-out name. The WHERE and
    # ORDER BY clauses must keep using ``ts`` because ClickHouse resolves them
    # against the source column, not the SELECT alias.
    query = f"""
        SELECT symbol, {CLICKHOUSE_TIMESTAMP_COLUMN} AS timestamp,
               open, high, low, close, volume
        FROM firstrate.{table}
        WHERE symbol IN %(symbols)s
          AND toDate({CLICKHOUSE_TIMESTAMP_COLUMN}) >= toDate(%(start_date)s)
          AND toDate({CLICKHOUSE_TIMESTAMP_COLUMN}) <= toDate(%(end_date)s)
          {session_filter}
        ORDER BY symbol, {CLICKHOUSE_TIMESTAMP_COLUMN}
    """

    query_parameters = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
    }
    result = client.query_df(query, parameters=query_parameters)
    exchange_timezone = run_config.get("exchange_timezone", DEFAULT_EXCHANGE_TIMEZONE)
    return _finalize_ohlcv_frame(result, exchange_timezone=exchange_timezone)


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


def _filter_frame(frame: pd.DataFrame, run_config: dict[str, Any]) -> pd.DataFrame:
    """Apply symbol, date, and session filters to a local OHLCV frame."""
    filtered = frame

    symbols = run_config.get("symbols")
    if symbols:
        # Symbol filtering runs first so date comparisons touch fewer rows.
        filtered = filtered[filtered["symbol"].isin(symbols)]

    if "start_date" in run_config or "end_date" in run_config:
        # Timestamps are already naive exchange-local, so calendar-date bounds
        # mean exchange trading dates. Materialize the date column once and
        # reuse it for both bounds.
        bar_dates = filtered["timestamp"].dt.date
        if "start_date" in run_config:
            start = pd.Timestamp(run_config["start_date"]).date()
            filtered = filtered[bar_dates.loc[filtered.index] >= start]
        if "end_date" in run_config:
            end = pd.Timestamp(run_config["end_date"]).date()
            filtered = filtered[bar_dates.loc[filtered.index] <= end]

    # CSV defaults to "full" (no time-of-day filter) because local files are
    # often daily bars stamped at midnight, which a "regular" default would drop
    # entirely. An explicit run.session is always honored.
    session = run_config.get("session", DEFAULT_CSV_SESSION)
    filtered = filtered[_session_mask(filtered["timestamp"], session)]

    return filtered.reset_index(drop=True)


def _session_mask(timestamps: pd.Series, session: str) -> pd.Series:
    """Return a boolean mask selecting bars inside one trading session.

    Parameters
    ----------
    timestamps
        Naive exchange-local timestamps, one per bar.
    session
        Key into ``SESSION_MINUTE_RANGES`` (``full``, ``extended``, or ``regular``).

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``timestamps`` selecting bars whose
        minute-of-day falls inside the session's inclusive bounds.
    """
    if session not in SESSION_MINUTE_RANGES:
        raise ValueError(f"Unsupported session filter: {session}")

    minute_range = SESSION_MINUTE_RANGES[session]
    if minute_range is None:
        return pd.Series(True, index=timestamps.index)

    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    return minute_of_day.between(minute_range[0], minute_range[1])


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


def _finalize_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    exchange_timezone: str,
) -> pd.DataFrame:
    """Select standard columns, validate identifiers, coerce types, and sort.

    Parameters
    ----------
    frame
        Raw loader output containing at least the standard OHLCV columns.
    exchange_timezone
        IANA timezone name used to convert timezone-aware timestamps to
        exchange-local wall-clock time. Naive timestamps are trusted to
        already be exchange-local.

    Returns
    -------
    pandas.DataFrame
        Standard OHLCV columns with stripped symbols, naive exchange-local
        timestamps, numeric price and volume columns, sorted by symbol and
        time with a fresh integer index.

    Raises
    ------
    KeyError
        If required OHLCV columns are missing.
    ValueError
        If symbols or timestamps are missing, or if duplicate
        ``(symbol, timestamp)`` bars exist after timezone normalization.
    """
    missing_columns = [
        column for column in OHLCV_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing OHLCV columns: {missing_columns}")

    finalized = frame.loc[:, OHLCV_COLUMNS].copy()

    # Whitespace-only or missing symbols would create phantom groups in every
    # per-symbol feature computation, so fail loudly instead.
    finalized["symbol"] = finalized["symbol"].astype("string").str.strip()
    if finalized["symbol"].isna().any() or (finalized["symbol"] == "").any():
        raise ValueError("OHLCV symbol values must be present and non-empty")

    finalized["timestamp"] = _parse_timestamps_as_exchange_local(
        finalized["timestamp"], exchange_timezone
    )
    if finalized["timestamp"].isna().any():
        raise ValueError("OHLCV timestamps must be present and valid")

    for column in NUMERIC_OHLCV_COLUMNS:
        finalized[column] = pd.to_numeric(finalized[column], errors="coerce")

    # Stable ordering is a contract for all feature functions.
    finalized = sort_by_symbol_and_time(finalized)

    # Duplicate bars double-count rows inside rolling windows and silently
    # skew every windowed feature, so they are a hard error. Checked after
    # timezone normalization because two differently-labeled input rows can
    # collapse onto the same exchange-local timestamp.
    if finalized.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate OHLCV bars found for symbol/timestamp pairs")

    return finalized


def _parse_timestamps_as_exchange_local(
    timestamps: pd.Series,
    exchange_timezone: str,
) -> pd.Series:
    """Parse timestamps and normalize them to naive exchange-local time.

    Naive inputs are trusted to already be exchange-local wall-clock time and
    pass through unchanged. Timezone-aware inputs (for example UTC exports)
    are converted to ``exchange_timezone`` and then stripped of their tzinfo,
    so every downstream stage sees one consistent naive representation.

    Parameters
    ----------
    timestamps
        Raw timestamp column: strings, datetimes, or a mix of naive and
        timezone-aware values.
    exchange_timezone
        IANA timezone name of the exchange, for example ``America/New_York``.

    Returns
    -------
    pandas.Series
        Naive exchange-local ``datetime64`` values aligned to the input index.
    """
    zone = ZoneInfo(exchange_timezone)

    try:
        parsed = pd.to_datetime(timestamps)
    except (TypeError, ValueError):
        # Mixed naive and aware inputs cannot be parsed in one vectorized
        # call; fall back to per-element parsing and normalize each value.
        parsed = timestamps.apply(pd.Timestamp)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(zone).dt.tz_localize(None)

    if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
        return parsed

    # Object dtype: element-wise mix of naive and aware timestamps.
    normalized = parsed.apply(
        lambda value: _one_timestamp_as_exchange_local(pd.Timestamp(value), zone)
    )
    return pd.to_datetime(normalized)


def _one_timestamp_as_exchange_local(
    timestamp: pd.Timestamp,
    zone: ZoneInfo,
) -> pd.Timestamp:
    """Normalize one timestamp to naive exchange-local wall-clock time."""
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert(zone).tz_localize(None)
