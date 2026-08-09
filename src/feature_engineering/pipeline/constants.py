"""Shared OHLCV column names, defaults, and SQL identifier rules.

This module is the single home for values that more than one pipeline stage
depends on, so a change here propagates everywhere at once.
"""

from __future__ import annotations

import re

import pandas as pd

# Standard loader output columns in stable order.
OHLCV_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]

# Identifier columns kept in engineered feature exports.
IDENTIFIER_COLUMNS = ["symbol", "ts"]
IDENTIFIER_COLUMN_SET = set(IDENTIFIER_COLUMNS)

# Price and numeric subsets used by cleaning rules.
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# ClickHouse table names must be simple identifiers before SQL interpolation.
SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Run defaults shared by config validation and data loading, so the two
# stages cannot silently disagree about what an omitted key means.
DEFAULT_CLICKHOUSE_TABLE = "stocks"
DEFAULT_SESSION = "rth"

# CSV files are often daily bars stamped at midnight, which an "rth" default
# would drop entirely, so CSV runs apply no session filter unless asked.
DEFAULT_CSV_SESSION = "full"

# Exchange timezone used to convert timezone-aware input timestamps to
# exchange-local wall-clock time. Naive inputs are assumed already local.
DEFAULT_EXCHANGE_TIMEZONE = "America/New_York"


def sort_by_symbol_and_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` in the project's canonical row order.

    Sorting by ``symbol`` then ``ts`` with a fresh integer index is the ordering
    contract every feature function relies on. All pipeline stages and engines
    call this one helper so the contract is defined in exactly one place.

    Parameters
    ----------
    frame
        Any DataFrame containing ``symbol`` and ``ts`` columns.

    Returns
    -------
    pandas.DataFrame
        New frame sorted by the identifier columns with index ``0..n-1``.
    """
    return frame.sort_values(IDENTIFIER_COLUMNS).reset_index(drop=True)
