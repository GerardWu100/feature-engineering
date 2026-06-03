"""Shared OHLCV column names and SQL identifier rules for pipeline stages."""

from __future__ import annotations

import re

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
