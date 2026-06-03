"""Clean invalid stock OHLCV rows before feature engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.pipeline.constants import NUMERIC_OHLCV_COLUMNS, PRICE_COLUMNS


def clean_ohlcv(
    frame: pd.DataFrame,
    data_quality: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows that violate basic OHLCV market-data invariants.

    Parameters
    ----------
    frame
        Raw OHLCV data with ``open``, ``high``, ``low``, ``close``, and
        ``volume`` columns.
    data_quality
        Optional rule toggles. Supported keys are
        ``drop_missing_numeric_values``, ``drop_zero_prices``,
        ``drop_high_lt_low``, and ``drop_ohlc_violations``. Missing keys
        default to ``True``.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Cleaned frame and a report with initial rows, final rows, and per-rule
        drop counts.
    """
    rules = data_quality or {}
    cleaned = frame.copy()
    report: dict[str, Any] = {
        "initial_rows": int(len(cleaned)),
        "rules": {},
    }

    # Drop rows with missing numeric OHLCV values first so later comparisons
    # against thresholds and ranges do not carry NaN-driven ambiguity.
    cleaned = _apply_drop_rule(
        cleaned,
        report,
        rule_name="drop_missing_numeric_values",
        enabled=rules.get("drop_missing_numeric_values", True),
        mask=cleaned[NUMERIC_OHLCV_COLUMNS].isna().any(axis=1),
        reason="open, high, low, close, and volume must be present",
    )

    # Price-based rules are evaluated after missing-value cleanup so each mask
    # reflects only concrete numeric rows that can be compared safely.
    cleaned = _apply_drop_rule(
        cleaned,
        report,
        rule_name="drop_zero_prices",
        enabled=rules.get("drop_zero_prices", True),
        mask=(cleaned[PRICE_COLUMNS] <= 0).any(axis=1),
        reason="open, high, low, and close must be positive prices",
    )

    cleaned = _apply_drop_rule(
        cleaned,
        report,
        rule_name="drop_high_lt_low",
        enabled=rules.get("drop_high_lt_low", True),
        mask=cleaned["high"] < cleaned["low"],
        reason="high must be greater than or equal to low",
    )

    cleaned = _apply_drop_rule(
        cleaned,
        report,
        rule_name="drop_ohlc_violations",
        enabled=rules.get("drop_ohlc_violations", True),
        mask=_ohlc_outside_range_mask(cleaned),
        reason="open and close must sit inside the low-high range",
    )

    # Capture final counts after all enabled rules so downstream logs can show
    # both rule-level drops and overall row retention.
    report["final_rows"] = int(len(cleaned))
    report["total_dropped"] = report["initial_rows"] - report["final_rows"]
    return cleaned.reset_index(drop=True), report


def _apply_drop_rule(
    frame: pd.DataFrame,
    report: dict[str, Any],
    *,
    rule_name: str,
    enabled: bool,
    mask: pd.Series,
    reason: str,
) -> pd.DataFrame:
    """Apply one boolean drop rule and record its effect."""
    if not enabled:
        report["rules"][rule_name] = {"enabled": False, "dropped": 0, "reason": reason}
        return frame

    # Align the mask to the current frame because earlier rules may have dropped
    # rows and changed which labels remain.
    aligned_mask = mask.reindex(frame.index, fill_value=False)
    dropped = int(aligned_mask.sum())
    report["rules"][rule_name] = {
        "enabled": True,
        "dropped": dropped,
        "reason": reason,
    }
    return frame.loc[~aligned_mask].copy()


def _ohlc_outside_range_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows where open or close sit outside the bar low-high range."""
    low_values = frame["low"]
    high_values = frame["high"]

    # A valid OHLC bar must include both open and close inside the low-high
    # envelope for that same row.
    open_outside_bar = (frame["open"] < low_values) | (frame["open"] > high_values)
    close_outside_bar = (frame["close"] < low_values) | (frame["close"] > high_values)
    return open_outside_bar | close_outside_bar
