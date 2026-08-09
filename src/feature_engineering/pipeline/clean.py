"""Clean invalid stock OHLCV rows before feature engineering."""

from __future__ import annotations

from collections.abc import Callable
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
    report: dict[str, Any] = {
        "initial_rows": len(frame),
        "rules": {},
    }

    # Every rule is a per-row predicate, so all masks can be computed against
    # the original frame and combined in one pass instead of materializing an
    # intermediate copy of the frame after each rule. Rules run in a fixed
    # order and each row is attributed to the first rule that would drop it,
    # which keeps the per-rule counts identical to sequential application.
    # NaN comparisons are False in pandas, so price rules stay unambiguous even
    # when the missing-value rule is disabled.
    rule_definitions: list[tuple[str, Callable[[pd.DataFrame], pd.Series], str]] = [
        (
            "drop_missing_numeric_values",
            lambda f: f[NUMERIC_OHLCV_COLUMNS].isna().any(axis=1),
            "open, high, low, close, and volume must be present",
        ),
        (
            "drop_zero_prices",
            lambda f: (f[PRICE_COLUMNS] <= 0).any(axis=1),
            "open, high, low, and close must be positive prices",
        ),
        (
            "drop_high_lt_low",
            lambda f: f["high"] < f["low"],
            "high must be greater than or equal to low",
        ),
        (
            "drop_ohlc_violations",
            _ohlc_outside_range_mask,
            "open and close must sit inside the low-high range",
        ),
    ]

    already_dropped = pd.Series(False, index=frame.index)
    for rule_name, build_mask, reason in rule_definitions:
        if not rules.get(rule_name, True):
            report["rules"][rule_name] = {
                "enabled": False,
                "dropped": 0,
                "reason": reason,
            }
            continue

        # Only count rows not already claimed by an earlier rule.
        new_drops = build_mask(frame) & ~already_dropped
        report["rules"][rule_name] = {
            "enabled": True,
            "dropped": int(new_drops.sum()),
            "reason": reason,
        }
        already_dropped |= new_drops

    cleaned = frame.loc[~already_dropped].reset_index(drop=True)

    # Capture final counts after all enabled rules so downstream logs can show
    # both rule-level drops and overall row retention.
    report["final_rows"] = len(cleaned)
    report["total_dropped"] = report["initial_rows"] - report["final_rows"]
    return cleaned, report


def _ohlc_outside_range_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows where open or close sit outside the bar low-high range."""
    low_values = frame["low"]
    high_values = frame["high"]

    # A valid OHLC bar must include both open and close inside the low-high
    # envelope for that same row.
    open_outside_bar = (frame["open"] < low_values) | (frame["open"] > high_values)
    close_outside_bar = (frame["close"] < low_values) | (frame["close"] > high_values)
    return open_outside_bar | close_outside_bar
