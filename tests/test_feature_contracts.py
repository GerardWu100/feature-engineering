"""Contract tests for feature function outputs.

A feature function that returns the wrong shape or index would silently
attach values to the wrong bars, so the engineering stage validates every
per-group result. These tests register throwaway feature functions that
violate each contract and assert the pipeline fails loudly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from feature_engineering.features.registry import REGISTRY, FeatureSpec
from feature_engineering.pipeline.engineer import compute_features


def _frame() -> pd.DataFrame:
    """Build a two-row single-symbol OHLCV frame for contract tests."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 09:31:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000.0, 1200.0],
        }
    )


def _config(feature_name: str, function_name: str) -> dict[str, Any]:
    """Return a minimal config dict enabling exactly one feature."""
    return {
        "features": {
            "parameters": [
                {"name": feature_name, "function": function_name, "enabled": True}
            ]
        }
    }


def _register_contract_feature(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    values: object,
) -> None:
    """Register a temporary feature returning fixed values, test-scoped."""

    def contract_feature(frame: pd.DataFrame, parameters: dict[str, Any]) -> object:
        return values

    monkeypatch.setitem(
        REGISTRY,
        function_name,
        FeatureSpec(
            function=contract_feature,
            category="returns",
            lookback=0,
            description="Temporary contract test feature.",
            calculation="fixed test values",
        ),
    )


def test_feature_output_wrong_length_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature result with the wrong length should fail loudly."""
    _register_contract_feature(monkeypatch, "wrong_length_feature", pd.Series([1.0]))

    with pytest.raises(ValueError, match="bad_length.*same length"):
        compute_features(_frame(), _config("bad_length", "wrong_length_feature"))


def test_feature_output_wrong_index_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature result with a mismatched index should fail loudly."""
    values = pd.Series([1.0, 2.0], index=[10, 11])
    _register_contract_feature(monkeypatch, "wrong_index_feature", values)

    with pytest.raises(ValueError, match="bad_index.*same index"):
        compute_features(_frame(), _config("bad_index", "wrong_index_feature"))


def test_feature_output_infinity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature may contain warm-up NaN values but never infinity."""
    values = pd.Series([np.nan, np.inf], index=[0, 1])
    _register_contract_feature(monkeypatch, "infinity_feature", values)

    with pytest.raises(ValueError, match="bad_infinity.*infinite"):
        compute_features(_frame(), _config("bad_infinity", "infinity_feature"))


def test_feature_output_non_series_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature functions must return a pandas Series."""
    _register_contract_feature(monkeypatch, "non_series_feature", [1.0, 2.0])

    with pytest.raises(ValueError, match="bad_type.*pandas Series"):
        compute_features(_frame(), _config("bad_type", "non_series_feature"))


def test_feature_output_non_numeric_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature functions must return numeric values, not strings."""
    values = pd.Series(["a", "b"], index=[0, 1])
    _register_contract_feature(monkeypatch, "string_feature", values)

    with pytest.raises(ValueError, match="bad_dtype.*numeric"):
        compute_features(_frame(), _config("bad_dtype", "string_feature"))
