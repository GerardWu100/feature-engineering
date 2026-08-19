"""Tests for pipeline config validation at the workflow boundary."""

from __future__ import annotations

import pytest
from feature_engineering.config import ConfigValidationError, validate_config


def _valid_csv_config() -> dict:
    """Build the smallest valid CSV config used by validator tests.

    Returns
    -------
    dict
        Plain config dictionary shaped like the parsed TOML file. Tests mutate
        one field at a time so each failure points to one rule.
    """
    return {
        "run": {
            "source": "csv",
            "input_path": "tests/data/toy_prices.csv",
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
            "output_formats": ["csv"],
            "output_dir": "outputs/toy",
        },
        "data_quality": {
            "drop_missing_numeric_values": True,
            "drop_zero_prices": True,
            "drop_high_lt_low": True,
            "drop_ohlc_violations": True,
        },
        "features": {
            "include_categories": [],
            "exclude_categories": [],
            "parameters": [
                {"name": "log_return", "function": "log_return", "enabled": True},
                {
                    "name": "ma_2",
                    "function": "moving_average",
                    "window": 2,
                    "enabled": True,
                },
            ],
        },
    }


def test_validate_config_accepts_minimal_csv_config() -> None:
    """A valid local CSV config should pass without mutation."""
    config = _valid_csv_config()

    validate_config(config)

    assert config["run"]["source"] == "csv"


def test_validate_config_rejects_unknown_feature_function() -> None:
    """Feature names should fail before the engineer stage indexes the registry."""
    config = _valid_csv_config()
    config["features"]["parameters"][0]["function"] = "not_a_feature"

    with pytest.raises(
        ConfigValidationError, match="features.parameters\\[0\\].function"
    ):
        validate_config(config)


def test_validate_config_rejects_duplicate_feature_column_names() -> None:
    """Duplicate output columns would silently overwrite data during engineering."""
    config = _valid_csv_config()
    config["features"]["parameters"][1]["name"] = "log_return"

    with pytest.raises(ConfigValidationError, match="Duplicate feature column"):
        validate_config(config)


def test_validate_config_rejects_unknown_category_filter() -> None:
    """Category filters should name categories that exist in the registry."""
    config = _valid_csv_config()
    config["features"]["include_categories"] = ["not_a_category"]

    with pytest.raises(ConfigValidationError, match="features.include_categories"):
        validate_config(config)


def test_validate_config_rejects_category_in_both_include_and_exclude() -> None:
    """A category in both lists is ambiguous and should fail early."""
    config = _valid_csv_config()
    config["features"]["include_categories"] = ["returns"]
    config["features"]["exclude_categories"] = ["returns"]

    with pytest.raises(ConfigValidationError, match="both include_categories"):
        validate_config(config)


def test_validate_config_rejects_invalid_output_format() -> None:
    """Output formats should stay limited to the formats export.py can write."""
    config = _valid_csv_config()
    config["run"]["output_formats"] = ["csv", "xlsx"]

    with pytest.raises(ConfigValidationError, match="run.output_formats"):
        validate_config(config)


def test_validate_config_rejects_feature_names_that_replace_identifiers() -> None:
    """A computed feature must not overwrite the symbol or timestamp key."""
    config = _valid_csv_config()
    config["features"]["parameters"][0]["name"] = "symbol"

    with pytest.raises(ConfigValidationError, match="reserved identifier column"):
        validate_config(config)


def test_validate_config_rejects_unknown_feature_parameter() -> None:
    """A misspelled feature setting should fail before loading any data."""
    config = _valid_csv_config()
    config["features"]["parameters"][1]["widnow"] = 20

    with pytest.raises(ConfigValidationError, match="unsupported parameter"):
        validate_config(config)


@pytest.mark.parametrize("parameter_name", ["fast", "slow", "signal"])
def test_validate_config_rejects_non_positive_macd_spans(parameter_name: str) -> None:
    """Every exponential-moving-average span must be a positive integer."""
    feature_item = {
        "name": "macd",
        "function": "macd_signal",
        "fast": 12,
        "slow": 26,
        "signal": 9,
    }
    feature_item[parameter_name] = 0
    config = _valid_csv_config()
    config["features"]["parameters"] = [feature_item]

    with pytest.raises(ConfigValidationError, match=f"{parameter_name}.*integer >= 1"):
        validate_config(config)


def test_validate_config_rejects_macd_fast_span_not_below_slow_span() -> None:
    """The fast average must react faster than the slow average."""
    config = _valid_csv_config()
    config["features"]["parameters"] = [
        {
            "name": "macd",
            "function": "macd_line",
            "fast": 26,
            "slow": 12,
        }
    ]

    with pytest.raises(ConfigValidationError, match="fast must be less than slow"):
        validate_config(config)


def test_validate_config_rejects_unknown_data_quality_rule() -> None:
    """A misspelled cleaning rule must not be silently ignored."""
    config = _valid_csv_config()
    config["data_quality"]["drop_negative_volum"] = True

    with pytest.raises(ConfigValidationError, match="unknown option"):
        validate_config(config)


def test_validate_config_accepts_one_sided_csv_date_filter() -> None:
    """CSV runs may keep only one side of the optional date filter."""
    config = _valid_csv_config()
    config["run"].pop("end_date")

    validate_config(config)

    assert config["run"]["start_date"] == "2024-01-02"


def test_validate_config_rejects_window_below_one() -> None:
    """Rolling-window feature configs should reject impossible window lengths."""
    config = _valid_csv_config()
    config["features"]["parameters"][1]["window"] = 0

    with pytest.raises(
        ConfigValidationError, match="features.parameters\\[1\\].window"
    ):
        validate_config(config)


def test_validate_config_rejects_clickhouse_without_symbols() -> None:
    """ClickHouse runs need at least one symbol for the parameterized query."""
    config = _valid_csv_config()
    config["run"]["source"] = "clickhouse"
    config["run"].pop("input_path")
    config["run"]["symbols"] = []
    config["run"]["table"] = "stocks"
    config["run"]["session"] = "regular"

    with pytest.raises(ConfigValidationError, match="run.symbols"):
        validate_config(config)
