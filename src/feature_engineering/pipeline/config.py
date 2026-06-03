"""Validate pipeline configuration before data loading starts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.features.registry import REGISTRY
from feature_engineering.pipeline.constants import SQL_IDENTIFIER_PATTERN

ALLOWED_DATA_SOURCES = {"clickhouse", "csv"}
ALLOWED_OUTPUT_FORMATS = {"csv", "parquet"}
ALLOWED_SESSIONS = {"extended", "full", "rth"}
REQUIRED_RUN_KEYS = {"output_dir", "output_formats", "source"}
POSITIVE_INTEGER_FEATURE_PARAMS = {"days", "periods", "window"}


class ConfigValidationError(ValueError):
    """Raised when a parsed TOML config cannot run the feature pipeline."""


def validate_config(config: dict[str, Any]) -> None:
    """Validate a parsed pipeline config in place without changing it.

    Parameters
    ----------
    config
        Plain dictionary returned by ``tomllib.load``. The expected top-level
        sections are ``run``, optional ``data_quality``, and optional
        ``features``.

    Raises
    ------
    ConfigValidationError
        If a required section, key, enum value, feature name, category filter,
        or numeric feature parameter is invalid.
    """
    run_config = _required_mapping(config, "run")
    _validate_run_config(run_config)

    data_quality_config = config.get("data_quality", {})
    _validate_data_quality_config(data_quality_config)

    features_config = config.get("features", {})
    _validate_features_config(features_config)


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required dictionary section from the top-level config."""
    section = config.get(key)
    if not isinstance(section, dict):
        raise ConfigValidationError(f"{key} must be a table.")

    return section


def _validate_run_config(run_config: dict[str, Any]) -> None:
    """Validate the ``[run]`` section used by every pipeline stage."""
    _require_keys(run_config, REQUIRED_RUN_KEYS, section_name="run")

    source = run_config["source"]
    if source not in ALLOWED_DATA_SOURCES:
        allowed_sources = sorted(ALLOWED_DATA_SOURCES)
        raise ConfigValidationError(f"run.source must be one of {allowed_sources}.")

    _validate_output_formats(run_config["output_formats"])
    _validate_non_empty_string(run_config["output_dir"], "run.output_dir")
    _validate_date_range(run_config)

    if source == "csv":
        _validate_csv_run_config(run_config)
        return

    _validate_clickhouse_run_config(run_config)


def _require_keys(
    section: dict[str, Any],
    required_keys: set[str],
    *,
    section_name: str,
) -> None:
    """Raise a clear error when a config table is missing required keys."""
    missing_keys = sorted(required_keys - set(section))
    if missing_keys:
        raise ConfigValidationError(f"{section_name} is missing keys: {missing_keys}.")


def _validate_output_formats(output_formats: Any) -> None:
    """Validate the list of dataset formats requested by ``export.py``."""
    if not isinstance(output_formats, list) or not output_formats:
        raise ConfigValidationError("run.output_formats must be a non-empty list.")

    invalid_formats: set[str] = set()
    for output_format in output_formats:
        if not isinstance(output_format, str):
            raise ConfigValidationError("run.output_formats must contain only strings.")
        if output_format not in ALLOWED_OUTPUT_FORMATS:
            invalid_formats.add(output_format)

    if invalid_formats:
        allowed_formats = sorted(ALLOWED_OUTPUT_FORMATS)
        raise ConfigValidationError(
            "run.output_formats contains unsupported values "
            f"{sorted(invalid_formats)}; allowed values are {allowed_formats}."
        )


def _validate_non_empty_string(value: Any, label: str) -> None:
    """Validate a config value that must be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{label} must be a non-empty string.")


def _validate_date_range(run_config: dict[str, Any]) -> None:
    """Validate optional inclusive start and end dates."""
    start_date = None
    end_date = None

    # CSV loading has historically allowed either bound independently. Validate
    # each bound that is present, and compare only when both are configured.
    if "start_date" in run_config:
        start_date = _parse_config_date(run_config["start_date"], "run.start_date")

    if "end_date" in run_config:
        end_date = _parse_config_date(run_config["end_date"], "run.end_date")

    if start_date is not None and end_date is not None and start_date > end_date:
        raise ConfigValidationError("run.start_date must be before run.end_date.")


def _parse_config_date(value: Any, label: str) -> pd.Timestamp:
    """Parse one config date and report the field name on failure."""
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{label} must be a valid date.") from exc


def _validate_csv_run_config(run_config: dict[str, Any]) -> None:
    """Validate settings required when ``run.source`` is ``csv``."""
    _require_keys(run_config, {"input_path"}, section_name="run")
    _validate_non_empty_string(run_config["input_path"], "run.input_path")


def _validate_clickhouse_run_config(run_config: dict[str, Any]) -> None:
    """Validate settings required when ``run.source`` is ``clickhouse``."""
    _require_keys(
        run_config,
        {"end_date", "start_date", "symbols"},
        section_name="run",
    )
    _validate_symbols(run_config["symbols"])

    table = str(run_config.get("table", "stocks"))
    if not SQL_IDENTIFIER_PATTERN.fullmatch(table):
        raise ConfigValidationError(f"run.table is not a safe SQL identifier: {table}.")

    session = run_config.get("session", "rth")
    if session not in ALLOWED_SESSIONS:
        allowed_sessions = sorted(ALLOWED_SESSIONS)
        raise ConfigValidationError(f"run.session must be one of {allowed_sessions}.")


def _validate_symbols(symbols: Any) -> None:
    """Validate ClickHouse symbols before they are used as query parameters."""
    if not isinstance(symbols, list) or not symbols:
        raise ConfigValidationError("run.symbols must be a non-empty list.")

    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ConfigValidationError("run.symbols cannot contain empty values.")


def _validate_data_quality_config(data_quality_config: Any) -> None:
    """Validate optional cleaning-rule toggles."""
    if data_quality_config is None:
        return

    if not isinstance(data_quality_config, dict):
        raise ConfigValidationError("data_quality must be a table.")

    for rule_name, enabled in data_quality_config.items():
        if not isinstance(enabled, bool):
            raise ConfigValidationError(
                f"data_quality.{rule_name} must be true or false."
            )


def _validate_features_config(features_config: Any) -> None:
    """Validate feature entries and category filters."""
    if features_config is None:
        return

    if not isinstance(features_config, dict):
        raise ConfigValidationError("features must be a table.")

    include_categories = _validate_category_list(
        features_config.get("include_categories", []),
        label="features.include_categories",
    )
    exclude_categories = _validate_category_list(
        features_config.get("exclude_categories", []),
        label="features.exclude_categories",
    )
    _validate_category_overlap(include_categories, exclude_categories)
    _validate_feature_items(features_config.get("params", []))


def _validate_category_list(raw_categories: Any, *, label: str) -> set[str]:
    """Validate one feature category filter list and return it as a set."""
    if not isinstance(raw_categories, list):
        raise ConfigValidationError(f"{label} must be a list.")

    available_categories = {spec.category for spec in REGISTRY.values()}
    category_set: set[str] = set()
    for index, category in enumerate(raw_categories):
        if not isinstance(category, str):
            raise ConfigValidationError(f"{label}[{index}] must be a string.")

        if category not in available_categories:
            raise ConfigValidationError(
                f"{label}[{index}] is unknown: {category}. "
                f"Available categories are {sorted(available_categories)}."
            )

        category_set.add(category)

    return category_set


def _validate_category_overlap(
    include_categories: set[str],
    exclude_categories: set[str],
) -> None:
    """Reject category filters that both include and exclude the same value."""
    overlapping_categories = sorted(include_categories & exclude_categories)
    if overlapping_categories:
        raise ConfigValidationError(
            "Feature categories cannot appear in both include_categories and "
            f"exclude_categories: {overlapping_categories}."
        )


def _validate_feature_items(raw_feature_items: Any) -> None:
    """Validate configured feature columns before engineering starts."""
    if not isinstance(raw_feature_items, list):
        raise ConfigValidationError("features.params must be a list.")

    active_feature_names: set[str] = set()
    for index, feature_item in enumerate(raw_feature_items):
        if not isinstance(feature_item, dict):
            raise ConfigValidationError(f"features.params[{index}] must be a table.")

        _validate_feature_item(feature_item, index)

        # Disabled entries stay in config as notes or experiments, but they do
        # not create output columns and therefore do not participate in duplicate
        # name checks.
        if not feature_item.get("enabled", True):
            continue

        feature_name = feature_item["name"]
        if feature_name in active_feature_names:
            raise ConfigValidationError(
                f"Duplicate feature column name: {feature_name}."
            )

        active_feature_names.add(feature_name)


def _validate_feature_item(feature_item: dict[str, Any], index: int) -> None:
    """Validate one ``[[features.params]]`` table."""
    label = f"features.params[{index}]"
    _require_keys(feature_item, {"fn", "name"}, section_name=label)
    _validate_non_empty_string(feature_item["name"], f"{label}.name")
    _validate_non_empty_string(feature_item["fn"], f"{label}.fn")

    function_name = feature_item["fn"]
    if function_name not in REGISTRY:
        raise ConfigValidationError(f"{label}.fn is unknown: {function_name}.")

    if "enabled" in feature_item and not isinstance(feature_item["enabled"], bool):
        raise ConfigValidationError(f"{label}.enabled must be true or false.")

    for parameter_name in POSITIVE_INTEGER_FEATURE_PARAMS:
        if parameter_name in feature_item:
            _validate_positive_integer(
                feature_item[parameter_name],
                label=f"{label}.{parameter_name}",
            )


def _validate_positive_integer(value: Any, *, label: str) -> None:
    """Validate a feature parameter that must be an integer greater than zero."""
    if not isinstance(value, int) or value < 1:
        raise ConfigValidationError(f"{label} must be an integer >= 1.")
