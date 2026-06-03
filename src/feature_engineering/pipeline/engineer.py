"""Compute configured stock features by category."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.features.registry import REGISTRY, FeatureSpec
from feature_engineering.pipeline.constants import IDENTIFIER_COLUMNS

RESERVED_FEATURE_KEYS = {"name", "fn", "enabled"}


def compute_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute enabled feature columns for each symbol independently.

    Parameters
    ----------
    frame
        Clean OHLCV data sorted by symbol and timestamp.
    config
        Config dict containing ``features.params`` entries. Optional
        ``include_categories`` and ``exclude_categories`` lists filter features
        by their registry category.

    Returns
    -------
    pandas.DataFrame
        Identifier columns plus computed feature columns. Raw OHLCV columns are
        intentionally omitted from the output to keep the feature dataset small.
    """
    sorted_frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # Keep identifier columns separate from computed feature columns so exports
    # stay compact and clearly signal what each row represents.
    output = sorted_frame.loc[:, IDENTIFIER_COLUMNS].copy()
    selected_feature_configs = _selected_feature_configs(config)

    for feature_config in selected_feature_configs:
        column_name, spec, feature_params = _resolve_feature(feature_config)
        output[column_name] = _compute_feature_series_by_symbol(
            sorted_frame,
            spec=spec,
            params=feature_params,
        )

    return output


def _compute_feature_series_by_symbol(
    frame: pd.DataFrame,
    *,
    spec: FeatureSpec,
    params: dict[str, Any],
) -> pd.Series:
    """Apply one feature function independently to each symbol.

    Parameters
    ----------
    frame
        Clean OHLCV data sorted by symbol and timestamp.
    spec
        Registry entry holding the concrete feature function and metadata.
    params
        Function-specific settings from one config item, such as ``window`` or
        ``days``.

    Returns
    -------
    pandas.Series
        One feature column aligned to ``frame.index``. Each symbol is computed
        in isolation so rolling windows and lags cannot cross ticker boundaries.
    """
    values_by_symbol: list[pd.Series] = []

    # Compute each ticker separately. This makes the isolation rule explicit and
    # avoids hiding the feature call behind a groupby/apply lambda.
    for _symbol, symbol_frame in frame.groupby("symbol", sort=False):
        symbol_values = spec.fn(symbol_frame, params)
        values_by_symbol.append(symbol_values)

    if not values_by_symbol:
        return pd.Series(index=frame.index, dtype="float64")

    # Concatenation preserves the original row labels from each symbol slice.
    # Reindexing restores exact frame order even if pandas changes grouping
    # internals in a future release.
    return pd.concat(values_by_symbol).reindex(frame.index)


def _selected_feature_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled feature config entries after category filtering."""
    feature_config = config.get("features", {})
    include_categories = set(feature_config.get("include_categories", []))
    exclude_categories = set(feature_config.get("exclude_categories", []))
    selected_feature_configs: list[dict[str, Any]] = []

    # Walk in config order so output column order matches the user's parameter
    # list, which makes export files easier to compare run-to-run.
    for feature_item in feature_config.get("params", []):
        if not feature_item.get("enabled", True):
            continue

        # Registry metadata supplies the category used by include/exclude rules.
        spec = REGISTRY[feature_item["fn"]]
        if not _category_is_selected(
            spec.category,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        ):
            continue

        selected_feature_configs.append(feature_item)

    return selected_feature_configs


def _category_is_selected(
    category: str,
    *,
    include_categories: set[str],
    exclude_categories: set[str],
) -> bool:
    """Return whether one feature category passes include/exclude filtering."""

    # Include rules are applied first: an empty include list means "allow all".
    if include_categories and category not in include_categories:
        return False

    # Exclude rules always win after include filtering.
    if category in exclude_categories:
        return False

    return True


def _resolve_feature(
    feature_config: dict[str, Any],
) -> tuple[str, FeatureSpec, dict[str, Any]]:
    """Look up one feature config entry and split function params from metadata."""
    function_name = feature_config["fn"]
    if function_name not in REGISTRY:
        raise KeyError(f"Unknown feature function: {function_name}")

    column_name = feature_config["name"]
    spec = REGISTRY[function_name]

    # Keep only function-specific settings such as ``window`` or ``days``.
    params = {
        key: value
        for key, value in feature_config.items()
        if key not in RESERVED_FEATURE_KEYS
    }
    return column_name, spec, params
