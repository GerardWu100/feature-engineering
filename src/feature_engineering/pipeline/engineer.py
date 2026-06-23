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
        by their registry category. Optional ``features.reset_by_session``
        (bool, default ``False``) makes row-count windows and forward shifts
        reset at each calendar day, which prevents intraday features from
        crossing the overnight gap. Leave it ``False`` for daily bars, where one
        row already is one day.

    Returns
    -------
    pandas.DataFrame
        Identifier columns plus computed feature columns. Raw OHLCV columns are
        intentionally omitted from the output to keep the feature dataset small.
    """
    sorted_frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
    selected_feature_configs = _selected_feature_configs(config)
    resolved_features = [_resolve_feature(item) for item in selected_feature_configs]

    # Choose the isolation boundary. Always isolate by symbol. When the run is
    # intraday and asks for session resets, also isolate by calendar day so a
    # 20-bar window at 09:30 cannot reach back into the prior session.
    reset_by_session = bool(config.get("features", {}).get("reset_by_session", False))

    return apply_resolved_features(
        sorted_frame,
        resolved_features=resolved_features,
        reset_by_session=reset_by_session,
    )


def apply_resolved_features(
    sorted_frame: pd.DataFrame,
    *,
    resolved_features: list[tuple[str, FeatureSpec, dict[str, Any]]],
    reset_by_session: bool,
) -> pd.DataFrame:
    """Compute already-resolved feature columns on a pre-sorted frame.

    This is the shared core used by both the one-shot :func:`compute_features`
    and the cached ``FeatureEngine`` wrapper. Splitting it out lets the engine
    resolve specs once in its constructor and reuse them on every ``transform``.

    Parameters
    ----------
    sorted_frame
        Clean OHLCV data already sorted by symbol and timestamp.
    resolved_features
        List of ``(column_name, spec, params)`` tuples from :func:`_resolve_feature`.
    reset_by_session
        When ``True``, isolate features by symbol and calendar day so intraday
        windows do not cross the overnight gap.

    Returns
    -------
    pandas.DataFrame
        Identifier columns plus one column per resolved feature.
    """
    # Keep identifier columns separate from computed feature columns so exports
    # stay compact and clearly signal what each row represents.
    output = sorted_frame.loc[:, IDENTIFIER_COLUMNS].copy()
    group_keys = _feature_group_keys(sorted_frame, reset_by_session=reset_by_session)

    for column_name, spec, params in resolved_features:
        output[column_name] = _compute_feature_series_by_group(
            sorted_frame,
            spec=spec,
            params=params,
            group_keys=group_keys,
        )

    return output


def _feature_group_keys(
    frame: pd.DataFrame,
    *,
    reset_by_session: bool,
) -> list[Any]:
    """Return the groupby keys that define the feature-isolation boundary.

    Parameters
    ----------
    frame
        Sorted OHLCV frame.
    reset_by_session
        When ``True``, append a per-calendar-day key so intraday windows reset
        each session. When ``False``, isolate by ``symbol`` only.

    Returns
    -------
    list
        Arguments suitable for ``frame.groupby(...)``. The session key is a
        Series (not a column name) so it never lands in the output frame.
    """
    if not reset_by_session:
        return ["symbol"]

    # Calendar date of each bar in its own (exchange-local) timestamp. Grouping
    # on this Series resets windows at midnight without adding a stored column.
    session_date = frame["ts"].dt.normalize()
    session_date.name = "_session_date"
    return ["symbol", session_date]


def _compute_feature_series_by_group(
    frame: pd.DataFrame,
    *,
    spec: FeatureSpec,
    params: dict[str, Any],
    group_keys: list[Any],
) -> pd.Series:
    """Apply one feature function independently within each isolation group.

    Parameters
    ----------
    frame
        Clean OHLCV data sorted by symbol and timestamp.
    spec
        Registry entry holding the concrete feature function and metadata.
    params
        Function-specific settings from one config item, such as ``window`` or
        ``bars``.
    group_keys
        Groupby keys from :func:`_feature_group_keys`. ``["symbol"]`` isolates by
        ticker; ``["symbol", session_date]`` also isolates by trading day.

    Returns
    -------
    pandas.Series
        One feature column aligned to ``frame.index``. Each group is computed in
        isolation so rolling windows and lags cannot cross ticker boundaries, or
        day boundaries when session resets are enabled.
    """
    values_by_group: list[pd.Series] = []

    # Compute each group separately. This makes the isolation rule explicit and
    # avoids hiding the feature call behind a groupby/apply lambda.
    for _group_key, group_frame in frame.groupby(group_keys, sort=False):
        group_values = spec.fn(group_frame, params)
        values_by_group.append(group_values)

    if not values_by_group:
        return pd.Series(index=frame.index, dtype="float64")

    # Concatenation preserves the original row labels from each group slice.
    # Reindexing restores exact frame order even if pandas changes grouping
    # internals in a future release.
    return pd.concat(values_by_group).reindex(frame.index)


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

    # Keep only function-specific settings such as ``window`` or ``bars``.
    params = {
        key: value
        for key, value in feature_config.items()
        if key not in RESERVED_FEATURE_KEYS
    }
    return column_name, spec, params
