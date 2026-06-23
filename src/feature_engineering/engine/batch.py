"""Cached batch feature engine for repeated DataFrame transforms.

``FeatureEngine`` is the convenience wrapper for research and backtesting. It
resolves the configured feature list once in the constructor and caches the
result, so a loop that calls ``transform`` on many frames (for example one
symbol-day at a time in a backtest) does not re-walk and re-validate the config
on every call.

For a single one-shot computation, ``feature_engineering.compute_features`` is
simpler. For live, bar-by-bar streaming use the ``OnlineFeatureEngine`` in
``engine/online.py`` instead.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.features.registry import FeatureSpec
from feature_engineering.pipeline.engineer import (
    _resolve_feature,
    _selected_feature_configs,
    apply_resolved_features,
)


class FeatureEngine:
    """Resolve a feature config once, then transform many OHLCV frames.

    Parameters
    ----------
    config
        Config dict with a ``features`` section, the same shape ``config.toml``
        parses into. The ``run`` section is not required: this engine only reads
        ``features.params``, ``features.include_categories``,
        ``features.exclude_categories``, and ``features.reset_by_session``.

    Attributes
    ----------
    feature_names
        Output column names this engine will produce, in config order.

    Notes
    -----
    Feature specs are resolved in ``__init__``. An unknown ``fn`` raises
    immediately, so a misconfigured engine fails at construction rather than on
    the first transform.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        features_config = config.get("features", {})
        self._reset_by_session = bool(features_config.get("reset_by_session", False))

        # Resolve and cache the selected feature specs once. This is the work we
        # avoid repeating on every transform call.
        selected_feature_configs = _selected_feature_configs(config)
        self._resolved_features: list[tuple[str, FeatureSpec, dict[str, Any]]] = [
            _resolve_feature(item) for item in selected_feature_configs
        ]

    @property
    def feature_names(self) -> list[str]:
        """Return the output feature column names in config order."""
        return [column_name for column_name, _spec, _params in self._resolved_features]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute the cached feature set for one OHLCV frame.

        Parameters
        ----------
        frame
            Clean OHLCV data with ``symbol``, ``ts``, and OHLCV columns. It does
            not need to be pre-sorted; the engine sorts internally.

        Returns
        -------
        pandas.DataFrame
            Identifier columns (``symbol``, ``ts``) plus one column per
            configured feature, identical to ``compute_features`` output.
        """
        sorted_frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
        return apply_resolved_features(
            sorted_frame,
            resolved_features=self._resolved_features,
            reset_by_session=self._reset_by_session,
        )
