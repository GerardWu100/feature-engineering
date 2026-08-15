"""Small registry for categorized stock feature functions.

The registry is the project's feature menu. Each registered feature has a
Python function, a category, and short metadata that can be exported with the
dataset.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

FeatureCallable = Callable[[pd.DataFrame, dict], pd.Series]
Lookback = int | Callable[[dict], int]

REGISTRY: dict[str, FeatureSpec] = {}


def as_feature_column(values: pd.Series) -> pd.Series:
    """Return one feature column without a Series name.

    Feature functions return their series through this helper as a contract:
    the configured column name (not the intermediate pandas name) is the only
    name a feature value ever carries, both inside the pipeline and for direct
    callers using feature functions standalone.
    """
    values.name = None
    return values


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one feature function available to the pipeline.

    Parameters
    ----------
    function
        Callable that receives one symbol's OHLCV data and a parameter dict.
    category
        Group name used by config filters, such as ``returns`` or ``trend``.
    lookback
        Number of rows or minutes usually needed before a feature becomes valid.
    description
        Plain-language description for the exported feature catalog.
    calculation
        Compact formula or calculation summary for documentation.
    """

    function: FeatureCallable
    category: str
    lookback: Lookback
    description: str
    calculation: str

    def resolve_lookback(self, parameters: dict) -> int:
        """Return the concrete lookback for a configured feature column."""
        if callable(self.lookback):
            return int(self.lookback(parameters))
        return int(self.lookback)


def register(
    *,
    category: str,
    lookback: Lookback,
    description: str,
    calculation: str,
) -> Callable[[FeatureCallable], FeatureCallable]:
    """Register a feature function under its Python function name."""

    def decorator(function: FeatureCallable) -> FeatureCallable:
        # The function name is the stable config key, for example "log_return".
        REGISTRY[function.__name__] = FeatureSpec(
            function=function,
            category=category,
            lookback=lookback,
            description=description,
            calculation=calculation,
        )
        return function

    return decorator


_FEATURE_MODULES = (
    "feature_engineering.engineering.features.returns",
    "feature_engineering.engineering.features.targets",
    "feature_engineering.engineering.features.trend",
    "feature_engineering.engineering.features.volatility",
    "feature_engineering.engineering.features.volume",
)

for module_path in _FEATURE_MODULES:
    importlib.import_module(module_path)
