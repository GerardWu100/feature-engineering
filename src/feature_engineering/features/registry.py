"""Small registry for categorized stock feature functions.

The registry is the project's feature menu. Each registered feature has a
Python function, a category, and short metadata that can be exported with the
dataset.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import pandas as pd

FeatureCallable = Callable[[pd.DataFrame, dict], pd.Series]
Lookback = int | Callable[[dict], int]

REGISTRY: dict[str, "FeatureSpec"] = {}


def as_feature_column(values: pd.Series) -> pd.Series:
    """Return one feature column without a Series name for clean DataFrame joins."""
    values.name = None
    return values


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one feature function available to the pipeline.

    Parameters
    ----------
    fn
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

    fn: FeatureCallable
    category: str
    lookback: Lookback
    description: str
    calculation: str

    def resolve_lookback(self, params: dict) -> int:
        """Return the concrete lookback for a configured feature column."""
        if callable(self.lookback):
            return int(self.lookback(params))
        return int(self.lookback)


def register(
    *,
    category: str,
    lookback: Lookback,
    description: str,
    calculation: str,
) -> Callable[[FeatureCallable], FeatureCallable]:
    """Register a feature function under its Python function name."""

    def decorator(fn: FeatureCallable) -> FeatureCallable:
        # The function name is the stable config key, for example "log_return".
        REGISTRY[fn.__name__] = FeatureSpec(
            fn=fn,
            category=category,
            lookback=lookback,
            description=description,
            calculation=calculation,
        )
        return fn

    return decorator


_FEATURE_MODULES = (
    "feature_engineering.features.returns",
    "feature_engineering.features.trend",
    "feature_engineering.features.volatility",
    "feature_engineering.features.volume",
)

for module_path in _FEATURE_MODULES:
    importlib.import_module(module_path)
