"""Categorized stock feature functions.

Every feature can be called directly, pandas-style, with keyword parameters:

    >>> from feature_engineering.engineering.features import moving_average, vwap
    >>> ma20 = moving_average(single_symbol_frame, window=20)
    >>> session_vwap = vwap(single_symbol_frame)

Each function expects one symbol's OHLCV frame sorted by time and returns a
Series aligned to the frame's index. The config-driven pipeline calls the same
functions through the registry, so direct calls and pipeline runs share one
implementation.
"""

from feature_engineering.engineering.features.registry import (
    REGISTRY,
    FeatureSpec,
    as_feature_column,
)
from feature_engineering.engineering.features.returns import log_return, simple_return
from feature_engineering.engineering.features.targets import (
    next_n_bar_realized_volatility,
    next_n_bar_return,
)
from feature_engineering.engineering.features.trend import (
    macd_histogram,
    macd_line,
    macd_signal,
    moving_average,
    price_vs_moving_average,
    rate_of_change,
    relative_strength_index,
)
from feature_engineering.engineering.features.volatility import (
    average_true_range,
    bar_range_percent,
    rolling_standard_deviation,
)
from feature_engineering.engineering.features.volume import (
    dollar_volume,
    price_vs_vwap,
    volume_change,
    volume_ratio,
    vwap,
)

__all__ = [
    "REGISTRY",
    "FeatureSpec",
    "as_feature_column",
    "average_true_range",
    "bar_range_percent",
    "dollar_volume",
    "log_return",
    "macd_histogram",
    "macd_line",
    "macd_signal",
    "moving_average",
    "next_n_bar_realized_volatility",
    "next_n_bar_return",
    "price_vs_moving_average",
    "price_vs_vwap",
    "rate_of_change",
    "relative_strength_index",
    "rolling_standard_deviation",
    "simple_return",
    "volume_change",
    "volume_ratio",
    "vwap",
]
