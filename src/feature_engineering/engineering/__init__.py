"""Feature engineering: build feature datasets from stock OHLCV bars.

The stages run in this order:

    load_ohlcv -> clean_ohlcv -> compute_features -> save_features

``features/`` holds the actual feature functions, one file per category
(returns, targets, trend, volatility, volume). ``load_features`` pulls a
stored dataset back from disk.
"""

from feature_engineering.engineering.clean import clean_ohlcv
from feature_engineering.engineering.compute import compute_features
from feature_engineering.engineering.load import load_ohlcv
from feature_engineering.engineering.store import load_features, save_features

__all__ = [
    "clean_ohlcv",
    "compute_features",
    "load_features",
    "load_ohlcv",
    "save_features",
]
