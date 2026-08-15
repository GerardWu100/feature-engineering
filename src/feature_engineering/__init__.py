"""Feature engineering and evaluation for stock OHLCV research workflows.

The package has two major parts:

1. ``engineering`` - build features: load OHLCV bars, clean them, compute the
   configured feature columns, and store/pull the resulting datasets.
2. ``evaluation`` - test features: measure how much evidence each feature
   carries about a target (information coefficients, regression, quantiles,
   and plots).

The package can be used two ways:

1. As a command-line pipeline (``run.py`` / ``feature-pipeline``) that reads a
   TOML config, loads data, and writes feature files to disk.
2. As an in-memory library inside a research or trading process. Import the
   stage functions directly and pass pandas DataFrames, with no file I/O:

   >>> from feature_engineering import compute_features, clean_ohlcv
   >>> cleaned, report = clean_ohlcv(raw_ohlcv_frame)
   >>> features = compute_features(cleaned, config_dict)

``compute_features`` is the pure transform at the heart of the pipeline: it
takes a clean OHLCV frame plus a config dict and returns a feature frame without
touching disk, which is what a live bot or backtest loop needs.

The config passed to the in-memory functions is the same plain dict shape that
``tomllib`` produces from ``config.toml``. See ``config.py`` for the exact keys
and ``validate_config`` to check a config before use.
"""

from feature_engineering.cli import run_pipeline
from feature_engineering.config import validate_config
from feature_engineering.engineering.clean import clean_ohlcv
from feature_engineering.engineering.compute import compute_features
from feature_engineering.engineering.features.registry import (
    REGISTRY,
    FeatureSpec,
    register,
)
from feature_engineering.engineering.load import load_ohlcv
from feature_engineering.engineering.store import (
    build_feature_catalog,
    load_features,
    save_features,
)
from feature_engineering.evaluation import evaluate_features

__all__ = [
    "REGISTRY",
    "FeatureSpec",
    "build_feature_catalog",
    "clean_ohlcv",
    "compute_features",
    "evaluate_features",
    "load_features",
    "load_ohlcv",
    "register",
    "run_pipeline",
    "save_features",
    "validate_config",
]
