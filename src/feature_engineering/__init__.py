"""Feature-engineering package for stock OHLCV research workflows.

This package can be used two ways:

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
``tomllib`` produces from ``config.toml``. See ``pipeline/config.py`` for the
exact keys and ``validate_config`` to check a config before use.
"""

from feature_engineering.features.registry import REGISTRY, FeatureSpec, register
from feature_engineering.pipeline.clean import clean_ohlcv
from feature_engineering.pipeline.cli import run_pipeline
from feature_engineering.pipeline.config import validate_config
from feature_engineering.pipeline.engineer import compute_features
from feature_engineering.pipeline.export import build_feature_catalog, export_features
from feature_engineering.pipeline.load import load_ohlcv

__all__ = [
    "REGISTRY",
    "FeatureSpec",
    "register",
    "validate_config",
    "load_ohlcv",
    "clean_ohlcv",
    "compute_features",
    "export_features",
    "build_feature_catalog",
    "run_pipeline",
]
