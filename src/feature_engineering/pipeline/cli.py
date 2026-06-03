"""Command-line interface for the simple feature engineering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

from feature_engineering.pipeline.clean import clean_ohlcv
from feature_engineering.pipeline.config import validate_config
from feature_engineering.pipeline.engineer import compute_features
from feature_engineering.pipeline.export import export_features
from feature_engineering.pipeline.load import load_ohlcv


def main() -> None:
    """Run the simple load-clean-engineer-export workflow from a TOML config."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Simple stock feature engineering: load -> clean -> engineer -> export.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        config = load_config(args.config)
        run_pipeline(config)
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        sys.exit(1)


def load_config(config_path: Path) -> dict[str, Any]:
    """Read one TOML config file into a plain dictionary."""
    with config_path.open("rb") as file:
        return tomllib.load(file)


def run_pipeline(config: dict[str, Any]) -> dict[str, Path]:
    """Execute the full simple pipeline and return written output paths.

    Parameters
    ----------
    config
        Parsed TOML config with ``run``, ``data_quality``, and ``features``
        sections.

    Returns
    -------
    dict[str, pathlib.Path]
        Output paths written by the export stage.
    """
    log = logging.getLogger(__name__)

    # Validate the external boundary once so the pipeline stages can focus on
    # loading, cleaning, engineering, and exporting rather than config shape.
    validate_config(config)

    log.info("Loading OHLCV data")
    raw = load_ohlcv(config)
    if raw.empty:
        raise ValueError("No rows loaded. Check source, symbols, and date range.")

    log.info("Cleaning %d rows", len(raw))
    cleaned, quality_report = clean_ohlcv(raw, config.get("data_quality"))
    if cleaned.empty:
        raise ValueError("All rows were dropped during cleaning.")

    log.info("Computing configured features")
    featured = compute_features(cleaned, config)

    log.info("Exporting %d rows", len(featured))
    paths = export_features(featured, config)

    log.info("Quality report: %s", json.dumps(quality_report, sort_keys=True))
    for label, path in paths.items():
        log.info("Wrote %s: %s", label, path)

    return paths


if __name__ == "__main__":
    main()
