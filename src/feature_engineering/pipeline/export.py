"""Export engineered features and a compact feature catalog."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering.features.registry import REGISTRY
from feature_engineering.pipeline.constants import IDENTIFIER_COLUMN_SET


def export_features(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    """Write feature outputs requested by config.

    Parameters
    ----------
    frame
        Engineered feature dataset.
    config
        Config dict with ``run.output_dir``, ``run.output_formats``, and
        ``features.params``.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of output type to written path.
    """
    run_config = config["run"]
    output_dir = Path(run_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use one timestamped stem for all run artifacts so CSV, Parquet, and
    # summaries from the same execution are easy to match.
    version = run_config.get("version", "dev")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"features_v{version}_{timestamp}"
    paths = _write_dataset_outputs(
        frame,
        output_dir=output_dir,
        stem=stem,
        output_formats=run_config.get("output_formats", ["parquet"]),
    )

    catalog = build_feature_catalog(frame, config)
    catalog_path = output_dir / "feature_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    paths["catalog_csv"] = catalog_path

    # Persist one machine-readable summary to support quick post-run checks
    # without opening the full dataset files.
    summary_path = output_dir / f"run_summary_v{version}_{timestamp}.json"
    summary = _build_run_summary(frame, paths)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary_json"] = summary_path

    return paths


def build_feature_catalog(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build a small catalog for the feature columns in an exported dataset."""
    active_by_name = {
        item["name"]: item
        for item in config.get("features", {}).get("params", [])
        if item.get("enabled", True)
    }
    rows: list[dict[str, Any]] = []

    for column in frame.columns:
        if column in IDENTIFIER_COLUMN_SET or column not in active_by_name:
            continue

        # Resolve registry metadata so the catalog captures both the configured
        # column name and the underlying formula definition.
        item = active_by_name[column]
        spec = REGISTRY[item["fn"]]
        rows.append(
            {
                "name": column,
                "fn": item["fn"],
                "category": spec.category,
                "lookback": spec.resolve_lookback(item),
                "description": spec.description,
                "calculation": spec.calculation,
            }
        )

    return pd.DataFrame(rows)


def _write_dataset_outputs(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
    output_formats: list[str],
) -> dict[str, Path]:
    """Write feature dataset files for the requested output formats."""
    paths: dict[str, Path] = {}

    # CSV and Parquet are handled independently so users can request one or both
    # formats from config without changing pipeline code.
    if "csv" in output_formats:
        csv_path = output_dir / f"{stem}.csv"
        frame.to_csv(csv_path, index=False)
        paths["csv"] = csv_path

    if "parquet" in output_formats:
        parquet_path = output_dir / f"{stem}.parquet"
        frame.to_parquet(parquet_path, index=False)
        paths["parquet"] = parquet_path

    return paths


def _build_run_summary(
    frame: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the compact JSON summary written beside exported datasets."""

    # Keep raw schema and feature-name lists together so downstream notebooks can
    # inspect one file to understand both structure and generated outputs.
    feature_columns = [
        column for column in frame.columns if column not in IDENTIFIER_COLUMN_SET
    ]
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "features": feature_columns,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
