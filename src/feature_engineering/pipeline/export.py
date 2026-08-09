"""Export engineered features and a compact feature catalog."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering.features.registry import REGISTRY
from feature_engineering.pipeline.constants import IDENTIFIER_COLUMN_SET
from feature_engineering.pipeline.engineer import selected_feature_configs


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
    summary = _build_run_summary(frame, paths, config=config, generated_at=timestamp)
    # default=str keeps the dump robust if a config value is a TOML date/time
    # object rather than a string.
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    paths["summary_json"] = summary_path

    return paths


def build_feature_catalog(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build a small catalog for the feature columns in an exported dataset."""
    # Use the same selection rule as feature computation so the catalog always
    # describes exactly the columns the pipeline produced.
    active_by_name = {item["name"]: item for item in selected_feature_configs(config)}
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
    *,
    config: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    """Build the compact JSON summary written beside exported datasets.

    The summary serves two jobs:

    1. Reproducibility - it records when the run happened and the exact config
       that produced it, so a later reader can recreate the dataset.
    2. Feature health - it reports per-feature null counts and value ranges, so
       a broken or all-null feature is visible without opening the dataset.

    Parameters
    ----------
    frame
        Engineered feature dataset (identifier columns plus feature columns).
    paths
        Mapping of output type to written path.
    config
        Parsed pipeline config. Stored verbatim for reproducibility. It does not
        contain secrets; ClickHouse credentials live in environment variables.
    generated_at
        Run timestamp string shared with the output filenames.

    Returns
    -------
    dict
        JSON-serializable summary.
    """
    feature_columns = [
        column for column in frame.columns if column not in IDENTIFIER_COLUMN_SET
    ]
    return {
        "generated_at": generated_at,
        "rows": len(frame),
        "columns": list(frame.columns),
        "features": feature_columns,
        # Per-symbol row counts catch silently empty or short tickers.
        "rows_per_symbol": _rows_per_symbol(frame),
        # Per-feature health lets a reader spot all-null or constant features.
        "feature_health": _feature_health(frame, feature_columns),
        # Config snapshot makes the run reproducible from this one file.
        "config": config,
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def _rows_per_symbol(frame: pd.DataFrame) -> dict[str, int]:
    """Return a symbol -> row-count mapping, or empty when no symbol column."""
    if "symbol" not in frame.columns:
        return {}

    counts = frame["symbol"].value_counts()
    return {str(symbol): int(count) for symbol, count in counts.items()}


def _feature_health(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Return null counts and value ranges for each feature column.

    Leading nulls from warmup windows are expected; a ``null_count`` equal to
    the row count means the feature produced nothing and should be investigated.
    """
    total_rows = len(frame)
    health: dict[str, dict[str, Any]] = {}

    for column in feature_columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        all_null = null_count == total_rows

        # min/mean/max skip NaN by default, so no filtered copy is needed. They
        # are None when the column is entirely null so the JSON stays valid and
        # unambiguous (NaN is not legal JSON).
        health[column] = {
            "null_count": null_count,
            "null_pct": round(null_count / total_rows, 4) if total_rows else None,
            "min": None if all_null else float(series.min()),
            "mean": None if all_null else float(series.mean()),
            "max": None if all_null else float(series.max()),
        }

    return health
