# GUIDE - pipeline/

## Part 1 - Conceptual Explanation

`pipeline/` owns the simple data workflow:

```text
config.py -> load.py -> clean.py -> engineer.py -> export.py
```

`config.py` validates the parsed TOML dictionary before the workflow starts. It checks the data source, source-specific required keys, output formats, feature names, duplicate enabled output columns, category filters, and positive integer feature parameters. This keeps config-boundary failures in one place.

`load.py` reads OHLCV data from either a local CSV or ClickHouse. ClickHouse values are passed through query parameters, while the table name is validated as a simple SQL identifier before use. `clean.py` removes rows that violate basic market-data rules, including missing numeric OHLCV values. `engineer.py` applies configured feature functions one symbol at a time, using `feature_engineering.features.registry` as the menu. `export.py` writes the dataset, a feature catalog, and a small run summary.

Per-symbol feature computation is an important time-series boundary. A rolling average, lagged return, or forward target for one ticker must never use another ticker's rows. The engineer stage sorts by symbol and timestamp, keeps `symbol` and `ts` as identifiers, then builds each feature column from independent symbol slices before re-aligning the result to the sorted frame.

The pipeline intentionally does not contain feature math. That keeps data movement separate from quantitative formulas.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `cli.py` | Parses CLI arguments, loads TOML config, and runs the workflow. |
| `config.py` | Validates parsed config before loading, cleaning, engineering, or exporting. |
| `constants.py` | Shared OHLCV column names and SQL identifier rules used by multiple stages. |
| `load.py` | Loads OHLCV data from CSV or ClickHouse, with validation around ClickHouse query inputs. |
| `clean.py` | Drops invalid or missing OHLCV rows and returns a quality report. |
| `engineer.py` | Computes enabled feature columns with category filters and explicit per-symbol isolation. |
| `export.py` | Writes CSV/Parquet outputs, `feature_catalog.csv`, and run summary JSON. |

Start with `cli.py` to understand the full run sequence.

## Part 3 - Short Journal

- 2026-04-24: Pipeline stages were reduced to load, clean, engineer, and export; delete mode, transforms, options branching, diagnostics, and metadata sidecars were removed.
- 2026-04-26: Cleaning now drops rows with missing numeric OHLCV values, ClickHouse loading validates query boundaries, and exports use microsecond timestamps.
- 2026-04-26: Refined pipeline internals for readability by extracting small helper functions and naming session-time constants; behavior and public workflow are unchanged.
- 2026-05-14: Feature computation now uses an explicit per-symbol loop instead of a grouped callback so the ticker-isolation boundary is easier to audit.
- 2026-05-14: Added a config validation stage so bad TOML inputs fail before data loading starts.
- 2026-05-19: Centralized shared OHLCV column names and SQL identifier rules in `constants.py` so loader, cleaner, engineer, exporter, and validator read the same contract.
- 2026-06-23: `engineer.py` gained an optional `features.reset_by_session` switch that also isolates features by calendar day, so intraday row-count windows and forward shifts do not cross the overnight gap. `export.py` run summaries now embed the full config snapshot, rows per symbol, and per-feature null/min/mean/max health. `load.py` documents the adjusted-price and exchange-local-timestamp data contract.
