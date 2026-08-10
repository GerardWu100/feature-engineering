# GUIDE - pipeline/

## Part 1 - Conceptual Explanation

`pipeline/` owns the data workflow:

```text
config.py -> load.py -> clean.py -> engineer.py -> export.py
```

`config.py` validates the parsed TOML dictionary before the workflow starts. It
checks the data source, source-specific required keys, output formats, feature
names, duplicate enabled output columns, category filters, and positive integer
feature parameters. Keeping these checks at one boundary gives later stages a
stable configuration contract.

`load.py` reads OHLCV data from either a local CSV or ClickHouse. ClickHouse
values are passed through query parameters, while the table name is validated
as a simple SQL identifier before use. Both paths normalize timestamps to
naive, exchange-local wall-clock time (`run.exchange_timezone`, default
US/Eastern). Naive inputs are trusted as local; timezone-aware inputs are
converted. Both paths strip and validate symbols and reject duplicate
`(symbol, timestamp)` bars, which would otherwise double-count rows in rolling
windows. The `run.session` filter applies to both sources. CSV runs default to
`full` because daily files are stamped at midnight; ClickHouse runs default to
`regular`.

`clean.py` removes rows that violate basic market-data rules. `engineer.py`
applies configured feature functions one symbol at a time, using
`feature_engineering.features.registry` as the feature menu. `export.py` writes
the dataset, feature catalog, and run summary.

Per-symbol feature computation is an important time-series boundary. A rolling
average, lagged return, or forward target for one ticker must never use another
ticker's rows. The engineer stage sorts by symbol and timestamp, keeps
`symbol` and `timestamp` as identifiers, computes each feature on independent
symbol slices, and then aligns the result to the sorted frame.

The pipeline intentionally does not contain feature math. That keeps data movement separate from quantitative formulas.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `cli.py` | Parses CLI arguments, loads TOML config, and runs the workflow. |
| `config.py` | Validates parsed config before loading, cleaning, engineering, or exporting. |
| `constants.py` | Shared OHLCV column names, ClickHouse defaults, SQL identifier rules, and the canonical symbol/time sort helper. |
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
- 2026-08-09: Hardened the load and engineer boundaries (ported from the May architecture-hardening branch): timestamps are normalized to naive exchange-local time via `run.exchange_timezone`, the `run.session` filter now applies to CSV loads (defaulting to `full` so daily files survive), duplicate `(symbol, timestamp)` bars and empty symbols are rejected at load time, cleaning drops negative volume, and every per-group feature result is validated for type, length, index alignment, and infinities before it joins the output.
- 2026-08-09: Deduplicated cross-stage contracts: ClickHouse table/session defaults live in `constants.py`, the allowed-session set is derived from the loader's `SESSION_FILTER_SQL` mapping, and the symbol/time sort is one shared helper. `engineer.py` exposes `selected_feature_configs` and `resolve_feature` publicly (engines and the exporter reuse them) and builds the groupby once per run instead of once per feature. `clean.py` combines all rule masks in one pass with no intermediate frame copies; per-rule drop counts are unchanged.
