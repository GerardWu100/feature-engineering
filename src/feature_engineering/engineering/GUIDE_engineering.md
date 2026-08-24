# GUIDE - engineering/

## Part 1 - Conceptual Explanation

This folder turns raw OHLCV bars into a feature dataset:

```text
load -> clean -> compute -> store
```

Load reads bars from a CSV file or ClickHouse, filters them by symbol, date
range, and trading session, and standardizes the columns to:
`symbol, timestamp, open, high, low, close, volume`.

Clean applies the configured data-quality checks and records what it drops.
Compute runs each enabled feature separately for each symbol. When session
resets are enabled, it also starts rolling windows and forward targets afresh
for each trading day, so they do not cross the overnight gap. Store writes the
dataset, a feature catalog, and a run summary. A stored run can also be loaded
back into memory.

`config.toml` controls the symbols, dates, features, output names, and feature
parameters. The same feature can appear more than once with different names
and parameters—for example, a 20-bar and a 50-bar moving average.

Every stage works with pandas DataFrames, so research code can import a stage
directly without writing files.

The loader rejects missing timestamps and duplicate symbol-timestamp keys.
Feature computation rejects duplicate output names and names that would replace
the `symbol` or `timestamp` identifier columns.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `load.py` | `load_ohlcv`: load from CSV or ClickHouse, filter by session and date, and handle time zones. |
| `clean.py` | `clean_ohlcv`: apply row-level data-quality rules and return a drop report. |
| `compute.py` | `compute_features`: resolve configured features from the registry and compute them by symbol. Also provides `target_column_names`, which lists a run’s forward-looking columns. |
| `store.py` | `save_features`: write the dataset, catalog, and run summary. `load_features`: load a stored run. `build_feature_catalog`: build the catalog. |
| `constants.py` | Shared column names, defaults, and the canonical symbol-and-time sort order. |
| `features/` | Feature formulas grouped by category; see `features/GUIDE_features.md`. |

Feature functions take keyword parameters directly, for example
`moving_average(frame, window=20)`. `compute.py` passes each configured
parameter block to the same function. Config validation rejects unknown
parameters before data is loaded.

`compute.py` also exposes `selected_feature_configs` and `resolve_feature` for
code that needs the active feature list without computing it. `store.py` uses
these when building the catalog.

`target_column_names(config)` returns the output columns registered as targets.
The user chooses column names in `features.parameters`, so a name by itself
does not reveal whether a column uses future data. Pass this list to
`evaluate_features(..., target_columns=...)` to keep targets out of the feature
set. With the shipped default `exclude_categories = ["target"]`, the list is
empty because that run produces no target columns.

## Part 3 - Short Journal

- 2026-08-15: Created `engineering/` from the former `pipeline/` and `features/` subpackages so the package has two clear parts: build features here, test them in `evaluation/`. Renamed `engineer.py` to `compute.py` and `export.py` to `store.py`, and added `load_features` so stored runs can be loaded without running the pipeline again.
- 2026-08-19: Loading now rejects missing timestamps, and computation protects identifier columns and duplicate feature names even when called directly.
- 2026-08-24: ClickHouse loads now re-check the session window after converting timestamps to exchange-local time, so a source column stored in another time zone fails at load time. A naive column holding UTC values still cannot be detected from the data and remains a contract on the source table. Added `target_column_names` so evaluation can exclude forward-looking columns.
