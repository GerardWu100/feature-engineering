# GUIDE - engineering/

## Part 1 - Conceptual Explanation

`engineering/` builds the feature dataset. It turns raw OHLCV bars into stored
features in four stages:

```text
load -> clean -> compute -> store
```

Load reads bars from a CSV file or the ClickHouse database, filters by symbol,
date range, and trading session, and standardizes the schema to
`symbol, timestamp, open, high, low, close, volume`. Clean drops invalid rows
using the config's data-quality rules. Compute applies each enabled feature
from `features/` per symbol (and per session day when `reset_by_session` is
on), so rolling windows do not mix tickers or cross the overnight gap. Store
writes the dataset, a feature catalog, and a run summary to the output
directory. It can also load a stored run back into memory.

Users control the run from `config.toml`: which symbols and dates to load,
which features to compute, each feature column's name, and each feature's
parameters. The same feature function can appear more than once with different
names and parameters, such as a 20-bar and a 50-bar moving average.

Each stage is a plain function over pandas DataFrames. Research code can import
any stage directly and skip writing files.

Loading rejects missing timestamps and duplicate symbol-timestamp keys. Feature
computation rejects duplicate output names and names that would replace the
`symbol` or `timestamp` identifier columns.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `load.py` | `load_ohlcv`: CSV/ClickHouse loading, session and date filtering, timezone handling. |
| `clean.py` | `clean_ohlcv`: config-driven row-level data-quality rules plus a drop report. |
| `compute.py` | `compute_features`: resolve configured features against the registry and compute them per symbol. |
| `store.py` | `save_features` (dataset + catalog + run summary), `load_features` (pull a stored run back), `build_feature_catalog`. |
| `constants.py` | Shared column names, defaults, and the canonical `sort_by_symbol_and_time` ordering contract. |
| `features/` | The feature formulas by category; see `features/GUIDE_features.md`. |

Feature functions take keyword parameters directly
(`moving_average(frame, window=20)`). `compute.py` unpacks each config entry's
parameters into that same call. Config validation rejects stray parameter names
before loading data.

`compute.py` exposes `selected_feature_configs` and `resolve_feature` for code
that needs the active feature list without computing anything (the catalog in
`store.py` uses this).

## Part 3 - Short Journal

- 2026-08-15: Created `engineering/` from the former `pipeline/` and `features/` subpackages so the package has two clear parts: build features here, test them in `evaluation/`. Renamed `engineer.py` to `compute.py` and `export.py` to `store.py`, and added `load_features` so stored runs can be pulled back without re-running the pipeline.
- 2026-08-19: Loading now rejects missing timestamps, and computation protects identifier columns and duplicate feature names even when called directly.
