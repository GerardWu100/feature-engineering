# Project Overview - Simple Feature Engineering

## Purpose

This project builds categorized stock OHLCV features for quantitative research.

OHLCV means open, high, low, close, and volume. The code intentionally follows one small workflow:

```text
validate config -> load data -> clean invalid rows -> compute categorized features -> export files
```

## Architecture

The implementation has one project namespace package with two main subpackages:

| Package | Responsibility |
|---|---|
| `feature_engineering.features/` | Feature formulas grouped by research category. |
| `feature_engineering.pipeline/` | Config validation, data loading, cleaning, feature application, and export. |

The pipeline does not contain feature math. Feature functions do not load or save files. This keeps responsibilities easy to trace.

## Feature Categories

| Category | Meaning | Examples |
|---|---|---|
| `returns` | Price change over time. | `log_return`, `simple_return` |
| `target` | Forward-looking labels for model training. | `next_n_day_return` |
| `trend` | Direction or momentum. | `moving_average`, `price_vs_sma`, `rate_of_change` |
| `volatility` | Size and instability of price movement. | `rolling_std`, `bar_range_pct` |
| `volume` | Trading activity and participation. | `volume_ratio`, `dollar_volume`, `volume_change` |

The `target` category should usually be excluded from live feature sets because it uses future information. For intraday rows, the forward return target uses the current bar close as the denominator, not the current day's final close.

## Inputs

The pipeline accepts either:

- ClickHouse stock OHLCV data from `firstrate.stocks`, or
- a local CSV with columns `symbol`, `ts`, `open`, `high`, `low`, `close`, and `volume`.

The parsed TOML config is validated before data loading. The validator catches
unsupported data sources, bad output formats, unknown feature functions,
duplicate enabled output feature names, unknown category filters, category
filter overlap, and non-positive integer parameters such as `window`, `periods`,
and `days`.

## Outputs

Outputs are written to `output_dir` from `config.toml`:

| Output | Purpose |
|---|---|
| `features_v{version}_{timestamp_with_microseconds}.parquet` | Main machine-readable feature dataset. |
| `features_v{version}_{timestamp_with_microseconds}.csv` | Inspection-friendly feature dataset. |
| `feature_catalog.csv` | Feature names, categories, formulas, and descriptions. |
| `run_summary_v{version}_{timestamp}.json` | Row counts, feature list, and written paths. |

## Important Assumptions

- Features are computed per symbol, so one ticker's history never enters another ticker's feature values.
- Config validation is a boundary check. After it passes, pipeline stages assume required config keys and feature names are valid.
- Rows are sorted by `symbol` and `ts` before feature computation.
- Cleaning drops clearly invalid OHLCV rows: missing numeric values, non-positive prices, `high < low`, and open/close outside the low-high range.
- Rolling windows are row-count windows in the simplified project. A 20-row moving average means the previous 20 observed bars for that symbol.
