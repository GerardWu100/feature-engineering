# Project Overview - Feature Engineering and Evaluation

## Purpose

This project computes categorized stock OHLCV features for quantitative
research and evaluates them against targets.

OHLCV means open, high, low, close, and volume. The build workflow is:

```text
validate config -> load data -> clean invalid rows -> compute categorized features -> store files
```

## Architecture

The implementation has two major parts plus a thin workflow boundary:

| Package | Responsibility |
|---|---|
| `feature_engineering.engineering/` | Build features: data loading, cleaning, feature computation, and storing/pulling datasets. |
| `feature_engineering.engineering.features/` | Feature formulas grouped by research category. |
| `feature_engineering.evaluation/` | Test features: information coefficients, Newey-West regression, quantile analysis, and plots. |
| `feature_engineering.config` / `feature_engineering.cli` | Config validation and the command-line workflow. |

The workflow stages do not contain feature math. Feature functions do not load
or save files. Evaluation consumes the finished feature frame and never
computes features. These boundaries make each stage easy to inspect and test.

## Feature Categories

| Category | Meaning | Examples |
|---|---|---|
| `returns` | Price change over time. | `log_return`, `simple_return` |
| `target` | Forward-looking labels for model training. | `next_n_bar_return`, `next_n_bar_realized_volatility` |
| `trend` | Direction or momentum. | `moving_average`, `price_vs_moving_average`, `rate_of_change`, `relative_strength_index`, `macd_line`, `macd_signal`, `macd_histogram` |
| `volatility` | Size and instability of price movement. | `rolling_standard_deviation`, `bar_range_percent`, `average_true_range` |
| `volume` | Trading activity and participation. | `volume_ratio`, `dollar_volume`, `volume_change`, `vwap`, `price_vs_vwap` |

The `target` category should usually be excluded from live feature sets because it uses future information. The forward target `next_n_bar_return` is a simple return over a fixed number of bars: `close[t+bars] / close[t] - 1`. The volatility target `next_n_bar_realized_volatility` is the sample standard deviation of the next `bars` one-bar log returns, so models can predict instability instead of direction. For intraday bars, enable `features.reset_by_session` so the forward shift and rolling windows do not cross the overnight gap.

## User Control

Everything a user changes lives in `config.toml`, not in Python:

- which feature functions run (`[[features.parameters]]` blocks, `enabled`),
- what each output column is called (`name`),
- each feature's parameters (`window`, `bars`, `fast`/`slow`/`signal`, ...),
- category filters (`include_categories` / `exclude_categories`).

The same function can appear several times under different names with
different parameters.

## Inputs

The pipeline accepts either:

- ClickHouse stock OHLCV data from `firstrate.stocks`, or
- a local CSV with columns `symbol`, `timestamp`, `open`, `high`, `low`, `close`, and `volume`.

The parsed TOML configuration is validated before data loading. The validator catches
unsupported data sources, bad output formats, unknown feature functions,
duplicate enabled output feature names, unknown category filters, category
filter overlap, and non-positive integer parameters such as `window`, `periods`,
and `bars`.

## Outputs

Outputs are written to `output_dir` from `config.toml`:

| Output | Purpose |
|---|---|
| `features_v{version}_{timestamp_with_microseconds}.parquet` | Main machine-readable feature dataset. |
| `features_v{version}_{timestamp_with_microseconds}.csv` | Inspection-friendly feature dataset. |
| `feature_catalog.csv` | Feature names, categories, formulas, and descriptions. |
| `run_summary_v{version}_{timestamp}.json` | Run timestamp, full config snapshot, rows per symbol, per-feature health (nulls and ranges), and written paths. |

`load_features(output_dir)` pulls the newest stored dataset back into a
DataFrame; pass `run_stem` to pick a specific run.

## Important Assumptions

- Features are computed per symbol, so one ticker's history never enters another ticker's feature values.
- Config validation is a boundary check. After it passes, pipeline stages assume required config keys and feature names are valid.
- Rows are sorted by `symbol` and `timestamp` before feature computation.
- Cleaning drops clearly invalid OHLCV rows: missing numeric values, non-positive prices, `high < low`, and open/close outside the low-high range.
- Rolling windows count rows. A 20-row moving average means the previous 20
  observed bars for that symbol.
- For intraday bars, `features.reset_by_session` also isolates features by calendar day so windows and forward shifts never cross the overnight gap.
- Data contract: prices are assumed split- and dividend-adjusted, and `timestamp` is assumed to be in the exchange's local wall-clock time (US equities: US/Eastern). The ClickHouse session filter and the intraday reset both rely on the timestamp assumption.
