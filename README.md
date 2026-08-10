# Simple Feature Engineering

This project computes categorized stock features from OHLCV market data.

OHLCV means **open**, **high**, **low**, **close**, and **volume**. The workflow is deliberately small:

```text
load data -> clean invalid rows -> compute categorized features -> export files
```

## What the Project Does

- Loads stock OHLCV data from ClickHouse or a local CSV (comma-separated values)
  file.
- Validates the run config before loading data, so bad feature names, category
  filters, output formats, and impossible windows fail with clear messages.
- Cleans impossible market-data rows.
- Computes features by category:
  - `returns`: price change features (log and simple returns).
  - `trend`: direction and momentum (moving average, price versus moving
    average, rate of change, Relative Strength Index, and Moving Average
    Convergence/Divergence line, signal, and histogram).
  - `volatility`: price movement size and instability (rolling standard
    deviation of returns, bar range percent, average true range).
  - `volume`: trading activity (relative volume, dollar volume, volume change,
    volume-weighted average price, and price versus volume-weighted average
    price).
  - `target`: forward-looking labels for supervised learning
    (`next_n_bar_return` for direction, `next_n_bar_realized_volatility` for
    volatility: the standard deviation of the next N one-bar log returns).
- Exports feature data to Parquet and/or CSV.
- Writes a small `feature_catalog.csv` with feature names, categories, formulas, and descriptions.
- Evaluates features against targets (`feature_engineering.evaluation`) with
  information coefficients, Newey-West regression, quantile analysis, and
  plots. See [Testing Features Against Targets](#testing-features-against-targets).

## Install

```bash
uv sync
```

## Run

Database run:

```bash
uv run python run.py --config config.toml
```

Installed script:

```bash
uv run feature-pipeline --config config.toml
```

For ClickHouse, create `.env` in the project root:

```bash
CLICKHOUSE_HOST=127.0.0.1
CLICKHOUSE_PORT=50050
CLICKHOUSE_USER=your_user
CLICKHOUSE_PASSWORD=your_password
CLICKHOUSE_SECURE=false
CLICKHOUSE_VERIFY=false
```

For a local CSV run, set this in `config.toml`:

```toml
[run]
source = "csv"
input_path = "data/raw/prices.csv"
```

The CSV must include:

```text
symbol,timestamp,open,high,low,close,volume
```

## Project Layout

```text
feature-engineering/
├── run.py
├── config.toml
├── src/
│   └── feature_engineering/
│       ├── features/
│       │   ├── returns.py
│       │   ├── trend.py
│       │   ├── volatility.py
│       │   ├── volume.py
│       │   └── registry.py
│       ├── engine/
│       │   ├── batch.py
│       │   └── online.py
│       ├── pipeline/
│       │   ├── config.py
│       │   ├── load.py
│       │   ├── clean.py
│       │   ├── engineer.py
│       │   ├── export.py
│       │   └── cli.py
│       └── evaluation/
│           ├── ic.py
│           ├── regression.py
│           ├── quantiles.py
│           ├── summary.py
│           └── plots.py
└── tests/
```

## Adding a Feature

1. Put the function in the matching category file under `src/feature_engineering/features/`.
2. Decorate it with `@register(...)`.
3. Add a `[[features.parameters]]` entry in `config.toml`.
4. For live use, add a constant-time accumulator in `engine/online.py` and
   register it in `ONLINE_FEATURE_FACTORIES`. The equivalence test then checks
   it against the batch version.

Example:

```toml
[[features.parameters]]
name = "moving_average_20"
function = "moving_average"
window = 20
enabled = true
```

## Category Filters

Use category filters to run grouped feature sets without editing Python:

```toml
[features]
include_categories = ["returns", "trend"]
exclude_categories = ["target"]
```

An empty `include_categories` list allows every enabled feature. The
`exclude_categories` rule is applied after the include rule, so an excluded
category is always removed from the result.

## Intraday Session Reset

Rolling features use row-count windows. With intraday bars, a window could
otherwise reach across the overnight gap: a 20-bar average at 09:30 could
include the previous session's final bars. Enable a per-day reset so rolling
windows and forward targets stay inside one calendar day:

```toml
[features]
reset_by_session = true
```

Leave it `false` (the default) for daily bars, where one row already represents
one day. This setting relies on `timestamp` being in the exchange's local time;
see [Data Contract](#data-contract).

## Data Contract

The loader assumes the following:

- Prices are split- and dividend-adjusted. The pipeline does not adjust for
  corporate actions, so unadjusted prices would turn a split into a fake return.
- Naive `timestamp` values are already in the exchange's local wall-clock time
  (US equities: US/Eastern). Timezone-aware values, such as UTC exports, are
  converted to `run.exchange_timezone` and stored without timezone information.
  The session filter and intraday reset therefore operate on exchange-local
  time.
- One row per symbol per bar. Duplicate `(symbol, timestamp)` bars fail the load
  loudly because they would double-count rows inside every rolling window.

The `run.session` filter (`regular`, `extended`, `full`) applies to both data
sources. CSV runs default to `full` because daily files are stamped at
midnight; ClickHouse runs default to `regular`.

## Use as a Module

The stages are importable for in-memory use inside research or trading code, with
no file I/O:

```python
from feature_engineering import clean_ohlcv, compute_features

cleaned, report = clean_ohlcv(raw_ohlcv_frame)
features = compute_features(cleaned, config_dict)
```

`config_dict` is the same plain dict shape that `config.toml` parses into.

For repeated use, two engines run the registered features without resolving the
configuration on every call:

```python
from feature_engineering import FeatureEngine, OnlineFeatureEngine

# Research or backtest: resolve the config once, then transform many frames.
engine = FeatureEngine(config_dict)
features = engine.transform(cleaned)

# Live trading: constant-time work per bar. Feed one bar at a time.
live = OnlineFeatureEngine(config_dict)  # rejects forward-looking targets
for bar in stream:  # bar has symbol, timestamp, OHLCV keys
    values = live.update(bar)  # -> {feature_name: value}
```

The online accumulators reproduce the batch feature formulas. An equivalence
test in `tests/test_engines.py` enforces that contract. Forward-looking
`target` features cannot be served online: train with batch output that includes
the target, then serve live features without it.

## Testing Features Against Targets

After computing features, `feature_engineering.evaluation` tests whether a
feature is associated with or predicts a target. The one-call entry point is:

```python
from feature_engineering import evaluate_features

table = evaluate_features(
    features,  # the frame compute_features returned
    "next_20bar_realized_volatility",  # target column to test against
    target_horizon_bars=20,  # horizon, so inference covers the overlap
)
```

The table has one row per feature, ranked by absolute t-statistic:

- `mean_time_series_ic`: time-series Spearman information coefficient (rank
  correlation between the feature now and the target later), averaged across
  symbols. This is descriptive, not a significance test.
- `beta`, `t_statistic`, `p_value`: regression of the target on the feature
  after per-symbol standardization, with Driscoll-Kraay standard errors. The
  estimator applies a Newey-West kernel to per-timestamp score sums. It handles
  both serial correlation from overlapping forward windows and dependence
  between symbols at the same time. This is the significance column.
- `quantile_spread`: mean target in the top feature quintile minus the bottom.

Screening many features creates a multiple-testing problem: with 20 features,
one t-statistic near 2 can appear by chance. Treat the table as a ranking and
confirm any result out of sample.

The pieces are importable individually (`time_series_ic`,
`cross_sectional_ic`, `rolling_ic`, `newey_west_regression`,
`target_by_feature_quantile`) from `feature_engineering.evaluation`.

The same module provides plots:

```python
from feature_engineering.evaluation import (
    violin_by_quantile,
    spread_rows_by_state,
    rolling_ic_panels,
)

violin_by_quantile(
    features, "rolling_standard_deviation_20", "next_20bar_realized_volatility"
)
spread_rows_by_state(
    features, ["rsi_14", "rolling_standard_deviation_20"], "next_1bar_return"
)
rolling_ic_panels(features, ["rsi_14"], "next_1bar_return", window=252)
```

Each function returns a Matplotlib figure. Pass `save_path=` to also write a PNG:

- `violin_by_quantile`: the target distribution inside each feature
  quintile, with the quartile bar and mean dot drawn on top.
- `spread_rows_by_state`: each feature split into low, neutral, and high states, one
  row per state showing the mean dot, middle-half bar, and 10th-90th line
  against the all-rows dashed line.
- `rolling_ic_panels`: trailing-window rank information coefficient through time per feature, with
  shading where the full-sample sign reverses.

## Config Validation

The pipeline validates `config.toml` before loading data. Important checks include:

- `run.source` is `csv` or `clickhouse`.
- `run.output_formats` contains only `csv` and/or `parquet`.
- ClickHouse runs include non-empty `symbols`, `start_date`, and `end_date`.
- Feature `function` values exist in the registry.
- Enabled feature `name` values are unique output columns.
- Category filters use real categories and do not both include and exclude the same category.
- Positive integer parameters such as `window`, `periods`, and `bars` are at least 1.

## Outputs

Outputs are written to `output_dir` from `config.toml`, for example `outputs/stocks/`.

- `features_v{version}_{timestamp_with_microseconds}.parquet`
- `features_v{version}_{timestamp_with_microseconds}.csv`
- `feature_catalog.csv`
- `run_summary_v{version}_{timestamp}.json` - run timestamp, the full config
  snapshot, rows per symbol, and per-feature health (null counts and value
  ranges) for reproducibility and quick checks.

## Tests

```bash
uv run pytest -q
```
