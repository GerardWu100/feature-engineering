# Feature Engineering

Computes stock features from OHLCV (open, high, low, close, volume) data and
evaluates them against targets.

## What it does

The package has two parts:

1. **Feature engineering** (`feature_engineering.engineering`) builds feature
   datasets: `load data -> clean invalid rows -> compute features -> store files`.
2. **Feature evaluation** (`feature_engineering.evaluation`) tests stored
   features against forward-looking targets.

- Loads OHLCV bars from ClickHouse (`firstrate.stocks`) or a local CSV file.
- Validates `config.toml` before loading data, so invalid feature names,
  category filters, output formats, and windows fail with a clear message.
- Drops impossible rows: missing values, non-positive prices, `high < low`, and
  open/close values outside the low-high range.
- Computes features by category: `returns`, `trend` (moving average, rate of
  change, RSI, MACD), `volatility` (rolling standard deviation, bar range,
  ATR), `volume` (relative volume, dollar volume, VWAP), and `target`
  (forward-looking labels for supervised learning). Feature names and
  parameters come from `config.toml`, so users can change them without editing
  Python code.
- Stores Parquet and/or CSV, a `feature_catalog.csv` describing each feature,
  and a `run_summary` JSON for reproducibility. `load_features` reads a stored
  run back into a DataFrame.
- Evaluates features against targets with information coefficients, Newey-West
  regression, quantile spreads, and plots. See `feature_engineering.evaluation`.

See `GUIDE_ROOT.md` and `PROJECT_OVERVIEW.md` for architecture and data-flow
details. `PROJECT_OVERVIEW.md`, under "Important Assumptions", covers adjusted
prices, timezone handling, and the one-row-per-bar assumption.

## Requirements

- Python 3.13.
- ClickHouse, only if `run.source = "clickhouse"`. Set these in `.env` (see
  `.env.example`): `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY`.
- No external service is needed for `run.source = "csv"`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python run.py --config config.toml   # database or CSV run, via run.py
uv run feature-pipeline --config config.toml  # same thing, installed script
uv run pytest -q                              # run the test suite
```

As a library, without file I/O:

```python
from feature_engineering import clean_ohlcv, compute_features, evaluate_features, load_features

cleaned, report = clean_ohlcv(raw_ohlcv_frame)
features = compute_features(cleaned, config_dict)
table = evaluate_features(features, "next_20bar_realized_volatility", target_horizon_bars=20)

stored = load_features("outputs/stocks")  # pull the newest stored run back
```

Every feature is also a plain pandas-style function. Call it directly with
keyword parameters and no config file:

```python
from feature_engineering.engineering.features import (
    moving_average, relative_strength_index, vwap, next_n_bar_return,
)

ma20 = moving_average(frame, window=20)
rsi = relative_strength_index(frame)          # default window=14
target = next_n_bar_return(frame, bars=5)
session_vwap = vwap(frame)
```

Each function expects one symbol's OHLCV frame sorted by time and returns a
Series aligned with the frame's index. The config file is only needed for the
command-line pipeline.


`config_dict` is the plain dict shape produced by parsing `config.toml`. See
the module docstring in `src/feature_engineering/__init__.py` for the full set
of importable pieces (`validate_config`, `save_features`, plots, and the
individual evaluation functions).

## Configuration

`config.toml` is the only configuration file:

- `[run]`: `source` (`clickhouse` or `csv`), `symbols`, `start_date`,
  `end_date`, `session` (`regular`, `extended`, `full`), `exchange_timezone`,
  `output_formats`, `output_dir`.
- `[data_quality]`: which invalid-row checks to apply.
- `[features]`: `include_categories` / `exclude_categories` to run a feature
  subset without editing Python, and `reset_by_session` so rolling windows and
  forward targets do not cross the overnight gap on intraday bars.
- `[[features.parameters]]`: one block per feature, naming its `function` and
  parameters (for example `window`, `bars`, `fast`/`slow`/`signal`).

To add a feature, write its function in the matching category file under
`src/feature_engineering/engineering/features/`, decorate it with
`@register(...)`, and add a `[[features.parameters]]` entry.

To rename a feature column or change its parameters, edit its
`[[features.parameters]]` block. `name` sets the output column name; the other
keys (`window`, `bars`, `fast`/`slow`/`signal`, ...) set the feature parameters.

## Layout

```text
run.py            entry point, delegates to feature_engineering.cli
config.toml       single run configuration
src/feature_engineering/
  engineering/    load, clean, compute, store/pull feature datasets
    features/     feature formulas by category (returns, targets, trend,
                  volatility, volume)
  evaluation/     feature-versus-target testing and plots
  config.py       config validation
  cli.py          the load -> clean -> compute -> store workflow
tests/            pytest suite, including a toy CSV fixture
```

## Output

The pipeline writes these files to `output_dir` (default `outputs/stocks/`):

- `features_v{version}_{timestamp}.parquet` and/or `.csv` — the feature data.
- `feature_catalog.csv` — feature names, categories, formulas, descriptions.
- `run_summary_v{version}_{timestamp}.json` — config snapshot, rows per
  symbol, and per-feature null counts and value ranges.

All rights reserved. See [LICENSE](LICENSE).
