# Feature Engineering

Builds stock features from OHLCV bars—open, high, low, close, and volume—and
tests those features against future targets.

## What it does

The package has two parts:

1. **Feature engineering** loads and cleans market data, computes the selected
   features, and stores the result.
2. **Feature evaluation** tests stored features against forward-looking targets.

The pipeline can:

- Load OHLCV bars from ClickHouse (`firstrate.stocks`) or a local CSV file.
- Validate `config.toml` before loading data. Invalid feature names, category
  filters, output formats, or windows fail early with a clear message.
- Drop impossible rows: rows with missing values, non-positive prices, a high
  below the low, or open/close values outside the low-high range.
- Compute features by category: returns, trend, volatility, volume, and target.
  The trend category includes moving averages, rate of change, relative
  strength index (RSI), and moving average convergence divergence (MACD). The
  feature names and parameters come from `config.toml`, so you can change them
  without editing Python code.
- Store Parquet and/or CSV data, a `feature_catalog.csv`, and a `run_summary`
  JSON file for reproducibility. `load_features` reads a stored run back into a
  DataFrame.
- Evaluate features with information coefficients, Newey-West regression,
  quantile spreads, and plots. See `feature_engineering.evaluation`.

See `GUIDE_ROOT.md` and `PROJECT_OVERVIEW.md` for the architecture and data
flow. The “Important Assumptions” section in `PROJECT_OVERVIEW.md` covers
adjusted prices, time zones, and the one-row-per-bar assumption.

## Requirements

- Python 3.13.
- ClickHouse only when `run.source = "clickhouse"`. Set these values in `.env`
  (see `.env.example`): `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`,
  `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, and
  `CLICKHOUSE_VERIFY`.
- No external service is needed when `run.source = "csv"`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python run.py --config config.toml   # database or CSV run
uv run feature-pipeline --config config.toml
uv run pytest -q
```

As a library, you can use the stages without writing files:

```python
from feature_engineering import (
    clean_ohlcv,
    compute_features,
    evaluate_features,
    load_features,
    target_column_names,
)

cleaned, report = clean_ohlcv(raw_ohlcv_frame)
features = compute_features(cleaned, config_dict)
table = evaluate_features(
    features,
    "next_20bar_realized_volatility",
    # Keep every forward-looking column out of the candidate feature set.
    target_columns=target_column_names(config_dict),
    target_horizon_bars=20,
)

stored = load_features("outputs/stocks")  # load the newest stored run
```

If a run keeps its target columns, always pass `target_columns`. Otherwise
`evaluate_features` may treat another future-based target as an ordinary
feature. It can then rank highly because it overlaps the target being tested,
not because it predicts anything.

Each feature is also available as a function. Call it directly with keyword
parameters; no config file is needed:

```python
from feature_engineering.engineering.features import (
    moving_average, relative_strength_index, vwap, next_n_bar_return,
)

ma20 = moving_average(frame, window=20)
rsi = relative_strength_index(frame)          # default window=14
target = next_n_bar_return(frame, bars=5)
session_vwap = vwap(frame)
```

Each function expects one symbol’s OHLCV frame sorted by time and returns a
Series with the same index. `config.toml` is only needed for the command-line
pipeline. `config_dict` is the plain dictionary produced by parsing that file.
See the module docstring in `src/feature_engineering/__init__.py` for the full
list of importable functions, including config validation, storage, plots, and
individual evaluation functions.

## Configuration

`config.toml` is the only configuration file:

- `[run]`: data source (`clickhouse` or `csv`), symbols, date range, session
  (`regular`, `extended`, or `full`), exchange time zone, output formats, and
  output directory.
- `[data_quality]`: the invalid-row checks to apply.
- `[features]`: feature categories to include or exclude, and whether rolling
  windows and forward targets reset at each session. Resetting prevents
  intraday calculations from crossing the overnight gap.
- `[[features.parameters]]`: one block per feature, including its function,
  output name, and parameters such as `window`, `bars`, `fast`, `slow`, and
  `signal`.

To add a feature, write its function in the matching category file under
`src/feature_engineering/engineering/features/`, decorate it with
`@register(...)`, and add a `[[features.parameters]]` entry.

To rename a feature or change its parameters, edit that feature’s configuration
block. `name` sets the output column name; the other keys set the feature’s
parameters.

## Layout

```text
run.py            entry point; delegates to feature_engineering.cli
config.toml       single run configuration
src/feature_engineering/
  engineering/    load, clean, compute, and store feature datasets
    features/     feature formulas by category
  evaluation/     feature-versus-target tests and plots
  config.py       config validation
  cli.py          load -> clean -> compute -> store workflow
tests/            pytest suite, including a toy CSV fixture
```

## Output

The pipeline writes these files to `output_dir` (default:
`outputs/stocks/`):

- `features_v{version}_{timestamp}.parquet` and/or `.csv`: feature data.
- `feature_catalog.csv`: feature names, categories, formulas, and descriptions.
- `run_summary_v{version}_{timestamp}.json`: a config snapshot, row counts by
  symbol, and null counts and value ranges for each feature.

All rights reserved. See [LICENSE](LICENSE).
