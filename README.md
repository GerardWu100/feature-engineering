# Feature Engineering

Computes categorized stock features from OHLCV (open, high, low, close,
volume) market data, for research and for live feature serving.

## What it does

The pipeline runs one small workflow:

```text
load data -> clean invalid rows -> compute categorized features -> export files
```

- Loads OHLCV bars from ClickHouse (`firstrate.stocks`) or a local CSV file.
- Validates `config.toml` before loading data, so bad feature names, category
  filters, output formats, or windows fail with a clear message instead of a
  silent bad run.
- Drops impossible rows: missing values, non-positive prices, `high < low`,
  open/close outside the low-high range.
- Computes features by category: `returns`, `trend` (moving average, rate of
  change, RSI, MACD), `volatility` (rolling standard deviation, bar range,
  ATR), `volume` (relative volume, dollar volume, VWAP), and `target`
  (forward-looking labels for supervised learning).
- Exports Parquet and/or CSV, plus a `feature_catalog.csv` describing every
  feature and a `run_summary` JSON for reproducibility.
- Evaluates features against targets (information coefficients, Newey-West
  regression, quantile spread, and plots) — see `feature_engineering.evaluation`.

Two engines share one set of feature formulas: `FeatureEngine` for batch
research and backtests, `OnlineFeatureEngine` for constant-time, one-bar-at-a-
time updates in live trading. An equivalence test (`tests/test_engines.py`)
checks that the online accumulators reproduce the batch formulas.

Architecture and data-flow details live in `GUIDE_ROOT.md` and
`PROJECT_OVERVIEW.md`. Assumptions about adjusted prices, timezone handling,
and one-row-per-bar are in `PROJECT_OVERVIEW.md` under "Important Assumptions".

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

As a library, with no file I/O:

```python
from feature_engineering import clean_ohlcv, compute_features, evaluate_features

cleaned, report = clean_ohlcv(raw_ohlcv_frame)
features = compute_features(cleaned, config_dict)
table = evaluate_features(features, "next_20bar_realized_volatility", target_horizon_bars=20)
```

`config_dict` is the plain dict shape `config.toml` parses into. See the
module docstring in `src/feature_engineering/__init__.py` for the full set of
importable pieces (`FeatureEngine`, `OnlineFeatureEngine`, `validate_config`,
plots, and the individual evaluation functions).

## Configuration

`config.toml` is the single control surface:

- `[run]`: `source` (`clickhouse` or `csv`), `symbols`, `start_date`,
  `end_date`, `session` (`regular`, `extended`, `full`), `exchange_timezone`,
  `output_formats`, `output_dir`.
- `[data_quality]`: which invalid-row checks to apply.
- `[features]`: `include_categories` / `exclude_categories` to run a feature
  subset without editing Python, and `reset_by_session` so rolling windows and
  forward targets do not cross the overnight gap on intraday bars.
- `[[features.parameters]]`: one block per feature, naming its `function` and
  parameters (for example `window`, `bars`, `fast`/`slow`/`signal`).

To add a feature: write the function in the matching file under
`src/feature_engineering/features/`, decorate it with `@register(...)`, and
add a `[[features.parameters]]` entry.

## Layout

```text
run.py            entry point, delegates to feature_engineering.pipeline.cli
config.toml       single run configuration
src/feature_engineering/
  features/       feature formulas by category
  engine/         FeatureEngine (batch), OnlineFeatureEngine (live)
  pipeline/       config validation, load, clean, engineer, export, CLI
  evaluation/     feature-versus-target testing and plots
tests/            pytest suite, including a toy CSV fixture
```

## Output

Written to `output_dir` (default `outputs/stocks/`):

- `features_v{version}_{timestamp}.parquet` and/or `.csv` — the feature data.
- `feature_catalog.csv` — feature names, categories, formulas, descriptions.
- `run_summary_v{version}_{timestamp}.json` — config snapshot, rows per
  symbol, and per-feature null counts and value ranges.

All rights reserved. See [LICENSE](LICENSE).
