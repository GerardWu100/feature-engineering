# Simple Feature Engineering

This project builds categorized stock features from OHLCV market data.

OHLCV means **open**, **high**, **low**, **close**, and **volume**. The workflow is intentionally small:

```text
load data -> clean invalid rows -> compute categorized features -> export files
```

## What It Does

- Loads stock OHLCV data from ClickHouse or a local CSV file.
- Validates the run config before loading data, so bad feature names, category
  filters, output formats, and impossible windows fail with clear messages.
- Cleans impossible market-data rows.
- Computes features by category:
  - `returns`: price change features.
  - `trend`: direction and momentum features.
  - `volatility`: price movement size and instability.
  - `volume`: trading activity features.
  - `target`: forward-looking labels for supervised learning.
- Exports feature data to Parquet and/or CSV.
- Writes a small `feature_catalog.csv` with feature names, categories, formulas, and descriptions.

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
symbol,ts,open,high,low,close,volume
```

## Project Layout

```text
feature-engineering/
├── run.py
├── main.py
├── config.toml
├── src/
│   └── feature_engineering/
│       ├── features/
│       │   ├── returns.py
│       │   ├── trend.py
│       │   ├── volatility.py
│       │   ├── volume.py
│       │   └── registry.py
│       └── pipeline/
│           ├── config.py
│           ├── load.py
│           ├── clean.py
│           ├── engineer.py
│           ├── export.py
│           └── cli.py
└── tests/
```

## Adding A Feature

1. Put the function in the matching category file under `src/feature_engineering/features/`.
2. Decorate it with `@register(...)`.
3. Add a `[[features.params]]` entry in `config.toml`.

Example:

```toml
[[features.params]]
name = "ma_20"
fn = "moving_average"
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

An empty `include_categories` list allows all enabled features. `exclude_categories` removes matching categories after the include rule.

## Config Validation

The pipeline validates `config.toml` before loading data. Important checks include:

- `run.source` is `csv` or `clickhouse`.
- `run.output_formats` contains only `csv` and/or `parquet`.
- ClickHouse runs include non-empty `symbols`, `start_date`, and `end_date`.
- Feature `fn` values exist in the registry.
- Enabled feature `name` values are unique output columns.
- Category filters use real categories and do not both include and exclude the same category.
- Positive integer parameters such as `window`, `periods`, and `days` are at least 1.

## Outputs

Outputs are written to `output_dir` from `config.toml`, for example `outputs/stocks/`.

- `features_v{version}_{timestamp_with_microseconds}.parquet`
- `features_v{version}_{timestamp_with_microseconds}.csv`
- `feature_catalog.csv`
- `run_summary_v{version}_{timestamp}.json`

## Tests

```bash
uv run pytest -q
```
