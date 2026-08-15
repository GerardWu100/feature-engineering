# GUIDE - Root-Level Files

## Part 1 - Conceptual Explanation

The root folder is intentionally thin. It holds the command wrapper, the single
user-facing configuration file, package metadata, tests, and project guides.

The main workflow is:

```text
config.toml
  -> run.py
  -> feature_engineering.cli
  -> feature_engineering.engineering.load
  -> feature_engineering.engineering.clean
  -> feature_engineering.engineering.compute
  -> feature_engineering.engineering.features (registry and category files)
  -> feature_engineering.engineering.store
  -> outputs/stocks/
```

The package has two major parts: `engineering` builds and stores feature
datasets, `evaluation` tests them against targets.

`config.toml` is the control surface. It chooses the data source, symbols, date
range, output formats, cleaning rules, and the feature list, including each
feature column's name and parameters. Feature-category filters live in
`[features]`.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `run.py` | Root command wrapper. Delegates to `feature_engineering.cli.main`. |
| `config.toml` | Single feature engineering config. |
| `pyproject.toml` | Package metadata, dependencies, and console script definitions. |
| `README.md` | User-facing overview and run instructions. |
| `PROJECT_STRUCTURE.md` | Compact layout reference. |
| `PROJECT_OVERVIEW.md` | High-level architecture explanation. |
| `src/` | Importable implementation code. |
| `tests/` | Focused tests for feature math, pipeline behavior, and package structure. |

Start at `run.py` for execution, then read `src/feature_engineering/cli.py` for the top-level workflow.

## Part 3 - Short Journal

- 2026-04-24: Simplified the project to one categorized stock OHLCV feature pipeline and removed options, diagnostics, transforms, and backtesting proof-of-concepts from the main package.
- 2026-04-26: Moved implementation packages under the `feature_engineering` namespace to avoid generic top-level imports.
- 2026-08-09: Removed the redundant entry-point wrappers (`main.py`, `src/main.py`, `src/run.py`) and the duplicate `feature-engineering` console script. `uv run python run.py` and the `feature-pipeline` script are the two remaining ways to run the CLI.
- 2026-08-10: Added the `next_n_bar_realized_volatility` volatility target and the `feature_engineering.evaluation` subpackage (information coefficients, Newey-West regression, quantile analysis, and violin / spread-row / rolling-IC plots) for testing features against targets after computation.
- 2026-08-10: Renamed every non-standard abbreviation in the user-facing surface so the config and data contract read as plain words: config key `fn` -> `function`, `[[features.params]]` -> `[[features.parameters]]`, column `ts` -> `timestamp` (the ClickHouse column is still `ts` and is aliased in the query), session `rth` -> `regular`, and the feature functions `rolling_std` -> `rolling_standard_deviation`, `bar_range_pct` -> `bar_range_percent`, `price_vs_sma` -> `price_vs_moving_average`, `next_n_bar_realized_vol` -> `next_n_bar_realized_volatility`. MACD and VWAP stay as-is because they are universally accepted finance terms.
- 2026-08-15: Restructured the package into two major parts, `engineering/` (formerly `features/` + `pipeline/`) and `evaluation/`, with `config.py` and `cli.py` at the package root. Moved targets into `features/targets.py`, renamed export to `save_features`, added `load_features`, and removed the unused `engine/` subpackage.
