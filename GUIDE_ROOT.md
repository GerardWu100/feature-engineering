# GUIDE - Root-Level Files

## Part 1 - Conceptual Explanation

The root folder is intentionally thin. It holds the command wrappers, the single user-facing config, package metadata, tests, and project guides.

The main workflow is:

```text
config.toml
  -> run.py
  -> feature_engineering.pipeline.cli
  -> feature_engineering.pipeline.load
  -> feature_engineering.pipeline.clean
  -> feature_engineering.pipeline.engineer
  -> feature_engineering.features.registry and category files
  -> feature_engineering.pipeline.export
  -> outputs/stocks/
```

The project now supports one core use case: stock OHLCV feature engineering. Optional subsystems such as options features, transform steps, diagnostics, and backtesting proof-of-concepts were removed so the codebase matches the simpler research workflow.

`config.toml` is the control surface. It chooses the data source, symbols, date range, output formats, cleaning rules, and feature list. Feature category filters live in `[features]`.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `run.py` | Root command wrapper. Delegates to `feature_engineering.pipeline.cli.main`. |
| `config.toml` | Single feature engineering config. |
| `pyproject.toml` | Package metadata, dependencies, and console script definitions. |
| `README.md` | User-facing overview and run instructions. |
| `PROJECT_STRUCTURE.md` | Compact layout reference. |
| `PROJECT_OVERVIEW.md` | High-level architecture explanation. |
| `src/` | Importable implementation code. |
| `tests/` | Focused tests for feature math, pipeline behavior, and simplified structure. |

Start at `run.py` for execution, then read `src/feature_engineering/pipeline/cli.py` for the top-level workflow.

## Part 3 - Short Journal

- 2026-04-24: Simplified the project to one categorized stock OHLCV feature pipeline and removed options, diagnostics, transforms, and backtesting proof-of-concepts from the main package.
- 2026-04-26: Moved implementation packages under the `feature_engineering` namespace to avoid generic top-level imports.
- 2026-08-09: Removed the redundant entry-point wrappers (`main.py`, `src/main.py`, `src/run.py`) and the duplicate `feature-engineering` console script. `uv run python run.py` and the `feature-pipeline` script are the two remaining ways to run the CLI.
- 2026-08-10: Added the `next_n_bar_realized_vol` volatility target and the `feature_engineering.evaluation` subpackage (information coefficients, Newey-West regression, quantile analysis, and violin / spread-row / rolling-IC plots) for testing features against targets after computation.
