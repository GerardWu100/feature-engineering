# GUIDE - feature_engineering/

## Part 1 - Conceptual Explanation

`feature_engineering/` is the project package. It keeps this code from exposing
generic top-level import names such as `features` or `evaluation`.

The package has two major parts plus a thin workflow boundary:

```text
feature_engineering/
├── engineering/    # Part 1: build features
│   └── features/   # the feature formulas, one file per category
├── evaluation/     # Part 2: test features against targets
├── config.py       # validate config.toml before anything runs
└── cli.py          # load -> clean -> compute -> store workflow
```

`engineering/` builds feature datasets: it loads OHLCV bars, cleans invalid
rows, computes the configured feature columns per symbol, and stores the
result on disk. It also provides the matching call for reading a stored run
back again. `engineering/features/` holds the formulas, grouped by category:
returns, targets (forward-looking labels), trend, volatility, and volume.

`evaluation/` tests whether computed features predict their targets using
information coefficients, Newey-West regression, quantile analysis, a one-call
summary table, and plots.

This name matters when the project is installed or used from notebooks. Python
searches import locations in order, so a generic import such as `features` can
resolve to another package or local folder. An import such as
`feature_engineering.engineering.features` points back to this project.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `__init__.py` | Defines the public package exports. |
| `engineering/` | Load, clean, compute, and store/pull feature datasets. |
| `engineering/features/` | Feature formulas and registry metadata. |
| `evaluation/` | Feature-versus-target testing: IC, Newey-West regression, quantiles, summary table, plots. |
| `config.py` | Config validation (`validate_config`) at the workflow boundary. |
| `cli.py` | CLI entry point (`main`, `run_pipeline`, `load_config`). |

Start with `cli.py` to follow execution, then read
`engineering/features/registry.py` to see how configured feature names resolve
to formulas, and `evaluation/` for feature testing after computation.

## Part 3 - Short Journal

- 2026-04-26: Added the `feature_engineering` package to reduce import-name collisions in installed and notebook workflows.
- 2026-05-14: Added an explicit config-validation boundary.
- 2026-08-10: Added the `evaluation/` subpackage (IC, Newey-West regression, quantile analysis, plots) and the `next_n_bar_realized_volatility` volatility target.
- 2026-08-15: Restructured into two major parts: `engineering/` (formerly `features/` + `pipeline/`) and `evaluation/`. Moved forward-looking targets into `features/targets.py`, renamed export to `save_features`, added `load_features` to pull stored datasets back, and removed the unused `engine/` subpackage (batch and online engines).
