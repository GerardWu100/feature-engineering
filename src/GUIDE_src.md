# GUIDE - src/

## Part 1 - Conceptual Explanation

`src/` is the import root for the project. It contains the project package:

```text
src/
└── feature_engineering/
    ├── features/
    ├── engine/
    ├── pipeline/
    └── evaluation/
```

`feature_engineering/` is the named import boundary that keeps this project's
modules separate from generic packages named `features` or `pipeline`.

`feature_engineering/features/` owns feature math. Each file is a category: returns, trend, volatility, or volume. The registry imports those category files and exposes the feature menu used by config.

`feature_engineering/pipeline/` owns the workflow. It validates config, loads OHLCV data, cleans invalid rows, computes configured feature columns per symbol, and exports the result.

`feature_engineering/evaluation/` owns feature testing after computation:
information coefficients, Newey-West regression, quantile analysis, and plots.

The split is deliberately simple:

```text
feature_engineering.pipeline   = config validation, data movement, and orchestration
feature_engineering.features   = quantitative formulas
feature_engineering.engine     = batch and online execution of registered features
feature_engineering.evaluation = does a computed feature predict its target?
```

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `feature_engineering/` | Package containing the project implementation. |
| `feature_engineering/features/` | Categorized stock feature functions and registry. |
| `feature_engineering/engine/` | Cached batch `FeatureEngine` and incremental `OnlineFeatureEngine`. |
| `feature_engineering/pipeline/` | Config validation, load, clean, engineer, export, and CLI workflow code. |
| `feature_engineering/evaluation/` | Feature-versus-target testing: information coefficients, Newey-West regression, quantiles, and plots. |

Read `feature_engineering/pipeline/cli.py` first to understand execution, then `feature_engineering/features/registry.py` to see the available features.

## Part 3 - Short Journal

- 2026-04-24: Replaced the old multi-subsystem package layout with two packages: `features` and `pipeline`.
- 2026-04-26: Wrapped `features` and `pipeline` in the `feature_engineering` package to reduce generic import-name collisions.
- 2026-05-14: Added `pipeline/config.py` so config-boundary validation is separate from stage logic.
- 2026-06-23: Added the `feature_engineering/engine/` subpackage (batch + online feature engines) for in-memory research and live-streaming use.
- 2026-08-10: Added the `feature_engineering/evaluation/` subpackage for testing features against targets (IC, Newey-West regression, quantiles, plots).
