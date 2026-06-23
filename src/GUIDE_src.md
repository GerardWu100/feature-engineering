# GUIDE - src/

## Part 1 - Conceptual Explanation

`src/` is the import root for the project. It contains the project namespace package:

```text
src/
└── feature_engineering/
    ├── features/
    └── pipeline/
```

`feature_engineering/` is the namespace package. A namespace package is the named import boundary that keeps this project's modules separate from generic packages named `features` or `pipeline`.

`feature_engineering/features/` owns feature math. Each file is a category: returns, trend, volatility, or volume. The registry imports those category files and exposes the feature menu used by config.

`feature_engineering/pipeline/` owns the workflow. It validates config, loads OHLCV data, cleans invalid rows, computes configured feature columns per symbol, and exports the result.

The split is deliberately simple:

```text
feature_engineering.pipeline = config validation, data movement, and orchestration
feature_engineering.features = quantitative formulas
```

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `main.py` | Packaged compatibility wrapper. |
| `run.py` | Packaged CLI wrapper. |
| `feature_engineering/` | Namespaced package for the project implementation. |
| `feature_engineering/features/` | Categorized stock feature functions and registry. |
| `feature_engineering/engine/` | Cached batch `FeatureEngine` and incremental `OnlineFeatureEngine`. |
| `feature_engineering/pipeline/` | Config validation, load, clean, engineer, export, and CLI workflow code. |

Read `feature_engineering/pipeline/cli.py` first to understand execution, then `feature_engineering/features/registry.py` to see the available features.

## Part 3 - Short Journal

- 2026-04-24: Replaced the old multi-subsystem package layout with two packages: `features` and `pipeline`.
- 2026-04-26: Wrapped `features` and `pipeline` in the `feature_engineering` namespace package.
- 2026-05-14: Added `pipeline/config.py` so config-boundary validation is separate from stage logic.
- 2026-06-23: Added the `feature_engineering/engine/` subpackage (batch + online feature engines) for in-memory research and live-streaming use.
