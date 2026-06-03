# GUIDE - feature_engineering/

## Part 1 - Conceptual Explanation

`feature_engineering/` is the project namespace package. It prevents this code from exposing generic top-level import names such as `features` and `pipeline`.

The package has two responsibilities below it:

```text
feature_engineering/
├── features/
└── pipeline/
```

`features/` contains quantitative formulas. `pipeline/` contains the workflow boundary and data movement: validate config, load OHLCV data, clean invalid rows, compute configured formulas, and write outputs.

This boundary matters when the project is installed or used from notebooks. Python searches import locations in order. A generic import such as `features` can accidentally resolve to another package or local folder. An import such as `feature_engineering.features` points back to this project.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `__init__.py` | Marks the project namespace package. |
| `features/` | Feature formulas and registry metadata. |
| `pipeline/` | CLI workflow, config validation, loading, cleaning, feature computation, and export. |

Start with `pipeline/cli.py` for execution flow, then read `features/registry.py` to see how configured feature names resolve to formulas.

## Part 3 - Short Journal

- 2026-04-26: Added the `feature_engineering` namespace package to reduce import-name collisions in installed and notebook workflows.
- 2026-05-14: Added an explicit config-validation boundary under `pipeline/`.
