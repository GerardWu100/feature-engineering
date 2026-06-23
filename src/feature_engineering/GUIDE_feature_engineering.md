# GUIDE - feature_engineering/

## Part 1 - Conceptual Explanation

`feature_engineering/` is the project namespace package. It prevents this code from exposing generic top-level import names such as `features` and `pipeline`.

The package has three responsibilities below it:

```text
feature_engineering/
├── features/
├── engine/
└── pipeline/
```

`features/` contains quantitative formulas. `pipeline/` contains the workflow boundary and data movement: validate config, load OHLCV data, clean invalid rows, compute configured formulas, and write outputs. `engine/` provides two ways to run the registered features: a cached batch `FeatureEngine` for research and a true O(1) `OnlineFeatureEngine` for live, bar-by-bar use.

This boundary matters when the project is installed or used from notebooks. Python searches import locations in order. A generic import such as `features` can accidentally resolve to another package or local folder. An import such as `feature_engineering.features` points back to this project.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `__init__.py` | Marks the project namespace package. |
| `features/` | Feature formulas and registry metadata. |
| `engine/` | Cached batch `FeatureEngine` and incremental `OnlineFeatureEngine`. |
| `pipeline/` | CLI workflow, config validation, loading, cleaning, feature computation, and export. |

Start with `pipeline/cli.py` for execution flow, then read `features/registry.py` to see how configured feature names resolve to formulas. Read `engine/` for the in-memory research and live-streaming entry points.

## Part 3 - Short Journal

- 2026-04-26: Added the `feature_engineering` namespace package to reduce import-name collisions in installed and notebook workflows.
- 2026-05-14: Added an explicit config-validation boundary under `pipeline/`.
- 2026-06-23: Added the `engine/` subpackage so features can be run as a cached batch transform or as O(1) incremental live updates, alongside the file-writing CLI pipeline.
