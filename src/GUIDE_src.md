# GUIDE - src/

## Part 1 - Conceptual Explanation

`src/` is the project's import root. It contains the project package:

```text
src/
└── feature_engineering/
    ├── engineering/
    │   └── features/
    ├── evaluation/
    ├── config.py
    └── cli.py
```

`feature_engineering/` is the named package that keeps this project's modules
separate from generic packages named `features` or `evaluation`.

The split is simple:

```text
feature_engineering.engineering = build features: load, clean, compute, store/pull
feature_engineering.engineering.features = the quantitative formulas by category
feature_engineering.evaluation  = does a computed feature predict its target?
feature_engineering.config      = validate config.toml at the boundary
feature_engineering.cli         = the load -> clean -> compute -> store workflow
```

`engineering/features/` contains the feature formulas. Each file covers one
category: returns, targets (forward-looking labels), trend, volatility, or
volume. The registry imports those category files and exposes the feature menu
used by config.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `feature_engineering/` | Package containing the project implementation. |
| `feature_engineering/engineering/` | Load, clean, compute, and store/pull feature datasets. |
| `feature_engineering/engineering/features/` | Categorized stock feature functions and registry. |
| `feature_engineering/evaluation/` | Feature-versus-target testing: information coefficients, Newey-West regression, quantiles, and plots. |
| `feature_engineering/config.py` | Config-boundary validation. |
| `feature_engineering/cli.py` | CLI workflow code. |

Read `feature_engineering/cli.py` first to follow execution, then
`feature_engineering/engineering/features/registry.py` to see the available
features.

## Part 3 - Short Journal

- 2026-04-24: Replaced the old multi-subsystem package layout with two packages: `features` and `pipeline`.
- 2026-04-26: Wrapped the implementation in the `feature_engineering` package to reduce generic import-name collisions.
- 2026-05-14: Added a dedicated config module so config-boundary validation is separate from stage logic.
- 2026-08-10: Added the `feature_engineering/evaluation/` subpackage for testing features against targets (IC, Newey-West regression, quantiles, plots).
- 2026-08-15: Restructured into two major parts, `engineering/` and `evaluation/`, and removed the unused `engine/` subpackage.
