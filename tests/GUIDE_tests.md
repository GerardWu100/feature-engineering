# GUIDE - tests/

## Part 1 - Conceptual Explanation

The tests protect the simplified project shape and the formulas that are easy to break.

The suite uses small toy data so expected values can be checked by hand. It avoids database dependencies by testing CSV loading and pure pipeline stages directly.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `test_config_validation.py` | Checks config-boundary errors before the pipeline loads data. |
| `test_simple_project_structure.py` | Verifies the repo exposes the `feature_engineering` namespace package and the registry categories are small. |
| `test_feature_math.py` | Checks return, target, trend, volatility, and volume formulas against manual calculations, including leakage-safe target math. |
| `test_simple_pipeline.py` | Checks CSV loading, OHLCV cleaning, category-filtered feature engineering, symbol-isolated rolling windows, and output writing. |

Run everything with:

```bash
uv run pytest -q
```

## Part 3 - Short Journal

- 2026-04-24: Replaced broad platform tests with focused tests for the simplified feature engineering workflow.
- 2026-04-26: Added regression coverage for the namespace layout, missing numeric OHLCV cleaning, volatility window semantics, and microsecond output names.
- 2026-05-14: Added regression coverage that rolling feature state resets at symbol boundaries.
- 2026-05-14: Added config validation tests so bad feature names, output formats, category filters, and impossible windows fail at the workflow boundary.
