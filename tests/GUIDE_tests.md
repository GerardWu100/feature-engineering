# GUIDE - tests/

## Part 1 - Conceptual Explanation

The tests protect the project structure and the formulas most likely to break.

The suite uses small toy data, so expected values can be checked by hand. It
avoids database dependencies by testing CSV loading and pipeline stages
directly.

## Part 2 - Code Reference

| File | Purpose |
|---|---|
| `test_config_validation.py` | Checks config-boundary errors before the pipeline loads data, including misspelled options, unsafe output names, and invalid feature spans. |
| `test_simple_project_structure.py` | Verifies the two-part `engineering`/`evaluation` package layout and keeps the registry categories small. |
| `test_feature_math.py` | Checks return, target, trend, volatility, and volume formulas against manual calculations, including leakage-safe target math. |
| `test_simple_pipeline.py` | Checks CSV loading, OHLCV cleaning, category-filtered feature computation, symbol-isolated rolling windows, and saving plus pulling back stored feature datasets. |
| `test_feature_contracts.py` | Checks every registered feature obeys the shared output contract (index alignment, no name, no mutation). |
| `test_schema_contracts.py` | Checks the loader's standardized OHLCV schema, required timestamps, unique keys, and cleaning contract. |
| `test_time_semantics.py` | Checks timezone and session-filter handling in the loader. |
| `test_evaluation.py` | Checks IC, Newey-West regression, and quantile statistics against synthetic data with known relationships. |
| `test_evaluation_plots.py` | Smoke tests: every evaluation figure builds and saves, with panel and label invariants pinned. |

Run everything with:

```bash
uv run pytest -q
```

## Part 3 - Short Journal

- 2026-04-24: Replaced broad platform tests with focused tests for the simplified feature engineering workflow.
- 2026-04-26: Added regression coverage for the namespace layout, missing numeric OHLCV cleaning, volatility window semantics, and microsecond output names.
- 2026-05-14: Added regression coverage that rolling feature state resets at symbol boundaries.
- 2026-05-14: Added config validation tests so bad feature names, output formats, category filters, and impossible windows fail at the workflow boundary.
- 2026-08-10: Added `test_evaluation.py` (IC, Newey-West regression, quantile buckets on synthetic known-relationship data) and `test_evaluation_plots.py` (figure smoke tests), plus forward realized-volatility target math in `test_feature_math.py`.
- 2026-08-15: Removed `test_engines.py` together with the `engine/` subpackage; added a `load_features` round-trip test.
- 2026-08-19: Added edge-case coverage for missing timestamps, unsafe feature names, misspelled settings, invalid spans and regression lags, insufficient volatility windows, and empty finite plot samples.
