# Architecture Review And Bug Hunt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the likely bugs in the current OHLCV feature pipeline, criticize the architecture directly, and refactor the project into a safer, typed, testable research tool.

**Architecture:** Keep the current good idea: a small pipeline package for data movement and a features package for quantitative formulas. Strengthen the weak parts by adding explicit configuration, schema, timestamp, feature-output, and artifact contracts at system boundaries.

**Tech Stack:** Python 3.12, pandas, NumPy, ClickHouse Connect, TOML via `tomllib`, pytest, ruff, uv.

---

## Definitions

- **Architecture** means the main boundaries of the system: which modules own loading, cleaning, feature math, exporting, configuration, and validation.
- **Contract** means an explicit rule between modules. Example: every feature function must return one `pandas.Series` with the same index as the input symbol frame.
- **Schema** means the required columns and data types of a dataset. For this project, the core schema is `symbol`, `ts`, `open`, `high`, `low`, `close`, and `volume`.
- **Lookahead bias** means a feature or label accidentally uses information that would not have been known at prediction time.
- **OHLCV** means open, high, low, close, and volume market data.

## Current Architecture Critique

The project is better than a typical junior codebase because it is small and the main split is sensible:

```text
feature_engineering.pipeline = loading, cleaning, orchestration, exporting
feature_engineering.features = quantitative formulas
```

The problem is that the split is not protected by strong contracts. Most functions accept `dict[str, Any]`, so invalid configuration fails late with raw `KeyError` or silently changes behavior. Data schema validation is mixed into loading and cleaning instead of being a first-class boundary. Timestamp and session semantics are implicit, which is dangerous in intraday finance. Export artifacts do not fully describe how they were produced, so a later notebook can easily use the wrong dataset.

### Confirmed Weak Points From Code Reading

| Area | Evidence | Criticism | Risk |
|---|---|---|---|
| Config boundary | `src/feature_engineering/pipeline/cli.py` returns raw TOML dicts, then pipeline stages index into them. | No typed config object, no central validation, and no clear defaults contract. | Bad configs fail late, sometimes with unhelpful errors. |
| Feature selection | `src/feature_engineering/pipeline/engineer.py` looks up `REGISTRY[item["fn"]]` before `_resolve_feature()` gives a friendly unknown-feature error. | Validation is split and inconsistent. | Unknown functions can raise raw `KeyError` before the intended error path. |
| Feature output contract | `compute_features()` assigns whatever each feature returns into the output frame. | There is no length, index, duplicate-name, or infinite-value validation. | A broken feature can silently misalign rows or overwrite another feature column. |
| Timestamp semantics | `src/feature_engineering/pipeline/load.py` uses `toHour(ts)` in ClickHouse and `.dt.date` for CSV filtering. | Exchange timezone and bar calendar are not explicit. | Regular trading hour filters and daily targets can be wrong if timestamps are UTC or mixed timezone. |
| Data quality | `clean_ohlcv()` validates prices but not duplicate `(symbol, ts)` rows or negative volume. | Market-data invariants are incomplete. | Rolling windows can double-count duplicated bars, and volume features can produce misleading values. |
| Target placement | `next_n_day_return()` lives in `returns.py` with category `target`. | Formula category and leakage category are mixed. | A developer may include targets accidentally or misunderstand live-feature safety. |
| Artifacts | `feature_catalog.csv` is overwritten on every run. | Run artifacts are not fully immutable or self-describing. | Later analysis can pair a dataset with the wrong catalog. |
| Entrypoints | Root `run.py` and `main.py` duplicate packaged `src/run.py` and `src/main.py`. | There are more public entrypoints than the project needs. | Packaging/import behavior is harder to reason about. |

## Bug Hunt Targets

These are the first bugs and failure modes to prove with tests before refactoring:

1. Unknown feature function names should raise a clear `ValueError` with the configured feature name and function name.
2. Duplicate output feature names should be rejected before computation.
3. Unsupported output formats should be rejected instead of silently writing only catalog and summary files.
4. Empty output formats should be rejected.
5. Duplicate `(symbol, ts)` bars should be rejected or explicitly cleaned with a documented rule.
6. Negative volume should be rejected or cleaned.
7. Feature functions that return the wrong length or wrong index should fail loudly.
8. Feature functions that return positive or negative infinity should fail loudly while preserving legitimate warmup `NaN` values.
9. ClickHouse regular trading hour filtering should be tested against the intended exchange timezone.
10. Target generation should be tested on sparse intraday data so the "future day close" convention is explicit.

## Target Architecture

The refactored structure should look like this:

```text
src/feature_engineering/
├── features/
│   ├── registry.py
│   ├── returns.py
│   ├── targets.py
│   ├── trend.py
│   ├── volatility.py
│   └── volume.py
└── pipeline/
    ├── cli.py
    ├── clean.py
    ├── config.py
    ├── engineer.py
    ├── export.py
    ├── load.py
    └── schema.py
```

### Design Rules

1. Validate at system boundaries, then trust internal callers.
2. Convert TOML dictionaries into typed configuration objects once in `pipeline/config.py`.
3. Keep OHLCV schema validation in `pipeline/schema.py`.
4. Keep feature formulas pure: one symbol frame in, one aligned `Series` out.
5. Keep target features separate from live input features.
6. Make every exported run self-describing through an immutable manifest.
7. Remove duplicate entrypoints and keep one canonical command path.

## Implementation Tasks

### Task 1: Capture Current Baseline

**Files:**
- Read: `README.md`
- Read: `GUIDE_ROOT.md`
- Read: `src/feature_engineering/pipeline/GUIDE_pipeline.md`
- Read: `src/feature_engineering/features/GUIDE_features.md`
- Read: `tests/GUIDE_tests.md`

- [ ] Run the current tests.

```bash
uv run pytest -q
```

Expected result before refactoring:

```text
15 passed
```

- [ ] Run the current linter.

```bash
uv run ruff check .
```

Expected result before refactoring:

```text
All checks passed!
```

- [ ] Write down any changed baseline result in the final implementation notes if the output differs.

### Task 2: Add Failing Bug Tests For Config And Feature Contracts

**Files:**
- Create: `tests/test_config_contracts.py`
- Create: `tests/test_feature_contracts.py`
- Modify only after tests fail: `src/feature_engineering/pipeline/engineer.py`
- Modify only after tests fail: `src/feature_engineering/pipeline/export.py`

- [ ] Add tests that prove the current weak contracts.

Required test cases:

| Test | Expected final behavior |
|---|---|
| `test_unknown_feature_function_has_clear_error` | Raises `ValueError` containing the feature column name and unknown function name. |
| `test_duplicate_feature_names_are_rejected` | Raises `ValueError` before feature computation starts. |
| `test_unsupported_output_format_is_rejected` | Raises `ValueError` listing allowed values `csv` and `parquet`. |
| `test_empty_output_formats_are_rejected` | Raises `ValueError` explaining that at least one dataset format is required. |
| `test_feature_output_wrong_length_is_rejected` | Raises `ValueError` naming the bad feature. |
| `test_feature_output_wrong_index_is_rejected` | Raises `ValueError` naming the bad feature. |
| `test_feature_output_infinity_is_rejected` | Raises `ValueError` naming the bad feature and infinity problem. |

- [ ] Run the new tests and confirm they fail for the expected reasons.

```bash
uv run pytest tests/test_config_contracts.py tests/test_feature_contracts.py -q
```

Expected result at this point:

```text
FAILED
```

### Task 3: Introduce Typed Configuration

**Files:**
- Create: `src/feature_engineering/pipeline/config.py`
- Modify: `src/feature_engineering/pipeline/cli.py`
- Modify: `src/feature_engineering/pipeline/load.py`
- Modify: `src/feature_engineering/pipeline/engineer.py`
- Modify: `src/feature_engineering/pipeline/export.py`
- Modify: `tests/test_config_contracts.py`

- [ ] Create dataclasses in `pipeline/config.py`.

Required objects:

| Object | Responsibility |
|---|---|
| `RunConfig` | Source, table, input path, symbols, date range, session, output formats, output directory, version. |
| `DataQualityConfig` | Boolean cleaning rules. |
| `FeatureParam` | Output name, registry function name, enabled flag, and function-specific params. |
| `FeatureConfig` | Include categories, exclude categories, and feature params. |
| `PipelineConfig` | Root object containing run, data quality, and features. |

- [ ] Add `load_pipeline_config(config_path: Path) -> PipelineConfig`.

Validation rules:

| Rule | Error |
|---|---|
| `source` is not `csv` or `clickhouse` | `ValueError("run.source must be one of: csv, clickhouse")` |
| CSV source has no `input_path` | `ValueError("run.input_path is required when run.source is csv")` |
| ClickHouse source has no symbols | `ValueError("run.symbols must contain at least one symbol")` |
| Duplicate feature names | `ValueError("Duplicate feature names: ...")` |
| Unknown feature function | `ValueError("Unknown feature function for feature ...")` |
| Unsupported output format | `ValueError("Unsupported output format ... Allowed: csv, parquet")` |
| Empty output formats | `ValueError("run.output_formats must contain at least one dataset format")` |

- [ ] Update pipeline stages to accept `PipelineConfig` or the specific sub-config object they need.

The desired dependency direction is:

```text
cli.py -> config.py -> load.py -> clean.py -> engineer.py -> export.py
```

`load.py`, `engineer.py`, and `export.py` should not parse raw TOML dictionaries.

- [ ] Run focused tests.

```bash
uv run pytest tests/test_config_contracts.py -q
```

Expected final result:

```text
passed
```

### Task 4: Centralize OHLCV Schema And Market-Data Validation

**Files:**
- Create: `src/feature_engineering/pipeline/schema.py`
- Modify: `src/feature_engineering/pipeline/load.py`
- Modify: `src/feature_engineering/pipeline/clean.py`
- Create or modify: `tests/test_schema_contracts.py`
- Modify: `src/feature_engineering/pipeline/GUIDE_pipeline.md`

- [ ] Move shared OHLCV constants into `schema.py`.

Required constants:

```text
OHLCV_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]
IDENTIFIER_COLUMNS = ["symbol", "ts"]
NUMERIC_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
```

- [ ] Add `finalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame`.

Contract:

1. Require all OHLCV columns.
2. Parse `ts` to pandas datetime.
3. Coerce price and volume columns to numeric.
4. Strip whitespace from symbols.
5. Reject missing or empty symbols.
6. Sort by `symbol` and `ts`.
7. Reset the row index.

- [ ] Add duplicate-bar validation.

Rule:

```text
No duplicate (symbol, ts) rows may enter feature computation.
```

Raise:

```text
ValueError("Duplicate OHLCV bars found for symbol/timestamp pairs")
```

- [ ] Extend cleaning to handle invalid volume.

Rule:

```text
volume must be greater than or equal to zero
```

Reasoning:

Zero volume can be valid for some illiquid bars, but negative volume is invalid market data. Features that divide by rolling average volume must still turn undefined divisions into `NaN`.

- [ ] Run focused tests.

```bash
uv run pytest tests/test_schema_contracts.py tests/test_simple_pipeline.py -q
```

Expected final result:

```text
passed
```

### Task 5: Make Feature Computation Deterministic And Guarded

**Files:**
- Modify: `src/feature_engineering/pipeline/engineer.py`
- Modify: `tests/test_feature_contracts.py`
- Modify: `src/feature_engineering/pipeline/GUIDE_pipeline.md`

- [ ] Replace `groupby(...).apply(...)` with an explicit per-symbol loop.

Required behavior:

1. Preserve sorted row order.
2. Compute each feature on exactly one symbol at a time.
3. Concatenate the returned per-symbol series.
4. Reindex back to the sorted frame index.

- [ ] Validate every feature result.

Rules:

| Rule | Allowed? |
|---|---|
| Same index as the symbol frame | Required |
| Same length as the symbol frame | Required |
| Warmup `NaN` values | Allowed |
| Positive infinity or negative infinity | Rejected |
| Non-Series return value | Rejected |

- [ ] Keep raw OHLCV columns out of the exported feature dataset unless a config option is intentionally added later.

Reason:

The current compact output is fine for a first pipeline. The bug is not missing raw columns; the bug is weak validation.

- [ ] Run focused tests.

```bash
uv run pytest tests/test_feature_contracts.py tests/test_feature_math.py -q
```

Expected final result:

```text
passed
```

### Task 6: Separate Target Features From Return Features

**Files:**
- Create: `src/feature_engineering/features/targets.py`
- Modify: `src/feature_engineering/features/returns.py`
- Modify: `src/feature_engineering/features/registry.py`
- Modify: `src/feature_engineering/features/GUIDE_features.md`
- Modify: `tests/test_feature_math.py`
- Modify: `tests/test_simple_project_structure.py`

- [ ] Move `next_n_day_return()` from `returns.py` to `targets.py`.

Preserve behavior:

```text
target_t = future_day_close / current_bar_close - 1
```

- [ ] Register `feature_engineering.features.targets` in `registry.py`.

- [ ] Add tests for target safety.

Required cases:

| Case | Expected behavior |
|---|---|
| Intraday current bar before close | Uses current bar close as denominator. |
| Last available future sessions | Produces `NaN` where the future day is unavailable. |
| Target excluded by default config | Target column is absent unless category filters allow it. |

- [ ] Update docs to explain that targets are labels, not live input signals.

- [ ] Run focused tests.

```bash
uv run pytest tests/test_feature_math.py tests/test_simple_project_structure.py -q
```

Expected final result:

```text
passed
```

### Task 7: Make Time And Session Semantics Explicit

**Files:**
- Modify: `src/feature_engineering/pipeline/config.py`
- Modify: `src/feature_engineering/pipeline/load.py`
- Create or modify: `tests/test_time_semantics.py`
- Modify: `config.toml`
- Modify: `README.md`
- Modify: `src/feature_engineering/pipeline/GUIDE_pipeline.md`

- [ ] Add an exchange timezone setting.

Config key:

```toml
[run]
exchange_timezone = "America/New_York"
```

- [ ] Use the exchange timezone consistently for:

1. CSV date filtering.
2. ClickHouse session filtering.
3. Daily target grouping.

- [ ] For ClickHouse, avoid assuming that `toHour(ts)` uses the intended exchange timezone.

Implementation direction:

```text
Use ClickHouse timezone-aware conversion in the SQL filter, or fetch a date range wide enough and apply session filtering in pandas after timestamp normalization.
```

Recommendation:

```text
For correctness, filter sessions in pandas after loading unless ClickHouse timestamp timezone is proven and documented.
```

- [ ] Add tests for:

| Case | Expected behavior |
|---|---|
| UTC timestamp representing 09:30 New York time | Included in regular trading hours. |
| UTC timestamp representing 08:00 New York time | Excluded from regular trading hours. |
| Naive timestamp in CSV | Treated as exchange-local time and documented. |

- [ ] Run focused tests.

```bash
uv run pytest tests/test_time_semantics.py tests/test_simple_pipeline.py -q
```

Expected final result:

```text
passed
```

### Task 8: Improve Export Artifacts And Run Reproducibility

**Files:**
- Modify: `src/feature_engineering/pipeline/export.py`
- Modify: `tests/test_simple_pipeline.py`
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `src/feature_engineering/pipeline/GUIDE_pipeline.md`

- [ ] Write timestamped feature catalogs.

Current weak behavior:

```text
feature_catalog.csv is overwritten on every run.
```

Target behavior:

```text
feature_catalog_v{version}_{timestamp}.csv
feature_catalog_latest.csv
```

- [ ] Expand the run summary into a manifest.

Required manifest fields:

| Field | Meaning |
|---|---|
| `version` | Run version from config. |
| `created_at_utc` | Artifact creation time in UTC. |
| `source` | CSV or ClickHouse. |
| `input_path` | CSV path when source is CSV. |
| `table` | ClickHouse table when source is ClickHouse. |
| `symbols` | Symbols requested. |
| `start_date` | Start date requested. |
| `end_date` | End date requested. |
| `exchange_timezone` | Timezone used for date and session logic. |
| `rows` | Output row count. |
| `columns` | Output column names. |
| `features` | Feature columns. |
| `feature_params` | Enabled feature configuration. |
| `outputs` | Written output paths. |
| `config_hash` | Stable hash of the loaded config. |

- [ ] Keep the old summary name only if it is renamed to the clearer manifest name.

Preferred final artifact:

```text
run_manifest_v{version}_{timestamp}.json
```

- [ ] Run focused tests.

```bash
uv run pytest tests/test_simple_pipeline.py -q
```

Expected final result:

```text
passed
```

### Task 9: Remove Duplicate Entrypoints

**Files:**
- Modify: `pyproject.toml`
- Keep: `run.py`
- Remove: `main.py`
- Remove: `src/main.py`
- Remove: `src/run.py`
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `src/GUIDE_src.md`
- Modify: `tests/test_simple_project_structure.py`

- [ ] Keep one canonical installed command.

Final command:

```bash
uv run feature-pipeline --config config.toml
```

- [ ] Keep root `run.py` only as a local wrapper.

Reason:

It is useful for local research:

```bash
uv run python run.py --config config.toml
```

- [ ] Remove `feature-engineering = "main:main"` from `pyproject.toml`.

- [ ] Remove `py-modules = ["main", "run"]` from `pyproject.toml`.

- [ ] Run structure tests.

```bash
uv run pytest tests/test_simple_project_structure.py -q
```

Expected final result:

```text
passed
```

### Task 10: Update Documentation And Guides

**Files:**
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `PROJECT_OVERVIEW.md`
- Modify: `PROJECT_STRUCTURE.md`
- Modify: `src/GUIDE_src.md`
- Modify: `src/feature_engineering/GUIDE_feature_engineering.md`
- Modify: `src/feature_engineering/features/GUIDE_features.md`
- Modify: `src/feature_engineering/pipeline/GUIDE_pipeline.md`
- Modify: `tests/GUIDE_tests.md`

- [ ] Update the project layout to include `config.py`, `schema.py`, and `targets.py`.

- [ ] Document the new config validation rules.

- [ ] Document timestamp conventions.

Required wording:

```text
Naive CSV timestamps are interpreted as exchange-local timestamps using run.exchange_timezone.
```

- [ ] Document the target safety rule.

Required wording:

```text
Target features are labels for supervised learning and must be excluded from live input feature sets.
```

- [ ] Document artifact names and manifest fields.

- [ ] Run documentation-sensitive tests if they exist; otherwise run the full test suite.

```bash
uv run pytest -q
```

Expected final result:

```text
passed
```

### Task 11: Full Verification

**Files:**
- All modified source, tests, config, and docs.

- [ ] Run the full test suite.

```bash
uv run pytest -q
```

Expected final result:

```text
passed
```

- [ ] Run ruff.

```bash
uv run ruff check .
```

Expected final result:

```text
All checks passed!
```

- [ ] Run the toy pipeline with the checked-in toy config.

```bash
uv run python run.py --config tests/data/toy_config.toml
```

Expected final result:

```text
The command exits with status 0 and writes CSV, Parquet, catalog, and manifest artifacts under outputs/toy.
```

- [ ] Inspect the toy manifest.

Required checks:

1. `rows` equals the number of exported feature rows.
2. `features` matches the configured feature columns.
3. `config_hash` is present.
4. `exchange_timezone` is present.
5. `outputs` points to files that exist.

- [ ] Run git status.

```bash
git status --short
```

Expected final result:

```text
Only intentional source, test, docs, config, and generated-output changes are listed.
```

### Task 12: Commit The Improvement

**Files:**
- Stage all intentional changes.

- [ ] Stage the work.

```bash
git add .
```

- [ ] Commit the work.

```bash
git commit -m "refactor: harden feature pipeline architecture"
```

Expected final result:

```text
Git creates one commit containing the architecture hardening work.
```

## Implementation Priority

Do the work in this order:

1. Tests for bugs and contracts.
2. Typed config.
3. Schema validation.
4. Feature output validation.
5. Target separation.
6. Timezone and session semantics.
7. Export manifest.
8. Entrypoint cleanup.
9. Documentation.
10. Full verification and commit.

This order matters because typed config and schema validation are the foundation. Refactoring exports or entrypoints first would make the project look cleaner without making the quantitative outputs safer.

## Self-Review

- The plan focuses on the current project scope: one stock OHLCV feature pipeline.
- The plan does not add options, backtesting, diagnostics, model training, or a web app.
- Each suspected bug has a test target before implementation.
- Each architecture criticism maps to at least one implementation task.
- The plan preserves the useful simple split between `pipeline` and `features`.
- The plan removes duplicate public entrypoints because this project does not need backward compatibility unless explicitly requested.
