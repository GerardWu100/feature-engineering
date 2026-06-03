# Refactor Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the correctness, packaging, documentation, and category-selection issues found during the refactor review.

**Architecture:** Keep the new `src/app`, `src/engine`, `src/diagnostics`, and `src/research` domain layout. Fix the only data-contract bug in the incremental cache path, keep generated artifacts out of package builds, repair stale guide text, and add lightweight category-based feature selection without changing existing `[[features.params]]` semantics.

**Tech Stack:** Python 3.12, pandas, setuptools package discovery, TOML config, `uv`, `pytest`, `ruff`.

---

## File Structure

Modify these files:

- `src/engine/pipeline/engineer.py`: add date-window filtering when returning cached feature rows.
- `tests/test_pipeline_guards.py`: add a regression test proving narrower cached date ranges return only requested dates.
- `pyproject.toml`: exclude generated/runtime artifact packages from setuptools discovery.
- `.gitignore`: ignore `dist/`, build artifacts, diagnostics images, feature-quality generated outputs, and nested research outputs consistently.
- `src/GUIDE_src.md`: replace the stale old tree with the new domain-package tree.
- `tests/GUIDE_tests.md`: correct fake filenames created by broad text replacement.
- `GUIDE_ROOT.md`, `PROJECT_OVERVIEW.md`, `PROJECT_STRUCTURE.md`, `README.md`: clean prose corruption where dotted import names replaced normal English words.
- `src/engine/pipeline/config_models.py`: validate optional category filters in `[features]`.
- `src/engine/pipeline/engineer.py`: apply category filters after `enabled` filtering.
- `tests/test_config_models.py`: add config-validation tests for category filters.
- `tests/test_pipeline_guards.py`: add compute test for category filtering.
- `src/engine/features/GUIDE_features.md`, `src/engine/pipeline/GUIDE_pipeline.md`, `README.md`: document category-based feature selection.

Create these files:

- `src/app/GUIDE_app.md`: document the application/command-line interface package.

Do not modify:

- `mynotes.md`.
- Any generated output content except through `.gitignore` or package-discovery rules.

---

### Task 1: Fix Incremental Cache Date Filtering

**Files:**
- Modify: `src/engine/pipeline/engineer.py`
- Test: `tests/test_pipeline_guards.py`

- [ ] **Step 1: Add the failing regression test**

Add this test method to `TestEngineerIncrementalGuards` in `tests/test_pipeline_guards.py`:

```python
    def test_compute_features_filters_cached_rows_to_requested_date_range(self) -> None:
        """A reusable cache must still be sliced to the config date window."""
        ts = pd.to_datetime(
            [
                "2024-01-02 09:30:00",
                "2024-01-03 09:30:00",
                "2024-01-04 09:30:00",
            ]
        ).tz_localize("America/New_York")
        cleaned_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "ts": ts[1:2],
                "open": [101.0],
                "high": [102.0],
                "low": [100.0],
                "close": [101.5],
                "volume": [1100.0],
            }
        )
        existing_df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "ts": ts,
                "log_return": [0.01, 0.02, 0.03],
            }
        )
        config = {
            "run": {
                "asset_class": "stocks",
                "table": "stocks",
                "symbols": ["AAPL"],
                "start_date": "2024-01-03",
                "end_date": "2024-01-03",
            },
            "features": {
                "params": [
                    {"name": "log_return", "fn": "log_return", "enabled": True},
                ]
            },
        }

        result = compute_features(cleaned_df, config, existing_df=existing_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ts"], ts[1])
        self.assertEqual(result.iloc[0]["log_return"], 0.02)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/test_pipeline_guards.py::TestEngineerIncrementalGuards::test_compute_features_filters_cached_rows_to_requested_date_range -q
```

Expected: fail because the current cached-symbol branch returns all three cached rows for `AAPL`.

- [ ] **Step 3: Add a date-window helper**

In `src/engine/pipeline/engineer.py`, add this helper after `_get_date_range()`:

```python
def _filter_to_config_window(
    df: pd.DataFrame,
    asset_class: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return rows whose feature date is inside the requested config window.

    Parameters
    ----------
    df : pd.DataFrame
        Cached or newly merged feature output.
    asset_class : str
        ``"options"`` uses ``trade_date``; all other asset classes use ``ts``.
    start_date : pd.Timestamp
        Inclusive start date from the run config.
    end_date : pd.Timestamp
        Inclusive end date from the run config.

    Returns
    -------
    pd.DataFrame
        A copy of the input limited to the requested date window.
    """
    if asset_class == "options":
        dates = pd.to_datetime(df["trade_date"]).dt.normalize()
    else:
        dates = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()

    in_window = (dates >= start_date) & (dates <= end_date)
    return df.loc[in_window].copy()
```

- [ ] **Step 4: Use the helper in the all-cached branch**

Replace the all-cached branch in `_apply_incremental()` with:

```python
    if not new_symbols:
        # All symbols are already cached, so return the requested symbol/date slice.
        logger.info(
            "All symbols cached — returning cache slice, no computation needed."
        )
        symbol_slice = existing_df[existing_df["symbol"].isin(config_symbols_set)]
        result = _filter_to_config_window(
            df=symbol_slice,
            asset_class=asset_class,
            start_date=config_start,
            end_date=config_end,
        )
        result = result.reset_index(drop=True)
        return result
```

- [ ] **Step 5: Use the helper after partial cache merge**

After sorting `result` in the partial-new-symbol path, filter by date before returning:

```python
    result = result.sort_values(sort_key).reset_index(drop=True)
    result = _filter_to_config_window(
        df=result,
        asset_class=asset_class,
        start_date=config_start,
        end_date=config_end,
    )
    result = result.reset_index(drop=True)
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
uv run pytest tests/test_pipeline_guards.py -q
```

Expected: all tests in `tests/test_pipeline_guards.py` pass.

- [ ] **Step 7: Commit**

```bash
git add src/engine/pipeline/engineer.py tests/test_pipeline_guards.py
git commit -m "fix: filter incremental cache by requested date range"
```

---

### Task 2: Exclude Generated Artifacts From Builds

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Confirm current build includes generated output modules**

Run:

```bash
uv build --sdist --wheel
```

Expected current bad evidence: build output includes files under `research/backtesting_frameworks_poc/outputs/`, such as `LeanRegressionAlgorithm.py` or `zipline_root/extension.py`.

- [ ] **Step 2: Add package-discovery excludes**

In `pyproject.toml`, update `[tool.setuptools.packages.find]` to include an `exclude` list:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = [
    "app*",
    "engine*",
    "diagnostics*",
    "research*",
]
exclude = [
    "outputs*",
    "*.outputs*",
    "*.models*",
]
```

- [ ] **Step 3: Strengthen generated-file ignores**

In `.gitignore`, add these generated-artifact patterns:

```gitignore
# Local build artifacts
dist/

# Generated diagnostics artifacts.
src/diagnostics/stats/outputs/*.png
src/diagnostics/feature_quality/outputs/*.csv
src/diagnostics/feature_quality/outputs/*.json
src/diagnostics/feature_quality/outputs/*.parquet

# Generated framework POC nested artifacts.
src/research/backtesting_frameworks_poc/outputs/**
!src/research/backtesting_frameworks_poc/outputs/
!src/research/backtesting_frameworks_poc/outputs/.gitkeep
```

Keep the existing root-level and old-path ignore rules until the refactor commit is complete. They are harmless and protect older local artifacts.

- [ ] **Step 4: Rebuild and inspect wheel contents**

Run:

```bash
rm -rf build dist
uv build --sdist --wheel
uv run python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel_path = next(Path("dist").glob("*.whl"))
with ZipFile(wheel_path) as wheel:
    names = sorted(wheel.namelist())

bad_names = [
    name
    for name in names
    if "/outputs/" in name or "/models/" in name
]
print("bad_names", bad_names)
PY
```

Expected:

```text
bad_names []
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: exclude generated artifacts from package builds"
```

---

### Task 3: Repair Stale Docs And Add App Guide

**Files:**
- Modify: `src/GUIDE_src.md`
- Modify: `tests/GUIDE_tests.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `PROJECT_OVERVIEW.md`
- Modify: `PROJECT_STRUCTURE.md`
- Modify: `README.md`
- Create: `src/app/GUIDE_app.md`

- [ ] **Step 1: Fix the `src/` tree guide**

Replace the tree in `src/GUIDE_src.md` with:

```text
src/
├── main.py
├── run.py
├── transform.py
├── app/
│   └── cli/
├── engine/
│   ├── features/
│   └── pipeline/
├── diagnostics/
│   ├── stats/
│   ├── robustness/
│   └── feature_quality/
├── research/
│   └── backtesting_frameworks_poc/
└── outputs/
    └── stocks/
```

Then update the paragraph below it so it says the old root-level workflow folders were grouped under `app`, `engine`, `diagnostics`, and `research`.

- [ ] **Step 2: Fix fake test filenames**

In `tests/GUIDE_tests.md`, replace these fake names:

```text
test_correlation_microstructure_engine.features.py
test_classical_ohlcv_engine.features.py
test_research.backtesting_frameworks_poc.py
```

with the actual filenames:

```text
test_correlation_microstructure_features.py
test_classical_ohlcv_features.py
test_backtesting_frameworks_poc.py
```

- [ ] **Step 3: Clean broad-replacement prose corruption**

Search:

```bash
rg -n "engine\\.pipeline|engine\\.features|diagnostics\\.stats" README.md GUIDE_ROOT.md PROJECT_OVERVIEW.md PROJECT_STRUCTURE.md src tests
```

For prose, use normal English:

```text
engine.pipeline -> pipeline
engine.features -> features
diagnostics.stats -> stats diagnostics
```

Keep dotted names when they are real import paths, code references, ClickHouse table names, or command examples, for example:

```text
from engine.pipeline.exporter import export
diagnostics.stats.run_stats
CREATE TABLE IF NOT EXISTS engine.features.stocks
```

- [ ] **Step 4: Create `src/app/GUIDE_app.md`**

Create this guide:

```markdown
# GUIDE - app/

## Part 1 - Conceptual Explanation

`app/` contains application-facing entrypoints. The code here parses command-line arguments, loads validated configuration, sets up logging, and then delegates real data work to `engine/`.

The split is intentional:

```text
app/cli/        user commands and orchestration
engine/         feature computation and data pipeline logic
diagnostics/    read-only analysis tools
research/       experiments and proof-of-concept code
```

This keeps command-line concerns separate from feature math. A command can change its flags or logging behavior without changing how a feature is computed.

## Part 2 - Code Reference

| Path | Purpose |
|---|---|
| `cli/pipeline.py` | Implements the main feature pipeline command used by root `run.py`, `src/run.py`, and the `feature-pipeline` console script. |
| `cli/transform.py` | Implements the post-feature transform command used by root `transform.py`, `src/transform.py`, and the `feature-transform` console script. |
| `cli/__init__.py` | Marks the CLI folder as a Python package. |

Start with `cli/pipeline.py` for the full load-clean-engineer-export-metadata workflow. Start with `cli/transform.py` for normalization and winsorization after features already exist.

## Part 3 - Short Journal

- 2026-04-24: The second refactor grouped command-line entrypoints under `app/` so CLI orchestration no longer sits beside core feature-engineering code.
```

- [ ] **Step 5: Run docs search again**

Run:

```bash
rg -n "src/(cli|features|pipeline|stats|robustness|feature_quality|backtesting_frameworks_poc)|test_.*engine\\.features|test_research\\.backtesting" README.md GUIDE_ROOT.md PROJECT_OVERVIEW.md PROJECT_STRUCTURE.md src tests
```

Expected: no stale old-path matches and no fake test filenames.

- [ ] **Step 6: Commit**

```bash
git add README.md GUIDE_ROOT.md PROJECT_OVERVIEW.md PROJECT_STRUCTURE.md src/GUIDE_src.md tests/GUIDE_tests.md src/app/GUIDE_app.md
git commit -m "docs: repair refactor guide references"
```

---

### Task 4: Add Category-Based Feature Selection

**Files:**
- Modify: `src/engine/pipeline/config_models.py`
- Modify: `src/engine/pipeline/engineer.py`
- Modify: `tests/test_config_models.py`
- Modify: `tests/test_pipeline_guards.py`
- Modify: `README.md`
- Modify: `src/engine/features/GUIDE_features.md`
- Modify: `src/engine/pipeline/GUIDE_pipeline.md`

- [ ] **Step 1: Add config-model tests for category filters**

In `tests/test_config_models.py`, add tests to `TestPipelineConfigValidation`:

```python
    def test_pipeline_config_accepts_feature_category_filters(self) -> None:
        """Feature config may include category include/exclude filters."""
        raw_config = self._valid_pipeline_config()
        raw_config["features"]["include_categories"] = ["returns", "trend"]
        raw_config["features"]["exclude_categories"] = ["target"]

        parsed = parse_pipeline_config(raw_config)
        parsed_dict = parsed.to_dict()

        self.assertEqual(
            parsed_dict["features"]["include_categories"],
            ["returns", "trend"],
        )
        self.assertEqual(
            parsed_dict["features"]["exclude_categories"],
            ["target"],
        )

    def test_pipeline_config_rejects_unknown_feature_category(self) -> None:
        """Unknown categories should fail before a pipeline run starts."""
        raw_config = self._valid_pipeline_config()
        raw_config["features"]["include_categories"] = ["not_a_category"]

        with self.assertRaisesRegex(ConfigValidationError, "not_a_category"):
            parse_pipeline_config(raw_config)

    def test_pipeline_config_rejects_category_in_both_include_and_exclude(self) -> None:
        """A category cannot be both selected and excluded."""
        raw_config = self._valid_pipeline_config()
        raw_config["features"]["include_categories"] = ["returns"]
        raw_config["features"]["exclude_categories"] = ["returns"]

        with self.assertRaisesRegex(ConfigValidationError, "both include_categories and exclude_categories"):
            parse_pipeline_config(raw_config)
```

If `_valid_pipeline_config()` does not currently include `features`, add only the smallest `features` table needed by the existing test helper style.

- [ ] **Step 2: Extend the typed config model**

In `FeatureColumnConfig` no change is needed. In `PipelineConfig`, add fields:

```python
    include_categories: list[str]
    exclude_categories: list[str]
```

In `PipelineConfig.to_dict()`, change the `features` block to:

```python
            "features": {
                "params": [feature.to_dict() for feature in self.features],
                "include_categories": list(self.include_categories),
                "exclude_categories": list(self.exclude_categories),
            },
```

- [ ] **Step 3: Validate category filters**

In `src/engine/pipeline/config_models.py`, add:

```python
def _available_feature_categories() -> set[str]:
    """Return every category declared by registered feature specs."""
    return {str(spec.category) for spec in REGISTRY.values()}


def _validate_feature_category_filters(
    features_cfg: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate optional category include/exclude lists from the features config."""
    available_categories = _available_feature_categories()
    include_categories = _validate_category_list(
        label="features.include_categories",
        raw_value=features_cfg.get("include_categories", []),
        available_categories=available_categories,
    )
    exclude_categories = _validate_category_list(
        label="features.exclude_categories",
        raw_value=features_cfg.get("exclude_categories", []),
        available_categories=available_categories,
    )

    overlap = sorted(set(include_categories) & set(exclude_categories))
    if overlap:
        raise ConfigValidationError(
            "Feature categories cannot appear in both include_categories and "
            f"exclude_categories: {overlap}"
        )

    return include_categories, exclude_categories


def _validate_category_list(
    label: str,
    raw_value: Any,
    available_categories: set[str],
) -> list[str]:
    """Validate one feature-category filter list."""
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ConfigValidationError(f"{label} must be a list of strings.")

    categories: list[str] = []
    seen: set[str] = set()
    for idx, value in enumerate(raw_value):
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(f"{label}[{idx}] must be a non-empty string.")
        category = value.strip()
        if category not in available_categories:
            raise ConfigValidationError(
                f"{label}[{idx}]='{category}' is not a registered feature category. "
                f"Available categories: {sorted(available_categories)}"
            )
        if category in seen:
            raise ConfigValidationError(f"{label} contains duplicate category '{category}'.")
        seen.add(category)
        categories.append(category)

    return categories
```

Then call it inside `parse_pipeline_config()`:

```python
    features_cfg = raw_config.get("features", {})
    if features_cfg is not None and not isinstance(features_cfg, Mapping):
        raise ConfigValidationError("features must be a table if provided.")
    features_cfg = features_cfg or {}
    feature_items = features_cfg.get("params", [])
    include_categories, exclude_categories = _validate_feature_category_filters(features_cfg)
    features = _validate_feature_entries(feature_items, run.asset_class)
```

Pass both fields into `PipelineConfig(...)`.

- [ ] **Step 4: Run config-model tests and verify they pass**

Run:

```bash
uv run pytest tests/test_config_models.py -q
```

Expected: all config model tests pass.

- [ ] **Step 5: Add compute test for category filtering**

Add this test to `TestEngineerIncrementalGuards` or a new `TestEngineerCategoryFilters` class in `tests/test_pipeline_guards.py`:

```python
class TestEngineerCategoryFilters(unittest.TestCase):
    """Verify category filters select feature groups before computation."""

    def test_compute_features_filters_by_registered_category(self) -> None:
        """Only included categories should appear in the output."""
        ts = pd.to_datetime(
            ["2024-01-02 09:30:00", "2024-01-02 09:31:00"]
        ).tz_localize("America/New_York")
        cleaned_df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "ts": ts,
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        )
        config = {
            "run": {
                "asset_class": "stocks",
                "table": "stocks",
                "symbols": ["AAPL"],
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
            },
            "features": {
                "include_categories": ["returns"],
                "exclude_categories": [],
                "params": [
                    {"name": "log_return", "fn": "log_return", "enabled": True},
                    {"name": "ma_20", "fn": "moving_average", "window": 20, "enabled": True},
                ],
            },
        }

        result = compute_features(cleaned_df, config)

        self.assertIn("log_return", result.columns)
        self.assertNotIn("ma_20", result.columns)
```

- [ ] **Step 6: Implement category filtering in the engineer**

In `src/engine/pipeline/engineer.py`, add:

```python
def _filter_feature_params_by_category(
    feature_params: list[dict],
    include_categories: list[str],
    exclude_categories: list[str],
) -> tuple[list[dict], list[str]]:
    """
    Filter feature configs using registered feature categories.

    Parameters
    ----------
    feature_params : list[dict]
        Enabled feature config entries.
    include_categories : list[str]
        If non-empty, keep only features whose registry category is in this list.
    exclude_categories : list[str]
        Drop features whose registry category is in this list.

    Returns
    -------
    tuple[list[dict], list[str]]
        Filtered feature configs and the names skipped by category rules.
    """
    filtered_params: list[dict] = []
    skipped_names: list[str] = []
    include_set = set(include_categories)
    exclude_set = set(exclude_categories)

    for item in feature_params:
        fn_name = item["fn"]
        category = REGISTRY[fn_name].category
        include_rejects = include_set and category not in include_set
        exclude_rejects = category in exclude_set
        if include_rejects or exclude_rejects:
            skipped_names.append(item["name"])
            continue
        filtered_params.append(item)

    return filtered_params, skipped_names
```

In `compute_features()`, after enabled filtering and before the `not feature_params` check, add:

```python
    features_cfg = config.get("features", {})
    include_categories = features_cfg.get("include_categories", [])
    exclude_categories = features_cfg.get("exclude_categories", [])
    feature_params, category_skipped_names = _filter_feature_params_by_category(
        feature_params=feature_params,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
    )
    if category_skipped_names:
        logger.info(
            "Skipping %d feature(s) by category filters: %s",
            len(category_skipped_names),
            category_skipped_names,
        )
```

- [ ] **Step 7: Run targeted engineer tests**

Run:

```bash
uv run pytest tests/test_pipeline_guards.py -q
```

Expected: all pipeline guard tests pass.

- [ ] **Step 8: Document category filters**

In `README.md`, under the feature-addition section, add:

```markdown
You can also run a subset by feature category:

```toml
[features]
include_categories = ["returns", "trend"]
exclude_categories = ["target"]
```

`include_categories` is optional. When it is empty, all categories are eligible. `exclude_categories` always removes matching categories after the include rule. Category names come from each feature's registry metadata and are exported in the feature catalog.
```

In `src/engine/features/GUIDE_features.md`, add a short category table using the current registry categories:

```markdown
| Category | Intent |
|---|---|
| `returns` | Backward-looking return measurements. |
| `target` | Forward-looking labels for research, not live model inputs. |
| `trend` | Directional persistence and moving-average style signals. |
| `mean_reversion` | Distance-from-fair-value and overbought/oversold signals. |
| `risk_volatility` | Realized range, dispersion, and volatility estimates. |
| `liquidity_flow` | Volume, VWAP, and participation-style signals. |
| `microstructure` | Bar-level trading pressure and spread proxies. |
| `cross_session` | Overnight and session-boundary effects. |
| `serial_dependence` | Autocorrelation-style persistence checks. |
| `cross_feature_interaction` | Relationships between two observed series. |
| `options_volatility` | Implied-volatility level and term-structure signals. |
| `options_liquidity` | Options-chain liquidity and spread signals. |
| `options_sentiment` | Put/call and positioning-style options signals. |
```

In `src/engine/pipeline/GUIDE_pipeline.md`, document that the engineer first removes disabled entries, then applies category include/exclude filters, then computes the surviving feature list.

- [ ] **Step 9: Run targeted tests**

Run:

```bash
uv run pytest tests/test_config_models.py tests/test_pipeline_guards.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/engine/pipeline/config_models.py src/engine/pipeline/engineer.py tests/test_config_models.py tests/test_pipeline_guards.py README.md src/engine/features/GUIDE_features.md src/engine/pipeline/GUIDE_pipeline.md
git commit -m "feat: support category-based feature selection"
```

---

### Task 5: Final Verification And Refactor Commit

**Files:**
- No new implementation files.
- Commit all remaining planned changes.

- [ ] **Step 1: Clean build outputs before final status**

Run:

```bash
rm -rf build dist
```

- [ ] **Step 2: Run full verification**

Run:

```bash
uv run ruff check .
uv run pytest -q
uv run python run.py --help
uv run python transform.py --help
uv run feature-pipeline --help
uv run feature-transform --help
uv build --sdist --wheel
```

Expected:

- Ruff passes.
- Pytest reports all tests passed.
- All CLI help commands print usage and exit successfully.
- Build succeeds.

- [ ] **Step 3: Verify wheel has no generated outputs**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel_path = next(Path("dist").glob("*.whl"))
with ZipFile(wheel_path) as wheel:
    names = sorted(wheel.namelist())

bad_names = [
    name
    for name in names
    if "/outputs/" in name or "/models/" in name
]
if bad_names:
    raise SystemExit(f"Generated artifacts found in wheel: {bad_names}")
print("Wheel artifact check passed.")
PY
```

Expected:

```text
Wheel artifact check passed.
```

- [ ] **Step 4: Review working tree**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended source, test, docs, config, notebook, and path-move changes are present. `dist/`, `build/`, `__pycache__/`, and generated outputs should not appear as untracked files.

- [ ] **Step 5: Commit final cleanup if anything remains unstaged**

If Task 1-4 commits were made separately, stage and commit any remaining docs or config cleanup:

```bash
git add .
git commit -m "docs: finalize domain refactor cleanup"
```

If all changes were already committed task-by-task and `git status --short` is clean, skip this step.

---

## Self-Review

Spec coverage:

- Incremental cache date bug: covered by Task 1 with a failing test and implementation.
- Package build including generated outputs: covered by Task 2 with wheel inspection.
- Stale docs and fake filenames: covered by Task 3.
- Feature category separation: covered by Task 4 through config validation, runtime filtering, tests, and docs.
- Final verification: covered by Task 5.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps remain.

Type consistency:

- Category filters are always `list[str]`.
- Runtime feature params remain `list[dict]`.
- The existing `[[features.params]]` config shape remains intact.
