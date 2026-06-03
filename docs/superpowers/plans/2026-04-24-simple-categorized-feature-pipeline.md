# Simple Categorized Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the project to a simple stock OHLCV feature engineering pipeline with categorized feature files.

**Architecture:** The project will keep one linear workflow: load raw OHLCV data, clean invalid rows, compute configured features by category, and export the dataset. Feature math will live in `src/features/`, pipeline stages will live in `src/pipeline/`, and optional diagnostics, options, transforms, and backtesting proof-of-concepts will be removed from the main code path.

**Tech Stack:** Python 3.12, pandas, pyarrow, ClickHouse Connect, TOML configuration, pytest, uv.

---

### Task 1: Add Structure Tests

**Files:**
- Create: `tests/test_simple_project_structure.py`

- [ ] **Step 1: Write failing tests**

Create tests that assert the new simplified modules exist, the old platform modules are absent, and the registry exposes only simple stock feature categories.

- [ ] **Step 2: Run the structure tests**

Run: `uv run pytest tests/test_simple_project_structure.py -q`

Expected: fail because the simplified modules do not exist yet and the old modules still exist.

### Task 2: Create Category-Based Feature Package

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/returns.py`
- Create: `src/features/trend.py`
- Create: `src/features/volatility.py`
- Create: `src/features/volume.py`
- Create: `src/features/registry.py`
- Remove: `src/engine/features/`

- [ ] **Step 1: Move only the simple stock features**

Implement returns, trend, volatility, and volume functions in category files. Keep target labels in `returns.py`, but register forward-looking labels with category `target`.

- [ ] **Step 2: Keep registry small**

Expose `REGISTRY`, `FeatureSpec`, and `register`. Import only the four category files.

- [ ] **Step 3: Run feature tests**

Run: `uv run pytest tests/test_simple_project_structure.py tests/test_feature_math.py -q`

Expected: pass after import paths and feature tests match the new structure.

### Task 3: Create Simple Pipeline Package

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/load.py`
- Create: `src/pipeline/clean.py`
- Create: `src/pipeline/engineer.py`
- Create: `src/pipeline/export.py`
- Remove: `src/engine/pipeline/`
- Remove: `src/app/`

- [ ] **Step 1: Keep one linear pipeline**

Implement loading, cleaning, feature engineering, and export as four separate files. Remove delete mode, transform mode, options branching, incremental cache reuse, and metadata catalog complexity.

- [ ] **Step 2: Preserve category filters**

Keep `include_categories` and `exclude_categories` in `[features]` because they make grouped feature selection simple.

- [ ] **Step 3: Run pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -q`

Expected: pass with simple OHLCV toy data and no ClickHouse dependency.

### Task 4: Simplify Entrypoints, Config, and Packaging

**Files:**
- Modify: `run.py`
- Modify: `main.py`
- Modify: `src/run.py`
- Modify: `src/main.py`
- Modify: `config.toml`
- Modify: `pyproject.toml`
- Remove: `transform.py`
- Remove: `src/transform.py`
- Remove: `transform_config.toml`
- Remove: `config_options.toml`

- [ ] **Step 1: Make `run.py` the only workflow**

Root `run.py` delegates to the simple pipeline CLI. `main.py` remains a compatibility wrapper that calls the same function.

- [ ] **Step 2: Make `config.toml` the only config**

Keep stocks/OHLCV settings, data quality rules, category filters, and a compact feature list.

- [ ] **Step 3: Trim dependencies and console scripts**

Remove packages used only by deleted diagnostics, options, transforms, and backtesting proof-of-concepts.

### Task 5: Remove Extra Modules and Tests

**Files:**
- Remove: `src/diagnostics/`
- Remove: `src/research/`
- Remove obsolete tests tied to removed modules
- Keep focused tests for feature math, cleaning, engineering, export, config, and CLI.

- [ ] **Step 1: Delete modules outside the simple workflow**

Remove optional subsystems from the tracked codebase.

- [ ] **Step 2: Replace broad old tests with focused simple tests**

The test suite should protect the new intended behavior, not the old platform behavior.

### Task 6: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `src/GUIDE_src.md`
- Modify or create: `src/features/GUIDE_features.md`
- Modify or create: `src/pipeline/GUIDE_pipeline.md`
- Modify: `tests/GUIDE_tests.md`
- Remove stale guides for deleted folders

- [ ] **Step 1: Rewrite docs around the simple mental model**

Document the flow: load, clean, engineer, export.

- [ ] **Step 2: Explain feature categories**

Define returns, trend, volatility, volume, and target categories.

### Task 7: Verify and Commit

**Files:**
- All modified files

- [ ] **Step 1: Run tests**

Run: `uv run pytest -q`

- [ ] **Step 2: Run the pipeline on toy/local data if practical**

Use a small local CSV path in config or a temporary test fixture path when ClickHouse is unavailable.

- [ ] **Step 3: Run lint if available**

Run: `uv run ruff check .`

- [ ] **Step 4: Commit**

Run:

```bash
git add .
git commit -m "refactor: simplify categorized feature pipeline"
```

