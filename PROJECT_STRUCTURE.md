# Project Structure

The project uses a small Python `src/` layout.

```text
feature-engineering/
├── run.py
├── config.toml
├── pyproject.toml
├── README.md
├── GUIDE_ROOT.md
├── PROJECT_OVERVIEW.md
├── PROJECT_STRUCTURE.md
├── src/
│   ├── GUIDE_src.md
│   └── feature_engineering/
│       ├── __init__.py
│       ├── GUIDE_feature_engineering.md
│       ├── features/
│       │   ├── GUIDE_features.md
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── returns.py
│       │   ├── trend.py
│       │   ├── volatility.py
│       │   └── volume.py
│       ├── engine/
│       │   ├── GUIDE_engine.md
│       │   ├── __init__.py
│       │   ├── batch.py
│       │   └── online.py
│       └── pipeline/
│           ├── GUIDE_pipeline.md
│           ├── __init__.py
│           ├── cli.py
│           ├── config.py
│           ├── constants.py
│           ├── load.py
│           ├── clean.py
│           ├── engineer.py
│           └── export.py
└── tests/
    ├── GUIDE_tests.md
    ├── test_config_validation.py
    ├── test_engines.py
    ├── test_feature_math.py
    ├── test_simple_pipeline.py
    └── test_simple_project_structure.py
```

## Root

| Path | Purpose |
|---|---|
| `run.py` | Root wrapper for the pipeline CLI. |
| `config.toml` | Single config file for stock OHLCV feature runs. |
| `pyproject.toml` | Package metadata, dependencies, and console scripts. |

## Source Packages

| Package | Responsibility |
|---|---|
| `feature_engineering/features/` | Pure categorized feature formulas. |
| `feature_engineering/engine/` | Cached batch `FeatureEngine` and O(1) incremental `OnlineFeatureEngine`. |
| `feature_engineering/pipeline/` | Validate config, load, clean, engineer, export workflow. |

## Data Flow

```text
config.toml
  -> feature_engineering.pipeline.cli
  -> feature_engineering.pipeline.config
  -> feature_engineering.pipeline.load
  -> feature_engineering.pipeline.clean
  -> feature_engineering.pipeline.engineer
       -> feature_engineering.features.registry
       -> feature_engineering.features category modules
  -> feature_engineering.pipeline.export
  -> outputs/stocks/
```
