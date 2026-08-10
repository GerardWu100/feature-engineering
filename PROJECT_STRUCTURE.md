# Project Structure

The project uses a small Python `src/` layout: importable code lives under
`src/feature_engineering/`, while tests and run-time configuration stay at the
repository root.

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
│       ├── pipeline/
│       │   ├── GUIDE_pipeline.md
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── config.py
│       │   ├── constants.py
│       │   ├── load.py
│       │   ├── clean.py
│       │   ├── engineer.py
│       │   └── export.py
│       └── evaluation/
│           ├── GUIDE_evaluation.md
│           ├── __init__.py
│           ├── ic.py
│           ├── regression.py
│           ├── quantiles.py
│           ├── summary.py
│           └── plots.py
└── tests/
    ├── GUIDE_tests.md
    ├── test_config_validation.py
    ├── test_engines.py
    ├── test_evaluation.py
    ├── test_evaluation_plots.py
    ├── test_feature_math.py
    ├── test_simple_pipeline.py
    └── test_simple_project_structure.py
```

## Root

| Path | Purpose |
|---|---|
| `run.py` | Root wrapper for the pipeline command-line interface. |
| `config.toml` | Single configuration file for stock OHLCV feature runs. |
| `pyproject.toml` | Package metadata, dependencies, and console-script definitions. |

## Source Packages

| Package | Responsibility |
|---|---|
| `feature_engineering/features/` | Pure categorized feature formulas. |
| `feature_engineering/engine/` | Cached batch `FeatureEngine` and constant-time `OnlineFeatureEngine`. |
| `feature_engineering/pipeline/` | Validate config, load, clean, engineer, export workflow. |
| `feature_engineering/evaluation/` | Feature-versus-target testing: information coefficients, Newey-West regression, quantiles, and plots. |

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
