# Project Structure

The project uses a small Python `src/` layout. Importable code lives under
`src/feature_engineering/`; tests and run-time configuration stay at the
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
│       ├── cli.py
│       ├── config.py
│       ├── engineering/
│       │   ├── GUIDE_engineering.md
│       │   ├── __init__.py
│       │   ├── load.py
│       │   ├── clean.py
│       │   ├── compute.py
│       │   ├── store.py
│       │   ├── constants.py
│       │   └── features/
│       │       ├── GUIDE_features.md
│       │       ├── __init__.py
│       │       ├── registry.py
│       │       ├── returns.py
│       │       ├── targets.py
│       │       ├── trend.py
│       │       ├── volatility.py
│       │       └── volume.py
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
    ├── test_evaluation.py
    ├── test_evaluation_plots.py
    ├── test_feature_contracts.py
    ├── test_feature_math.py
    ├── test_schema_contracts.py
    ├── test_simple_pipeline.py
    ├── test_simple_project_structure.py
    └── test_time_semantics.py
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
| `feature_engineering/engineering/` | Build features: load, clean, compute, store/pull datasets. |
| `feature_engineering/engineering/features/` | Pure categorized feature formulas. |
| `feature_engineering/evaluation/` | Feature-versus-target testing: information coefficients, Newey-West regression, quantiles, and plots. |
| `feature_engineering/config.py` | Validate `config.toml` at the workflow boundary. |
| `feature_engineering/cli.py` | The load -> clean -> compute -> store workflow. |

## Data Flow

```text
config.toml
  -> feature_engineering.cli
  -> feature_engineering.config
  -> feature_engineering.engineering.load
  -> feature_engineering.engineering.clean
  -> feature_engineering.engineering.compute
       -> feature_engineering.engineering.features.registry
       -> feature_engineering.engineering.features category modules
  -> feature_engineering.engineering.store
  -> outputs/stocks/
```
