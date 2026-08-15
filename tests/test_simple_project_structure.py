"""Tests for the two-part project structure.

These tests protect the main design decision: the package is split into two
parts, ``engineering`` (build and store features, grouped by feature category)
and ``evaluation`` (test features against targets), with no optional platform
modules in the main package.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "feature_engineering"


def test_engineering_and_evaluation_packages_exist() -> None:
    """The public implementation should be split into engineering and evaluation."""
    expected_paths = [
        PACKAGE_ROOT / "engineering" / "features" / "returns.py",
        PACKAGE_ROOT / "engineering" / "features" / "targets.py",
        PACKAGE_ROOT / "engineering" / "features" / "trend.py",
        PACKAGE_ROOT / "engineering" / "features" / "volatility.py",
        PACKAGE_ROOT / "engineering" / "features" / "volume.py",
        PACKAGE_ROOT / "engineering" / "features" / "registry.py",
        PACKAGE_ROOT / "engineering" / "load.py",
        PACKAGE_ROOT / "engineering" / "clean.py",
        PACKAGE_ROOT / "engineering" / "compute.py",
        PACKAGE_ROOT / "engineering" / "store.py",
        PACKAGE_ROOT / "engineering" / "constants.py",
        PACKAGE_ROOT / "evaluation" / "ic.py",
        PACKAGE_ROOT / "evaluation" / "regression.py",
        PACKAGE_ROOT / "evaluation" / "quantiles.py",
        PACKAGE_ROOT / "evaluation" / "plots.py",
        PACKAGE_ROOT / "evaluation" / "summary.py",
        PACKAGE_ROOT / "config.py",
        PACKAGE_ROOT / "cli.py",
    ]

    missing_paths = [path for path in expected_paths if not path.exists()]

    assert missing_paths == []


def test_removed_modules_do_not_remain_in_main_package() -> None:
    """Removed subsystems should not reappear in the main package."""
    removed_paths = [
        PACKAGE_ROOT / "engine",
        PACKAGE_ROOT / "pipeline",
        PACKAGE_ROOT / "features",
        SRC_ROOT / "engine",
        SRC_ROOT / "diagnostics",
        SRC_ROOT / "research",
        SRC_ROOT / "app",
        SRC_ROOT / "features",
        SRC_ROOT / "pipeline",
        PROJECT_ROOT / "transform.py",
        PROJECT_ROOT / "transform_config.toml",
        PROJECT_ROOT / "config_options.toml",
    ]

    remaining_paths = [path for path in removed_paths if path.exists()]

    assert remaining_paths == []


def test_registry_exposes_expected_categories_only() -> None:
    """The registry should expose exactly the documented feature categories."""
    from feature_engineering.engineering.features.registry import REGISTRY

    categories = {spec.category for spec in REGISTRY.values()}

    assert categories == {"returns", "target", "trend", "volatility", "volume"}
