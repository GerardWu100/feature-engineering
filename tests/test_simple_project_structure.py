"""Tests for the intentionally small project structure.

These tests protect the main design decision of the simplification pass: the
project should expose one stock OHLCV feature pipeline, grouped by feature
category, without optional platform modules in the main package.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "feature_engineering"


def test_simple_feature_and_pipeline_packages_exist() -> None:
    """The public implementation should be split into simple feature and pipeline packages."""
    expected_paths = [
        PACKAGE_ROOT / "features" / "returns.py",
        PACKAGE_ROOT / "features" / "trend.py",
        PACKAGE_ROOT / "features" / "volatility.py",
        PACKAGE_ROOT / "features" / "volume.py",
        PACKAGE_ROOT / "features" / "registry.py",
        PACKAGE_ROOT / "pipeline" / "config.py",
        PACKAGE_ROOT / "pipeline" / "constants.py",
        PACKAGE_ROOT / "pipeline" / "load.py",
        PACKAGE_ROOT / "pipeline" / "clean.py",
        PACKAGE_ROOT / "pipeline" / "engineer.py",
        PACKAGE_ROOT / "pipeline" / "export.py",
    ]

    missing_paths = [path for path in expected_paths if not path.exists()]

    assert missing_paths == []


def test_removed_platform_modules_do_not_remain_in_main_package() -> None:
    """Optional subsystems should not remain in the simplified main package."""
    removed_paths = [
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


def test_registry_exposes_simple_categories_only() -> None:
    """The registry should expose only the categories used by the simple pipeline."""
    from feature_engineering.features.registry import REGISTRY

    categories = {spec.category for spec in REGISTRY.values()}

    assert categories == {"returns", "target", "trend", "volatility", "volume"}
