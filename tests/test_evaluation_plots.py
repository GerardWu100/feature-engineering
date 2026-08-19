"""Smoke tests for the evaluation plot functions.

Plot code fails in two ways: it raises, or it silently draws the wrong thing.
These tests catch the first kind (every figure must build and save on synthetic
data) and pin the cheap invariants of the second kind (panel counts, axis
formatting mode, flag-versus-continuous state handling).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from feature_engineering.evaluation import (
    rolling_ic_panels,
    spread_rows_by_state,
    violin_by_quantile,
)

RANDOM_SEED = 11


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates so tests cannot leak memory."""
    yield
    plt.close("all")


@pytest.fixture()
def frame() -> pd.DataFrame:
    """Two symbols, 200 bars, one real signal, one flag, one forward target."""
    rng = np.random.default_rng(RANDOM_SEED)
    blocks = []
    timestamps = pd.date_range("2024-01-01", periods=200, freq="D")
    for name in ("AAA", "BBB"):
        signal = rng.standard_normal(200)
        blocks.append(
            pd.DataFrame(
                {
                    "symbol": name,
                    "timestamp": timestamps,
                    "signal": signal,
                    "flag": (signal > 1.0).astype(float),
                    "fwd_return": 0.01 * signal + 0.02 * rng.standard_normal(200),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def test_violin_by_quantile_builds_and_saves(frame: pd.DataFrame, tmp_path) -> None:
    """The violin figure should build, save a non-empty PNG, and label buckets."""
    out = tmp_path / "violin.png"
    figure = violin_by_quantile(
        frame, "signal", "fwd_return", quantiles=4, save_path=out
    )

    assert out.exists() and out.stat().st_size > 0
    panel = figure.axes[0]
    # Tick labels carry the bucket name and its sample size (100 rows per
    # symbol x 2 symbols / 4 buckets = 50 per bucket).
    tick_labels = [label.get_text() for label in panel.get_xticklabels()]
    assert [label.split("\n")[0] for label in tick_labels] == ["Q1", "Q2", "Q3", "Q4"]
    assert all(label.split("\n")[1].startswith("n=") for label in tick_labels)


def test_violin_by_quantile_rejects_no_finite_pairs() -> None:
    """An empty finite sample should produce a clear analysis error."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "signal": [np.nan, np.inf],
            "target": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="No rows have both"):
        violin_by_quantile(frame, "signal", "target")


def test_spread_rows_handles_continuous_and_flag_features(
    frame: pd.DataFrame, tmp_path
) -> None:
    """Continuous features get 3 state rows, flags get 2, on one shared axis."""
    out = tmp_path / "spread.png"
    figure = spread_rows_by_state(
        frame, ["signal", "flag"], "fwd_return", save_path=out
    )

    assert out.exists() and out.stat().st_size > 0
    panel = figure.axes[0]
    tick_labels = [label.get_text() for label in panel.get_yticklabels()]
    assert tick_labels == ["low", "neutral", "high", "off", "on"]


def test_rolling_ic_panels_one_panel_per_feature(frame: pd.DataFrame, tmp_path) -> None:
    """The stability figure should stack one panel per requested feature."""
    out = tmp_path / "rolling.png"
    figure = rolling_ic_panels(
        frame, ["signal", "flag"], "fwd_return", window=60, save_path=out
    )

    assert out.exists() and out.stat().st_size > 0
    assert len(figure.axes) == 2


def test_spread_rows_survives_heavily_tied_feature() -> None:
    """A tied feature that collapses a tercile bin must degrade, not crash."""
    rng = np.random.default_rng(RANDOM_SEED)
    # One symbol, half the values identical: the lower tercile edge collapses,
    # leaving two bins that map to low/high.
    tied = np.concatenate([np.zeros(50), np.linspace(1.0, 10.0, 50)])
    working = pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "tied_feature": tied,
            "fwd_return": rng.standard_normal(100) * 0.02,
        }
    )

    figure = spread_rows_by_state(working, ["tied_feature"], "fwd_return")
    tick_labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert tick_labels == ["low", "high"]


def test_plot_functions_reject_empty_feature_list(frame: pd.DataFrame) -> None:
    """An empty feature list is a caller mistake and must raise, not draw."""
    with pytest.raises(ValueError):
        spread_rows_by_state(frame, [], "fwd_return")
    with pytest.raises(ValueError):
        rolling_ic_panels(frame, [], "fwd_return", window=60)
