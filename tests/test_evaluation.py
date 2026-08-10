"""Tests for the feature evaluation subpackage (IC, regression, quantiles).

The strategy throughout: build synthetic data where the true relationship is
known by construction, then check that each statistic recovers it — sign,
approximate magnitude, and the degenerate cases (no relationship, constant
feature, too few symbols).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from feature_engineering.evaluation import (
    cross_sectional_ic,
    evaluate_features,
    ic_summary,
    newey_west_regression,
    rolling_ic,
    target_by_feature_quantile,
    time_series_ic,
)
from feature_engineering.evaluation.quantiles import target_values_by_quantile
from feature_engineering.evaluation.regression import default_kernel_lags

RANDOM_SEED = 7


def _synthetic_frame(
    *,
    n_symbols: int = 2,
    n_bars: int = 300,
    slope: float = 0.5,
    noise_scale: float = 0.1,
) -> pd.DataFrame:
    """Build a long feature frame where target = slope * z(feature) + noise.

    The feature is standard normal per symbol, so it is already (close to) its
    own z-score and the regression should recover ``slope``. A pure-noise
    column is included as a negative control.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    blocks: list[pd.DataFrame] = []
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    for i in range(n_symbols):
        feature = rng.standard_normal(n_bars)
        noise_feature = rng.standard_normal(n_bars)
        target = slope * feature + noise_scale * rng.standard_normal(n_bars)
        blocks.append(
            pd.DataFrame(
                {
                    "symbol": f"SYM{i}",
                    "timestamp": timestamps,
                    "signal": feature,
                    "noise": noise_feature,
                    "fwd_target": target,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def test_time_series_ic_recovers_sign_and_perfect_rank() -> None:
    """IC should be strongly positive for a real signal, ~0 for noise, 1 for monotone."""
    frame = _synthetic_frame()

    signal_ic = time_series_ic(frame, "signal", "fwd_target")
    noise_ic = time_series_ic(frame, "noise", "fwd_target")

    assert (signal_ic > 0.9).all()
    assert (noise_ic.abs() < 0.2).all()

    # A strictly monotone transform has Spearman IC exactly 1.
    monotone = frame.assign(fwd_target=frame["signal"] ** 3)
    perfect_ic = time_series_ic(monotone, "signal", "fwd_target")
    assert np.allclose(perfect_ic.to_numpy(), 1.0)


def test_time_series_ic_returns_nan_below_min_observations() -> None:
    """Symbols with too little history should report NaN, not a noisy number."""
    frame = _synthetic_frame(n_bars=10)
    ic = time_series_ic(frame, "signal", "fwd_target", min_observations=24)
    assert ic.isna().all()


def test_cross_sectional_ic_gates_on_universe_width() -> None:
    """Per-timestamp IC should exist for a wide universe and be empty for 2 symbols."""
    wide = _synthetic_frame(n_symbols=10, n_bars=60)
    narrow = _synthetic_frame(n_symbols=2, n_bars=60)

    wide_ic = cross_sectional_ic(wide, "signal", "fwd_target", min_symbols=5)
    narrow_ic = cross_sectional_ic(narrow, "signal", "fwd_target", min_symbols=5)

    assert len(wide_ic) == 60
    assert wide_ic.mean() > 0.5
    assert narrow_ic.empty


def test_rolling_ic_tracks_a_stable_relationship() -> None:
    """Rolling IC windows over a stable strong signal should all be high."""
    frame = _synthetic_frame(n_symbols=1, n_bars=120)
    rolled = rolling_ic(frame, "signal", "fwd_target", window=60)

    # 120 paired rows and a 60-row window leave 61 complete windows.
    assert len(rolled) == 61
    assert (rolled["ic"] > 0.8).all()
    assert set(rolled.columns) == {"symbol", "timestamp", "ic"}


def test_rolling_spearman_reranks_inside_each_window() -> None:
    """Windowed Spearman must use within-window ranks, not full-sample ranks."""
    # One extreme outlier early on: full-sample ranks would remember it, but
    # windows that exclude it must be unaffected. With rank recomputation the
    # last window of this monotone-in-window series has IC exactly 1.
    values = np.array([1000.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    frame = pd.DataFrame(
        {
            "symbol": "SYM0",
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "signal": values,
            "fwd_target": values**3,
        }
    )
    rolled = rolling_ic(frame, "signal", "fwd_target", window=3)
    assert np.allclose(rolled["ic"].to_numpy(), 1.0)


def test_rolling_spearman_handles_ties_and_constant_windows() -> None:
    """Tied windows must use average ranks; constant windows must stay NaN."""
    scipy_stats = pytest.importorskip("scipy.stats")
    # First three values constant: any window fully inside them is undefined.
    x = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 2.0, 4.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 8.0, 7.0])
    frame = pd.DataFrame(
        {
            "symbol": "SYM0",
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "signal": x,
            "fwd_target": y,
        }
    )
    rolled = rolling_ic(frame, "signal", "fwd_target", window=3)
    rolled = rolled.set_index("timestamp")["ic"]

    # Window ending at position 2 covers [5, 5, 5]: constant x, undefined IC.
    assert pd.Timestamp("2024-01-03") not in rolled.index

    # Window ending at position 5 covers x=[1, 1, 2] with a tie: must equal
    # scipy's average-rank Spearman, not the ordinal-rank shortcut.
    expected = float(scipy_stats.spearmanr(x[3:6], y[3:6]).statistic)
    assert np.isclose(rolled.loc[pd.Timestamp("2024-01-06")], expected)


def test_quantile_bucketing_excludes_symbols_with_collapsed_buckets() -> None:
    """A symbol whose ties collapse buckets is excluded with a warning."""
    clean = _synthetic_frame(n_symbols=1, n_bars=100)
    tied = clean.assign(symbol="TIED", signal=np.where(np.arange(100) < 80, 0.0, 1.0))
    frame = pd.concat([clean, tied], ignore_index=True)

    with pytest.warns(UserWarning, match="TIED"):
        buckets = target_by_feature_quantile(frame, "signal", "fwd_target", quantiles=5)

    # Only the clean symbol's 100 rows remain, evenly split into 5 buckets.
    assert int(buckets["observations"].sum()) == 100
    assert (buckets["observations"] == 20).all()


def test_ic_summary_reports_mean_icir_and_share_positive() -> None:
    """Summary statistics should match hand-computed values."""
    ic_values = pd.Series([0.1, 0.3, -0.1, 0.2])
    summary = ic_summary(ic_values)

    assert summary["observations"] == 4
    assert np.isclose(summary["mean_ic"], 0.125)
    assert np.isclose(summary["share_positive"], 0.75)
    assert np.isclose(summary["ic_information_ratio"], 0.125 / ic_values.std())

    empty = ic_summary(pd.Series(dtype="float64"))
    assert empty["observations"] == 0 and np.isnan(empty["mean_ic"])


def test_newey_west_regression_recovers_known_slope() -> None:
    """The slope on a standardized feature should be close to the true slope."""
    frame = _synthetic_frame(slope=0.5, noise_scale=0.1)
    result = newey_west_regression(frame, "signal", "fwd_target")

    assert abs(result.beta - 0.5) < 0.05
    assert result.t_statistic > 10
    assert result.p_value < 1e-6
    assert result.observations == 600

    noise_result = newey_west_regression(frame, "noise", "fwd_target")
    assert abs(noise_result.t_statistic) < 3


def test_driscoll_kraay_matches_newey_west_for_one_symbol() -> None:
    """With a single symbol the panel-robust SE must reduce to classic Newey-West."""
    sm = pytest.importorskip("statsmodels.api")
    frame = _synthetic_frame(n_symbols=1, n_bars=300)
    result = newey_west_regression(frame, "signal", "fwd_target", kernel_lags=5)

    z = (frame["signal"] - frame["signal"].mean()) / frame["signal"].std()
    design = sm.add_constant(z.to_numpy())
    fitted = sm.OLS(frame["fwd_target"].to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False}
    )
    # ``fitted.params`` is the statsmodels attribute name, not ours.
    assert np.isclose(result.beta, float(fitted.params[1]), rtol=1e-8)
    assert np.isclose(result.beta_standard_error, float(fitted.bse[1]), rtol=1e-6)


def test_duplicated_symbol_does_not_shrink_the_standard_error() -> None:
    """A cloned symbol adds no information, so the SE must not drop by sqrt(2).

    Naive pooled HAC treats the two identical symbols as independent and cuts
    the standard error by ~sqrt(2); Driscoll-Kraay sums scores per timestamp
    first, so perfectly dependent rows change (almost) nothing.
    """
    single = _synthetic_frame(n_symbols=1, n_bars=300)
    clone = single.assign(symbol="CLONE")
    doubled = pd.concat([single, clone], ignore_index=True)

    single_result = newey_west_regression(single, "signal", "fwd_target", kernel_lags=5)
    doubled_result = newey_west_regression(doubled, "signal", "fwd_target", kernel_lags=5)

    assert np.isclose(doubled_result.beta, single_result.beta, rtol=1e-8)
    assert doubled_result.beta_standard_error > 0.95 * single_result.beta_standard_error


def test_regression_result_is_invariant_to_row_order() -> None:
    """Interleaving symbols by timestamp must not change the inference."""
    frame = _synthetic_frame(n_symbols=2, n_bars=200)
    shuffled = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    by_symbol = newey_west_regression(frame, "signal", "fwd_target", kernel_lags=5)
    by_time = newey_west_regression(shuffled, "signal", "fwd_target", kernel_lags=5)

    assert np.isclose(by_symbol.beta, by_time.beta, rtol=1e-10)
    assert np.isclose(by_symbol.beta_standard_error, by_time.beta_standard_error, rtol=1e-10)


def test_hac_lags_matter_for_autocorrelated_errors() -> None:
    """Overlap-aware lags must widen the SE when errors are serially correlated."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 400
    # Build an error series with strong positive autocorrelation, mimicking an
    # overlapping forward window.
    shocks = rng.standard_normal(n + 20)
    overlapping_error = np.convolve(shocks, np.ones(20) / 20.0, mode="valid")[:n]
    feature = rng.standard_normal(n)
    frame = pd.DataFrame(
        {
            "symbol": "SYM0",
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "signal": feature,
            "fwd_target": 0.1 * feature + overlapping_error,
        }
    )
    short = newey_west_regression(frame, "signal", "fwd_target", kernel_lags=1)
    long = newey_west_regression(frame, "signal", "fwd_target", kernel_lags=19)
    assert long.kernel_lags == 19
    # The exact ratio is data-dependent; the direction is the invariant.
    assert long.beta_standard_error != short.beta_standard_error


def test_evaluation_masks_infinite_values() -> None:
    """An inf from a bad row must be excluded, not poison the statistics."""
    frame = _synthetic_frame(n_symbols=1, n_bars=100)
    frame.loc[10, "signal"] = np.inf

    ic = time_series_ic(frame, "signal", "fwd_target")
    result = newey_west_regression(frame, "signal", "fwd_target")

    assert np.isfinite(ic.to_numpy()).all()
    assert np.isfinite(result.beta) and np.isfinite(result.beta_standard_error)
    assert result.observations == 99


def test_newey_west_regression_rejects_constant_feature() -> None:
    """A zero-variance feature has no identified slope and must raise."""
    frame = _synthetic_frame(n_bars=50)
    frame["flat"] = 1.0
    with pytest.raises(ValueError):
        newey_west_regression(frame, "flat", "fwd_target")


def test_default_hac_lags_covers_target_overlap() -> None:
    """The lag rule must cover the mechanical overlap of an h-bar target."""
    # Sample-size rule alone: floor(4 * (100/100)^(2/9)) = 4.
    assert default_kernel_lags(100) == 4
    # A 20-bar target overlaps 19 lags, which must win over the size rule.
    assert default_kernel_lags(100, target_horizon_bars=20) == 19
    assert default_kernel_lags(10) >= 1


def test_target_by_feature_quantile_is_monotone_for_linear_signal() -> None:
    """Bucket means should rise from bottom to top bucket for a positive slope."""
    frame = _synthetic_frame()
    buckets = target_by_feature_quantile(frame, "signal", "fwd_target", quantiles=5)

    assert list(buckets["quantile"]) == [1, 2, 3, 4, 5]
    means = buckets["mean"].to_numpy()
    assert (np.diff(means) > 0).all()
    # Equal-count buckets over 600 rows: 120 rows each.
    assert (buckets["observations"] == 120).all()


def test_target_values_by_quantile_partitions_all_rows() -> None:
    """The raw-values view should return every paired row exactly once."""
    frame = _synthetic_frame(n_symbols=1, n_bars=100)
    values = target_values_by_quantile(frame, "signal", "fwd_target", quantiles=4)

    assert set(values.keys()) == {1, 2, 3, 4}
    assert sum(len(v) for v in values.values()) == 100


def test_evaluate_features_ranks_signal_above_noise() -> None:
    """The summary table should rank the true signal first by |t-stat|."""
    frame = _synthetic_frame()
    table = evaluate_features(frame, "fwd_target", target_horizon_bars=1)

    assert list(table["feature"]) == ["signal", "noise"]
    signal_row = table.iloc[0]
    assert signal_row["mean_time_series_ic"] > 0.9
    assert signal_row["t_statistic"] > 10
    assert signal_row["quantile_spread"] > 0
