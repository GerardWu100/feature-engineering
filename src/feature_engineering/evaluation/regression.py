"""Predictive regression of a forward target with panel-robust errors.

Why this exists next to the IC functions: a correlation says whether a
relationship exists; a regression says how large it is in target units and lets
you test it. Two dependence problems make naive inference wrong here:

1. Overlap through time. Forward targets computed every bar overlap: a 20-bar
   forward return shares 19 bars with the next row's target — so consecutive
   errors are serially correlated, so ordinary least squares (OLS) standard
   errors are far too small.
2. Dependence across symbols. Volatility and returns have a large market-common
   component, so two symbols' errors at the same timestamp are correlated.
   Pooling symbols and treating rows as independent shrinks the standard error
   by roughly the square root of the number of symbols without adding real
   information.

The fix used here is Driscoll-Kraay standard errors: sum each timestamp's
score (regressor times residual) across symbols first, then apply the
Newey-West kernel to that single time series of summed scores. Contemporaneous
cross-symbol correlation lands inside each per-timestamp sum, and serial
correlation is handled by the kernel. With one symbol this reduces exactly to
classic Newey-West. Robust errors change inference, not the coefficient: the
slope estimate is plain pooled OLS either way.

Model, per (feature, target) pair, pooled across symbols:

    target_t = alpha + beta * z(feature_t) + error_t

where z(feature) is the feature standardized per symbol (mean 0, standard
deviation 1 within each symbol's history). Standardizing makes betas comparable
across features: beta is "expected change in the target per one standard
deviation move in the feature." Two caveats the caller should know: the target
is not per-symbol scaled, so symbols with larger target variance dominate the
pooled slope; and because the z-score sees the full sample, the fitted beta is
a screening statistic, never a live forecast coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RegressionResult:
    """Result of one predictive regression.

    Parameters
    ----------
    beta
        Slope: expected target change per one standard deviation of the
        feature. In the target's units (decimal return or decimal volatility).
    beta_standard_error
        Driscoll-Kraay standard error of the slope (Newey-West kernel over
        per-timestamp summed scores, robust to serial and cross-symbol
        correlation).
    t_statistic
        beta / beta_standard_error. Values beyond roughly plus or minus 2 are
        conventionally treated as distinguishable from zero, but screening many
        features guarantees that some large t-statistics will appear by chance.
    p_value
        Two-sided p-value for beta under the normal approximation.
    alpha
        Intercept: expected target when the feature is at its mean.
    r_squared
        In-sample fraction of target variance explained. For single features
        on noisy financial targets, values near zero are normal; the t-stat,
        not the R-squared, is the decision number.
    observations
        Number of pooled rows used.
    kernel_lags
        Newey-West truncation lag actually used by the kernel.
    """

    beta: float
    beta_standard_error: float
    t_statistic: float
    p_value: float
    alpha: float
    r_squared: float
    observations: int
    kernel_lags: int


def default_kernel_lags(n: int, *, target_horizon_bars: int | None = None) -> int:
    """Choose a Newey-West truncation lag.

    Two considerations, take the larger:

    1. The Newey-West rule of thumb driven by sample size:
       floor(4 * (n / 100)^(2/9)).
    2. The known overlap of the target: a target looking ``h`` bars ahead
       makes errors correlated up to lag h - 1 by construction, so the lag
       window must cover at least that.

    Parameters
    ----------
    observations
        Number of time periods available to the kernel (distinct timestamps,
        not pooled rows).
    target_horizon_bars
        Forward horizon of the target in bars, when known. ``None`` falls back
        to the sample-size rule alone.

    Returns
    -------
    int
        Truncation lag, at least 1.
    """
    size_rule = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    overlap_rule = 0 if target_horizon_bars is None else int(target_horizon_bars) - 1
    return max(1, size_rule, overlap_rule)


def newey_west_regression(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    kernel_lags: int | None = None,
    target_horizon_bars: int | None = None,
    standardize: bool = True,
) -> RegressionResult:
    """Regress a forward target on one feature with panel-robust inference.

    The coefficient is pooled OLS across all symbols; the standard error is
    Driscoll-Kraay (see module docstring), which stays honest when several
    symbols move together and when forward-target windows overlap. Rows are
    grouped by ``timestamp`` internally, so the caller's row order does not
    affect the result.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, ``timestamp``, feature, and target
        columns. Rows from all symbols are pooled after per-symbol
        standardization. Rows with non-finite feature or target values
        (NaN or infinity, e.g. from a log return over a zero close) are
        excluded.
    feature
        Feature column name (the predictor).
    target
        Forward target column name (the outcome).
    kernel_lags
        Newey-West truncation lag for the kernel. ``None`` (default) chooses
        automatically via :func:`default_kernel_lags` on the number of distinct
        timestamps.
    target_horizon_bars
        Forward horizon of the target in bars, e.g. 20 for
        ``next_20bar_realized_volatility``. Only used when ``kernel_lags`` is ``None``,
        to make sure the automatic lag covers the mechanical overlap.
    standardize
        When ``True`` (default), z-score the feature within each symbol so the
        slope reads "target units per one standard deviation of feature" and is
        comparable across features. When ``False``, the raw feature is used and
        the slope is in target-units per feature-unit.

    Returns
    -------
    RegressionResult
        Slope, Driscoll-Kraay standard error, t-statistic, p-value, intercept,
        R-squared, sample size, and the kernel lag used.

    Raises
    ------
    ValueError
        If fewer than 3 paired observations remain, or the feature is constant
        (zero variance) so no slope is identified.
    """
    for column in ("timestamp", feature, target):
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} not found in the feature frame.")

    paired = frame.loc[:, ["symbol", "timestamp", feature, target]].copy()
    # Infinities (e.g. a log return over a zero close) survive dropna, then
    # silently poison the fit, so mask them to NaN before pairing.
    paired[[feature, target]] = paired[[feature, target]].replace(
        [np.inf, -np.inf], np.nan
    )
    paired = paired.dropna(subset=[feature, target])
    if len(paired) < 3:
        raise ValueError(
            f"Regression of {target!r} on {feature!r} needs at least 3 paired "
            f"observations, got {len(paired)}."
        )

    if standardize:
        # z-score within each symbol: symbols can live on different feature
        # scales (e.g. dollar volume), and pooling raw values would let the
        # cross-symbol level differences masquerade as a time-series signal.
        def _zscore(values: pd.Series) -> pd.Series:
            spread = values.std()
            if not np.isfinite(spread) or spread == 0:
                return pd.Series(np.nan, index=values.index)
            return (values - values.mean()) / spread

        predictor = paired.groupby("symbol")[feature].transform(_zscore)
    else:
        predictor = paired[feature]

    regression_frame = pd.DataFrame(
        {
            "timestamp": paired["timestamp"],
            "predictor": predictor,
            "outcome": paired[target],
        }
    ).dropna()
    if regression_frame.empty or regression_frame["predictor"].nunique() < 2:
        raise ValueError(
            f"Feature {feature!r} has no variation after standardization; "
            "the slope is not identified."
        )

    n = len(regression_frame)
    n_timestamps = int(regression_frame["timestamp"].nunique())
    lags = (
        int(kernel_lags)
        if kernel_lags is not None
        else default_kernel_lags(n_timestamps, target_horizon_bars=target_horizon_bars)
    )

    beta_vector, standard_errors, r_squared = _driscoll_kraay_ols(
        regression_frame, lags=lags
    )
    beta = float(beta_vector[1])
    beta_standard_error = float(standard_errors[1])
    t_statistic = beta / beta_standard_error
    p_value = float(2.0 * stats.norm.sf(abs(t_statistic)))

    return RegressionResult(
        beta=beta,
        beta_standard_error=beta_standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
        alpha=float(beta_vector[0]),
        r_squared=r_squared,
        observations=n,
        kernel_lags=lags,
    )


def _driscoll_kraay_ols(
    regression_frame: pd.DataFrame,
    *,
    lags: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit pooled OLS and compute Driscoll-Kraay standard errors.

    The sandwich estimator is Var(beta) = B^-1 S B^-1 with B = X'X and S the
    Newey-West (Bartlett-weighted) long-run covariance of the per-timestamp
    summed scores h_t = sum over symbols at t of x_it * e_it. Summing within a
    timestamp first is what makes the estimator robust to cross-symbol
    correlation; the kernel over timestamps handles serial correlation.

    Parameters
    ----------
    regression_frame
        Columns ``timestamp``, ``predictor``, ``outcome``; rows already filtered to
        finite values.
    lags
        Bartlett kernel truncation lag, in timestamps.

    Returns
    -------
    tuple
        ``(beta_vector, standard_errors, r_squared)`` where ``beta_vector`` is
        ``[intercept, slope]`` and ``standard_errors`` aligns with it.
    """
    design = np.column_stack(
        [
            np.ones(len(regression_frame)),
            regression_frame["predictor"].to_numpy(dtype=float),
        ]
    )
    outcome = regression_frame["outcome"].to_numpy(dtype=float)

    bread = design.T @ design
    beta_vector = np.linalg.solve(bread, design.T @ outcome)
    residuals = outcome - design @ beta_vector

    total_variance = float(((outcome - outcome.mean()) ** 2).sum())
    r_squared = (
        1.0 - float((residuals**2).sum()) / total_variance
        if total_variance > 0
        else np.nan
    )

    # Scores per row, then summed within each timestamp in time order. After
    # this sum, each timestamp contributes one 2-vector regardless of how many
    # symbols traded, so contemporaneous cross-symbol correlation is inside
    # h_t rather than wrongly treated as independent information.
    scores = design * residuals[:, np.newaxis]
    score_frame = pd.DataFrame(scores, index=regression_frame["timestamp"].to_numpy())
    summed_scores = score_frame.groupby(level=0).sum().sort_index().to_numpy()

    n_timestamps = len(summed_scores)
    effective_lags = min(lags, n_timestamps - 1)
    long_run = summed_scores.T @ summed_scores
    for lag in range(1, effective_lags + 1):
        weight = 1.0 - lag / (effective_lags + 1.0)
        cross = summed_scores[lag:].T @ summed_scores[:-lag]
        long_run += weight * (cross + cross.T)

    bread_inverse = np.linalg.inv(bread)
    covariance = bread_inverse @ long_run @ bread_inverse
    standard_errors = np.sqrt(np.diag(covariance))
    return beta_vector, standard_errors, r_squared
