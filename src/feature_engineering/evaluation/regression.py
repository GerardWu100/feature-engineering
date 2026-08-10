"""Predictive regression of a forward target on a feature, with Newey-West errors.

Why this exists next to the IC functions: a correlation says whether a
relationship exists; a regression says how large it is in target units and lets
you test it. But forward targets computed every bar overlap — a 20-bar forward
return shares 19 bars with the next row's target — so consecutive regression
errors are strongly serially correlated, and ordinary least squares (OLS)
standard errors are far too small. Newey-West standard errors (also called HAC:
heteroskedasticity- and autocorrelation-consistent) fix the inference, not the
coefficients: the slope estimate is unchanged, only its uncertainty is
corrected.

Model, per (feature, target) pair, pooled across symbols:

    target_t = alpha + beta * z(feature_t) + error_t

where z(feature) is the feature standardized per symbol (mean 0, standard
deviation 1 within each symbol's history). Standardizing makes betas comparable
across features: beta is "expected change in the target per one standard
deviation move in the feature."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionResult:
    """Result of one predictive regression.

    Parameters
    ----------
    beta
        Slope: expected target change per one standard deviation of the
        feature. In the target's units (decimal return or decimal volatility).
    beta_se
        Newey-West standard error of the slope.
    t_stat
        beta / beta_se. Values beyond roughly +-2 are conventionally treated
        as distinguishable from zero, but remember multiple testing: screening
        many features guarantees some large t-statistics by chance.
    p_value
        Two-sided p-value for beta under the Newey-West covariance.
    alpha
        Intercept: expected target when the feature is at its mean.
    r_squared
        In-sample fraction of target variance explained. For single features
        on noisy financial targets, values near zero are normal; the t-stat,
        not the R-squared, is the decision number.
    n
        Number of pooled observations used.
    hac_lags
        Newey-West truncation lag actually used.
    """

    beta: float
    beta_se: float
    t_stat: float
    p_value: float
    alpha: float
    r_squared: float
    n: int
    hac_lags: int


def default_hac_lags(n: int, *, target_horizon_bars: int | None = None) -> int:
    """Choose a Newey-West truncation lag.

    Two considerations, take the larger:

    1. The Newey-West rule of thumb driven by sample size:
       floor(4 * (n / 100)^(2/9)).
    2. The known overlap of the target: a target looking ``h`` bars ahead
       makes errors correlated up to lag h - 1 by construction, so the lag
       window must cover at least that.

    Parameters
    ----------
    n
        Number of observations in the regression.
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
    hac_lags: int | None = None,
    target_horizon_bars: int | None = None,
    standardize: bool = True,
) -> RegressionResult:
    """Regress a forward target on one feature with Newey-West inference.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, feature, and target columns. Rows
        from all symbols are pooled after per-symbol standardization.
    feature
        Feature column name (the predictor).
    target
        Forward target column name (the outcome).
    hac_lags
        Newey-West truncation lag. ``None`` (default) chooses automatically
        via :func:`default_hac_lags`.
    target_horizon_bars
        Forward horizon of the target in bars, e.g. 20 for
        ``next_20bar_realized_vol``. Only used when ``hac_lags`` is ``None``,
        to make sure the automatic lag covers the mechanical overlap.
    standardize
        When ``True`` (default), z-score the feature within each symbol so the
        slope reads "target units per one standard deviation of feature" and is
        comparable across features. When ``False``, the raw feature is used and
        the slope is in target-units per feature-unit.

    Returns
    -------
    RegressionResult
        Slope, Newey-West standard error, t-statistic, p-value, intercept,
        R-squared, sample size, and the lag used.

    Raises
    ------
    ValueError
        If fewer than 3 paired observations remain, or the feature is constant
        (zero variance) so no slope is identified.
    """
    for column in (feature, target):
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} not found in the feature frame.")

    paired = frame.loc[:, ["symbol", feature, target]].dropna(subset=[feature, target])
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
        {"predictor": predictor, "outcome": paired[target]}
    ).dropna()
    if regression_frame.empty or regression_frame["predictor"].nunique() < 2:
        raise ValueError(
            f"Feature {feature!r} has no variation after standardization; "
            "the slope is not identified."
        )

    n = len(regression_frame)
    lags = (
        int(hac_lags)
        if hac_lags is not None
        else default_hac_lags(n, target_horizon_bars=target_horizon_bars)
    )

    design = sm.add_constant(regression_frame["predictor"].to_numpy())
    model = sm.OLS(regression_frame["outcome"].to_numpy(), design)
    fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    return RegressionResult(
        beta=float(fitted.params[1]),
        beta_se=float(fitted.bse[1]),
        t_stat=float(fitted.tvalues[1]),
        p_value=float(fitted.pvalues[1]),
        alpha=float(fitted.params[0]),
        r_squared=float(fitted.rsquared),
        n=n,
        hac_lags=lags,
    )
