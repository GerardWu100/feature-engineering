"""Information coefficient (IC) between a feature and a forward target.

The information coefficient is the correlation between a feature observed now
and a target realized later. Because the pipeline's targets are forward-looking
(``next_n_bar_return``, ``next_n_bar_realized_volatility``), correlating row-aligned
columns already compares "signal today" with "outcome tomorrow"; no extra
shifting is needed here.

Two IC designs exist and answer different questions:

- Time-series IC (one symbol through time): "when this symbol's feature is
  high, does that symbol's future return tend to be high?"
- Cross-sectional IC (many symbols at one timestamp): "at this moment, do the
  symbols with the higher feature values deliver the higher future returns?"
  This is the classic equity-factor IC, but it needs a reasonably wide
  universe; with a handful of symbols each per-timestamp correlation is noise.

Rank (Spearman) correlation is the default everywhere because features and
returns are heavy-tailed; a single outlier can dominate a Pearson correlation
while ranks are unaffected.

A note on significance: the descriptive statistics here (mean IC, IC information ratio) do not
correct for the serial dependence created by overlapping forward windows. For
inference use ``regression.newey_west_regression``, which handles that overlap
through heteroskedasticity- and autocorrelation-consistent standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

VALID_IC_METHODS = ("spearman", "pearson")

# Minimum symbols required at one timestamp before a cross-sectional IC is
# computed. Below this, a per-date correlation is dominated by noise.
DEFAULT_MIN_SYMBOLS_PER_TIMESTAMP = 5

# Minimum paired observations required before any correlation is reported.
DEFAULT_MIN_OBSERVATIONS = 24


def _validate_method(method: str) -> None:
    """Raise if the correlation method is not supported."""
    if method not in VALID_IC_METHODS:
        raise ValueError(f"method must be one of {VALID_IC_METHODS}, got {method!r}.")


def _paired(frame: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """Return only the rows where both feature and target are present.

    Warm-up rows (feature NaN) and end-of-sample rows (forward target NaN)
    carry no information about the relationship, so they are dropped before
    any correlation. Infinities (e.g. a log return over a zero close) survive
    ``dropna`` and would silently corrupt correlations, so they are masked to
    NaN first.
    """
    for column in (feature, target):
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} not found in the feature frame.")
    paired = frame.loc[:, ["symbol", "timestamp", feature, target]].copy()
    paired[[feature, target]] = paired[[feature, target]].replace(
        [np.inf, -np.inf], np.nan
    )
    return paired.dropna(subset=[feature, target])


def time_series_ic(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    method: str = "spearman",
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> pd.Series:
    """Compute one IC per symbol across that symbol's whole history.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, ``timestamp``, feature, and target columns.
    feature
        Feature column name.
    target
        Forward target column name.
    method
        ``"spearman"`` (rank, default) or ``"pearson"`` (linear).
    min_observations
        Symbols with fewer paired (feature, target) rows than this return NaN
        instead of an unstable correlation.

    Returns
    -------
    pandas.Series
        IC per symbol, indexed by symbol name. Values lie in [-1, 1].
    """
    _validate_method(method)
    paired = _paired(frame, feature, target)

    def _one_symbol_ic(symbol_frame: pd.DataFrame) -> float:
        if len(symbol_frame) < min_observations:
            return np.nan
        return float(symbol_frame[feature].corr(symbol_frame[target], method=method))

    return paired.groupby("symbol").apply(_one_symbol_ic).rename("ic")


def cross_sectional_ic(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    method: str = "spearman",
    min_symbols: int = DEFAULT_MIN_SYMBOLS_PER_TIMESTAMP,
) -> pd.Series:
    """Compute one IC per timestamp across the symbols present at that time.

    This is the classic factor-model IC: at each timestamp, correlate the
    feature values of all symbols with their subsequent target values. Use it
    only when the universe is wide enough; timestamps with fewer than
    ``min_symbols`` symbols are skipped.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, ``timestamp``, feature, and target columns.
    feature
        Feature column name.
    target
        Forward target column name.
    method
        ``"spearman"`` (default) or ``"pearson"``.
    min_symbols
        Minimum symbols with valid data at a timestamp for that timestamp to
        produce an IC value.

    Returns
    -------
    pandas.Series
        IC per timestamp, indexed by ``timestamp``, only for timestamps that meet
        ``min_symbols``. Empty if no timestamp qualifies.
    """
    _validate_method(method)
    paired = _paired(frame, feature, target)

    # Group sizes first, so undersized timestamps are dropped in one pass
    # instead of producing NaN placeholders.
    sizes = paired.groupby("timestamp").size()
    valid_timestamps = sizes.index[sizes >= min_symbols]
    qualified = paired[paired["timestamp"].isin(valid_timestamps)]
    if qualified.empty:
        return pd.Series(dtype="float64", name="ic")

    per_timestamp = qualified.groupby("timestamp").apply(
        lambda g: float(g[feature].corr(g[target], method=method))
    )
    return per_timestamp.rename("ic")


def rolling_ic(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    window: int,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute a trailing-window time-series IC per symbol.

    This is the stability view: a full-sample IC can hide a relationship that
    held in one regime and reversed in another. Each value answers "over the
    trailing ``window`` bars ending here, what was the IC?"

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, ``timestamp``, feature, and target columns.
    feature
        Feature column name.
    target
        Forward target column name.
    window
        Trailing window length in paired observations (rows with both values).
    method
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``symbol``, ``timestamp``, ``ic``. One row per (symbol, window end);
        the first ``window - 1`` rows of each symbol are omitted because their
        windows are incomplete.
    """
    _validate_method(method)
    if window < 2:
        raise ValueError("rolling_ic requires window >= 2.")
    paired = _paired(frame, feature, target)

    results: list[pd.DataFrame] = []
    for symbol, symbol_frame in paired.groupby("symbol"):
        ordered = symbol_frame.sort_values("timestamp")
        if method == "spearman":
            # Rolling Spearman = rolling Pearson on full-sample ranks is NOT
            # exact (ranks must be recomputed inside each window), so compute
            # each window honestly. Windows are small enough in research use.
            values = _honest_rolling_rank_corr(
                ordered[feature], ordered[target], window
            )
        else:
            values = (
                ordered[feature]
                .rolling(window, min_periods=window)
                .corr(ordered[target])
            )
        block = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": ordered["timestamp"].to_numpy(),
                "ic": values.to_numpy(),
            }
        )
        results.append(block.dropna(subset=["ic"]))

    if not results:
        return pd.DataFrame(columns=["symbol", "timestamp", "ic"])
    return pd.concat(results, ignore_index=True)


def _honest_rolling_rank_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Spearman correlation over each trailing window, ranks recomputed per window.

    Ranking once over the full sample and then rolling a Pearson correlation
    over those ranks looks similar but is a different statistic: a window's
    internal ordering, not its position in the full sample, is what Spearman
    measures. This helper re-ranks inside every window with average-tie ranks
    (``scipy.stats.rankdata``), matching the full-sample Spearman that pandas
    computes, and leaves NaN where either window has fewer than two distinct
    values (the correlation is undefined there; ordinal tie-breaking would
    instead leak the time position of tied rows into the statistic).
    Complexity is O(n * window log window), fine for research-scale data.
    """
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    n = len(x_values)
    out = np.full(n, np.nan)
    for end in range(window - 1, n):
        start = end - window + 1
        x_window = x_values[start : end + 1]
        y_window = y_values[start : end + 1]
        if len(np.unique(x_window)) < 2 or len(np.unique(y_window)) < 2:
            continue
        x_ranks = rankdata(x_window)
        y_ranks = rankdata(y_window)
        x_centered = x_ranks - x_ranks.mean()
        y_centered = y_ranks - y_ranks.mean()
        denominator = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        if denominator == 0:
            continue
        out[end] = (x_centered * y_centered).sum() / denominator
    return pd.Series(out, index=x.index)


def ic_summary(ic_values: pd.Series) -> dict[str, float]:
    """Summarize a series of IC values (per-timestamp or per-window).

    Parameters
    ----------
    ic_values
        A series of IC observations, e.g. the output of
        :func:`cross_sectional_ic` or one symbol's slice of :func:`rolling_ic`.

    Returns
    -------
    dict
        ``mean_ic``: average IC. ``ic_standard_deviation``: standard deviation across
        observations. ``ic_information_ratio``: information coefficient information ratio,
        mean_ic / ic_standard_deviation — a signal-to-noise measure of the IC itself.
        ``share_positive``: fraction of observations with IC > 0.
        ``observations``: number of IC values summarized.

    Notes
    -----
    The naive t-statistic mean/std*sqrt(n) is deliberately not reported:
    overlapping forward windows make consecutive IC observations dependent,
    which inflates that t-statistic. Use Newey-West regression for inference.
    """
    clean = ic_values.dropna()
    n = len(clean)
    if n == 0:
        return {
            "mean_ic": np.nan,
            "ic_standard_deviation": np.nan,
            "ic_information_ratio": np.nan,
            "share_positive": np.nan,
            "observations": 0,
        }
    mean_ic = float(clean.mean())
    ic_standard_deviation = float(clean.std())
    ic_information_ratio = (
        mean_ic / ic_standard_deviation if ic_standard_deviation > 0 else np.nan
    )
    return {
        "mean_ic": mean_ic,
        "ic_standard_deviation": ic_standard_deviation,
        "ic_information_ratio": ic_information_ratio,
        "share_positive": float((clean > 0).mean()),
        "observations": n,
    }
