"""Target behavior across feature quantile buckets.

A single IC number can hide the shape of a relationship: a feature can be
useless in its middle range and predictive only in its extremes, or monotonic
up to a point and then reverse. Bucketing the feature into quantiles and
looking at the target's distribution inside each bucket exposes that shape.
This is also the data behind the violin and quantile plots in ``plots.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = 5


def _paired_rows(frame: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """Return rows where feature and target are both finite.

    Infinities (e.g. a log return over a zero close) survive ``dropna`` and
    would land in the extreme buckets as if they were real observations, so
    they are masked to NaN first.
    """
    for column in (feature, target):
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} not found in the feature frame.")
    paired = frame.loc[:, ["symbol", feature, target]].copy()
    paired[[feature, target]] = paired[[feature, target]].replace(
        [np.inf, -np.inf], np.nan
    )
    return paired.dropna(subset=[feature, target])


def _bucket_labels(
    paired: pd.DataFrame,
    feature: str,
    *,
    quantiles: int,
    by_symbol: bool,
) -> pd.Series:
    """Assign each row a 1..quantiles bucket label, or NaN when not comparable.

    Bucket numbers are only comparable across symbols when every symbol formed
    the full set of ``quantiles`` buckets. When repeated feature values force
    ``pd.qcut`` to collapse bins for a symbol (``duplicates="drop"``), that
    symbol's "bucket 3 of 3" would silently pool with other symbols'
    "bucket 3 of 5" and bias the top-minus-bottom spread. Such symbols are
    excluded entirely (labels NaN) and a warning reports how many.
    """

    def _bucket_one_series(values: pd.Series) -> pd.Series:
        buckets = pd.qcut(values, q=quantiles, labels=False, duplicates="drop")
        # A collapsed bin count means this symbol's bucket numbers would not
        # line up with the other symbols'; exclude it rather than mislabel it.
        if buckets.dropna().nunique() < quantiles:
            return pd.Series(np.nan, index=values.index)
        return buckets + 1

    if by_symbol:
        labels = paired.groupby("symbol")[feature].transform(_bucket_one_series)
        dropped_symbols = sorted(paired.loc[labels.isna(), "symbol"].unique().tolist())
        if dropped_symbols:
            warnings.warn(
                f"Feature {feature!r}: {len(dropped_symbols)} symbol(s) "
                f"({', '.join(map(str, dropped_symbols))}) had too many repeated "
                f"values to form {quantiles} distinct buckets and were excluded "
                "from the quantile analysis.",
                stacklevel=3,
            )
        return labels

    labels = _bucket_one_series(paired[feature])
    if labels.isna().all():
        warnings.warn(
            f"Feature {feature!r} has too many repeated values to form "
            f"{quantiles} distinct buckets; all rows were excluded.",
            stacklevel=3,
        )
    return labels


def target_by_feature_quantile(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    quantiles: int = DEFAULT_QUANTILES,
    by_symbol: bool = True,
) -> pd.DataFrame:
    """Summarize the target inside each feature quantile bucket.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, feature, and target columns.
    feature
        Feature column name used for bucketing.
    target
        Forward target column name being summarized.
    quantiles
        Number of equal-count buckets. 5 (quintiles) by default.
    by_symbol
        When ``True`` (default), bucket each symbol against its own history so
        "quantile 5" means "high for that symbol", then pool the buckets. When
        ``False``, bucket the pooled sample directly; only sensible when all
        symbols share the feature's scale. Symbols whose repeated feature
        values cannot form all ``quantiles`` buckets are excluded with a
        warning, because their bucket numbers would not line up with the other
        symbols'.

    Returns
    -------
    pandas.DataFrame
        One row per bucket (1 = lowest feature values), columns:
        ``quantile``, ``n``, ``mean``, ``median``, ``std``, ``q10``, ``q90``
        of the target, and ``feature_min``/``feature_max`` showing the actual
        feature range the bucket covers. The interesting read is whether
        ``mean`` moves monotonically from bucket 1 to bucket ``quantiles``.

    Raises
    ------
    ValueError
        If ``quantiles`` < 2 or there are not enough distinct feature values
        to form the requested buckets.
    """
    if quantiles < 2:
        raise ValueError("target_by_feature_quantile requires quantiles >= 2.")

    paired = _paired_rows(frame, feature, target)
    if paired.empty:
        raise ValueError(
            f"No rows have both {feature!r} and {target!r}; nothing to bucket."
        )

    bucket_labels = _bucket_labels(
        paired, feature, quantiles=quantiles, by_symbol=by_symbol
    )
    working = paired.assign(quantile=bucket_labels).dropna(subset=["quantile"])
    if working["quantile"].nunique() < 2:
        raise ValueError(
            f"Feature {feature!r} has too few distinct values to form "
            f"{quantiles} buckets."
        )

    def _summarize(bucket_frame: pd.DataFrame) -> pd.Series:
        target_values = bucket_frame[target]
        return pd.Series(
            {
                "n": len(bucket_frame),
                "mean": float(target_values.mean()),
                "median": float(target_values.median()),
                "std": float(target_values.std()),
                "q10": float(target_values.quantile(0.10)),
                "q90": float(target_values.quantile(0.90)),
                "feature_min": float(bucket_frame[feature].min()),
                "feature_max": float(bucket_frame[feature].max()),
            }
        )

    summary = (
        working.groupby("quantile")
        .apply(_summarize)
        .reset_index()
        .astype({"quantile": int, "n": int})
        .sort_values("quantile", ignore_index=True)
    )
    return summary


def target_values_by_quantile(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    quantiles: int = DEFAULT_QUANTILES,
    by_symbol: bool = True,
) -> dict[int, np.ndarray]:
    """Return the raw target values inside each feature quantile bucket.

    Same bucketing rules as :func:`target_by_feature_quantile`, but instead of
    summary statistics this returns the underlying target observations per
    bucket — the input a violin or distribution plot needs.

    Returns
    -------
    dict
        ``{bucket_number: array_of_target_values}``, bucket 1 = lowest feature
        values.
    """
    if quantiles < 2:
        raise ValueError("target_values_by_quantile requires quantiles >= 2.")

    paired = _paired_rows(frame, feature, target)
    bucket_labels = _bucket_labels(
        paired, feature, quantiles=quantiles, by_symbol=by_symbol
    )
    working = paired.assign(quantile=bucket_labels).dropna(subset=["quantile"])
    return {
        int(bucket): bucket_frame[target].to_numpy(dtype=float)
        for bucket, bucket_frame in working.groupby("quantile")
    }
