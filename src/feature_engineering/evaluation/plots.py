"""Feature-versus-target visualizations for the evaluation workflow.

Three views, from raw distribution to time stability:

1. ``violin_by_quantile``: the target's full distribution inside each feature
   quantile bucket, with the quartile bar and mean dot drawn on top so the
   smooth violin body cannot overstate how much data it summarizes.
2. ``spread_rows_by_state``: many features on one shared axis, each feature
   split into low / neutral / high states, each state drawn as one row with a
   mean dot, a thick 25th-75th percentile bar, and a thin 10th-90th line. This
   grammar follows the Sun Life assessment submission, where it replaced
   per-state violins: with many states side by side, the one comparison that
   matters — each state against the others — must be a straight read.
3. ``rolling_ic_panels``: each feature's trailing-window rank IC through time,
   with shading on the side of zero opposite the full-sample value, so time
   spent with the relationship "running backwards" is visible.

Colour convention (Okabe-Ito colour-blind-safe hues, validated for colour
vision deficiency separation): vermillion = low state, grey = neutral,
blue = high state and anything measured against a return target, green =
anything measured against a volatility target. Every mark also carries a text
label, so no information is carried by colour alone.

All functions return a ``matplotlib.figure.Figure`` and never call
``plt.show()``; pass ``save_path`` to write a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from feature_engineering.evaluation.ic import rolling_ic, time_series_ic
from feature_engineering.evaluation.quantiles import target_values_by_quantile

# Okabe-Ito hues. LOW/HIGH/RETURN/VOLATILITY pass the colour-vision-deficiency
# separation checks; NEUTRAL is a deliberate recessive grey midpoint (dark
# enough for 3:1 contrast on white) used only for "neutral" states.
LOW_COLOUR = "#D55E00"  # vermillion
NEUTRAL_COLOUR = "#8C8C8C"
HIGH_COLOUR = "#0072B2"  # blue, also the default return-target hue
VOLATILITY_COLOUR = "#009E73"  # green, reserved for volatility targets
RULE_COLOUR = "#444444"

STATE_COLOURS = {
    "low": LOW_COLOUR,
    "neutral": NEUTRAL_COLOUR,
    "high": HIGH_COLOUR,
    "off": NEUTRAL_COLOUR,
    "on": LOW_COLOUR,
}

FIGURE_DPI = 200


def _target_colour(target: str) -> str:
    """Return green for volatility targets, blue otherwise.

    Keeping the two target families in different hues means a return claim and
    a volatility claim can never be confused across figures.
    """
    return VOLATILITY_COLOUR if "vol" in target.lower() else HIGH_COLOUR


def _percent_axis(axis: plt.Axes, which: str = "y") -> None:
    """Format one axis as whole percentages (0.0123 -> '1.2%')."""
    formatter = FuncFormatter(lambda value, _pos: f"{value * 100.0:.1f}%")
    if which == "y":
        axis.yaxis.set_major_formatter(formatter)
    else:
        axis.xaxis.set_major_formatter(formatter)


def _tidy_spines(axis: plt.Axes) -> None:
    """Hide the top and right spines so the data outranks the frame."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _maybe_save(figure: Figure, save_path: str | Path | None) -> None:
    """Write the figure to ``save_path`` when one is given."""
    if save_path is not None:
        figure.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")


def violin_by_quantile(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    quantiles: int = 5,
    percent: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Draw the target's distribution inside each feature quantile bucket.

    Each bucket gets a violin (smoothed density) plus honest marks on top: a
    thick 25th-75th percentile bar, a white median tick, and a mean dot. The
    marks stop a smooth violin shape from being read as more information than
    the underlying observations contain. Bucket sample sizes are printed under
    each violin for the same reason.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, feature, and target columns.
    feature
        Feature column used for bucketing (per-symbol quantiles).
    target
        Forward target column being drawn.
    quantiles
        Number of equal-count buckets. Bucket 1 holds the lowest feature values.
    percent
        Format the target axis as percentages (right for return and volatility
        targets stored as decimals).
    save_path
        Optional path; when given the figure is also written as a PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values_by_bucket = target_values_by_quantile(
        frame, feature, target, quantiles=quantiles
    )
    buckets = sorted(values_by_bucket)
    data = [values_by_bucket[bucket] for bucket in buckets]
    colour = _target_colour(target)

    figure, panel = plt.subplots(
        figsize=(11.0, 6.5), dpi=FIGURE_DPI, constrained_layout=True
    )

    parts = panel.violinplot(
        data, positions=range(len(buckets)), showextrema=False, widths=0.85
    )
    for body in parts["bodies"]:
        body.set_facecolor(colour)
        body.set_alpha(0.35)

    # Honest marks over each violin: middle half, median tick, mean dot.
    for position, bucket_values in enumerate(data):
        first, median, third = np.percentile(bucket_values, [25, 50, 75])
        panel.plot(
            [position, position],
            [first, third],
            color=colour,
            linewidth=6.0,
            alpha=0.8,
            solid_capstyle="butt",
            zorder=3,
        )
        panel.plot(
            position,
            median,
            marker="_",
            markersize=11,
            color="white",
            markeredgewidth=2.2,
            zorder=4,
        )
        panel.plot(
            position,
            float(np.mean(bucket_values)),
            marker="o",
            markersize=7.5,
            color=RULE_COLOUR,
            zorder=5,
        )

    unconditional = float(frame.loc[:, [feature, target]].dropna()[target].mean())
    panel.axhline(
        unconditional, color=RULE_COLOUR, linestyle="--", linewidth=1.4, zorder=1
    )

    # Sample size rides in the tick label so a smooth violin body cannot be
    # read as more observations than it actually summarizes.
    panel.set_xticks(range(len(buckets)))
    panel.set_xticklabels(
        [f"Q{bucket}\nn={len(values_by_bucket[bucket])}" for bucket in buckets]
    )
    panel.set_xlabel(f"{feature} quantile (Q1 = lowest values)")
    panel.set_ylabel(target)
    if percent:
        _percent_axis(panel, which="y")
    panel.grid(axis="y", alpha=0.25)
    _tidy_spines(panel)

    handles = [
        Line2D([], [], marker="o", color=RULE_COLOUR, linestyle="none", label="mean"),
        Line2D([], [], color=colour, linewidth=6.0, alpha=0.8, label="middle half"),
        Line2D(
            [],
            [],
            color=RULE_COLOUR,
            linestyle="--",
            linewidth=1.4,
            label=f"all rows: {unconditional * 100.0:.2f}%",
        ),
    ]
    panel.legend(handles=handles, loc="best", fontsize=9, framealpha=0.93)
    figure.suptitle(f"{target} by {feature} quantile", fontsize=13)

    _maybe_save(figure, save_path)
    return figure


def _tercile_states(values: pd.Series) -> pd.Series:
    """Label each observation low / neutral / high against its own symbol-free series.

    Flag-like series (only two distinct values) are labelled off / on instead,
    because terciles of a 0/1 column are meaningless.
    """
    distinct = values.dropna().unique()
    if len(distinct) <= 2:
        on_value = max(distinct)
        return values.map(
            lambda v: "on" if v == on_value else ("off" if pd.notna(v) else np.nan)
        )
    buckets = pd.qcut(values, q=3, labels=["low", "neutral", "high"], duplicates="drop")
    return buckets.astype(object)


def _spread_row(
    panel: plt.Axes, values: np.ndarray, height: float, colour: str
) -> None:
    """One state's outcomes as three marks on a single line.

    Thin line: 10th-90th percentile. Thick bar: 25th-75th (the middle half).
    Dot: the mean, which is the statistic a screen actually tests.
    """
    low, first, third, high = np.percentile(values, [10, 25, 75, 90])
    panel.plot(
        [low, high], [height] * 2, color=colour, linewidth=1.5, alpha=0.6, zorder=2
    )
    panel.plot(
        [first, third],
        [height] * 2,
        color=colour,
        linewidth=7.0,
        alpha=0.55,
        zorder=3,
        solid_capstyle="butt",
    )
    panel.plot(
        float(values.mean()), height, marker="o", markersize=8.5, color=colour, zorder=4
    )


def spread_rows_by_state(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    percent: bool = True,
    limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Draw every feature's low/neutral/high states on one shared target axis.

    For each feature, observations are split into per-symbol terciles (or
    off/on for flag-like features) and each state becomes one spread row. What
    to look for: a mean dot that sits clear of the dashed all-rows line while
    the bars around it do not — that is a state with predictive content.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, feature, and target columns.
    features
        Feature columns to draw, one block each, in the given order.
    target
        Forward target column shared by every block.
    percent
        Format the target axis as percentages.
    limits
        Optional horizontal view as decimals, e.g. ``(-0.09, 0.09)``. ``None``
        lets matplotlib choose.
    save_path
        Optional PNG output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not features:
        raise ValueError("spread_rows_by_state needs at least one feature.")

    outcome_frame = frame.dropna(subset=[target])
    unconditional = float(outcome_frame[target].mean())

    # Height scales with the number of feature blocks: one header + up to
    # three state rows + a gap is about 1.3 inches per feature.
    figure_height = max(4.0, 1.5 + 1.3 * len(features))
    figure, panel = plt.subplots(
        figsize=(13.0, figure_height), dpi=FIGURE_DPI, constrained_layout=True
    )
    panel.axvline(
        unconditional, color=RULE_COLOUR, linestyle="--", linewidth=1.4, zorder=1
    )

    height = 0.0
    ticks: list[float] = []
    tick_labels: list[str] = []

    for feature in features:
        # A header row carrying the feature name, so state rows underneath
        # need only say which state they are.
        panel.annotate(
            feature,
            xy=(0.0, height),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            va="center",
            ha="left",
        )
        height -= 1.0

        paired = frame.dropna(subset=[feature, target])
        states = paired.groupby("symbol")[feature].transform(_tercile_states)
        order = (
            ("off", "on")
            if set(states.dropna().unique()) <= {"off", "on"}
            else ("low", "neutral", "high")
        )
        for state_name in order:
            values = paired.loc[states == state_name, target].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            _spread_row(panel, values, height, STATE_COLOURS[state_name])
            ticks.append(height)
            tick_labels.append(state_name)
            height -= 1.0
        height -= 0.5  # gap between feature blocks

    panel.set_yticks(ticks)
    panel.set_yticklabels(tick_labels, fontsize=8.5)
    panel.set_ylim(height, 1.0)
    if limits is not None:
        panel.set_xlim(*limits)
    panel.set_xlabel(target)
    panel.grid(axis="x", alpha=0.25)
    _tidy_spines(panel)
    if percent:
        _percent_axis(panel, which="x")

    handles = [
        Line2D([], [], marker="o", color=RULE_COLOUR, linestyle="none", label="mean"),
        Line2D(
            [], [], color=RULE_COLOUR, linewidth=7.0, alpha=0.55, label="middle half"
        ),
        Line2D(
            [],
            [],
            color=RULE_COLOUR,
            linewidth=1.5,
            alpha=0.6,
            label="10th to 90th percentile",
        ),
        Line2D(
            [],
            [],
            color=RULE_COLOUR,
            linestyle="--",
            linewidth=1.4,
            label=f"all rows: {unconditional * 100.0:.2f}%",
        ),
    ]
    panel.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.93)
    figure.suptitle(f"{target} by feature state", fontsize=13)

    _maybe_save(figure, save_path)
    return figure


def rolling_ic_panels(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    window: int,
    save_path: str | Path | None = None,
) -> Figure:
    """Draw each feature's trailing-window rank IC through time, one panel each.

    A full-sample IC is an average; this shows what it averages over. Each
    panel has a solid line at zero, a dashed line at the feature's full-sample
    IC, and vermillion shading over the entire side of zero opposite that
    full-sample value: time the line spends in the shading is time the
    reported relationship was running backwards.

    Parameters
    ----------
    frame
        Long feature frame with ``symbol``, ``ts``, feature, and target columns.
    features
        Feature columns to draw, one stacked panel each, shared time axis.
    target
        Forward target column. Return targets draw in blue, volatility targets
        in green.
    window
        Trailing IC window length in paired observations.
    save_path
        Optional PNG output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not features:
        raise ValueError("rolling_ic_panels needs at least one feature.")

    colour = _target_colour(target)
    figure, panels = plt.subplots(
        len(features),
        1,
        figsize=(15.0, 2.5 * len(features)),
        dpi=FIGURE_DPI,
        constrained_layout=True,
        sharex=True,
    )
    panels = np.atleast_1d(panels)

    for panel, feature in zip(panels, features):
        rolled = rolling_ic(frame, feature, target, window=window)
        full_sample = float(time_series_ic(frame, feature, target).mean())

        for _symbol, symbol_rows in rolled.groupby("symbol"):
            panel.plot(
                symbol_rows["ts"],
                symbol_rows["ic"],
                linewidth=1.5,
                color=colour,
                alpha=0.9,
            )
            # Direct label at the line end so multiple symbols stay tellable
            # apart without a colour-per-symbol legend.
            last = symbol_rows.iloc[-1]
            panel.annotate(
                str(last["symbol"]),
                xy=(last["ts"], last["ic"]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8.5,
                color=RULE_COLOUR,
                va="center",
            )

        panel.set_ylim(-1.05, 1.05)
        # Everything on the opposite side of zero from the full-sample value:
        # the region where the reported relationship runs backwards.
        wrong_side = (0.0, 1.05) if full_sample < 0 else (-1.05, 0.0)
        panel.axhspan(*wrong_side, color=LOW_COLOUR, alpha=0.10, linewidth=0)

        panel.axhline(0.0, color=RULE_COLOUR, linewidth=1.2)
        panel.axhline(full_sample, color=RULE_COLOUR, linestyle="--", linewidth=1.2)
        panel.set_ylabel("rank IC", fontsize=9)
        panel.grid(alpha=0.25)
        share_same_sign = (
            float((np.sign(rolled["ic"]) == np.sign(full_sample)).mean())
            if len(rolled) and full_sample != 0
            else np.nan
        )
        share_text = (
            f", same sign in {share_same_sign:.0%} of {len(rolled):,} windows"
            if np.isfinite(share_same_sign)
            else ""
        )
        panel.set_title(
            f"{feature}  -  against {target}   |   "
            f"full sample {full_sample:+.3f}{share_text}",
            fontsize=10,
            loc="left",
        )
        _tidy_spines(panel)

    panels[-1].set_xlabel(f"end of the trailing {window}-bar window")
    figure.suptitle(
        "Trailing rank IC of each feature with the target; "
        "shading marks where the full-sample sign reverses",
        fontsize=13,
    )

    _maybe_save(figure, save_path)
    return figure
