# GUIDE - engine/

## Part 1 - Conceptual Explanation

`engine/` holds two ways to run the registered features, for two different usage
patterns. Neither contains feature math; both delegate to the formulas in
`features/`.

| Engine | File | Use case | Cost per update |
|---|---|---|---|
| `FeatureEngine` | `batch.py` | Research / backtest. Validate and resolve the feature list once, then `transform(df) -> df` many times. | Full vectorized recompute over the frame. |
| `OnlineFeatureEngine` | `online.py` | Live trading. Feed one bar at a time with `update(bar) -> {feature: value}`. | O(1) (bounded) per bar, independent of session length. |

The key invariant connecting them: **the batch feature functions are the single
source of truth for the math.** The online accumulators are a second
implementation tuned for speed, and `tests/test_engines.py` feeds identical bars
through both and asserts equality to floating-point tolerance. If you add a
feature, add its online accumulator and the equivalence test guards the match.

### Why the online engine can be O(1)

- Sums over a window keep a running total plus a fixed-size deque, so adding a
  bar and dropping the one leaving the window is constant work (`_RollingSum`).
- EMA-style features (RSI and ATR via Wilder smoothing, MACD via span EMA) are
  naturally recursive: the next value depends only on the previous value and the
  new bar (`_Ema`).
- VWAP keeps two running cumulative sums and resets at each session.

### Targets are not available online

Forward-looking `target` features (for example `next_n_bar_return`) need future
bars, so `OnlineFeatureEngine` rejects them at construction. Train on batch
output that includes the target; serve live features without it.

### Session resets

When `features.reset_by_session` is true, the online engine rebuilds a symbol's
accumulators when the bar's calendar date changes, mirroring the batch
per-session isolation. This is also what makes VWAP a per-day figure intraday.

## Part 2 - Code Reference

| Path | Key contents |
|---|---|
| `batch.py` | `FeatureEngine`: caches resolved specs, `transform`, `feature_names`. |
| `online.py` | Accumulator classes, `ONLINE_FEATURE_FACTORIES`, `OnlineFeatureEngine` with `update`, `stream_frame`, `reset`, `feature_names`. |

`FeatureEngine` reuses `pipeline.engineer.apply_resolved_features`, so batch and
one-shot `compute_features` share one code path.

## Part 3 - Short Journal

- 2026-06-23: Added the `engine/` subpackage: `FeatureEngine` (cached batch
  wrapper) and `OnlineFeatureEngine` (O(1) incremental updates). Online
  accumulators are held to the batch math by equivalence tests.
