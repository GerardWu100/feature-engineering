# GUIDE - features/

## Part 1 - Conceptual Explanation

`features/` contains pure stock OHLCV feature functions. Each function receives
one symbol's time-sorted data and returns a pandas `Series` with the same index.

Files are organized by category:

| File | Category | Input columns | Meaning |
|---|---|---|---|
| `returns.py` | `returns` | close | Backward-looking price changes. |
| `targets.py` | `target` | close (future rows) | Forward-looking labels for supervised learning. |
| `trend.py` | `trend` | close | Direction and momentum. |
| `volatility.py` | `volatility` | full candle (open, high, low, close) | Size and instability of price movement. |
| `volume.py` | `volume` | candle + volume | Trading activity and liquidity context. |
| `registry.py` | feature menu | - | Maps config function names to real functions and metadata. |

The `target` category is special: a target is the value a model tries to
predict. Targets live in their own file, `targets.py`, because they look into
the future and must never be used as live input signals.

`next_n_bar_return` is a forward simple return over a fixed number of bars (rows), not calendar days:

$$
\text{target}_t = \frac{C_{t+n}}{C_t} - 1
$$

Here $C_t$ is the close at the current row and $n$ is the configured horizon in bars (`bars`). A bar is one row of the input: a daily bar on daily data, a one-minute bar on one-minute data. The final $n$ rows are `NaN` because their future close is unavailable. For intraday data, enable `reset_by_session` (see `engineering/compute.py`) so the forward shift does not cross the overnight gap.

`next_n_bar_realized_volatility` is the volatility counterpart: instead of
direction, it labels how unstable price will be. It is the sample standard
deviation of the next $n$ one-bar log returns:

$$
\text{target}_t = \operatorname{std}\left(r_{t+1}, \ldots, r_{t+n}\right),
\qquad r_{t+k} = \ln\frac{C_{t+k}}{C_{t+k-1}}
$$

The value is per-bar volatility in decimal-return units, not annualized. It uses
the same sample standard deviation as the backward-looking
`rolling_standard_deviation` feature, so the pair answers one clear question:
can the previous window's statistic predict the next window's statistic?
`bars` must be at least 2 because the standard deviation of one return is
undefined.

## Part 2 - Code Reference

| File | Key contents |
|---|---|
| `registry.py` | `FeatureSpec`, `REGISTRY`, `register`, and `as_feature_column`. |
| `returns.py` | `log_return`, `simple_return`. |
| `targets.py` | `next_n_bar_return`, `next_n_bar_realized_volatility`. |
| `trend.py` | `moving_average`, `price_vs_moving_average`, `rate_of_change`, `relative_strength_index`, `macd_line`, `macd_signal`, `macd_histogram`. |
| `volatility.py` | `rolling_standard_deviation`, `bar_range_percent`, `average_true_range`. |
| `volume.py` | `volume_ratio`, `dollar_volume`, `volume_change`, `vwap`, `price_vs_vwap`. |

Add a new feature by placing it in the matching category file and decorating it with `@register(...)`.

## Part 3 - Short Journal

- 2026-04-24: Feature modules were reorganized by simple research category instead of by a larger engine/options architecture.
- 2026-04-26: `next_n_day_return` now uses the current bar close as the denominator to avoid intraday label leakage from the current day-end close.
- 2026-05-19: Added `as_feature_column` so every feature returns an unnamed Series through one shared helper instead of repeating `values.name = None`.
- 2026-06-23: Replaced `next_n_day_return` with `next_n_bar_return`, a plain forward N-bar simple return (`close[t+bars]/close[t] - 1`). The bar horizon plus the new `reset_by_session` engineer option removed the earlier hybrid intraday/daily target logic.
- 2026-06-23: Added `relative_strength_index` and Moving Average Convergence/Divergence (`macd_line`, `macd_signal`, `macd_histogram`) to `trend.py`, `average_true_range` to `volatility.py`, and `vwap`/`price_vs_vwap` to `volume.py`. The exponential-moving-average features drop warmup NaNs before smoothing (via `dropna`/`reindex`) so the recurrence seeds unambiguously and the constant-time online accumulators in `engine/online.py` reproduce them exactly.
- 2026-08-10: Added `next_n_bar_realized_volatility`, a forward realized-volatility target: the sample standard deviation of the next `bars` one-bar log returns, computed as a backward rolling std shifted back by `bars` so each row's window covers exactly the future returns and the final `bars` rows are `NaN`.
- 2026-08-09: All feature parameter defaults (`DEFAULT_MOVING_AVERAGE_WINDOW`, `DEFAULT_RATE_OF_CHANGE_PERIODS`, Relative Strength Index and Moving Average Convergence/Divergence constants) now live at the top of `trend.py` and are imported by the online engine, so an omitted configuration parameter means the same thing on both paths. `average_true_range` reuses the shared `_wilder_average` helper, and `macd_histogram` computes the convergence/divergence line once and feeds it to the signal helper.
- 2026-08-10: Renamed every non-standard abbreviation in the user-facing surface so the config and data contract read as plain words: config key `fn` -> `function`, `[[features.params]]` -> `[[features.parameters]]`, column `ts` -> `timestamp` (the ClickHouse column is still `ts` and is aliased in the query), session `rth` -> `regular`, and the feature functions `rolling_std` -> `rolling_standard_deviation`, `bar_range_pct` -> `bar_range_percent`, `price_vs_sma` -> `price_vs_moving_average`, `next_n_bar_realized_vol` -> `next_n_bar_realized_volatility`. MACD and VWAP stay as-is because they are universally accepted finance terms.
- 2026-08-15: Moved the forward-looking targets out of `returns.py` into their own `targets.py` so the category files match the config categories one-to-one.
