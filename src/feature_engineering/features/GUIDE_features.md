# GUIDE - features/

## Part 1 - Conceptual Explanation

`features/` contains pure stock OHLCV feature functions. A feature function receives one symbol's sorted data and returns a pandas `Series` aligned to the same index.

Files are organized by category:

| File | Category | Meaning |
|---|---|---|
| `returns.py` | `returns`, `target` | Price changes and forward-looking labels. |
| `trend.py` | `trend` | Direction and momentum. |
| `volatility.py` | `volatility` | Size and instability of price movement. |
| `volume.py` | `volume` | Trading activity and liquidity context. |
| `registry.py` | feature menu | Maps config function names to real functions and metadata. |

The `target` category is special. A target is the value a model tries to predict. `next_n_bar_return` is stored in `returns.py` because its formula is a return, but its category is `target` because it looks into the future.

`next_n_bar_return` is a forward simple return over a fixed number of bars (rows), not calendar days:

$$
\text{target}_t = \frac{C_{t+n}}{C_t} - 1
$$

Here $C_t$ is the close at the current row and $n$ is the configured horizon in bars (`bars`). A bar is one row of the input: a daily bar on daily data, a one-minute bar on one-minute data. The final $n$ rows are `NaN` because their future close is unavailable. For intraday data, enable `reset_by_session` (see `pipeline/engineer.py`) so the forward shift does not cross the overnight gap.

## Part 2 - Code Reference

| File | Key contents |
|---|---|
| `registry.py` | `FeatureSpec`, `REGISTRY`, `register`, and `as_feature_column`. |
| `returns.py` | `log_return`, `simple_return`, `next_n_bar_return`. |
| `trend.py` | `moving_average`, `price_vs_sma`, `rate_of_change`. |
| `volatility.py` | `rolling_std`, `bar_range_pct`. |
| `volume.py` | `volume_ratio`, `dollar_volume`, `volume_change`. |

Add a new feature by placing it in the matching category file and decorating it with `@register(...)`.

## Part 3 - Short Journal

- 2026-04-24: Feature modules were reorganized by simple research category instead of by a larger engine/options architecture.
- 2026-04-26: `next_n_day_return` now uses the current bar close as the denominator to avoid intraday label leakage from the current day-end close.
- 2026-05-19: Added `as_feature_column` so every feature returns an unnamed Series through one shared helper instead of repeating `values.name = None`.
- 2026-06-23: Replaced `next_n_day_return` with `next_n_bar_return`, a plain forward N-bar simple return (`close[t+bars]/close[t] - 1`). The bar horizon plus the new `reset_by_session` engineer option removed the earlier hybrid intraday/daily target logic.
