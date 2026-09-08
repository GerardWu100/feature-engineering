# Fable Thoughts - 2026-09-08

Strategic review of the feature-engineering project against the goal in
`AGENTS.md`: *"Feature engineering for quant trading; features built here must
fit into my quant research and trading pipeline."* No code was changed; this
file is the only addition. Evidence is the working tree at commit `0a04f52`,
`git log`, and one run of `uv run pytest -q` (91 passed in 2.55 s).

## 1. What changed since the 2026-08-31 review

Short answer: nothing that moves the project toward the goal. Three commits
landed after the Fable review was written:

| Commit | Date | What it contains |
|---|---|---|
| `7b6e3a5` | 2026-08-31 | One-line project goal added to `AGENTS.md`. |
| `d23fdb9` | 2026-09-01 | The Fable review file itself. |
| `0a04f52` | 2026-09-05 | The Astra review file, plus a formatting pass on 11 source and test files. |

I read the diff of `0a04f52` against `src/`: every hunk is an import wrapped
onto multiple lines, a quote style change, or a string joined onto one line.
No logic changed. The last substantive edits to the two files that matter most
for this review are `engineering/features/targets.py` (2026-08-15) and
`evaluation/summary.py` (2026-08-24). `outputs/` is still empty and
`git status` is clean.

Recommendations from the two prior reviews, checked against the tree:

| Recommendation | Fable 08-31 | Astra 09-05 | Acted on? |
|---|---|---|---|
| Name a first consumer and write its interface or acceptance contract | Step 1 | Step 1 | No. No contract document exists anywhere in the repo. |
| One thin end-to-end slice in the consuming project | Step 2 | Step 3 | No. Nothing under `~/projects` imports `feature_engineering` (unchanged since 08-31). |
| Chronological (out-of-sample) confirmation | Step 3, inside `evaluation/` | Step 4, possibly in the consumer repo | No. `evaluation/` still has no split, purge, or walk-forward code. |
| Pin input provenance (adjustment policy, column type, run id) | "Cheap now" list | Step 2 | No. `run_summary` still records config only (`store.py:180-228`). |
| Research ledger / promotion record | Section 4 | Step 5 | No. |
| Do not add more indicators | Explicit | Explicit | Yes, trivially: no indicators were added. |
| Survivorship-free universe via `ivydb.secprd` | Step 4 | Not adopted as fact | No, and see Section 3 on whether it is needed yet. |
| Live-parity harness | Step 5, "not urgent" | Step 6, "only when needed" | No, correctly deferred. |

The honest reading is that the project has been reviewed twice and worked on
zero times in between. The two reviews are broadly consistent, and the code is
in exactly the state they described. Both told the owner to stop polishing and
connect the component to something. That has not happened, so this review
tries to give a smaller, more concrete first move.

## 2. What the code does today

- **Registry:** 19 functions in `engineering/features/` (2 backward returns,
  2 forward targets, 7 trend, 3 volatility, 5 volume). The default
  `config.toml` enables both targets but then removes them through
  `exclude_categories = ["target"]` (`config.toml:63`), so the shipped run
  writes 17 predictor columns and zero targets.
- **Targets:** exactly two, both fixed-horizon and both measured from the
  close of bar $t$ to a fixed number of later rows:
  `next_n_bar_return` (`targets.py:25`, $C_{t+n}/C_t - 1$) and
  `next_n_bar_realized_volatility` (`targets.py:82`, sample standard
  deviation of the next $n$ one-bar log returns). Horizons count surviving
  rows after cleaning, not calendar time.
- **Pipeline:** `cli.run_pipeline` validates config, loads CSV or ClickHouse
  bars, drops invalid rows, computes features per symbol (optionally per
  symbol-day, `compute.py:50`), and stores Parquet/CSV plus catalog and
  summary. Every feature function receives one group's OHLCV frame and
  keyword parameters only (`compute.py:198`); nothing else can enter a formula.
- **Evaluation:** `evaluate_features` (`summary.py:22`) returns one row per
  feature with mean time-series Spearman information coefficient (IC, the
  rank correlation between feature now and target later), a pooled ordinary
  least squares slope with Driscoll-Kraay standard errors, a t-statistic,
  p-value, in-sample $R^2$, and a top-minus-bottom quantile spread, sorted by
  $|t|$. The Newey-West kernel lag is `max(size rule, target_horizon_bars - 1)`
  (`regression.py:122-124`). Plots: violin by quantile, spread rows by state,
  rolling IC panels.
- **Tests:** 86 test functions, 91 collected with parametrization, all pass.
  They cover formulas on hand-checkable toy data, contracts, config
  validation, time semantics, and synthetic known-relationship statistics.

## 3. The gap against the goal

The goal is integration. The prior reviews agree on the diagnosis (a good
component on a bench, no engine) and mostly agree on the cure (name a
consumer, write the contract, run one slice, confirm out of sample). Where
they differ:

| Point of disagreement | Fable 08-31 | Astra 09-05 | My position |
|---|---|---|---|
| Function count | "17 functions" | 19 registered, 17 selected by default | Astra is right; Fable counted the default output. |
| `firstrate.stocks` lacks delisted tickers; use `ivydb.secprd` | Stated as fact | Not verifiable from the repo, not adopted | True per the owner's own notes from other projects, but irrelevant until a cross-sectional question is chosen. The default research question here is within-symbol (per-symbol z-scores and quantiles, `regression.py:210-216`, `quantiles.py:92`), where a survivor-only pair of large caps is a limitation, not a bias that invalidates the screen. |
| Where out-of-sample confirmation lives | Add to `evaluation/` | Possibly the consumer repo | Add a minimal chronological split to `evaluation/` here. A consumer does not exist yet, so "put it in the consumer" defers it indefinitely. |
| ClickHouse as feature store | Proposed as a simpler option | Only after a second consumer | Astra. Local Parquet is fine for one consumer. |

My own addition. Both reviews say "name the consumer" first, and both are
right that "fits into my pipeline" has no completion condition without one.
But the owner's question for this review (do practitioners use labels that
decide *when* to enter and exit, not just fixed 10- or 20-bar returns?) points
at why naming a consumer has stalled: the consumer would be a trading rule,
and a trading rule needs a label shaped like a trading decision. The two
targets in the tree are shaped like a forecaster's homework, not a trader's
decision. Adding one decision-shaped label is squarely "feature engineering",
is small, and makes the consumer contract concrete because the label *is* the
entry-and-exit rule. That is the first step I recommend, ahead of the
contract document, because it gives the contract something to describe.

## 4. Targets and timing: fixed horizons versus decision-driven labels

Yes, practitioners use many labels other than a fixed $N$-bar forward return,
and the most used ones encode *when* a position would have been closed. A
fixed-horizon return assumes you hold for exactly $N$ bars regardless of what
price does. Real rules exit when a profit target, a stop, a signal reversal,
or a time limit is reached, whichever comes first. Labels that mimic that
answer the question a trader actually faces.

Notation used below: $C_t$ is the close of bar $t$; $r_{t,k} = C_{t+k}/C_t - 1$
is the forward simple return $k$ bars ahead; $\sigma_t$ is a trailing
volatility estimate known at $t$; $H$ is a maximum holding period in bars.

| Label family | What the label is | Question it answers | What this codebase lacks | Main statistical trap |
|---|---|---|---|---|
| **Triple-barrier** (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 3) | Scan forward from $t$; label $+1$ if $r_{t,k}$ first reaches $+u\sigma_t$, $-1$ if it first reaches $-d\sigma_t$, else the outcome at the time barrier $H$. | "If I enter here with a vol-scaled take-profit and stop, which one hits first?" | A forward scan up to $H$ bars with a trailing-volatility threshold. Targets today are single `shift(-bars)` calls. | Labels overlap for a *variable* number of bars, so the Newey-West lag must cover the worst case ($H-1$), and label "uniqueness" is low: consecutive rows share most of their future path. |
| **Meta-labeling** (same source) | Given a primary model's side (long/short), label $1$ if acting on it would have been profitable under the triple-barrier rule, else $0$. | "Should I take this trade, and how big?" (bet sizing, not direction) | A primary signal as an input column. Feature functions see only OHLCV (`compute.py:198`); there is no two-stage pipeline. | The secondary model is trained on trades the primary model chose, so its accuracy is conditional on that model; evaluate it only on the primary model's out-of-sample signals. |
| **Trend-scanning** (Lopez de Prado, *Machine Learning for Asset Managers*) | For each $t$, regress $C_{t..t+L}$ on time for several $L$; label is the sign (or t-value) of the slope for the $L$ with the largest $\lvert t \rvert$. | "Is a trend starting here, and how confident is that?" | A forward multi-window regression loop; a choice of $L$ range. | The label is itself an in-sample statistic of the future path, so overlap runs to the longest $L$; the horizon choice is data-snooped by construction. |
| **Time-to-event / first-passage** | Number of bars until $r_{t,k}$ first crosses a threshold (or $H$ if never). | "How long until this move plays out?" (exit timing, position half-life) | Same forward scan; integer output with censoring at $H$. | Censoring: rows that never hit get $H$, which is a lower bound, not the true time. Plain regression on censored counts is biased; survival methods are the correct tool. |
| **Volatility-scaled or volume-clock horizons** | Horizon $H_t$ varies per row: bars until cumulative dollar volume reaches a threshold, or $N$ scaled by $\sigma_t$. Often done by resampling into dollar or volume bars before any feature is computed. | "What happens over the next unit of *information*, not the next unit of clock time?" | Everything counts rows (`shift(-bars)`). Volume bars need a resampling stage between `load` and `compute`. | Variable horizon again means Newey-West lag at the maximum; and results on volume bars are not comparable to results on time bars, so IC tables from the two clocks cannot be mixed. |
| **Regime- or signal-conditioned exits** | Return from entry at $t$ to the first bar where an exit rule fires (e.g. close crosses below its moving average, RSI crosses 50, or a regime flag flips). | "What does this entry earn under my actual exit rule?" | A "scan forward until condition" primitive. The exit conditions themselves exist as features (`moving_average`, `relative_strength_index`). | Circularity: if the exit uses a moving average, evaluating the moving-average feature against that label is testing a feature against a function of itself. |
| **Reinforcement-learning execution timing** | No label. An agent learns a policy from simulated rewards (fills, costs, inventory) in an execution environment. | "Given the state, act now or wait?" | A simulator with costs and fills; entirely outside this repo's scope. | Overfitting to the simulator and to one regime; sample inefficiency; near-impossible to audit. Use only after a supervised label already works. |

**Recommendation: add the triple-barrier label first.** Reasons: it is the
standard answer to the owner's question; it directly encodes entry-and-exit
timing; it reuses the existing trailing volatility formula (the same one in
`volatility.py:30-66`); it needs no input beyond OHLCV so it fits the current
registry contract; and it is the base on which meta-labeling and
time-to-event labels are built (the same scan produces both). Trend-scanning
and volume clocks change the sampling or the horizon choice and are better
added once one decision-shaped label has gone through evaluation end to end.

Definition to implement. With $\sigma_t$ the sample standard deviation of
one-bar log returns over the trailing `volatility_window` bars ending at $t$
(known at $t$, so no leakage), $u$ the upper multiple, $d$ the lower
multiple, and $H$ the time barrier:

$$k^{\ast} = \min\{\,k \in 1..H : r_{t,k} \ge u\,\sigma_t \ \text{or}\ r_{t,k} \le -d\,\sigma_t\,\}$$

$$\text{label}_t = \begin{cases} +1 & k^{\ast} \text{ exists and the upper barrier was hit at } k^{\ast} \\ -1 & k^{\ast} \text{ exists and the lower barrier was hit at } k^{\ast} \\ 0 & \text{no barrier hit within } H \text{ bars} \end{cases}$$

Use closes, not highs and lows, for the first version so the label is
consistent with the two existing close-based targets and so a bar that
touches both barriers cannot produce an ambiguous label. The last $H$ rows of
each group, and any row whose $\sigma_t$ is not yet defined, are `NaN`.
Returning $0$ (rather than the sign of $r_{t,H}$, which the book also allows)
keeps "time ran out" observable, which is what an exit-timing rule wants.

Exact signature to register in
`src/feature_engineering/engineering/features/targets.py`:

```python
DEFAULT_TRIPLE_BARRIER_MAX_BARS = 20
DEFAULT_TRIPLE_BARRIER_MULTIPLE = 2.0
DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW = 20


@register(
    category="target",
    lookback=lambda parameters: int(
        parameters.get("volatility_window", DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW)
    ),
    description=(
        "Triple-barrier label: +1 if the upper volatility-scaled barrier is hit "
        "first, -1 if the lower barrier is hit first, 0 if neither within max_bars."
    ),
    calculation=(
        "first k<=max_bars with close_{t+k}/close_t-1 >= upper_multiple*sigma_t "
        "(+1) or <= -lower_multiple*sigma_t (-1); else 0; "
        "sigma_t = trailing std of log returns over volatility_window bars"
    ),
)
def triple_barrier_label(
    frame: pd.DataFrame,
    *,
    max_bars: int = DEFAULT_TRIPLE_BARRIER_MAX_BARS,
    upper_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    lower_multiple: float = DEFAULT_TRIPLE_BARRIER_MULTIPLE,
    volatility_window: int = DEFAULT_TRIPLE_BARRIER_VOLATILITY_WINDOW,
) -> pd.Series:
```

Concrete example with `max_bars=3`, `upper_multiple=lower_multiple=1.0`,
and $\sigma_t = 0.02$: closes `[100, 101, 103, 104, ...]` give
$r_{t,1}=0.010$, $r_{t,2}=0.030 \ge 0.02$, so the label at row 0 is $+1$
with $k^{\ast}=2$. A second function `triple_barrier_bars_to_hit` returning
$k^{\ast}$ (or `max_bars` when censored) falls out of the same loop and is
the natural time-to-event companion; add it in the same change if cheap.

Wiring notes, because the registry and validator will otherwise reject or
mis-describe it:

- Add `max_bars` and `volatility_window` to `POSITIVE_INTEGER_FEATURE_PARAMS`
  (`config.py:26`) and `{"triple_barrier_label": {"volatility_window": 3}}`
  to `FEATURE_PARAMETER_MINIMUMS` (`config.py:41`). The float multiples pass
  the signature check at `config.py:366` without changes.
- The output is numeric with `NaN` warm-up and tail rows, so it satisfies
  `_validate_feature_result` (`compute.py:216`).
- With `reset_by_session = true` the scan is confined to one symbol-day, the
  same behaviour the existing targets have.

How to evaluate it. Call
`evaluate_features(frame, "triple_barrier_20", target_columns=..., target_horizon_bars=max_bars)`.
Set `target_horizon_bars` to `max_bars` (20 in the example), not to a typical
holding time: the label at $t$ can depend on closes up to $t+H$, so
consecutive errors can be correlated up to lag $H-1$, and
`default_kernel_lags` (`regression.py:88`) turns `target_horizon_bars=H` into
exactly that lag. This is conservative when most labels resolve early, which
is the right side to err on. The pooled OLS then acts as a linear probability
model on a $\{-1,0,+1\}$ outcome: `beta` reads as "change in
$P(\text{up}) - P(\text{down})$ per one standard deviation of the feature",
and `quantile_spread` is the same quantity between the top and bottom
feature buckets. That is a legitimate screening statistic. Do not read
$R^2$ at all for this target.

## 5. Recommended next steps

Ordered; each says what it unblocks. Effort assumes one person with the
current toolchain.

1. **Add `triple_barrier_label` (and `triple_barrier_bars_to_hit`) with tests
   and a config block.** About half a day. Unblocks every later step by
   giving the project a decision-shaped target. Hand-checkable test on the
   five-row fixture in `tests/test_feature_math.py`; a second test that the
   last `max_bars` rows are `NaN` and that the label is unchanged when future
   rows beyond $k^{\ast}$ are perturbed (proves the "first hit" logic).
2. **Write the consumer contract, one page, in `docs/reference/`.** Half a
   day. Keep from both prior reviews. Must contain one timing sentence
   ("feature at close $t$, earliest fill at open $t+1$, label barriers
   measured from that fill") and the barrier parameters. Step 1 makes this
   concrete; without it the contract has nothing to specify.
3. **Run the screen once on real data and record it.** One day. AAPL and
   MSFT, regular session, the 17 predictors against `triple_barrier_20`,
   `target_horizon_bars=20`, results appended to a plain
   `docs/reference/research_ledger.md` with date, config stem, and verdict.
   Keep the ledger idea from Fable 08-31; it is minutes of work.
4. **Add a chronological split to `evaluation/`.** Two to three days. One
   function that screens on the first fraction of each symbol's history,
   purges the `max_bars` rows straddling the split, and re-runs
   `evaluate_features` on the remainder with per-symbol z-scores fitted on
   the training portion only. Keep from Fable 08-31; place it here, not in a
   consumer that does not exist.
5. **Shift the label origin to the next open.** Half a day, after step 2
   decides the timing sentence. A `from_next_open` flag on the targets, or a
   sibling target, so the label matches the executable price. Keep from both
   reviews' "targets that match trading decisions" lists.
6. **Only then: meta-labeling or a thin backtest slice** that turns labels
   into positions with a cost assumption. Several days. This is the first
   step that needs something outside this repo.

Drop for now: the ClickHouse feature store (Fable 08-31, item 3 of "later"),
the survivorship-free universe work (until a cross-sectional question is
chosen), the live-parity harness (until a live consumer exists), performance
work, and any new indicator. Keep: consumer contract, ledger, chronological
confirmation, decision-aligned targets.

## 6. What could go wrong

- **The label becomes a new parameter search.** $u$, $d$, $H$, and the
  volatility window are four knobs; sweeping them across 17 features is
  multiple testing on a new axis. Fix them in the contract before screening
  and record every variant in the ledger.
- **Degenerate label mix.** Large multiples make most labels $0$; small ones
  make nearly every label resolve in one or two bars. Report the class shares
  beside the screen and treat a $0$-share above roughly 70 percent or below
  10 percent as a signal to revisit the multiples, not as a result.
- **Close-only barriers understate stops.** Real stops trigger on intrabar
  lows. A close-based label is optimistic for the lower barrier. State this
  in the docstring and revisit with high/low once the close version works.
- **Cleaning changes the clock.** Dropped rows make $H$ bars cover more
  elapsed time (Astra's point stands). With `reset_by_session` off on
  intraday data, a barrier scan crosses the overnight gap, where the first
  bar's return is often the largest of the day.
- **Screening evidence still is not confirmation.** The Driscoll-Kraay
  errors are honest, but full-sample z-scores and quantiles remain in-sample
  (`regression.py:210-216`). Step 4 exists to stop a good t-statistic from
  being read as a deployable edge.
- **Third review, still zero integration.** The repeated pattern is careful
  reviews followed by no consumer. If step 1 through 3 are not done within a
  couple of sessions, the honest conclusion is that this repository is a
  finished library and the goal belongs to whichever project will call it.
- **Small mechanical hazards.** `plots._target_colour` (`plots.py:68`)
  chooses the volatility hue by the substring `"vol"`, so a target named with
  "volume" would be coloured as volatility; the `feature_catalog.csv` is
  still overwritten per directory while datasets accumulate (flagged in
  2026-05 and 2026-08, still true, `store.py:55`). Neither blocks the plan.

**TL;DR**

- **Request** - Review the project's progress since the 2026-08-31 and
  2026-09-05 reviews, restate the gap against the trading-pipeline goal, and
  answer whether practitioners use labels that decide entry and exit timing
  rather than fixed 10- or 20-bar returns.
- **Answer** - Nothing substantive changed: 19 registered functions, 2
  fixed-horizon targets, 91 passing tests, and no consumer, contract, split,
  or ledger. Practitioners do use decision-driven labels; add the
  triple-barrier label first with the signature above, evaluate it with
  `target_horizon_bars=max_bars`, then write the contract, run one recorded
  screen, and add a chronological split before anything else.

*Addendum, same day: step 1 was implemented after this review was written.
`targets.py` now registers `triple_barrier_label`, `triple_barrier_bars_to_exit`
(the function called `triple_barrier_bars_to_hit` above), and
`triple_barrier_exit_return`, all built on one `scan_triple_barrier` call, with
config validation, tests, and three config blocks. Steps 2 to 6 remain open.*
