# Fable Thoughts - 2026-08-31

This is an assessment of the feature-engineering project against the goal in
`AGENTS.md`: *"Feature engineering for quant trading; features built here must
fit into my quant research and trading pipeline."*

The feature-building half is in good shape. The pipeline-integration half has
not started. That is the main finding.

## 1. What the project does today

Based on the code, not just the documentation:

**Feature engineering.** A linear batch pipeline validates `config.toml`, loads
OHLCV bars from ClickHouse (`firstrate.stocks`) or a local CSV, removes invalid
rows, computes configured features per symbol, and writes Parquet or CSV files,
a feature catalog, and a JSON run summary. `load_features` reads a stored run.

The menu contains 17 functions in five groups: backward returns, forward
targets, trend indicators, volatility measures, and volume measures. They are
all standard technical indicators based only on each symbol's own bars. The
default config runs AAPL and MSFT over two years of regular-session bars.

**Evaluation.** The screening toolkit tests stored features against a forward
target using time-series, cross-sectional, and rolling Spearman information
coefficients; pooled ordinary-least-squares regression with Driscoll–Kraay
standard errors; quantile buckets; a ranked summary table; and three plot
types.

The statistics are careful for a project of this size. They handle rolling
window ranks, infinities, ties, mechanical overlap in forward targets, and
cross-symbol co-movement. The documentation also makes the important point
that the output ranks candidates; it does not prove that a feature will work
live.

**Engineering quality.** Strong. The project has boundary validation, explicit
data contracts, per-symbol isolation, duplicate-bar checks, clear session and
timezone rules, feature-output validation, hand-checkable tests, and thorough
docstrings.

Two facts the documentation does not state:

- `outputs/` is empty, and no other project under `~/projects` imports
  `feature_engineering`. Nothing downstream consumes the package today.
- Recent git history is mostly internal work: refactors, renames, hardening,
  and documentation. The feature menu and pipeline shape have been roughly
  stable since April.

## 2. The gap against the goal

**Built:**

- A correct, well-tested batch feature computer for a small set of symbols.
- A statistically careful in-sample screening toolkit.
- A clean library surface: `compute_features` is a pure in-memory transform
  that a backtest or live loop could call.

**Half-built:**

- **The research loop.** Screening exists, but there is no out-of-sample step:
  no chronological train/test split, walk-forward check, or way to show that a
  feature survived on data the screen never saw. The full-sample regression
  z-scores are explicitly screening statistics, not live coefficients. The
  evaluation code can nominate candidates, but not confirm them.
- **The cross-sectional path.** `cross_sectional_ic` requires at least five
  symbols per timestamp, but the config, data source, and default workflow use
  a short explicit symbol list. There is no universe definition. Since
  `firstrate.stocks` has no delisted tickers, a broader study using it has
  survivorship bias. The survivorship-free route on the ClickHouse server is
  `ivydb.secprd` total returns.
- **Live computation.** Batch and online feature engines were removed in the
  2026-08-15 simplification. `compute_features` could run on a trailing window,
  but nothing checks that windowed recomputation matches stored batch values.
  Per-feature warm-up requirements exist only as catalog metadata.

**Not started:**

- **A real consumer.** No modeling project, backtest, or trading system reads
  these features. The required interface is also unwritten: schema, storage,
  universe, timing convention, and adjustment policy. Until those are defined,
  “fits into my pipeline” cannot be tested.
- **Trading timing and costs.** Features use bar *t*'s close, and targets start
  at bar *t*'s close. That is fine for screening, but a trading consumer needs a
  rule such as “signal at the close of *t*, earliest execution at the open of
  *t+1*,” plus a cost model. Nothing encodes this today.
- **A research ledger.** Each run snapshots its own config, but there is no
  record across sessions of which features were screened or discarded. Because
  screening involves multiple testing, the evidence quietly weakens as sessions
  accumulate.

The project is a well-built component sitting on a bench. The goal requires
that component to be installed in an engine. The missing work is connective,
not internal.

## 3. Recommended next steps

1. **Name the first consumer and write the interface contract.** Choose the
   concrete project that will read these features first, then document the
   storage format, schema, universe, adjustment policy, timing convention
   (for example, “feature at close *t* is tradable at *t+1*”), and warm-up rules.
   Every later decision depends on this. Small: roughly a day of thinking and
   writing.

2. **Build one thin end-to-end slice.** In that consuming project, load a
   stored feature run, fit the simplest possible model—even a one-feature sign
   rule—and produce a walk-forward result. The goal is to expose joins,
   adjustment mismatches, timing, warm-up rows, and target handling, not to
   find a good strategy. Medium: a few days.

3. **Add out-of-sample confirmation to `evaluation/`.** Start with a
   chronological split: screen on the first portion and confirm on the held-out
   remainder. Report the split date in the output. This turns the evaluation
   code from a candidate-ranking tool into a source of usable evidence. The
   statistics are simple; keeping every fit and transform inside the training
   window is the hard part. Medium.

4. **Fix the data foundation before widening the universe.** For studies beyond
   a few hand-picked large caps, use the survivorship-free `ivydb.secprd` path
   and define the universe with effective-dated membership—for example, the top
   *N* symbols by dollar volume on each date—instead of a static ticker list.
   This is mandatory for cross-sectional conclusions, but can wait if the first
   consumer needs only named symbols. Medium-large.

5. **Add a live-parity harness when the trading side is real.** Feed bars one at
   a time, or as trailing windows, through `compute_features` and compare the
   result with the stored batch run. Derive each feature's minimum history from
   its catalog `lookback`. This is not urgent until a live consumer exists.
   Medium.

6. **Scale performance and storage last.** The per-symbol, per-feature Python
   loop and the rolling Spearman calculation are adequate for research-scale
   data. They may be slow for 1,000 symbols of minute bars, but there is no
   reason to optimize before a real workload demonstrates the problem.

Do not add more indicators yet. Seventeen is already more than the number of
features evaluated end-to-end, and every addition increases the multiple-testing
burden without moving the project closer to its goal.

## 4. What may be missing

**Decisions the code has already made for you:**

- **Time-series or cross-sectional research.** Per-symbol z-scores, time-series
  IC, two default symbols, and per-symbol quantile buckets all point toward
  asking whether a symbol's own indicator predicts its own future. That differs
  from ranking many stocks against one another. The two approaches need
  different data and portfolio construction, so choose explicitly. A
  cross-sectional study also invalidates the current data source.
- **Own-symbol technical indicators only.** Every feature uses one symbol's
  OHLCV history and nothing else. There is no market or sector context, calendar
  information, or cross-asset input. This is the biggest constraint on what the
  pipeline can find, yet it is not documented as a deliberate choice.
- **Horizons measured in bars, not time.** A 20-bar window covers different
  amounts of wall-clock time across sessions, halts, and bar sizes. That is a
  clean choice, but it prevents mixing frequencies in one frame and leaves no
  place for lower-frequency inputs such as fundamentals or macro data joined as
  of a bar.
- **Adjusted prices everywhere.** `dollar_volume` is adjusted close multiplied
  by raw volume, so it is not actual dollars traded. VWAP on adjusted prices is
  not the VWAP someone traded against. Back-adjusted prices also change when a
  new split or dividend appears, which makes old feature files hard to
  reproduce from the same query. The run summary does not capture that
  versioning problem.
- **Signal timing.** Features use bar *t*'s close and targets begin at that
  close. A backtest that trades at the same close is unrealistic. The contract
  needs one explicit timing sentence, and probably a target measured from
  *t+1*'s open.

**Assumptions likely to fail at scale:**

- The full date range fits in memory; there is no chunking.
- A naive `DateTime` that is actually UTC cannot be identified from the data.
  Verify the ClickHouse column type once and record it in the README.
- `feature_catalog.csv` is overwritten per output directory while datasets
  accumulate beside it. A later reader could pair a dataset with the wrong
  catalog. The 2026-05-05 architecture review flagged this, and it is still
  true.

**Simpler options worth considering:**

- Write features directly to ClickHouse. A `features` table keyed by symbol,
  timestamp, feature name, and run id would make price joins simple, give every
  consumer one access path, and remove local file management.
- Keep evaluation as a library used from short notebooks for each research
  question. It is already a reasonable size; it does not need to become a
  general framework.

**Cheap now, expensive later:**

- Write the interface contract before adding features.
- Start a research ledger with the session date, screened feature, and verdict.
  It takes minutes per session; reconstructing the multiple-testing exposure
  later may be impossible.
- Record the ClickHouse column-type verification.

## 5. Improvements that could come later

Ranked by how much they would serve the goal:

1. **Market-relative and cross-sectional features:** excess return over an
   index, rolling beta, sector-relative momentum, and cross-sectional ranks.
   This is where single-name technical indicators usually give way to broader
   equity research. It requires the survivorship-free data path first.
2. **Targets that match trading decisions:** open-to-open returns,
   volatility-scaled returns, and excess-over-market returns. These are cheap to
   add and make evaluation answer questions a portfolio actually faces.
3. **ClickHouse as the feature store:** one storage system for prices and
   features, with a natural home for run ids and lineage.
4. **Overnight and intraday decomposition:** close-to-open and open-to-close
   returns. This fits the existing session machinery.
5. **Experiment-tracking hooks:** add evaluation results to the run summary so
   each research session documents not only its config but also its findings.

Interesting, but not useful right now: more oscillator indicators, a plugin
system for third-party features, multi-exchange or crypto support (the
Hyperliquid project has its own stack), and performance work before a real
workload exists.

## 6. What could go wrong

- **The polishing trap is the biggest risk.** Recent work has improved internal
  quality through renames, contracts, docstrings, and guide files. That is good
  maintenance, but none of it has put a feature into a pipeline. The failure
  mode is an excellent component that nobody uses. For the next few sessions,
  count “a downstream project consumed a feature run” as progress.
- **Good statistics can hide survivorship bias.** The evaluation machinery is
  rigorous, which makes its results persuasive. Run it on a survivor-only
  universe and it can produce confident, wrong conclusions with excellent
  standard errors. Careful statistics cannot repair biased data.
- **Multiple testing can remain invisible.** Seventeen features, several
  targets, and repeated sessions make spurious t-statistics near 2 more likely.
  A docstring warning does not track that risk across sessions; a ledger does.
- **Batch and live results can diverge.** If a trading loop reimplements a
  feature instead of calling this package, the two versions will eventually
  differ. The pure-function design helps, and the parity test would make the
  protection enforceable.
- **The timestamp contract may be wrong.** Session filters, resets, and VWAP all
  depend on the ClickHouse `ts` column being exchange-local wall clock. The
  loader documents that this failure cannot be detected from the data. Verify
  it once and record the result.
- **The project can drift by accretion.** A new indicator, plot, or refactor may
  make sense on its own without moving the project toward its goal. Use the
  one-line goal in `AGENTS.md` as the test: if a task does not bring a feature
  closer to use in research or trading, it is maintenance. That is fine in
  moderation, but dangerous as the main diet.

*Written 2026-08-31. No code was modified; this file is the only change.*
