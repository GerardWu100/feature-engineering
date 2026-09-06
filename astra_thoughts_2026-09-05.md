# Astra thoughts — feature-engineering — 2026-09-05

The project already provides a compact, reusable stock-feature builder and a separate statistical screening library. The largest remaining step is to demonstrate that a named research or trading consumer can use its output under an agreed data and timing contract. Adding more indicators would contribute less than completing that connection.

The governing intent in `AGENTS.md`, reread after the implementation, is: “Feature engineering for quant trading; features built here must fit into my quant research and trading pipeline.” This specifies integration as the outcome. It does not specify an asset universe, trading horizon, first consumer, latency requirement, or profitability threshold. Recommendations below treat those as decisions to make, rather than inventing requirements.

## Scope and evidence

This is a strategic review of the current working tree, including its 14 pre-existing modified files. I read all 31 Python files under `src/` and `tests/`, the root wrapper, configuration and packaging files, README, all guides, architecture documents, three historical implementation plans, the earlier Fable review, and all notebook source cells and saved plain-text outputs. I also read the ignored toy configuration and eight-row CSV fixture. The notebook has 17 cells, including eight code cells. Its saved outputs are historical evidence, not a fresh execution.

Generated material was bounded: the dependency lockfile was sampled for its format and Python requirement; wheel/source distributions and installed package metadata were inventoried, not unpacked; notebook HTML duplicates were not separately inspected. The local `outputs/` directory contains no files. Environment secrets, virtual-environment dependencies, caches, Git internals, and other projects were not read. No database, external service, tests, notebook, or pipeline was executed. Accordingly, implementation presence is established; present runtime success, market-data coverage, performance, and use by other repositories are not established.

The quantitative research and methods skills informed the recommendations. No correctness audit, debugging, refactoring, or code change was performed. Historical plans are evidence of intent at their dates, not instructions to resume their work.

## 1. What the code actually does today

**Builds a feature dataset in one batch.** `run.py` delegates to `cli.main`; `cli.run_pipeline` validates configuration, loads bars, cleans them, computes selected columns, and stores results. Evaluation is not part of this command. The source is either a local comma-separated values (CSV) file or a ClickHouse query against `firstrate.{table}`. Both paths return the same seven columns: symbol, timestamp, open, high, low, close, and volume. OHLCV abbreviates those five price/volume fields.

`engineering/load.py` normalizes identifiers and timestamps, converts numeric columns, sorts each symbol's history, rejects duplicate symbol/timestamp keys, and applies source/date/session filters. Aware timestamps become naive exchange-local time; naive inputs are assumed already local. The source must supply adjusted prices. The package does not perform corporate-action adjustments, construct a historical eligible universe, or establish when a bar became available to a trader.

`engineering/clean.py` removes rows failing configurable numeric, price-range, or volume rules and returns counts explaining removals. This is row filtering, not gap reconstruction or a market-calendar service.

**Computes 19 registered functions: 17 predictors and two future targets.** `engineering/features/registry.py` imports five category modules. Their actual functions comprise two backward returns, seven trend indicators, three volatility measures, five volume measures, and two targets. Examples include relative strength index, moving average convergence divergence, average true range, and volume-weighted average price. All draw on the supplied symbol's own bars; no market, sector, fundamental, or external series enters the formulas.

`engineering/compute.py` resolves configured names and parameters, groups by symbol and optionally local calendar day, applies each function, checks result shape/index/numeric validity, and returns only identifiers plus calculated columns. Raw prices are omitted. The same functions can be imported directly; the computation itself does not require disk storage.

The shipped `config.toml` requests AAPL and MSFT during 2023–2024, regular-session data, both Parquet and CSV output, and no session reset. Although both target definitions are enabled, `exclude_categories = ["target"]` removes them: the configured output has 17 predictor columns, not 19 feature/target columns. This is configuration behavior, not evidence that a market-data run has completed.

The target functions in `features/targets.py` label either the simple close-to-future-close return or the sample standard deviation of the next specified number of one-bar log returns. Horizons count observed rows. Backward rolling standard deviation counts price rows, so its `window=20` uses 19 adjacent returns; the forward volatility target's `bars=20` uses 20 future returns. These are concrete definitions a consumer must understand when choosing horizons.

**Stores data with useful, incomplete provenance.** `engineering/store.py` writes timestamped datasets and a JSON summary containing configuration, symbol counts, feature missingness and ranges, and paths. A separate `feature_catalog.csv` describes formulas and lookbacks. `load_features` reads a requested run stem or selects a matching filename by sorting. These are practical research artifacts, but they do not pin source-data content, code revision, or a catalog immutable to that run.

**Screens relationships; it does not train deployable forecasts.** `evaluation/summary.py` returns one row per predictor, ranked by absolute regression t-statistic, a measure of estimated coefficient size relative to its uncertainty. `ic.py` calculates information coefficients (ICs): correlations through a symbol's history, across symbols at a timestamp, or inside trailing windows. Cross-sectional IC defaults to requiring at least five symbols, so the two-symbol default is naturally a time-series example.

`regression.py` standardizes each predictor over its supplied symbol history and fits a pooled single-predictor regression. Driscoll–Kraay standard errors account for dependence across symbols and through time; the Newey–West lag rule can incorporate a declared target horizon. Full-sample standardization makes this a screening fit, as its own documentation states. `quantiles.py` buckets values, generally within each symbol, and summarizes later outcomes. `plots.py` supplies distribution, low/neutral/high-state, and rolling-correlation figures.

Tests cover toy formulas, schema and time contracts, feature-result guards, storage round trips, synthetic statistical relationships, and plot construction. The notebook's saved output reports 19 registered functions, 56 retained toy rows, and an identical storage round trip. Its source ends by suggesting evaluation; it does not actually run a feature evaluation experiment.

## 2. The gap against the ultimate goal

| Goal-related capability | Current position | What remains |
|---|---|---|
| Reusable feature formulas | Built in source | Demonstrate use under a consumer's real input contract. |
| Configured batch production and reload | Built in source | Bind an exact run to input provenance and its consumer. |
| Candidate screening | Built in source | Record a real research decision and its limits. |
| Reproducible research evidence | Partial | Pin data/code versions, screening history, and untouched confirmation data. |
| Research-pipeline integration | A callable library exists | No named consumer or integration acceptance example is established inside this repository. |
| Trading-pipeline integration | Functions are reusable | Availability, execution timing, warm-up/state, and restart behavior need a concrete consumer agreement. |
| Confirmatory evaluation | Not implemented here | Chronological selection/confirmation and training-only learned transformations. |
| Market-wide stock selection | Components only | Historically eligible securities, suitable breadth, and cross-sectional evaluation choices. |

“Not implemented here” does not mean another project lacks the capability. The correct boundary may be to use this package from that project's existing evaluator, model, or execution system. The goal does not require this repository to own every stage.

The previous Fable review also identifies integration as the central gap. Its statements about other repositories and particular database tables cannot be verified from this review's evidence and are not adopted as facts. Current source contains 19 registered functions; that is distinct from the 17 predictors selected by default.

## 3. Concrete next steps, ordered by what they unblock

Effort estimates assume one developer, access to a suitable existing consumer, and available data. They describe implementation and validation work after this review; none was undertaken here.

1. **Choose one consumer and write its acceptance contract — roughly half a day to two days.** Specify whether the feature supports return prediction, volatility/risk estimation, or trade filtering; name the instrument set, bar interval, target, decision time, earliest execution time, expected output schema, and owner of downstream transforms. Decide whether consumption is an in-memory call or a pinned Parquet run. This comes first because “fits into my pipeline” otherwise has no observable completion condition. Success is one precise example the consumer agrees to accept.

2. **Pin a small input sample and its meaning — roughly one to three days, longer if source history needs investigation.** Capture the price/volume adjustment policy, timestamp convention, bar completeness, gaps, source version, and historical eligibility applicable to that consumer. Link input, configuration, code, catalog, and cleaning outcomes to one run identifier. This precedes empirical selection because an experiment is only reusable if the same inputs and meanings can be recovered. Keep ordinary local artifacts unless the consumer demonstrates a shared-database requirement.

3. **Complete one consumer-to-feature-to-decision example — roughly two to five days.** Use a small explicit predictor list and one target, load the pinned output or call the package, and make the consumer produce its normal research decision. Preserve target/predictor roles across the handoff. For a trading consumer, trace a completed bar through feature availability to a later executable price. The deliverable is an accepted connection with documented exclusions, not a claim of profitable alpha, meaning returns unexplained by the chosen benchmark. This exposes integration requirements before expanding the feature menu.

4. **Complete one chronological confirmation experiment — roughly three to seven days.** Declare the hypothesis, baseline, selection interval, later confirmation interval, and stopping rule before screening. Estimate scaling, thresholds, or model coefficients only on the training portion. Exclude training labels whose future intervals overlap confirmation; this exclusion is called purging. Record all attempted variants, including rejected ones. For volatility, compare against a forecast using recent observed volatility; for returns, use an explicitly chosen simple baseline. This converts a candidate ranking into evidence the consumer can assess. It may belong in the consumer repository rather than a new framework here.

5. **Define the promotion and retirement record — roughly one to three days.** For each adopted feature, preserve its economic purpose, exact definition, applicable universe/horizon, incremental benefit over the baseline, known limits, and conditions for removal. Distinguish research approval from trading approval. This follows a real experiment so the record describes actual decisions rather than speculative bureaucracy. Success includes the valid conclusion that no candidate deserves promotion.

6. **Add live parity or scale work only when the chosen consumer needs it — roughly three to ten days for parity; scale estimates require measurement.** Parity means research and live paths produce equivalent values given the same information. Specify initialization, session resets, corrected bars, and restarts; then compare replay with full-history computation. Recursive averages and cumulative volume-weighted prices need more than a simplistic “take the catalog lookback” rule. Measure workload and latency before choosing stateful computation, partitioning, or a new engine.

## 4. Decisions and alternatives worth making explicit

**The feature's role may matter more than its predictive score.** Volatility or trading-activity features can improve risk sizing or exclude unsuitable trading conditions without predicting return direction. Evaluate them against that purpose. A project serving trading does not need every feature to be a standalone trading signal.

**The default research question is mostly within-symbol.** Per-symbol standardization, historical quantiles, and mean time-series IC ask whether unusually high values for this stock predict its subsequent outcome. Ranking many stocks at the same instant is a different question. Choose before widening the universe or interpreting dollar-denominated indicators across securities.

**One session-reset flag embodies several economic choices.** It controls moving-window history, recursive indicators, cumulative volume-weighted price, and target boundaries together. A consumer may want overnight history for risk but a daily reset for an execution benchmark. Decide required meanings first; feature-specific policies become justified only when that concrete combination is needed.

**Cleaned rows are not a complete clock.** Dropping invalid bars makes a 20-row horizon span the next 20 surviving observations. Halts and gaps can lengthen elapsed time. Record whether the downstream decision is defined in observed bars, scheduled minutes, or trading sessions. Do not silently interchange them.

**Price continuity and execution units are separate requirements.** The loader assumes adjusted OHLC prices; `dollar_volume` multiplies the supplied close and supplied volume. Whether that represents actual traded currency depends on their adjustment relationship. Ask the source contract, rather than assuming raw execution-price meaning from a column name.

**A small shared library can be the finished product.** The public import surface already permits this. A consumer using a few proven functions may satisfy the goal better than a central feature database, automated feature search, or generalized research platform. Those expansions are interesting but currently lack a demonstrated goal-serving dependency.

## 5. Improvement directions beyond the current implementation

Prioritize **decision-aligned targets** first: return labels beginning at a feasible execution point, or risk labels matching the consumer's holding interval. Reuse existing consumer logic where available so definitions do not diverge.

Next consider **incremental information from context**: one market-relative return or session-state variable may add a different information source, while another oscillator often repackages the same price history. Require a specific economic hypothesis and point-in-time input, meaning data available at the historical decision time. These additions follow an established baseline and do not justify indiscriminate feature expansion.

Then consider **feature-set economy**: identify several indicators representing the same phenomenon and retain a small representative set unless a held-out comparison shows a benefit from more. This reduces interpretation and maintenance costs while making research search easier to account for.

Only a demonstrated second consumer should motivate **shared discovery and storage**. A run manifest and pinned local dataset can serve the first consumer. Central storage becomes valuable when multiple consumers require common identifiers, access, and refresh coordination; it is not intrinsically an improvement in predictive usefulness.

## 6. What could steer the project away from its goal

- **Internal completion substitutes for adoption.** More formulas, tests, and polished plots can accumulate while no consumer accepts a run. Count completed consumer decisions and reproducible experiments as progress.
- **Screening becomes deployment by implication.** Dependence-aware standard errors do not turn full-history scaling or historical quantile thresholds into available live inputs. Keep screening output explicitly exploratory until chronological confirmation and a consumer-specific implementation exist.
- **Target identity is lost at a file boundary.** `evaluate_features` relies on an explicit feature list or supplied `target_columns` to avoid treating another future label as a predictor. The exported table alone does not enforce that semantic distinction. Carry the role metadata into the consumer agreement.
- **Historical data assumptions remain unverifiable.** A static symbol list, adjustment policy, or naive timestamp convention can support a narrow experiment while failing a broader claim. Do not claim survivorship-free research or correct intraday availability without source evidence.
- **Apparent breadth is mostly repetition.** Many correlated indicators, overlapping targets, and co-moving stocks do not supply as many independent opportunities as their row counts suggest. Record attempted choices and assess stability on genuinely later periods.
- **Recomputed features have a different starting history.** Exponentially weighted averages and cumulative volume-weighted price retain initialization information. A small trailing buffer or restart can change the feature a model receives even when its column name stays identical.
- **A registry grows into a platform without demand.** Multi-asset schemas, distributed computation, and automated search add decisions and operating burden. They should follow demonstrated consumer needs, not an abstract ambition to support everything.

The practical finish line for the next phase is modest: one named consumer, one pinned feature definition and dataset, one explicit decision timeline, and one recorded chronological evaluation. A negative research result still advances the ultimate goal if the integration is reusable and the rejection is well founded.

**TL;DR**

- **Request** — Assess the current project against its trading-pipeline goal and recommend strategic next steps without changing code.
- **Answer** — The feature builder and screening library exist; prioritize a concrete consumer contract, reproducible handoff, and chronological confirmation before additional indicators or infrastructure.
