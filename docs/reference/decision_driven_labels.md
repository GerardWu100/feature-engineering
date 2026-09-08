# Decision-driven labels and timing features

This is a literature survey written on 2026-09-08 for the `feature-engineering` project. It covers the targets and features that quantitative equity researchers and machine-learning-for-trading practitioners use *instead of* a fixed N-bar forward return, with an emphasis on labels that let a model decide when to enter and when to exit rather than assuming a fixed holding period. It closes with a concrete recommendation for what to add to this pipeline next, written against the code that exists today.

Acronyms are defined on first use. Formulas appear as LaTeX where they are mathematical statements and as plain text where they are closer to pseudocode.

## 1. Comparison of label families

| Label type | Question it answers | What it needs | Main pitfall | Who uses it |
|---|---|---|---|---|
| Fixed N-bar return | "What is the return over exactly N bars?" | Close series | Assumes a holding period nobody trades; label variance tracks the local volatility regime, so base rates drift | Almost all academic cross-sectional asset pricing (Gu, Kelly and Xiu 2020) |
| Triple-barrier method (TBM) | "Which happens first: take-profit, stop-loss, or timeout?" | Path, a causal volatility estimate, three parameters | Overlapping labels break the independent-and-identically-distributed (IID) assumption; barrier widths are a hidden hyperparameter search; never shown to beat fixed-horizon out of sample | Lopez de Prado (2018); `mlfinlab` / `mlfinpy`; crypto machine-learning literature |
| Meta-labeling | "Given a primary rule already picked the side, should I take this trade and how big?" | A primary strategy plus TBM outcomes | Improves precision only; cannot create signal the primary model did not have | Joubert and Hudson and Thames (2022-2023) |
| Trend scanning | "Is this bar inside a statistically significant trend, and how strong?" | Forward-window search over horizons | The reported t-statistic is a selected maximum, so it is not t-distributed; heterogeneous data-chosen horizons make overlap worse than TBM | Lopez de Prado (2020); `mlfinlab`; almost no independent validation |
| First-passage / survival | "How long until a level is hit, and which level?" | Path plus right-censoring bookkeeping | Naive Kaplan-Meier is wrong under competing risks; per-bar expansion multiplies rows without adding information | Limit-order-book fill-probability research; `lifelines`, `scikit-survival`, XGBoost `survival:aft` |
| Event-clock horizons | "What happens over a fixed quantity of information, not calendar time?" | Trade-level or at least volume data | Threshold calibration leaks if fit on the full sample; each asset gets its own clock, breaking cross-sectional alignment | Lopez de Prado (2018) ch. 2; Easley, Lopez de Prado and O'Hara (2012) |
| Policy-evaluation labels | "What would my actual exit rule have earned starting here?" | A simulator for the exit rule | Label is a joint function of signal and rule parameters; changing one parameter invalidates the dataset | Hwang et al. (2023); Cheevirot et al. (2026) |
| Oracle / perfect-foresight exits | "What was the ex-post best exit?" | Full future path | Has a theorem against it: behavioural cloning regret is $O(T^2 \epsilon)$, quadratic in horizon | Used only as a soft teacher (Fang et al. 2021), never as a direct target |
| Optimal stopping / reinforcement learning | "Learn the continuation-versus-exit boundary directly" | A simulator or generative market model | One price path; reward shaping is the strategy; best-documented result is negative | Becker, Cheridito and Jentzen (2019); Moody and Saffell (2001) |

## 2. Triple-barrier method and meta-labeling

For an event starting at $t_0$, with $\sigma_{t_0}$ a volatility estimate known *at* $t_0$, and multipliers $pt, sl > 0$:

$$\text{upper} = p_{t_0}\,(1 + pt \cdot \sigma_{t_0}), \qquad \text{lower} = p_{t_0}\,(1 - sl \cdot \sigma_{t_0})$$

with a vertical barrier at $t_0 + h$, where $h$ is the maximum holding period in bars. Then

```
t1  = first time in (t0, t0+h] that any barrier is touched
ret = p_t1 / p_t0 - 1                       "return at touch"
bin = +1 upper first, -1 lower first, 0 vertical first
```

The volatility estimator in Lopez de Prado's snippet 3.1 (`getDailyVol`) is causal by construction: daily returns, then an exponentially weighted moving average (EWMA) standard deviation with span 100. The equivalent RiskMetrics form

$$\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2, \qquad \lambda = 0.94 \text{ daily}$$

has no estimated parameters at all, so it is leak-free by construction.

How `t1` and `ret` are actually used, which most write-ups skip:

- `t1` defines each label's **lifespan**, which drives concurrency, uniqueness weights, purging in cross-validation, and the time-to-event framing.
- `ret` is the **continuous companion** to the sign label: it keeps the magnitude of the move and feeds return-attribution sample weights.
- `t1 - t0` (bars to touch) is a first-passage duration and is **right-censored** at the vertical barrier: a value of $h$ means "at least $h$", not "exactly $h$".

**Meta-labeling.** A primary exogenous rule sets the side $s \in \{-1,+1\}$ and the entry. The machine-learning model predicts only $p = P(\text{trade is profitable} \mid \text{features}, s)$, a binary problem. Bet size (Lopez de Prado 2018, ch. 10):

$$z = \frac{p - 0.5}{\sqrt{p\,(1-p)}}, \qquad m = 2\,N(z) - 1$$

where $N$ is the standard normal cumulative distribution function and $m \in (-1,1)$. The position is $s \cdot m$: the primary model owns direction, the secondary owns conviction.

Sources: Lopez de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, ch. 3 — https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 · `mlfinpy` labelling docs — https://mlfinpy.readthedocs.io/en/latest/Labelling.html · Joubert, J. (2022), "Meta-Labeling: Theory and Framework", *Journal of Financial Data Science* 4(3), 31-44 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018 · code — https://github.com/hudson-and-thames/meta-labeling

## 3. Trend-scanning labels

For observation $t$, fit ordinary least squares (OLS) of price on time over each forward horizon $[t, t+L']$ for $L'$ in $[L_{\min}, L_{\max}]$:

$$p_s = b_0 + b_1 s + e_s, \qquad L^* = \arg\max_{L'} \left| \frac{\hat b_1(L')}{\mathrm{SE}(\hat b_1(L'))} \right|$$

and label by $\mathrm{sign}(t\text{-stat}(L^*))$, optionally three-class via a threshold. Outputs are `t1`, the t-value, the trend return and the bin. The t-value doubles as a sample weight or a regression target. The appeal is that the data, not the researcher, picks the horizon.

Criticisms, ordered by how well supported they are. (1) The reported t-statistic is the maximum over many overlapping regressions, so it is an uncorrected multiple-testing statistic and the threshold is not a significance level; this is sound reasoning but I found no peer-reviewed source stating it. (2) Horizons are heterogeneous and data-chosen, so the overlap structure is worse than TBM's and nobody has characterised concurrency under it. (3) The only benchmark that exists puts it at Sharpe ratio 0.00. (4) Searches of arXiv, OpenAlex and Semantic Scholar return zero independent evaluations.

Sources: Lopez de Prado, M. (2020), *Machine Learning for Asset Managers*, Cambridge University Press, section 5.2 · `mlfinlab` docs — https://random-docs.readthedocs.io/en/latest/implementations/labeling_trend_scanning.html · only peer-reviewed use: Boonpan, S. and Sarakorn, W. (2025), *Journal of Open Innovation* 10(3):100438 — https://doi.org/10.1016/j.joitmc.2024.100438 (no comparison against TBM) · better-validated alternative: Kovacevic, T., Mercep, A., Begusic, S. and Kostanjcar, Z. (2023), "Optimal Trend Labeling in Financial Time Series", *IEEE Access* 11 — https://doi.org/10.1109/ACCESS.2023.3303283

## 4. First-passage, time-to-event and survival targets

TBM already computes the first-touch time and then discards it. Survival analysis keeps it, giving a target pair $(T, D)$: $T$ = bars to first touch, $D$ = which barrier, with censoring at the vertical barrier.

For log price $X(t) = x_0 + \mu t + \sigma W(t)$, an upper level at distance $b = a - x_0 > 0$, and $\tau = \inf\{t : X(t) \ge a\}$, the first-passage density is

$$f_\tau(t) = \frac{b}{\sigma\sqrt{2\pi t^3}} \exp\left(-\frac{(b - \mu t)^2}{2\sigma^2 t}\right)$$

the Inverse Gaussian density. It is **defective** when $\mu \le 0$: $P(\tau < \infty) = \exp(2\mu b / \sigma^2) < 1$. A fraction of paths never touch, so the vertical barrier is not an arbitrary convention but the physics.

The cheapest usable form is a **discrete-time hazard model**: expand each trade into one row per bar it survives, then run ordinary logistic regression with a free intercept per bar index. The likelihood factorises into Bernoulli terms, so any binary classifier becomes a survival model. Alternatives: Cox proportional hazards, accelerated failure time (XGBoost ships `survival:aft`), and competing risks via cause-specific hazards plus the Aalen-Johansen cumulative incidence function.

Pitfall worth stating plainly: treating "stop-loss hit" as *censoring* for the take-profit event is wrong, because the trade could not still have hit take-profit later, and it systematically overstates the probability of take-profit. Also, concordance is not a profit-and-loss metric.

Sources: Arroyo, A., Cartea, A., Moreno-Pino, F. and Zohren, S. (2023/24), "Deep Attentive Survival Analysis in Limit Order Books", *Quantitative Finance* 24(1) — https://arxiv.org/abs/2306.05479 · Hwang, Y., Park, J., Lee, Y. and Lim, D.-Y. (2023), "Stop-loss adjusted labels for machine learning-based trading of risky assets", *Finance Research Letters* 58:104285 — https://doi.org/10.1016/j.frl.2023.104285 · Engle, R. and Russell, J. (1998), autoregressive conditional duration, *Econometrica* 66(5) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6809 · `lifelines` — https://github.com/CamDavidsonPilon/lifelines

**Gap worth flagging:** no published paper applies an explicit competing-risks model to triple-barrier labels. The pieces exist separately.

## 5. Event-clock and volatility-scaled horizons

Calendar time is a poor clock because information does not arrive uniformly. Clark (1973) modelled price as Brownian motion subordinated to an information clock, $P(t) = B(\theta(t))$; Ane and Geman (2000) found the cumulative number of trades recovers near-normality of returns. Sampling one observation per fixed increment of $\theta$ gives roughly constant information per observation.

Bar types (Lopez de Prado 2018, ch. 2): tick bars (fixed transaction count), volume bars (fixed shares), dollar bars (fixed notional, robust to price appreciation and splits, the default recommendation for equities), and imbalance and run bars (close when signed order flow exceeds its EWMA-forecast expectation). The CUSUM (cumulative sum) filter is an event sampler rather than a bar type: accumulate `S+ = max(0, S+ + r_t)` and `S- = min(0, S- + r_t)`, fire and reset when `|S| > h`.

Under a volume clock, ten bars is a fixed quantity of trading activity and the calendar duration is random. That is statistically desirable but silently breaks everything calendar-denominated: overnight gaps, borrow, per-day cost budgets, and Sharpe annualisation.

Volatility-normalized horizons are the simpler version of the same idea. Choose $h_t$ so the expected absolute move is a constant $c$: under a random walk $\sigma_t \sqrt{h_t} = c$, so

$$h_t = \left(\frac{c}{\sigma_t}\right)^2$$

Equivalently, and this is what TBM actually does, hold $h$ fixed and scale the barriers by $\sigma_t$. Either way label base rates become stationary across regimes. This is the single highest-value fix to a fixed-horizon target.

Pitfalls: imbalance and run bar thresholds rely on EWMA forecasts and are notoriously unstable; any threshold calibrated on the full sample leaks; a fixed dollar threshold produces far more bars in 2025 than in 2010, so index it to trailing traded value. Treat "dollar bars give IID Gaussian returns" as a direction, not a fact: Fayyaz et al. (2026) show apparent tick-bar advantages shrink substantially once bar counts are frequency-matched.

Sources: Clark, P. (1973), *Econometrica* 41(1) — https://conservancy.umn.edu/bitstreams/c0c15ef7-a81c-4354-89fd-89c8c5237986/download · Ane, T. and Geman, H. (2000), *Journal of Finance* 55(5) — https://www.lancaster.ac.uk/staff/izzeldin/AG.pdf · Easley, D., Lopez de Prado, M. and O'Hara, M. (2012), "The Volume Clock", *Journal of Portfolio Management* 39(1) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858 · Gradzki, P., Wojcik, P. and Lessmann, S. (2025), *Financial Innovation* 11 — https://doi.org/10.1186/s40854-025-00866-w · Fayyaz et al. (2026) — https://arxiv.org/abs/2608.26158

## 6. Signal- and regime-conditioned exits, and how to label them

The rules: exit on signal reversal; trailing and chandelier stops (`stop_long = max(High, N) - k * ATR_N`, defaults N = 22 and k = 3, which are convention with no peer-reviewed validation, where ATR is average true range); exit on volatility-regime change via a Markov-switching model or change-point detection.

Do stops even help? Kaminski and Lo define the *stopping premium* and prove it is always negative under a random walk: a stop with unforecastable returns is a pure tax. It is positive only under positive serial correlation or regime switching. Han, Zhou and Zhu confirm the momentum case: a 10 percent stop on value-weighted momentum (1926-2013) improves the worst month from -64.97 percent to -23.28 percent and more than doubles the Sharpe ratio. A stop-loss is a bet on autocorrelation, not free risk management.

Four ways to turn an exit rule into a supervised label:

1. **Policy-evaluation label.** Simulate the full exit rule forward from bar $i$ and set $y_i = \mathrm{PnL}(\text{rule}, i)$, or a binary version net of costs. TBM is the canonical special case. Pitfall: the label is a joint function of signal and rule parameters, so changing a parameter invalidates the dataset.
2. **Meta-labeling** (section 2). Pitfall: improves precision only.
3. **Per-bar "should I still be in this trade".** Rare, and rejected by practitioners because features do not support a prediction at every bar. Extreme imbalance and near-duplicate consecutive rows.
4. **Oracle / perfect-foresight exits.** The one option with a theorem against it: Ross, Gordon and Bagnell show naive behavioural cloning has regret $O(T^2 \epsilon)$, quadratic in horizon. Oracle-distillation architectures exist precisely because the student cannot imitate the oracle. A model that fits an oracle exit label well is fitting noise.

A quiet look-ahead trap: `statsmodels` reports *smoothed* regime probabilities by default, which use the full sample. Only *filtered* probabilities are tradable, and only with expanding-window maximum likelihood; the likelihood is multimodal so state labels flip between refits, and states must be re-sorted by $\sigma_k$ every time.

Sources: Kaminski, K. and Lo, A. (2014), "When Do Stop-Loss Rules Stop Losses?", *Journal of Financial Markets* 18 — https://www.smallake.kr/wp-content/uploads/2017/02/When_Do_Stop-Loss_Rules_Stop_Losses.pdf · Han, Y., Zhou, G. and Zhu, Y. (2016), "Taming Momentum Crashes", *Journal of Banking and Finance* — https://www.cicfconf.org/sites/default/files/paper_811.pdf · Ross, S., Gordon, G. and Bagnell, J. (2011), AISTATS — http://proceedings.mlr.press/v15/ross11a/ross11a.pdf · Fang, Y. et al. (2021), "Oracle Policy Distillation", AAAI — https://arxiv.org/abs/2103.10860 · Garleanu, N. and Pedersen, L. (2013), *Journal of Finance* — https://w4.stern.nyu.edu/facdir/lpederse/papers/DynamicTrading.pdf

## 7. Optimal stopping and reinforcement learning

Optimal stopping is the exact formulation of "when to exit". For a Markov state $X_n$ and payoff $g$,

$$V_n(x) = \max\Big( g(n,x),\; \mathbb{E}\big[V_{n+1}(X_{n+1}) \mid X_n = x\big] \Big)$$

and you exit at the first time immediate reward beats continuation value. The stopping boundary is the object you actually compute.

- **Longstaff and Schwartz (2001)**, least-squares Monte Carlo: backward-induct by regressing realized discounted continuation cash flows on basis functions of the current state.
- **Becker, Cheridito and Jentzen (2019)**, deep optimal stopping, *Journal of Machine Learning Research* 20(74): decompose the stopping time into per-date binary decisions, each a small network trained backward in time, with a dual martingale upper bound giving a confidence interval — https://jmlr.org/papers/v20/18-232.html
- **Leung and Li (2015)**, Ornstein-Uhlenbeck spread: optimal *double* stopping. The answer is a take-profit level plus an entry interval lying strictly above the stop-loss, and a higher stop-loss always implies a lower optimal take-profit. Use it as the analytic benchmark for any learned boundary — https://arxiv.org/abs/1411.5062
- **Carr and Lopez de Prado (2014)**, "Determining Optimal Trading Rules without Backtesting": derive optimal profit-taking and stop-loss pairs analytically under an Ornstein-Uhlenbeck process. This is the principled answer to "how do I pick the barrier multipliers without grid-searching them" — https://arxiv.org/pdf/1408.1159
- Execution ancestry: Bertsimas and Lo (1998); Almgren and Chriss (2000). The adjacent continuous-action version is deep hedging, Buehler, Gonon, Teichmann and Wood (2019) — https://arxiv.org/abs/1802.03042

Moody and Saffell (2001) optimize the differential Sharpe ratio directly, and their lesson that risk-adjusted reward beats profit reward still holds. Zhang, Zohren and Roberts (2019) build volatility scaling into the reward on 50 liquid futures — https://arxiv.org/abs/1911.10107. But you have one price path; reward shaping is the risk model; and the best-documented result is negative. Garg et al. (AAAI-22 workshop) attacked hold-or-sell liquidation as optimal stopping and found supervised forecasting, imitation learning and reinforcement learning all degraded performance versus a simple heuristic — https://arxiv.org/abs/2202.12578. Measure overfitting explicitly: Gort et al. (2022) — https://arxiv.org/abs/2209.05559.

## 8. Criticisms of the triple-barrier method, and 2023-2026 work

**Label overlap and the standard fixes.** Labels span $[t_{i,0}, t_{i,1}]$; two labels sharing any return are concurrent, so samples are not IID.

```
concurrency:        c_t     = sum_i 1_{t,i}
uniqueness:         u_{t,i} = 1_{t,i} / c_t
average uniqueness: ubar_i  = (sum_t u_{t,i}) / (sum_t 1_{t,i})       in (0, 1]
return attribution: wtilde_i = | sum_{t=t_i0}^{t_i1} r_{t-1,t} / c_t |
sequential bootstrap: draw j at step k with prob  ubar_j^(k) / sum_i ubar_i^(k)
```

The same logic drives purging plus embargo in cross-validation: drop training rows whose lifespan overlaps the test set.

**But the sequential bootstrap does not do much.** Peng (2025), a 100-seed replication, finds it does not move mean out-of-bag error meaningfully, with only 6 of 57 test cells significant at the 5 percent level and 4 of those flipping sign relative to a 3-seed run. The only robust effect is a modest variance reduction. https://arxiv.org/abs/2511.18065

**Barrier width is a risk-preference dial, not a tunable.** Fu, Kang, Hong and Kim (2024) produce two simultaneously valid label sets from the same data by genetic search: high-risk/high-profit and low-risk/low-drawdown — https://doi.org/10.3390/math12050780. The implication is sharper than "grid searching overfits": tuning barriers by validation accuracy accidentally and invisibly selects a risk preference. Kang (2025) grid-searched horizon by threshold over the whole sample, selecting on label balance, and landed on the grid boundary — https://arxiv.org/abs/2504.02249. Carr and Lopez de Prado (section 7) is the principled alternative.

**Look-ahead in the volatility estimate.** The `getDailyVol` estimator is causal and fine. Leakage enters through three habits: a single full-sample sigma; a centred rolling window; and multipliers tuned on the full sample. I could find zero published measurements of this. It is real, but folklore.

**Class imbalance and the vertical-barrier label.** Labelling the timeout as 0 gives a genuine three-class problem, but the zero class dominates when barriers are wide and the model learns "do nothing". Labelling it by the sign of the return restores balance but destroys the risk-management content, since a small drift and a full take-profit get the same label. There is no accepted way to set this. Related: intrabar path ambiguity, since open-high-low-close data cannot say whether the high or the low came first.

**Does TBM beat fixed-horizon labels out of sample?** Four studies test it and all lean neutral-to-negative.

| Study | Setting | Result |
|---|---|---|
| Kovacevic, Goluza, Mercep and Kostanjcar (2022), MIPRO | controlled fixed-horizon versus TBM | Neither dominates; correlation between classification metrics and financial performance is significant but low under both |
| Kili et al. (2025), *Applied Sciences* 15(24):13204 | 16 assets, 2000-2025 | TBM Sharpe -0.03, fixed horizon -0.29, trend scanning 0.00, all near zero |
| Stanciu (2026), SSRN 6519542 | 240 configurations | TBM loses to a naive binary target in every configuration |
| Lima (2026), SSRN 7177958 | US equity panel | Label lift positive, event-return lift negative, portfolio fails economically |

The honest framing: TBM is better motivated than fixed-horizon labelling and far more popular, but has never been shown to beat it out of sample in a controlled test. Two of the four are unrefereed preprints, so the evidence is thin in both directions. The burden of proof is not where the folklore puts it.

**Meta-labeling criticisms.** The founding evidence tests two technical strategies; out-of-sample precision rose from 0.17 to 0.20, and the headline accuracy jump is largely the meta-model learning to predict "do not trade" in an imbalanced problem. The published series is by one group in one journal. Baldisserri's objection is unrebutted: both models see identical features, so a well-trained end-to-end model should dominate, and his grid search found lower average Sharpe under meta-labeling — https://www.quantconnect.com/forum/discussion/14706/

**The best recent idea: label with the payoff, not the horizon return.** Three groups converge independently. Hwang et al. (2023) adjust labels for whether a stop would have fired first. Cheevirot et al. (2026) derive labels by forward-simulating the strategy's own payoff — https://doi.org/10.3390/a19060442 (take the labelling idea, discard the performance claim). Wang and Gu (2025) state the principle most cleanly: point-to-point labels ignore *interim risk*, the drawdown along the path — https://doi.org/10.1109/CAIT68620.2025.11424833

Also worth knowing: Song, Liu and Chen (2026), "The Label Horizon Paradox", find the best training label horizon often differs from the horizon you want to predict — https://arxiv.org/abs/2602.03395. Ma, Ventre and Polukarov (2022), denoised labels via self-supervised learning — https://arxiv.org/abs/2112.10139. Arian, Norouzi Mobarekeh and Seco (2024) find combinatorial purged cross-validation has the lowest probability of backtest overfitting, but in a synthetic environment and by a Lopez de Prado collaborator — https://doi.org/10.1016/j.knosys.2024.112477

Confirmed empty across arXiv, OpenAlex and Semantic Scholar: no survey of financial labelling schemes; no independent evaluation of trend scanning; nothing on look-ahead bias in the barrier-width volatility estimator.

## 9. Timing features (predictors, not labels)

**Nearness to the 52-week high.** $\text{ratio}_{52w} = P_t / \max_{s \in [t-252,\,t]} P_s$, in $(0,1]$. Answers how far into a completed advance a name sits. George, T. and Hwang, C.-Y. (2004), *Journal of Finance* 59(5), 2145-2176 — https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf. Market-level version: Li, J. and Yu, J. (2012), *Journal of Financial Economics* 104(2) — nearness to the 52-week high positively predicts aggregate returns, nearness to the historical high negatively. Pitfall: use split and dividend adjustment factors as of $t$, not as of today. That is the most common look-ahead in this code.

**Drawdown from the running peak.** $DD_t = P_t / \max_{s \le t} P_s - 1$, plus bars since the running max was last set. Note the identity: $\text{ratio}_{52w} = 1 + DD$ on a 252-day rolling max, so do not ship both unnormalized. Goldberg, L. and Mahmoud, O. (2017) — https://arxiv.org/abs/1404.7493. Pitfall: drawdown is a path functional of the running max, so it is strongly autocorrelated, non-Gaussian and mechanically correlated with realized volatility.

**Capital gains overhang**, the literal "where in a trade" feature: $g = 1 - R_{t-1}/P_{t-2}$ where $R$ is a turnover-weighted reference price over 260 weeks. Grinblatt, M. and Han, B. (2005), *Journal of Financial Economics* 78(2), 311-339 — https://www-2.rotman.utoronto.ca/facbios/file/momentum_JFE.pdf. Pitfall: cross-sectional correlation with the past one-year return is about 0.55, so it is partly a momentum re-encoding, and it needs five years of history, which silently imposes an age filter.

**Momentum age.** Intermediate-horizon $r_{12,7}$ versus recent $r_{6,2}$, with month $t-1$ always skipped. Novy-Marx, R. (2012), *Journal of Financial Economics* 103(3), 429-453. Counter-evidence: Goyal, A. and Wahal, S. (2015), *Journal of Financial and Quantitative Analysis* 50(6) — the echo holds in the US but there is no comparable evidence in 37 other countries. The cut points are tuned hyperparameters. A better single-number staleness proxy is the momentum gap, Huang, S. (2022), *Review of Financial Studies* 35(7), which negatively predicts subsequent momentum profits in 21 markets.

**Days since event.** A post-earnings-announcement-drift decay clock, commonly a ramp to zero over about 60 trading days. That window is practitioner convention, not a sourced number. Bernard, V. and Thomas, J. (1989, 1990). Pitfall: Compustat announcement dates are backfilled, and the variable is mechanically bounded and resets, so a tree will carve calendar quarters and learn seasonality.

**Volatility regime.** Realized-volatility percentile (trailing or expanding rank only); VIX level and percentile; volatility-of-volatility as a coefficient of variation of implied volatility, Baltussen, van Bekkum and van der Grient (2018), *Journal of Financial and Quantitative Analysis* 53(4) — https://personal.eur.nl/vanbekkum/2018JFQA_unknown_unknowns.pdf; filtered Markov-switching regime probability, Hamilton, J. (1989), *Econometrica* 57(2) — https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html. Volatility-managed portfolios: Moreira, A. and Muir, T. (2017), *Journal of Finance* 72(4) — https://www.nber.org/system/files/working_papers/w22208/w22208.pdf. Read the replication before relying on it: Cederburg, O'Doherty, Wang and Yan (2020), *Journal of Financial Economics* 138(1), find volatility management wins on Sharpe in 53 of 103 cases versus 50, a coin flip, and the spanning alpha is not implementable in real time — https://www.lehigh.edu/~xuy219/research/COWY.pdf

**Trend and persistence.** Prefer the variance ratio (Lo, A. and MacKinlay, A. 1988, *Review of Financial Studies* 1(1), 41-66) over the Hurst exponent, which reports $H \approx 0.55$ to $0.6$ for pure noise on a 250-day window without the Anis-Lloyd correction and has no sampling distribution. For a rolling OLS trend fit, note the identity

$$t^2 = \frac{R^2 (n-2)}{1 - R^2}$$

so at fixed window length $n$ the t-statistic and $R^2$ are the same feature. Ship the signed t-statistic, never $R^2$ and $|t|$ together. The best academic anchor is Schmidhuber, C. (2021), *Physica A* 566 — https://arxiv.org/pdf/2006.07847 — whose load-bearing finding is that tomorrow's expected return is a **cubic** polynomial of trend strength: a positive linear term (persistence) plus a negative cubic term (reversal). Trends tend to revert before they become statistically significant. A linear model or shallow tree averages the two regions to zero and concludes the feature is useless, so add a cubic term or bucket the feature.

**Path shape.** Information discreteness, Da, Z., Gurun, U. and Warachka, M. (2014), *Review of Financial Studies* 27(7), 2171-2218 — https://academicweb.nd.edu/~zda/Frog.pdf:

```
ID = sgn(PRET) * ( %neg - %pos )
PRET       = cumulative return over months t-12 .. t-2
%pos, %neg = fraction of daily returns positive / negative over the same window
```

$ID \in [-1, +1]$, and **low ID means continuous information and stronger, more persistent momentum**. Over six months momentum increases monotonically from -2.07 percent in the discrete portfolio to 5.94 percent in the continuous one. Use the normalized variant $ID_Z$ below large-cap, since zero-return frequency is an illiquidity proxy. ID is meaningless unconditionally: it must be interacted with or double-sorted on PRET. Also useful and much cheaper: the efficiency ratio, $|\text{cumulative return}| / \sum |\text{daily returns}|$, in $[0,1]$. Path signatures (Chevyrev and Kormilitzin 2016 — https://arxiv.org/abs/1603.03788) capture ordering and lead-lag via the Levy area, but a six-dimensional path truncated at level four already yields 1,555 terms.

**Position in range.** The stochastic oscillator %K, Bollinger %B and Williams %R are the same feature up to an affine transformation; pick one. Note that the 920-covariate benchmark in Gu, S., Kelly, B. and Xiu, D. (2020), *Review of Financial Studies* 33(5) — https://dachxiu.chicagobooth.edu/download/ML.pdf — contains no oscillator or range-position features at all. Adding them goes beyond that benchmark rather than replicating it.

**Cross-cutting requirement.** Gu, Kelly and Xiu rank all stock characteristics cross-sectionally period by period and map the ranks into $[-1, 1]$. Full-sample `rank(pct=True)` or fitting a scaler on the whole panel is the single most common leak in this entire family. Market-wide series (VIX, regime probability, momentum gap) instead need an expanding-window z-score, and they induce cross-sectional dependence on each date that invalidates naive cross-validation.

## 10. Addendum

**Variance risk premium.** The highest-value volatility feature not covered above: $VRP_t = VIX_t^2 - RV_t$ (Bollerslev, Tauchen and Zhou 2009, *Review of Financial Studies*). It is the cleanest measure of how expensive fear is right now, more predictive of market returns at one to three month horizons than either VIX or realized volatility alone, and it de-duplicates the two, which correlate about 0.8 daily. If you would otherwise keep both a realized-volatility percentile and a VIX percentile, keep the spread rather than the two levels.

**Fractional differentiation.** Raw price-level features are near-integrated of order one. Ordinary differencing restores stationarity but destroys memory, since every price depends on previous levels whereas returns have a memory cut-off. Fractional differentiation (Lopez de Prado 2018, ch. 5, after Hosking 1981) differences by a real $d \in (0,1)$, removing only the minimum integration needed for stationarity while preserving maximum memory. Relevant here because `price_vs_moving_average` and the distance-to-high features recommended below are all price-level quantities.

**Label-shuffle sanity test.** Shuffle the labels and refit. Any feature that still produces in-sample alpha is measuring the sample, not the market. Given how many of these features are path functionals with heavy autocorrelation, this catches more than it should.

## 11. Redundancy pairs to check before modelling

| Pair | Relationship |
|---|---|
| `distance_to_52w_high` versus $1 + DD$ on a 252-day window | Exact identity |
| Stochastic %K versus Williams %R | Perfectly collinear |
| Rolling $R^2$ versus $\lvert t \rvert$ at fixed window length | Same feature |
| Path efficiency ratio versus information discreteness | Close cousins |
| Amihud illiquidity versus Kyle's lambda | Near-substitutes; lambda adds little without tick data |
| Amihud illiquidity versus log market capitalisation | Correlation about -0.61; largely a size proxy |
| `rolling_standard_deviation` versus `average_true_range` | Both volatility scales; check before using both to set barriers |

## 12. What to add to this pipeline first

This project already has `next_n_bar_return` and `next_n_bar_realized_volatility`, and now also `triple_barrier_label`, `triple_barrier_bars_to_exit` and `triple_barrier_exit_return` in `src/feature_engineering/engineering/features/targets.py`. That implementation tests barriers against **closes only** (avoiding the intrabar ordering ambiguity at the cost of missing intrabar touches), derives $\sigma_t$ from the trailing `rolling_standard_deviation` rather than an EWMA with span 100, and sets the last `max_bars` rows to `NaN` so no row is labelled on a partial window. All three are defensible choices, and all three should be documented as choices rather than defaults. The ordering below is written against that state.

**Priority 1: a `label_uniqueness_weight` target column.** This is the highest value addition and it is not optional. Without it the triple-barrier targets that already exist will produce inflated in-sample fit and unreliable information coefficients, because neighbouring rows share up to `max_bars` of outcome path. It costs one pass over `bars_to_exit`:

```
For each row t with bars_to_exit[t] = k_t, the label's life is [t, t + k_t].
concurrency:  c_s    = number of rows t with t <= s <= t + k_t   (difference-array pass)
weight:       ubar_t = mean over s in [t, t + k_t] of (1 / c_s)  in (0, 1]
```

Pass `ubar` as `sample_weight` to any model, and use it to discount the effective sample size in the existing Newey-West evaluation. A useful diagnostic is `mean(ubar)`, which reports the fraction of the row count that is genuinely independent information. With `max_bars = 20` on daily bars sampled every row, expect roughly $1/20$.

**Priority 2: a causal EWMA option for the barrier volatility.** Add a `volatility_method` parameter with values `"rolling"` (current behaviour) and `"ewma"` (recommended):

$$\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2, \qquad \lambda = 0.94$$

Note $r_{t-1}$, not $r_t$: the estimate must be known at the entry bar. This has no estimated parameters, so it cannot leak, and it matches the reference convention more closely than a trailing sample standard deviation.

**Priority 3: a `meta_label` target.** Given a primary side column $s \in \{-1, +1\}$ derived from an existing trend feature such as `macd_histogram` or `price_vs_moving_average`:

```
meta_label_t = 1  if  side_t * triple_barrier_exit_return_t > cost_threshold
               0  otherwise
```

with `cost_threshold` a named constant holding the round-trip cost in decimal return units, for example 0.001. This is the target that actually answers "should I take this trade", and it is the only labelling idea in section 8 with published support behind it. Keep `side` a parameter of the target function so the label is reproducible from configuration alone.

**Priority 4: four timing features.** Cheap, daily-data-only, high research maturity, and all computable from the OHLCV columns already loaded:

```
distance_to_52w_high_t = close_t / max(close_s : s in [t-251, t]) - 1     in (-1, 0]
drawdown_from_peak_t   = close_t / cummax(close)_t - 1                     expanding
bars_since_high_t      = t - argmax(close_s : s in [t-251, t])             integer >= 0
path_efficiency_t      = |close_t / close_{t-n} - 1| / sum_{k=1..n} |r_{t-k+1}|   in [0, 1]
```

`path_efficiency` is the cheap stand-in for information discreteness and does not need the cross-sectional conditioning that ID does. Because `distance_to_52w_high` equals the drawdown on a 252-day window, check their correlation before shipping both: with an expanding `cummax` for drawdown they are genuinely different features, but if both use 252 days they are the same one.

**What not to add yet.** Trend-scanning labels, because no independent validation exists and the selected-maximum t-statistic is not the statistic it appears to be. Full survival targets, because `triple_barrier_bars_to_exit` already is the censored duration; add a competing-risks model only once there is a use for the cumulative incidence function. Oscillators, because they are collinear with each other and absent from the canonical machine-learning asset-pricing feature set.

**One thing to fix regardless of priority.** Document that `max_bars` is simultaneously a label definition and a risk preference, and that tuning it on validation accuracy silently selects a risk preference rather than merely overfitting. That is the cleanest lesson in the recent literature and the one most often missed.
