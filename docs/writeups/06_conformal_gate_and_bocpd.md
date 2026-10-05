# Conformal abstention gate and Bayesian online changepoint detection

Two additions from recent literature. Both are small, causal, tested modules
that sit beside the existing programs without changing any of their signatures.
Neither is wired into `compute_consensus` by default.

## 1. Conformal gate (`src/conformal_gate.py`)

**The gap it fills.** The consensus score says which way the votes lean and
nothing about how much the lean is worth. A score of +0.4 in a quiet market and
+0.4 inside a volatility shock look the same to `position_size`.

**What it does.** Builds a prediction interval for the next bar's return by
adaptive conformal inference (after Temporal Conformal Prediction,
arXiv 2507.05470, and Gibbs & Candes), then trades only when the whole interval
sits on one side of zero. Otherwise it abstains. The interval adapts online
after every bar, so it needs no exchangeability assumption, which returns do
not satisfy.

**Two things measured while building it, not assumed.**

- Step size matters. A decaying step (the TCP schedule) reached 85.5% coverage
  against a 90% target on a series whose volatility doubles mid-sample
  (mean of 8 seeds). A constant step reached 89.4-89.9%. The default is
  constant. A decaying schedule freezes the interval exactly when a regime
  change needs it to move.
- Coverage is marginal, not conditional. Over the whole sample the miss rate
  tracks alpha; inside a particular regime it can still be off.

**Held-out experiment** (`python -m src.risk.conformal_experiment`, six
synthetic seeds, forecast scale fit on the first 60% of decisions, scored on the
last 40%, 2 bps per unit of turnover):

| | held-out Sharpe (mean / median) | trades per week | abstain rate |
|---|---|---|---|
| existing neutral-band sizing | 0.14 / 0.19 | 1.25 | n/a |
| conformal gate, alpha 0.10 | 0.00 / 0.00 | 0.00 | 100% |
| conformal gate, alpha 0.25 | 0.00 / 0.00 | 0.00 | 100% |

Coverage held up (88% at a 90% target, 74% at a 75% target), so the gate is
calibrated. It simply never found a bar where the interval excluded zero.

**How to read that.** Three things at once, and the first matters most.

1. The existing baseline's held-out Sharpe is not distinguishable from zero. Six
   seeds range from -0.74 to +0.72. With roughly 500 held-out daily bars the
   standard error of a Sharpe estimate is about 0.7, so a mean of 0.14 is noise.
   The gate and the baseline agree: at a daily horizon on this synthetic data
   there is no edge a calibrated method will bet on.
2. An "interval excludes zero" rule is very strict by construction. Daily
   return volatility is about 1% and any realistic score explains a tiny share of
   it, so the forecast is a small fraction of the interval half-width. This rule
   will abstain on almost any daily-horizon return forecast. That is a property
   of the rule, not evidence the signals are worthless on other horizons.
3. The synthetic data have regime drift but were built to be hard to separate
   (see `src/synthetic_data.py`). This says the mechanism behaves, not that a
   real market has or lacks edge.

**What would make it useful.** Intraday bars (far more observations per day, and
the target use case here), and a soft variant that scales size by
|forecast| / interval width instead of a hard in/out cut. Both are listed in
ROADMAP.md. A related line of work trains conformal intervals around scheduled
macro releases ("Known Unknowns: Trading Scheduled Surprises with Conformal
Prediction", Springer 2026). I could not open that page (the fetch tool was
rate limited), so it is cited from its title and abstract listing only, and
nothing here depends on its details.

## 2. BOCPD (`src/bocpd.py`)

**The gap it fills.** The HMM needs a fitting window, decodes a fixed menu of
three states, and its smoothed path can rewrite the past. BOCPD (Adams &
MacKay 2007) keeps a posterior over the run length, the number of bars since the
last structural break. It updates in O(max run length) per bar, needs no refit,
and never revises history. The predictive is Student-t (conjugate
Normal-Inverse-Gamma), echoing the fat-tail logic of the Student-t HMM.

**Behavior, as tested** (`tests/test_bocpd.py`):

- Mean shift of 4 sigma: change probability above 0.9 within 10 bars, quiet
  before.
- Variance shift with the mean unchanged: detected.
- Stationary noise: false alarms on under 2% of bars.
- Strictly causal: appending wild future data leaves earlier outputs unchanged.

**A limit worth stating plainly.** It is not robust to single large outliers the
way the Student-t HMM is. The t is heavy-tailed only while a segment is young
(degrees of freedom grow with run length). Measured with standard-normal noise
and one spike at bar 250: 3-4 sigma is absorbed, 5 sigma is flagged then
forgiven within about 50 bars, 6-7 sigma is read as a real break and the run
length restarts. So act on persistence, not on one bar's jump.

**A property that is easy to trip on.** With a constant hazard, the probability
that the run length is exactly 0 equals the hazard at every step. It carries no
information. The module reports the mass on short run lengths instead
(`recent_change_prob(k)`).

**Intended use.** `regime_age_confidence(map_run_length)` maps run length to a
[0, 1] trust factor: a regime a few bars old has not earned much weight. It is
meant as a multiplier on the regime vote, not as a vote. That wiring is not done
yet. It needs the same held-out test the gate just got before it earns a place
in `compute_consensus`.

## Sources

- [Temporal Conformal Prediction (TCP), arXiv 2507.05470](https://arxiv.org/html/2507.05470v1)
- [Known Unknowns: Trading Scheduled Surprises with Conformal Prediction](https://link.springer.com/chapter/10.1007/978-3-032-15120-9_16) (not opened, see above)
- [Bayesian Online Changepoint Detection for Financial Time Series, ACM](https://dl.acm.org/doi/10.1145/3795154.3795291) (not opened; method taken from Adams & MacKay 2007)
- [Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection Methods, arXiv 2307.02375](https://arxiv.org/html/2307.02375v2) (not opened)
- Adams & MacKay (2007), Bayesian Online Changepoint Detection, arXiv 0710.3742
- Gibbs & Candes (2021), Adaptive Conformal Inference Under Distribution Shift
