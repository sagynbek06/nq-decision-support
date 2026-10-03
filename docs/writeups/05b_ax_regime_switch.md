# Ax-Style Regime Switching: A Trend/Reversion Sub-Signal for Program 1

**An addition to Phase 5 (Consensus Engine), alongside
[05a](05a_hurst_exponent.md).** This writeup covers the historical idea
behind this addition, the implementation (an OU half-life calibrator, a
kernel-regression trend slope, and the sub-signal that combines them), what
`validate_regime_edge` found when asked to check the premise empirically,
and an honest answer to the question that motivated building it: does this
system's regime detector actually support trusting a mean-reversion
sub-signal as much as this design leans on it?

Code: [`src/ou_half_life.py`](../../src/ou_half_life.py),
[`src/ax_regime_switch.py`](../../src/ax_regime_switch.py). Consensus
integration: [`src/consensus_engine.py`](../../src/consensus_engine.py)'s
`vote_from_regime_subsignal` and `compute_consensus`'s `regime_mode`
parameter. Tests:
[`tests/test_ou_half_life.py`](../../tests/test_ou_half_life.py),
[`tests/test_ax_regime_switch.py`](../../tests/test_ax_regime_switch.py).

## 1. Ax's historical approach

James Ax's Axcom Trading Advisors — the 1980s predecessor firm that became
Renaissance Technologies — built its earliest systematic strategies around
switching between two sub-strategies depending on detected market
conditions: trend-following when the market showed directional momentum,
mean-reversion when it didn't (Zuckerman, *The Man Who Solved the Market*,
2019). The insight wasn't "pick one strategy" but "detect which regime
you're in, and let that choice determine *which kind of bet* to make, not
just whether to make a directional bet at all."

This project's Program 1 already detects bull/bear/sideways regimes
(`src/regime_detection_robust.py`) and already feeds that label into the
consensus engine (`vote_from_regime`: bull=+1, bear=-1, sideways=0). What
it didn't do until now is anything *conditional* on the regime beyond that
bare label — bull and bear both just vote a fixed +1/-1 regardless of how
strong the trend actually looks, and sideways always votes exactly 0,
discarding any information about whether price is sitting above or below
its recent mean. `src/ax_regime_switch.py` adds that conditioning: given
the decoded regime, it computes a trend-following sub-signal (bull/bear)
from Program 3's kernel-regression trend, or a mean-reversion sub-signal
(sideways) from a newly-added OU half-life calibration, and `consensus_
engine.py`'s new `regime_mode="subsignal"` option lets the consensus engine
use it in place of the bare label.

## 2. Implementation

### 2.1 OU half-life calibration (`src/ou_half_life.py`)

`generate_ornstein_uhlenbeck` (added for the Hurst module, [05a](05a_hurst_exponent.md))
*simulates* a mean-reverting series from known parameters. This module does
the reverse: given an *observed* price series, it estimates how strongly it
is actually pulling back toward some level, via the standard OLS
calibration of the OU SDE's discretization — regress the increment on the
lagged level, `p_t - p_{t-1} = alpha + beta*p_{t-1} + epsilon_t`, then
`theta = -beta` and `mu = -alpha/beta` (where `E[increment] = 0`).
`half_life = ln(2)/theta` when `theta > 0`; both `half_life` and `mu` are
`None` when `theta <= 0` — there's no meaningful target level to report
when the fit found no reversion at all.

Recovery on genuine OU data (true theta=0.15, mu=100, n=2000, 7 seeds):

| seed | theta_hat | half_life (true=4.62) | mu_hat |
|---:|---:|---:|---:|
| 1 | 0.1519 | 4.56 | 99.91 |
| 2 | 0.1881 | 3.68 | 99.62 |
| 3 | 0.1450 | 4.78 | 100.19 |
| 4 | 0.1557 | 4.45 | 100.05 |
| 5 | 0.1482 | 4.68 | 100.16 |
| 6 | 0.1477 | 4.69 | 100.16 |
| 7 | 0.1283 | 5.40 | 99.74 |

Good recovery across the board, and it scales correctly across reversion
speeds too (theta=0.02 → 0.0177; 0.05 → 0.0497; 0.15 → 0.1519; 0.3 → 0.3017;
0.5 → 0.5013; all seed=1).

**A real subtlety, caught empirically rather than assumed away:** this
regression is exactly the (non-augmented) Dickey-Fuller unit-root test. Its
textbook finite-sample bias means a *true* random walk (theta=0 exactly)
does not give theta_hat centered on zero — across 50 random-walk seeds
(n=2000), **46/50 gave a spuriously positive theta_hat**, i.e. apparent
"mean reversion" that isn't really there. If this module stopped at "theta
> 0 means trust it," it would be wrong most of the time on pure noise. What
saves it is magnitude: the largest spurious theta across those 50 seeds was
**0.0084**, against a genuine OU signal's **0.1519** — more than an order
of magnitude apart, and the implied half-lives are just as far apart (a
few hundred days of spurious "reversion" vs. ~4.6 real days).
`tests/test_ou_half_life.py::test_pure_random_walk_mean_reversion_signal_is_negligible_versus_genuine_ou`
checks exactly this magnitude gap rather than asserting a sign that isn't
actually guaranteed. A second check uses a positive-feedback (GBM-like)
trending series, where the systematic relationship runs the other way —
that one does reliably clip to `theta <= 0`, confirmed on 10/10 seeds.

### 2.2 Trend slope, and a dead end this project's own prior documentation avoided

The trend sub-signal needs a slope from Program 3's kernel regression. The
obvious approach — call `fit_kernel_regression` and measure the slope of
its LOOCV-selected curve — turned out to reproduce a problem
[03_kernel_regression.md](03_kernel_regression.md) already diagnosed:
LOOCV reliably picks the smallest bandwidth on persistent price data, which
barely smooths at all. Tested directly rather than assumed: on a purely
bear-trending synthetic series (10 seeds, n=500), a slope measured off the
LOOCV-selected (bandwidth=2) curve got the **sign wrong on 2/10 seeds** —
not a rare edge case, a real failure rate, because the last-10-point slope
of an almost-unsmoothed curve is dominated by short-run noise, not the
actual trend. `kernel_trend_slope` instead uses a manually-chosen bandwidth
(20, this module's default) via `nadaraya_watson_regression` directly,
exactly the remedy `kernel_regression.py`'s own METHODOLOGICAL NOTE 2
already prescribes for "a smoothed trend ... legible" rather than a
minimal-reconstruction-error fit. On the same 10 seeds, bandwidth=20 gets
the sign right **10/10**, and more broadly correct on 99/100
trend/seed/length combinations tested (bull and bear, lengths 100-500).

The slope itself is standardized by the curve's own day-to-day volatility
(`std(diff(estimate))`) before use — the same standardize-by-a-residual-
scale idea `compute_deviation_signal` already uses, kept here so the result
is a dimensionless "trend strength" rather than a raw price-per-day number
with no natural scale to compare against the other votes.

### 2.3 `ax_regime_subsignal`: combining both into one scaled float

For bull/bear, the sub-signal is `tanh(kernel_trend_slope(price) / 2.0)` —
Ax's momentum sub-strategy, following the same direction Program 3's trend
already points.

For sideways, it's a confidence-weighted pull toward the OU-calibrated
`mu`: the current price's distance from `mu`, expressed in units of the OU
process's own stationary standard deviation (`sigma/sqrt(2*theta)`,
approximated from the calibration's residual scale and theta), then
tanh-squashed. A stronger theta means a tighter stationary distribution, so
the same price gap reads as a more confident signal — this is deliberately
not just a direction, it's a conviction that scales with how strongly
mean-reverting the window actually looked. Worked example (seed=1, OU fit:
theta=0.1552, mu=99.637):

| price | sub-signal |
|---|---:|
| mu + 0.5 | -0.140 |
| mu + 1.0 | -0.275 |
| mu + 2.0 | -0.511 |
| mu + 3.0 | -0.689 |
| mu + 5.0 | -0.888 |

(and the mirror image, positive, for price below `mu`). When `theta <= 0`
(no valid structure in the calibration window), the sub-signal is `0.0` —
an honest "no signal," not a guess dressed up as one.

Both branches land in `[-1, 1]` by construction, matching
`vote_from_order_flow` and `vote_from_kernel_deviation`'s tanh-squashed
scale, which is why `consensus_engine.vote_from_regime_subsignal` is a
pass-through (clipped defensively) rather than a second squashing
convention — all the scaling work happens here, once.

### 2.4 Consensus engine integration

`consensus_engine.py` gets `vote_from_regime_subsignal(sub_signal)` and a
`regime_mode` parameter on `compute_consensus` (default `"label"`, opt-in
`"subsignal"`). `vote_from_regime` is untouched and still the default —
`test_end_to_end_with_real_programs_1_through_4` passes with no changes,
and "label" mode remains available as the simpler baseline Phase 7 may want
to compare against. In `"subsignal"` mode, the caller passes the
already-computed sub-signal via a new `regime_subsignal` parameter;
`regime_label` is still required (ax_regime_subsignal needs it to decide
what to compute) but unused for the vote itself in this mode.

## 3. `validate_regime_edge`: does the premise actually hold?

Everything above assumes Program 1's regime labels carry real information:
that "sideways" precedes mean-reversion more often than chance, and
bull/bear precede continuation in their implied direction more often than
chance. `validate_regime_edge` tests this directly rather than assuming it,
using this project's established walk-forward discipline: for each of
`n_splits` expanding-window folds, fit a fresh `StudentTHMM` on only the
data before that fold's test block, decode regimes over train+test jointly
(so the test block's Viterbi path is informed by what came before, not
restarted cold), then check what actually happened next on every test day.
A bull/bear day "succeeds" if the next bar continues in the implied
direction; a sideways day "succeeds" if the next bar moves toward a `mu`
calibrated from a trailing window ending at that day. Significance is
assessed with a permutation test — 500 shuffles of which regime label
attaches to which (fixed, time-ordered) outcome, per the task spec — giving
a p-value for each edge against the implicit random-walk null of 50/50.

**On a pure random walk** (seed=11, n=1800, no real structure of any kind):
neither edge is significant, as it should be — `trend_p=0.331`,
`reversion_p=0.497`.

**On data built with real switching structure** — bull/bear blocks with
genuine drift, alternating with blocks that are genuinely
Ornstein-Uhlenbeck (mean-reverting, not just low-variance noise; see
`tests/test_ax_regime_switch.py`'s `_build_mixed_trend_reversion_series`
for why this distinction mattered) — the **trend edge comes through
clearly**: `trend_edge=0.080`, `trend_p=0.002`, from 809 pooled test-day
samples across 5 folds. Program 1's directional labels reliably predict
next-bar continuation on this data, robustly across every seed and
parameter variation tried during development.

**The reversion edge does not**: `reversion_edge=0.005`, `reversion_p=0.459`
— indistinguishable from the random-walk null, on data that was
deliberately built to contain real mean-reversion.

### 3.1 Diagnosing why, instead of shrugging

That gap is surprising enough to chase down rather than report flatly. Two
checks on the exact same fixture:

**Is the reversion signal even real, underneath the noise?** Yes. Measuring
reversion success against the *true* block boundaries directly (no HMM
involved — just: on days truly inside an OU block, did price move toward
that block's true anchor more often than chance), the success rate is
**58.7%** (n=443) — a genuine ~9-point edge over the 50% null. The
mean-reversion mechanism itself, and the OU calibration that measures it,
both work.

**So what's destroying it in the full pipeline?** Regime-label precision.
Fitting the same HMM on this fixture (non-walk-forward, just to inspect the
confusion matrix) gives only **40.8%** overall accuracy against true
labels — and of the days the HMM actually calls `"sideways"`, only
**25.9%** are truly from a reverting block. The other ~74% are mostly
mislabeled trend-block days. A real, ~9-point edge measured against labels
that are right barely a quarter of the time washes out to statistical
noise by the time it's pooled and tested — not because the edge isn't
there, but because the label used to find it mostly isn't pointing at it.

This directly extends a finding [01_regime_detection.md](01_regime_detection.md)
already reported: Program 1 is weak away from return shocks (44.7% overall
for the Student-t model, 42.4% specifically away from shocks). This writeup
adds the sharper, more specific version of that finding: the weakness isn't
evenly spread across bull/bear/sideways — on data that mixes trending and
reverting dynamics, `"sideways"` specifically is the label most degraded,
to the point that a real, checkable mean-reversion edge underneath it
becomes undetectable once you have to rely on the HMM to find the days it
applies to.

## 4. The sideways-dominance concern, answered directly

The task motivating this module raised a specific worry up front: sideways
is this system's most common detected regime (1,007/1,260 days per
[01_regime_detection.md](01_regime_detection.md)), so the mean-reversion
sub-signal — the harder of the two to validate — is also the one that will
fire by far the most often. Section 3 is the direct answer to whether that
should be trusted as much as it will be relied upon: **no, not on this
evidence.** The trend sub-signal's premise is confirmed, robustly, by
`validate_regime_edge`. The reversion sub-signal's premise is not — not
because the underlying mean-reversion math is wrong (Section 3.1 shows it
isn't), but because the regime label gating *when* to apply it is right
far too rarely for that real edge to survive contact with this system's
actual regime detector.

Practically, this means `regime_mode="subsignal"` is, right now, most
trustworthy exactly on the bull/bear days it's used least often (sideways
being the majority class), and least trustworthy on the days it fires most.
That's the opposite of reassuring for a sub-signal meant to be weighted
equally against order flow and kernel deviation in `compute_consensus`'s
default weights. This isn't a reason to revert the reversion branch — the
OU calibration and sub-signal math are correct and the `vote_from_regime`
label-only baseline remains available precisely for this kind of case — but
it is a reason to flag, explicitly, that trusting it in practice should
wait on either a better `"sideways"` label (a regime-detection improvement,
not a consensus-engine one) or a validated lower weight for the regime vote
when `regime_mode="subsignal"` is active during sideways stretches. Both
are Phase 7 walk-forward/weight-tuning questions, not something to guess at
here.

## 5. Where this leaves things

The OU half-life module, the trend slope, and the sub-signal scaling are
all validated the way this project validates its other numerical methods:
against known cases (OU recovery, the Dickey-Fuller bias magnitude check),
against a real dead end this project's own prior documentation predicted
and this module's tests now confirm empirically (the LOOCV bandwidth
sign-error rate), and against ground truth (the 58.7% isolated reversion
success rate). `validate_regime_edge` itself is the more important result:
it's not a rubber-stamp confirming the design works, it's a genuine,
non-trivial finding that one half of it — the half that fires most often —
doesn't yet hold up end-to-end, with a specific, checked diagnosis of why.
Whether to address that by improving Program 1's sideways precision,
down-weighting the subsignal-mode regime vote during sideways stretches, or
something else, is a decision for Phase 7's walk-forward tuning to make
with data, not a default to pick here.
