# Regime Detection: Gaussian Baseline vs. Student-t Emissions

**Program 1 of the NQ decision-support system.** This writeup covers the
baseline Gaussian HMM, why its core distributional assumption doesn't hold
on this data, the from-scratch Student-t HMM built to address it, and a
quantitative before/after comparison — including performance specifically
around extreme-return ("shock") days, which is where the two models
diverge most.

Code: [`src/regime_detection.py`](../../src/regime_detection.py) (baseline),
[`src/regime_detection_robust.py`](../../src/regime_detection_robust.py)
(Student-t variant). Data and diagnostics:
[`notebooks/01_synthetic_data.ipynb`](../../notebooks/01_synthetic_data.ipynb),
[`notebooks/02_gaussian_assumption_check.ipynb`](../../notebooks/02_gaussian_assumption_check.ipynb).

## 1. Baseline: 3-state Gaussian HMM

The baseline is a standard 3-component `GaussianHMM` (diagonal covariance,
`hmmlearn`) fit via Baum-Welch on daily log returns, decoded with Viterbi.
Hidden states are mapped to `bull` / `bear` / `sideways` post hoc: the
highest-mean state is `bull`; of the remaining two, the higher-volatility
state is `bear` and the lower-volatility one is `sideways`.

Baum-Welch is EM under the hood, so it's sensitive to initialization and
prone to local optima — this showed up concretely during development as a
pathological fit where two true regimes collapsed into duplicate states
that split the data by contiguous time block rather than by distribution.
The fix (`src/regime_detection.py`, `fit_hmm`) is 25 random restarts, kept
by training log-likelihood, which restores reliable recovery (>98% on a
synthetic sanity check with cleanly separated, Gaussian-generated regimes;
see `tests/test_regime_detection.py`).

That sanity check used regime parameters deliberately more separated than
anything claimed to be realistic. On the actual synthetic NQ series
(`data/synthetic_nq.csv`, generated with Student-t innovations and
realistic per-regime drift/vol — see notebook 01), the same model fit with
the same 25-restart procedure recovers the ground-truth regime for only
**29.1%** of days. The decoded state populations are also revealing:

| Detected regime | Days |
|---|---:|
| sideways | 1,007 |
| bull | 247 |
| bear | 6 |

A 6-day "bear" state is not a regime — it's the model dedicating an entire
mixture component to a handful of the most extreme return days, because a
Gaussian's likelihood is maximized by fitting a very tight, very
low-probability component to outliers rather than absorbing them into a
wider, more representative component. The remaining data gets smeared
across an oversized `sideways` bucket.

## 2. Diagnostic: does the Gaussian assumption hold?

`notebooks/02_gaussian_assumption_check.ipynb` tests this directly: for
each regime's returns, compute skewness, excess kurtosis, and the
Jarque-Bera statistic (H0: normally distributed), plus QQ-plots against a
normal reference.

**Against the true, ground-truth regime labels** (the relevant test of
whether the *data-generating process* is Gaussian within a regime, since
`data/synthetic_nq.csv` uses Student-t shocks with 5 degrees of freedom
throughout):

| Regime | n | Skewness | Excess kurtosis | Jarque-Bera p |
|---|---:|---:|---:|---:|
| bull | 670 | 0.656 | 3.546 | 2.1×10⁻⁸⁷ |
| bear | 242 | -0.085 | 1.874 | 1.8×10⁻⁸ |
| sideways | 348 | 0.697 | 3.996 | 4.1×10⁻⁵⁷ |

Every regime rejects normality decisively, with excess kurtosis well above
zero in all three. This is not an artifact of regime-detection error — it's
the actual shape of the data the HMM is being asked to model.

**Against the Gaussian model's own detected regimes**, the picture is
muddier for a specific reason: `bull` (n=247, JB p=0.22) and `bear` (n=6,
JB p=0.63) both fail to reject normality — but `bear`'s sample is too small
for the test to have any power, and `bull`'s visible QQ-plot curvature just
doesn't clear significance at that sample size. `sideways` (n=1,007, JB
p=3.0×10⁻⁷), the majority of the data, rejects unambiguously. Taken
together with the ground-truth result, the conclusion is that the fat
tails are real and pervasive, and the Gaussian baseline's own regime
assignment is partly a symptom of trying to fit them with the wrong
distribution.

## 3. Motivation for Student-t emissions

A Gaussian mixture/HMM has no mechanism to distinguish "a legitimately
different regime" from "an outlier under the current regime" — both pull
the component mean and inflate its variance by the same mechanism. A
Student-t emission adds a degrees-of-freedom parameter ν that controls
tail weight independently of the mean and scale, and — critically — comes
with a natural per-observation *reliability weight* through its
Gaussian-scale-mixture representation:

```
x_t | u_t, state k  ~  Normal(mu_k, sigma2_k / u_t)
u_t                 ~  Gamma(nu_k / 2, nu_k / 2)
```

An observation far from its state's center gets a small posterior u_t and
is automatically downweighted when re-estimating that state's mean and
variance — outliers get absorbed and characterized (via ν) rather than
either dragging the estimates around or requiring their own component.

`hmmlearn` only implements Gaussian/GMM emissions, so this required
writing Baum-Welch from scratch: log-space forward-backward and Viterbi,
plus an M-step that updates (mean, scale, ν) per state using the standard
EM-for-t-distributions equations (McLachlan & Peel, *Finite Mixture
Models*, 2000, Ch. 7) — the ν update in particular has no closed form and
is solved with 1-D root-finding each iteration.

Two implementation notes worth flagging for anyone extending this:

- **Correctness check used:** the log-likelihood of a correctly implemented
  EM must be non-decreasing every iteration. Early versions of this
  implementation violated that, traced to a broadcasting bug in a
  hand-rolled `logsumexp` (subtracting a per-row max against the wrong
  axis) that was invisible with a uniform transition matrix and only
  surfaced once the matrix became asymmetric. Monotonicity is a cheap,
  effective invariant to check for any EM implementation before trusting
  its output.
- **Performance:** a pure-Python per-timestep forward-backward loop is the
  unavoidable cost of not using a compiled library. Replacing
  `scipy.special.logsumexp` with a version specialized to the fixed 2D
  shapes used here cut per-iteration cost by ~8x; the model still fits the
  1,260-day dataset in well under a minute with 15 random restarts, versus
  a few seconds for `hmmlearn`'s compiled Gaussian implementation.

## 4. Before / after comparison

Both models fit to the same `data/synthetic_nq.csv` returns, decoded, and
scored against the true regime labels:

| | Gaussian baseline | Student-t | Δ |
|---|---:|---:|---:|
| Overall accuracy | 29.1% | 44.7% | +15.5 pts |
| Accuracy on the 20 most extreme \|return\| days | 40.0% | 65.0% | +25 pts |
| Accuracy in a ±3-day window around those days (n=125) | 22.4% | 65.6% | +43.2 pts |
| Accuracy away from those windows (n=1,135) | 29.9% | 42.4% | +12.5 pts |

The gap is largest exactly where the mechanism predicts it should be: in
the neighborhood of shock days, the Gaussian model's accuracy *drops below*
its away-from-shock baseline (22.4% vs. 29.9%) — a shock doesn't just get
misclassified itself, it visibly contaminates the surrounding days' state
assignment. The Student-t model does the opposite: it's *more* accurate
near shocks (65.6%) than away from them (42.4%), because in this dataset
shocks cluster inside bear regimes, and bear is the state the Student-t
model characterizes best — full-sized (505 detected days, vs. the
Gaussian's degenerate 6) and correctly identified as the fat-tailed,
high-volatility state (fitted ν≈4.7, vs. ν≈7.6–7.7 for the other two).

Note bear's fitted mean (-0.00083) isn't the most negative of the three
states here — sideways is slightly more negative (-0.00101). `label_states`
still identifies bear correctly because it uses volatility as the
tie-breaker among non-bull states, not mean rank: bear's defining
statistical signature in this data is elevated volatility and fat tails,
not the most negative average return over its assigned days.

## 5. Where this leaves things

44.7% is a real improvement but still far from good regime recovery in an
absolute sense. Two caveats worth being explicit about:

- Some of the ceiling here is a hard data problem, not a model problem:
  bull and sideways drift differs by a small fraction of a day's
  volatility, which is a low-power discrimination problem for *any*
  return-only model, Gaussian or Student-t. This is the same limitation
  noted when validating the baseline's HMM fitting logic
  (`tests/test_regime_detection.py` uses deliberately well-separated
  synthetic parameters for exactly this reason).
- Both models are single-feature (returns only). Program 2's planned
  order-flow features (`ROADMAP.md` Phase 2) are a more direct way to
  close the remaining gap than further refining the emission distribution.

The practical takeaway: switching emission families addressed the specific,
diagnosable failure mode found here (outlier-driven state collapse), with
the largest, most mechanistically explainable gains concentrated exactly
where that failure mode operates — around shock days — rather than being a
uniform, hard-to-interpret lift.
