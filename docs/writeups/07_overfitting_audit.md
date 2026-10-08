# Walk-Forward Backtest and Overfitting Audit

**Result, stated plainly: the reported configuration does not pass the
overfitting gate.** Its held-out Sharpe is +0.55 annualized, but after
deflating for the 189 trials logged in this project, its Deflated Sharpe Ratio
is −1.86 (p = 0.97). The gate also fails if only the seven walk-forward runs
are counted (DSR −0.43, p = 0.67). None of the seven walk-forward configurations
passes. Every calibration setting has a negative held-out Sharpe (the best is
−0.45), so none of them can pass either.

Code: [`src/backtest/walk_forward.py`](../../src/backtest/walk_forward.py),
[`src/backtest/overfitting.py`](../../src/backtest/overfitting.py),
[`src/backtest/run_experiments.py`](../../src/backtest/run_experiments.py).
Trial log: [`results/experiment_log.csv`](../../results/experiment_log.csv).
Tests: [`tests/test_walk_forward.py`](../../tests/test_walk_forward.py),
[`tests/test_overfitting.py`](../../tests/test_overfitting.py).

## 1. The gate

The gate is the `deflated-sharpe` package, version 0.1.0 (Apache-2.0, published
by Mnemox AI), pinned in `requirements.txt`. I read its source (`gates.py`)
before relying on it. The DSR, the p-value and the minimum backtest length are
computed by the package. This module passes it the right inputs.

**Units.** The package's docstring says the Sharpe ratio is annualized, but its
code takes a per-period Sharpe. Its standard error,
sqrt((1 − γ₃·SR + ((γ₄ − 1)/4)·SR²)/(T − 1)), is the standard error of a
per-period Sharpe estimated from T returns. Passing an annualized Sharpe
therefore inflates the z-score. For a strategy with per-period Sharpe 0.10
(annualized 1.59) on 252 daily returns, with the package defaults (skew 0,
kurtosis 3):

| Input | M = 1 | M = 100 |
|---|---|---|
| per-period Sharpe 0.10 | DSR +1.58, p 0.057 | DSR −0.78, p 0.78 |
| annualized Sharpe 1.59 | DSR +16.7, p ≈ 0 | DSR +15.2, p ≈ 0 |

The gate always receives the per-period Sharpe, and `tests/test_overfitting.py`
pins that convention.

**Expected maximum.** The package approximates the expected maximum of M normal
draws with the asymptotic formula sqrt(2 ln M) − (ln ln M + ln 4π)/(2 sqrt(2 ln M)).
It also takes the standard deviation of trial Sharpe ratios to be
sqrt(1/(T − 1)), its null value, rather than estimating it from the trials.
Bailey and López de Prado (2014) use a mixture of two normal quantiles for the
expected maximum. I used that form from memory here and did not check the
paper's text. At M = 189 the package's expected maximum is 2.59 and the
mixture's is 2.75. With the mixture, the reported trial's DSR is −2.02 (p 0.98)
at M = 189 and −0.66 (p 0.74) at M = 7. The verdict does not depend on this
choice.

**Trial count.** `num_trials` is the number of rows in the trial log: 182
calibration-sweep settings plus 7 walk-forward runs, M = 189. The gate takes the
reported trial by name. That trial is chosen by train Sharpe among the six
full-pipeline configurations in `run_experiments.py`, and that choice is counted
in M through the log. The 182 sweep settings share one consensus signal and
differ only in sizing, so they are closely related. Counting all of them is
conservative for that reason. The conclusion does not depend on it: at M = 7
the gate still fails.

## 2. The pipeline and its no-lookahead checks

`build_signals` computes every causal input once: the walk-forward regime (the
causal filter, not Viterbi), the expanding-window order-flow composite, the
one-sided kernel with each fold's bandwidth chosen only from prices before that
fold's test block, the Hurst band multiplier, the Ax sub-signal, WSS context,
and GEX. GEX is context only and is not part of the score. `run_config` then
applies one configuration: votes, weights (equal, or re-calibrated on resolved
history every 20 bars), consensus, sizing and the sized backtest. Decision bar t
uses only data up to t.

The guarantee is checked at two levels.

- `tests/test_walk_forward.py` plants a 50× price jump at a later bar (bars 200
  and 270) and checks that every earlier decision's score and position are
  bit-identical, for the full pipeline and the baseline. A companion test checks
  that the same jump does change some decisions at or after the bar, so the
  check is not vacuous.
- A signal-level check (scratch script, not committed) applies the same
  construction and compares every signal before bar 270: kernel deviation, order
  flow, order-flow features, GEX, Hurst multiplier, Ax sub-signal, regimes and
  WSS context. All are bit-identical.

Two problems came up while building the first version of the test, and both are
fixed. (1) The synthetic order-flow generator scaled volumes by full-sample
statistics, so a future outlier changed earlier bars. A causal option
(`causal_scales=True`, expanding statistics with fixed priors for the first 20
bars) fixes it. The default is kept, so analyses that use it are not causal,
including the calibration sweep (section 5). (2) The spike helper rebuilt every
return from closes, which changed every bar by rounding noise. The helper now
changes only the outlier bar and the two returns it touches. An earlier
diagnostic that rebuilt returns this way showed differences of 1e-13 to 1e-8
before the spike. Those differences came from the helper, not from a leak.

## 3. Configurations and results

Seven runs, all through the same machinery and the same causal inputs: the
original equal-weight three-vote consensus (`baseline_pre5a`, with no Hurst, Ax,
WSS or dynamic weights) and six full-pipeline variants, each differing from
`full_meanrev` in one setting. Sizing is the calibrated setting (neutral band
0.00, CDaR limit 1.00, which the sweep selects, section 5), except
`full_band010`, which uses a base band of 0.10. Costs are 2 bps per unit of
turnover. Sharpe ratios are annualized. The held-out window is the last 448
decisions, 2023-02-10 to 2024-10-29. The data are 1,260 synthetic bars dated
2020-01-02 to 2024-10-30.

| Trial | Train Sharpe | Held-out Sharpe | Held-out trades/week | Held-out total return |
|---|---:|---:|---:|---:|
| baseline_pre5a | 0.42 | −0.36 | 1.50 | −2.9% |
| full_meanrev | 0.42 | −0.36 | 1.50 | −2.9% |
| full_momentum | **1.14** | **+0.55** | 1.54 | +8.7% |
| full_dynamic | 0.34 | −1.10 | 1.43 | −7.8% |
| full_wss_vote | 0.42 | −0.36 | 1.50 | −1.7% |
| full_subsignal | 0.00 | −2.20 | 1.24 | −11.2% |
| full_band010 | 0.13 | −0.52 | 1.98 | −2.7% |

`full_momentum` is the reported trial: it has the highest train Sharpe of the six
full-pipeline configurations. The baseline is the reference and is not a
candidate. `full_meanrev` and the baseline have identical scores and positions.
At base band 0.00 the Hurst multiplier has no effect and WSS context is not a
vote. I checked both in the rerun, and a test pins the positions.

**Deflated Sharpe, plainly.** The reported trial's held-out per-period Sharpe is
+0.0347 over T = 448 returns, with skew +0.29 and raw kurtosis 14.5.

- M = 189: DSR −1.86, p = 0.97. Fails.
- M = 7: DSR −0.43, p = 0.67. Fails.
- Per-period Sharpe needed for p < 0.05 at T = 448: 0.204 (annualized 3.23) at
  M = 189, and 0.134 (annualized 2.12) at M = 7. The observed value is 0.035 per
  period (0.55 annualized).
- The package's minimum backtest length at this Sharpe and M = 189 is 14,881
  bars, which is about 59 years at 252 bars a year.

The other six walk-forward runs fail as well. Their DSRs at M = 189 are −3.07
(baseline and full_meanrev), −4.14 (full_dynamic), −3.07 (full_wss_vote), −6.58
(full_subsignal) and −3.29 (full_band010).

## 4. Deferred items, resolved

**Causal kernel.** Implemented and used by every walk-forward run. Each fold's
bandwidth is chosen by LOOCV on the prices before that fold's test block, and
every fold chose bandwidth 2 in this data. The full-sample kernel fit uses future
prices and is not used by the pipeline.

**Kernel reading: mean reversion or momentum.** Both were run as trials.
Momentum had the higher train Sharpe (1.14 against 0.42), and its held-out sign
agreed (+0.55 against −0.36). That is weak evidence. The two votes are exact
negatives of each other, so their next-bar hit rates sum to one: 0.496 for mean
reversion and 0.504 for momentum over 1,119 decisions. Neither reading predicts
the next bar in this data, and the DSR says the held-out difference is not
statistically meaningful. `compute_consensus` always reads its kernel input as
mean reversion. The momentum configuration negates that input in `run_config`,
and the consensus module itself is unchanged. The reported trial uses momentum
by configuration.

**WSS: vote or context.** The default is context. The vote variant changes every
decision, not only the event bars. On the 1,068 bars with no active event, the
surprise vote is exactly zero. It still joins the weighted average and the
agreement count, so each score is exactly 3/4 of its former value. Each position
is 9/16 of its former size when the CDaR throttle is at full size (the minimum
observed ratio), and up to 0.58 when the throttle is active (median 0.57). That
is a leverage cut, not a signal. The 51 bars with an active event all fall in
the held-out window. The held-out Sharpe is −0.36 in both runs, and the total
return is −1.7% instead of −2.9% because the positions are smaller. The four
events in `DEFAULT_EVENT_DATABASE` are illustrative 2024 dates, so this does not
test the event signal. Context stays the default until real annotated events
exist.

**Other components.** At base band 0.00 the Hurst multiplier has no effect, as
above. At base band 0.10 (`full_band010`) the Hurst multiplier is active. That
run's held-out Sharpe is −0.52 against −0.36, and it trades 1.98 times a week.
Dynamic weights (`full_dynamic`, held-out −1.10) and the Ax regime mode
(`full_subsignal`, −2.20) are worse on held-out. None is adopted.

## 5. What this does not show

- **The data are synthetic.** The 1,260 bars come from the repo's synthetic data
  notebook, and the event database is illustrative. The results say nothing
  about real NQ.
- **The held-out window is short and was not untouched.** 448 decisions is
  about 1.8 years at 252 bars a year, which is short for a Sharpe estimate. The
  calibration sweep's held-out columns were already in the committed report, and
  the sweep covers the same 448 bars. That is why its 182 rows are counted in M.
- **The calibration sweep is not fully causal.** Two of its inputs use future
  data. The order-flow scales are taken over the full sample, and one kernel
  bandwidth is chosen by LOOCV on prices up to the held-out start, so early
  training decisions use a bandwidth informed by later training prices. Re-running
  the sweep with the causal inputs used here (per-fold bandwidths and causal
  scales) selects the same setting, band 0.00 and CDaR limit 1.00. Its training
  Sharpe rises from 0.34 to 0.42, and its held-out Sharpe moves from −0.45 to
  −0.36. The committed report reproduces exactly from the current code (maximum
  difference 0.000). It has not been regenerated, so it still shows the
  non-causal numbers.
- **Package limits.** Version 0.1.0; section 1 covers units and the
  expected-maximum approximation. The package's README reports 27 tests. They
  are not in the installed wheel, and I did not run them.
- **Trial dependence.** The trials are closely related, so M = 189 overstates the
  number of independent trials, and the deflation is conservative. The gate still
  fails at M = 7.
- **Costs.** 2 bps per unit of turnover is an assumption, not a measurement.
- **Real data** is planned for ROADMAP.md Phase 6, according to the comment in
  `src/wss_signal.py`.

## 6. Reproduction

`data/` is gitignored, so the experiment needs a local `data/synthetic_nq.csv`,
generated by `notebooks/01_synthetic_data.ipynb`. The sweep numbers come from the
committed `reports/position_sizing_sweep.txt`. Then, from the project root:

```bash
python -m src.backtest.run_experiments
```

The script refuses to add the `walk_forward_v1` rows twice. The test suite runs
with `python -m pytest tests/ -v`, which gives 280 passed on this tree.
