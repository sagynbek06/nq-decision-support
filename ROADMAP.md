# Roadmap

Planned development phases for the NQ decision-support system, in order.

## Phase 1: HMM Regime Detection (Program 1) — ✅ Complete
Gaussian baseline hidden Markov model for market regime detection, followed by
more robust emission models (skew-t) to better handle fat tails and asymmetry
in returns.

**Outcome:** switching from Gaussian to a from-scratch Student-t emission HMM
raised regime detection accuracy from 29% to 45%, with the largest gains
around return shocks (see [docs/writeups/01_regime_detection.md](docs/writeups/01_regime_detection.md)).

## Phase 2: Order Flow Monitor (Program 2) — ✅ Complete
Volume Delta, Order Book Imbalance, and Marchenko-Pastur filtering to separate
signal from noise in order flow data.

**Outcome:** MP filtering is adapted to a single instrument (a self-referential
smoothed panel, not the textbook cross-sectional application) — validated
against known random matrix theory and recovered an injected common factor
at 0.84 correlation (see [docs/writeups/02_order_flow.md](docs/writeups/02_order_flow.md)).

## Phase 3: Kernel Regression (Program 3) — ✅ Complete
Nadaraya-Watson estimator for non-parametric price/level regression.

**Outcome:** LOOCV bandwidth selection reliably picks the smallest bandwidth
on NQ's persistent price series — correct for minimizing reconstruction
error, but not for a visually legible trend line, which needs a manually
chosen bandwidth instead (see [docs/writeups/03_kernel_regression.md](docs/writeups/03_kernel_regression.md)).

## Phase 4: Greeks Dashboard (Program 4) — ✅ Complete
Black-Scholes Greeks, Gamma Exposure (GEX), Vanna, and implied volatility skew
visualization.

**Outcome:** all Greeks validated against finite-difference derivatives and
put-call parity; GEX/VEX use the standard long-calls/short-puts dealer
convention on a synthetic options chain (see [docs/writeups/04_greeks_dashboard.md](docs/writeups/04_greeks_dashboard.md)).

## Phase 5: Consensus Engine — ✅ Complete
Weighted voting mechanism that combines signals across all four programs into
a single decision-support view.

**Outcome:** a 3-way directional vote (regime, order flow, kernel deviation)
with Program 4's GEX surfaced separately as volatility-regime context rather
than folded into the vote, since it isn't a directional signal (see
[docs/writeups/05_consensus_engine.md](docs/writeups/05_consensus_engine.md)).

### Phase 5 extensions — ✅ Complete
Hurst-modulated neutral band, Ax-style regime-conditioned sub-signal (opt-in
`regime_mode`), WSS surprise context, accuracy-weighted votes, and CDaR-throttled
position sizing with a train/held-out calibration sweep. Writeups 05a and 05b;
sizing results in `reports/position_sizing_sweep.txt`.

**Outcome:** the held-out sweep shows no reliable edge on synthetic daily data
(held-out Sharpe near zero or negative across settings). That is a finding, not
a bug; see writeup 06 for why a calibrated gate agrees.

### Phase 5.5: Uncertainty and online regime tools — ✅ Modules done, wiring open
`src/conformal_gate.py` (adaptive conformal intervals, abstain when the interval
contains zero) and `src/bocpd.py` (Bayesian online changepoint detection). See
[docs/writeups/06_conformal_gate_and_bocpd.md](docs/writeups/06_conformal_gate_and_bocpd.md).

Open items, in priority order:
1. Soft conformal sizing (size by |forecast| / interval width) instead of the hard in/out cut, tested held-out across seeds.
2. Intraday (1-minute) version of the sizing backtest. Everything in `src/risk/` is daily, while the stated goal is 1-5 trades per DAY; "trades per week" in the calibration is a daily-bar artifact.
3. Wire `bocpd.regime_age_confidence` as a multiplier on the regime vote, only if it improves held-out results.
4. Hawkes-process order-flow-imbalance forecasting for Program 2 (needs real or realistic tick data).
5. Conformal intervals around scheduled macro events, tied to the WSS event database.
6. Deflated Sharpe on the sizing sweep, via the existing `deflated-sharpe` library, before any result is described as an edge.

## Phase 6: Real Data Integration
Bloomberg API and Rithmic integration to replace historical/sample data with
live and production-grade market data feeds.

## Phase 7: Walk-Forward Hyperparameter Optimization
Walk-forward validation and optimization of model hyperparameters across all
programs to guard against overfitting.
