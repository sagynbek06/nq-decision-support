# Roadmap

Planned development phases for the NQ decision-support system, in order.

## Phase 1: HMM Regime Detection (Program 1) — ✅ Complete
Gaussian baseline hidden Markov model for market regime detection, followed by
more robust emission models (skew-t) to better handle fat tails and asymmetry
in returns.

**Outcome:** switching from Gaussian to a from-scratch Student-t emission HMM
raised regime detection accuracy from 29% to 45%, with the largest gains
around return shocks (see [docs/writeups/01_regime_detection.md](docs/writeups/01_regime_detection.md)).

## Phase 2: Order Flow Monitor (Program 2)
Volume Delta, Order Book Imbalance, and Marchenko-Pastur filtering to separate
signal from noise in order flow data.

## Phase 3: Kernel Regression (Program 3)
Nadaraya-Watson estimator for non-parametric price/level regression.

## Phase 4: Greeks Dashboard (Program 4)
Black-Scholes Greeks, Gamma Exposure (GEX), Vanna, and implied volatility skew
visualization.

## Phase 5: Consensus Engine
Weighted voting mechanism that combines signals across all four programs into
a single decision-support view.

## Phase 6: Real Data Integration
Bloomberg API and Rithmic integration to replace historical/sample data with
live and production-grade market data feeds.

## Phase 7: Walk-Forward Hyperparameter Optimization
Walk-forward validation and optimization of model hyperparameters across all
programs to guard against overfitting.
