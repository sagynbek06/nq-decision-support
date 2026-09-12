# NQ Decision Support

A quantitative decision-support system for trading NQ (Nasdaq-100 E-mini)
futures. It combines regime detection, order flow analysis, kernel
regression, and options-derived signals into a single consensus view.

## Decision Philosophy

This system informs decisions — it does not automate them. Every component
produces a signal or a probability, not an order. The human trader synthesizes
those signals, applies judgment and risk management, and makes the final
call. No part of this project is intended to place trades autonomously.

## Status

- **Phase 1: HMM Regime Detection — in progress**
- Phase 2: Order Flow Monitor — not started
- Phase 3: Kernel Regression — not started
- Phase 4: Greeks Dashboard — not started
- Phase 5: Consensus Engine — not started
- Phase 6: Real Data Integration — not started
- Phase 7: Walk-Forward Hyperparameter Optimization — not started

See [ROADMAP.md](ROADMAP.md) for details on each phase.

## Project Structure

```
src/               production code (tested, importable modules)
notebooks/         exploratory Jupyter notebooks
docs/writeups/     portfolio-facing writeups and documentation
tests/             pytest test suite
```
