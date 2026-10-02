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

- **Phase 1: HMM Regime Detection — complete**
- **Phase 2: Order Flow Monitor — complete**
- **Phase 3: Kernel Regression — complete**
- **Phase 4: Greeks Dashboard — complete**
- **Phase 5: Consensus Engine — complete**
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
