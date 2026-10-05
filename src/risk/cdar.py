"""
Conditional Drawdown at Risk (CDaR).

CDaR at level alpha is the mean of the worst alpha-fraction of drawdowns on
an equity curve: take every per-bar drawdown, keep the largest ceil(alpha*n)
of them, and average. Value at Risk asks how bad a drawdown gets at a given
confidence level; CDaR asks how bad drawdowns get on average once they are
in the tail. It is the drawdown analogue of Expected Shortfall (Chekhlov,
Uryasev & Zabarankin, 2005).

Drawdowns are fractions of the running peak (0 at a new high, 0.15 after a
15% decline), not dollar amounts, so CDaR is comparable across equity curves
of different sizes. The peak is measured from the start of the whole curve;
a trailing window only decides which drawdown observations count.
"""

import math

import numpy as np


def drawdown_series(equity):
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return np.array([], dtype=float)
    peak = np.maximum.accumulate(equity)
    return (peak - equity) / peak


def compute_cdar(drawdowns, alpha):
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    d = np.asarray(drawdowns, dtype=float)
    if d.size == 0:
        return 0.0
    k = max(1, math.ceil(round(alpha * d.size, 9)))  # guards float products like 0.07 * 100 == 7.000000000000001
    worst = np.partition(d, d.size - k)[d.size - k:]
    return float(worst.mean())


def trailing_cdar(equity, alpha, window):
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    return compute_cdar(drawdown_series(equity)[-window:], alpha)
