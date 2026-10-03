"""
Ornstein-Uhlenbeck half-life calibration.

`src/synthetic_data.py`'s `generate_ornstein_uhlenbeck` *simulates* a
mean-reverting level series from known parameters. This module does the
reverse: given an *observed* price/level series, estimate how strongly (if
at all) it is pulling back toward some long-run mean, via the textbook OLS
calibration of the OU SDE's discretization.

The continuous-time OU process is
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
Discretized at dt=1 (one bar per step), its conditional mean is linear in
the lagged level, so theta, mu, and the innovation variance all drop out of
a single OLS regression of the increment on the lagged level:
    X_t - X_{t-1} = alpha + beta * X_{t-1} + epsilon_t
    theta = -beta,   mu = -alpha / beta   (the level where E[increment] = 0)
This is the same regression used as the (non-augmented) Dickey-Fuller unit-
root test -- a fact with a real consequence documented below and in
tests/test_ou_half_life.py.

theta <= 0 means the fit found no mean-reverting structure at all (the
series is flat, trending, or explosive rather than pulling back toward a
level) -- there's no meaningful half-life or target level in that case, so
`half_life` and `mu` are reported as `None` rather than a negative or
infinite number that looks like a real estimate.

CAVEAT -- the Dickey-Fuller finite-sample bias. Under a true random walk
(theta=0 exactly), the OLS estimator of beta here does *not* center on
zero in finite samples: its well-known finite-sample (Dickey-Fuller)
distribution is biased toward beta < 0, i.e. toward a spuriously *positive*
theta_hat (apparent mean reversion that isn't really there). Empirically
(50 seeds, n=2000, see docs/writeups/05b_ax_regime_switch.md), this gave
theta_hat > 0 on 46/50 pure random walks -- so "theta > 0" alone is not a
reliable test for "this series really mean-reverts." What *is* reliable is
magnitude: the spurious theta from pure noise topped out at 0.0084 across
those 50 seeds, versus 0.13-0.19 recovered from genuine OU data simulated
with theta=0.15 -- more than an order of magnitude apart, and implying a
wildly different half-life (hundreds of days, vs. the true ~4.6). Callers
relying on this module for a trading signal (see `src/ax_regime_switch.py`)
should treat a small theta_hat as noise regardless of its sign, not just
reject theta_hat <= 0.
"""

import numpy as np


def calibrate_ou_half_life(prices):
    """
    OLS-calibrate an Ornstein-Uhlenbeck process from an observed price (or
    other level) series: regress p_t - p_{t-1} on p_{t-1}.

    Returns a dict:
        theta      -- mean-reversion speed (-beta); can be <= 0.
        mu         -- implied long-run mean, or None if theta <= 0.
        half_life  -- ln(2)/theta in bars, or None if theta <= 0.
        alpha, beta -- the raw OLS coefficients.
        residuals  -- regression residuals (delta - fitted), for callers
                      that want the fit's implied innovation scale.
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 3:
        raise ValueError("calibrate_ou_half_life needs at least 3 price points")

    p_lag = p[:-1]
    delta = np.diff(p)
    design = np.column_stack([np.ones_like(p_lag), p_lag])
    coeffs, *_ = np.linalg.lstsq(design, delta, rcond=None)
    alpha, beta = coeffs
    residuals = delta - design @ coeffs

    theta = float(-beta)
    if theta > 0:
        half_life = float(np.log(2) / theta)
        mu = float(-alpha / beta)
    else:
        half_life = None
        mu = None

    return {
        "theta": theta,
        "mu": mu,
        "half_life": half_life,
        "alpha": float(alpha),
        "beta": float(beta),
        "residuals": residuals,
    }
