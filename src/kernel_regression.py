"""
Program 3: Kernel Regression (Nadaraya-Watson).

Non-parametric local regression of price against time: a Nadaraya-Watson
estimator weights every observation by a Gaussian kernel of its distance to
the evaluation point, producing a smooth, locally-adaptive "fair value"
curve. Unlike a fixed-window moving average, the estimator uses every point
in the sample at every location, with influence that decays smoothly with
distance rather than cutting off sharply at a window edge.

The deviation of actual price from this curve -- standardized by the local
residual scale -- is exposed as a signal: large positive/negative deviations
flag price trading unusually far from its local non-parametric trend.

METHODOLOGICAL NOTE 1 -- this is a retrospective smoother, not a causal
(real-time) estimator. The standard Nadaraya-Watson estimator at time t uses
every observation in the sample, including points *after* t ("future"
information relative to that point). That's appropriate for understanding
and visualizing where price sat relative to its local trend historically,
but it is not valid as-is for a live trading signal -- using it that way
would leak lookahead information. The causal variant is
`causal_nadaraya_watson` below, which the risk backtest uses; this function
remains the retrospective fit for visualization. Walk-forward discipline for
the remaining programs is still explicitly Phase 7's job (see ROADMAP.md).

METHODOLOGICAL NOTE 2 -- `loocv_select_bandwidth`'s bandwidth is the one
that minimizes one-step reconstruction error, which is the textbook-correct
answer to "what bandwidth generalizes best," but is *not* the same question
as "what bandwidth gives a visually useful fair-value trend line." Asset
prices are highly persistent (close to a random walk): each day's price
tells you almost everything about tomorrow's starting point, so LOOCV
reliably prefers the smallest bandwidth on a candidate grid, which barely
smooths at all (verified empirically on this project's synthetic NQ data --
LOOCV selects the grid's minimum, and the resulting curve is nearly
indistinguishable from raw price). That's a correct answer to the
optimization problem it's solving, not a bug, but it means the
"fair-value trend" and "mean-reversion deviation signal" framing in this
module's docstring is best served by a deliberately larger, manually-chosen
bandwidth via `nadaraya_watson_regression(x, y, bandwidth=...)` directly,
not by `fit_kernel_regression`'s LOOCV default, when the goal is a smoothed
trend a human would find legible rather than a minimal-error reconstruction.
"""

import numpy as np
import pandas as pd


def gaussian_kernel(u):
    """Standard Gaussian kernel, K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)."""
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def nadaraya_watson_regression(x, y, bandwidth, x_eval=None):
    """
    Nadaraya-Watson kernel regression estimate:
        m_hat(x0) = sum_i K((x0 - x_i) / h) * y_i / sum_i K((x0 - x_i) / h)

    `x`, `y` are the observed data; `x_eval` are the points to evaluate the
    estimate at (defaults to `x` itself, i.e. the in-sample fit).
    Vectorized as one (len(x_eval), len(x)) weight matrix -- fine at the
    scale of a few thousand points; would need a windowed/approximate
    approach at much larger scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_eval is None:
        x_eval = x
    x_eval = np.asarray(x_eval, dtype=float)

    weights = gaussian_kernel((x_eval[:, None] - x[None, :]) / bandwidth)
    weight_sums = weights.sum(axis=1)
    return (weights @ y) / weight_sums


def loocv_select_bandwidth(x, y, bandwidths):
    """
    Pick the bandwidth from `bandwidths` that minimizes leave-one-out
    cross-validation error for the Nadaraya-Watson estimator -- the
    standard approach to bandwidth selection for kernel regression (an
    analogue of cross-validating a hyperparameter anywhere else). Each
    point's held-out prediction excludes its own (zero-distance, maximum-
    weight) contribution, so shrinking the bandwidth toward zero is
    correctly penalized as overfitting rather than trivially rewarded.

    Returns (best_bandwidth, best_loocv_mse, scores) where `scores` is a
    dict of {bandwidth: loocv_mse} for every candidate, for inspection.

    On persistent (near-random-walk) series like asset price levels, this
    reliably selects the smallest bandwidth in `bandwidths` -- see
    METHODOLOGICAL NOTE 2 in this module's docstring before assuming the
    selected value makes a good visual trend line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    scores = {}
    for h in bandwidths:
        weights = gaussian_kernel((x[:, None] - x[None, :]) / h)
        np.fill_diagonal(weights, 0.0)  # leave-one-out
        weight_sums = weights.sum(axis=1)
        valid = weight_sums > 1e-300
        loo_estimate = np.full(len(x), np.nan)
        loo_estimate[valid] = (weights[valid] @ y) / weight_sums[valid]
        scores[h] = float(np.nanmean((y[valid] - loo_estimate[valid]) ** 2))

    best_h = min(scores, key=scores.get)
    return best_h, scores[best_h], scores


def compute_deviation_signal(y, estimate):
    """
    Standardized deviation of actual values from the kernel regression
    estimate: (y - estimate) / std(residuals). Positive values mean price
    trading above its local non-parametric trend, negative means below.
    """
    y = np.asarray(y, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    residuals = y - estimate
    std = residuals.std()
    if std == 0:
        return np.zeros_like(residuals)
    return residuals / std


DEFAULT_BANDWIDTH_GRID = np.array([2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135])


def fit_kernel_regression(price, bandwidths=DEFAULT_BANDWIDTH_GRID):
    """
    End-to-end: a price series (indexed 0..T-1 by trading day) -> LOOCV-
    selected bandwidth -> Nadaraya-Watson trend estimate -> standardized
    deviation signal.

    Returns a dict with `bandwidth`, `loocv_mse`, `loocv_scores`,
    `estimate`, and `signal`.

    The LOOCV-selected bandwidth minimizes reconstruction error, not
    visual smoothness -- see METHODOLOGICAL NOTE 2 above. For a chart-
    legible trend line, call `nadaraya_watson_regression` directly with a
    manually chosen (larger) bandwidth instead of relying on this
    function's automatic selection.
    """
    price = np.asarray(price, dtype=float)
    x = np.arange(len(price), dtype=float)

    bandwidth, loocv_mse, loocv_scores = loocv_select_bandwidth(x, price, bandwidths)
    estimate = nadaraya_watson_regression(x, price, bandwidth)
    signal = compute_deviation_signal(price, estimate)

    return {
        "bandwidth": bandwidth,
        "loocv_mse": loocv_mse,
        "loocv_scores": loocv_scores,
        "estimate": estimate,
        "signal": signal,
    }


def causal_nadaraya_watson(y, bandwidth):
    """
    One-sided Nadaraya-Watson: estimate[t] is a kernel-weighted average of
    y[0..t] only. The lag mask zeroes every weight on a later point, so the
    estimate at t cannot depend on anything after t.
    """
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(y))
    lag = idx[:, None] - idx[None, :]
    weights = gaussian_kernel(lag / bandwidth) * (lag >= 0)
    return (weights @ y) / weights.sum(axis=1)


def causal_deviation_signal(y, bandwidth, standardization_window=252, min_periods=20):
    """
    Lookahead-free counterpart of fit_kernel_regression's deviation signal:
    each bar's residual from its one-sided estimate, divided by the trailing
    standard deviation of residuals. NaN until `min_periods` residuals exist.
    Standardizing by the full-sample std, as fit_kernel_regression does, would
    fold future residuals into every bar's signal.
    """
    y = np.asarray(y, dtype=float)
    residuals = y - causal_nadaraya_watson(y, bandwidth)
    trailing_std = pd.Series(residuals).rolling(standardization_window, min_periods=min_periods).std().to_numpy()
    return np.divide(residuals, trailing_std, out=np.full(len(y), np.nan), where=trailing_std > 0)
