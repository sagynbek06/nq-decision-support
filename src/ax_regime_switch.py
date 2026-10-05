"""
Program 1 upgrade: Ax-style regime-conditioned trend/mean-reversion sub-signal.

James Ax's Axcom Trading Advisors (the forerunner to Renaissance
Technologies' Medallion fund) built its earliest systems around switching
between a trend-following sub-strategy and a mean-reversion sub-strategy
depending on detected market regime, rather than running one fixed strategy
regardless of conditions (Zuckerman, *The Man Who Solved the Market*, 2019).
This module applies that idea on top of this project's existing Program 1
(`src.regime_detection_robust`'s Student-t HMM): instead of handing the
consensus engine a bare bull/bear/sideways label (`vote_from_regime`), it
reads the decoded regime and computes a *regime-conditioned* directional
sub-signal -- trend-following when the state is bull/bear, mean-reverting
when it's sideways -- so the consensus engine's "regime" vote reflects not
just *which* state the market is in but *what Ax's switching logic would
have done about it*.

This is a new module, not an edit to `regime_detection_robust.py` or
`kernel_regression.py`: both already have their own test coverage this is
meant to build on top of without disturbing, and both are left untouched.

HONESTY UP FRONT, before any of the mechanics below: this module's
`validate_regime_edge` function was built specifically to check whether the
premise behind the reversion half of this design -- that Program 1's
"sideways" label actually precedes mean-reversion more often than chance --
holds up end-to-end. It does not, reliably, on the walk-forward tests this
module's test suite runs (see `validate_regime_edge`'s docstring and
docs/writeups/05b_ax_regime_switch.md for the full finding and diagnosis).
That finding is reported here rather than hidden, per this project's
practice throughout, and it directly bears on how much weight the sideways
sub-signal should be given relative to the trend sub-signal -- a real
concern given sideways is also this system's most common detected regime
(docs/writeups/01_regime_detection.md: 1,007/1,260 days).
"""

import numpy as np

from src.kernel_regression import nadaraya_watson_regression

DEFAULT_TREND_BANDWIDTH = 20
DEFAULT_TREND_LOOKBACK = 10
DEFAULT_OU_WINDOW = 100
DEFAULT_N_PERMUTATIONS = 500


def kernel_trend_slope(price, bandwidth=DEFAULT_TREND_BANDWIDTH, lookback=DEFAULT_TREND_LOOKBACK):
    """
    Local trend slope of Program 3's Nadaraya-Watson fair-value curve, used
    as the Ax trend sub-strategy's directional input for bull/bear regimes:
    an OLS slope over the most recent `lookback` points of the curve,
    standardized by the curve's own day-to-day volatility -- the same
    standardize-by-a-residual-scale idea `kernel_regression.
    compute_deviation_signal` already uses for its deviation signal, kept
    here so this stays a dimensionless "trend strength" rather than a raw
    price-per-day number with no natural scale.

    Deliberately does NOT use `fit_kernel_regression`'s LOOCV-selected
    bandwidth. `kernel_regression.py`'s own METHODOLOGICAL NOTE 2 already
    documents that LOOCV reliably selects the smallest bandwidth on
    persistent price data, which barely smooths at all -- and empirically,
    that turned out to matter here, not just cosmetically: feeding this
    function a LOOCV-selected (bandwidth=2) curve on purely-trending
    synthetic price got the sign of the trend WRONG on 2/10 bear-trend
    seeds, because the last-`lookback`-point slope of a barely-smoothed
    curve is dominated by short-run noise, not the underlying trend. A
    manually-chosen, larger bandwidth fixes this: bandwidth=20 (this
    module's default) gave the correct sign on 99/100 trend/seed/length
    combinations tested (bull and bear trends, series lengths 100-500) --
    see docs/writeups/05b_ax_regime_switch.md for the full comparison. This
    mirrors kernel_regression.py's own documented guidance for "a smoothed
    trend ... legible" rather than a minimal-reconstruction-error fit.
    """
    price = np.asarray(price, dtype=float)
    x = np.arange(len(price), dtype=float)
    estimate = nadaraya_watson_regression(x, price, bandwidth)

    lb = min(lookback, len(estimate))
    recent = estimate[-lb:]
    t = np.arange(lb, dtype=float)
    slope = np.polyfit(t, recent, 1)[0]

    local_vol = np.std(np.diff(estimate))
    if local_vol == 0:
        return 0.0
    return float(slope / local_vol)


def ax_regime_subsignal(regime_label, price, ou_result=None,
                         trend_scale=2.0, reversion_scale=2.0,
                         trend_bandwidth=DEFAULT_TREND_BANDWIDTH, trend_lookback=DEFAULT_TREND_LOOKBACK):
    """
    Ax-style regime-conditioned sub-signal: a single float in [-1, 1],
    already tanh-squashed to match `vote_from_order_flow`'s and
    `vote_from_kernel_deviation`'s scale in `src.consensus_engine` -- see
    `consensus_engine.vote_from_regime_subsignal`, which passes this value
    straight through rather than squashing it a second time.

    Trend sub-signal (regime_label in {"bull", "bear"}): Ax's momentum
    sub-strategy. Sign and magnitude follow `kernel_trend_slope(price)` --
    a trend-FOLLOWING bet in the same direction Program 3's kernel
    regression already sees, not a second, independent read of direction.
    `regime_label` itself doesn't change the computation here (bull and
    bear both just follow the measured slope); it's accepted for symmetry
    with the sideways branch and so callers can pass one function the
    decoded label plus whatever inputs it needs, matching the task's
    "takes a regime label plus Program 3's slope or the OU module's output"
    framing.

    Reversion sub-signal (regime_label == "sideways"): Ax's mean-reversion
    sub-strategy. Points from the current price (the last value in `price`)
    toward `ou_result["mu"]` (from `src.ou_half_life.calibrate_ou_half_life`),
    scaled by how many of the OU process's own stationary standard
    deviations away the price currently sits -- sigma/sqrt(2*theta), the
    textbook stationary variance of an OU process, approximated from the
    calibration's residual scale and theta. A stronger theta (faster
    reversion) means a tighter stationary distribution, so the same price
    deviation reads as a more confident signal -- not just direction, but
    conviction, scaled by how strongly mean-reverting the window actually
    was. If `ou_result["half_life"]` is None (no valid mean-reverting
    structure was found in the calibration window), this returns 0.0 --
    Ax's mean-reversion sub-strategy has nothing to trade without a valid
    fit, so this is an honest "no signal," not a guess.

    Raises ValueError for any other `regime_label`, and if `ou_result` is
    missing for a sideways call.
    """
    if regime_label in ("bull", "bear"):
        slope = kernel_trend_slope(price, bandwidth=trend_bandwidth, lookback=trend_lookback)
        return float(np.tanh(slope / trend_scale))

    if regime_label == "sideways":
        if ou_result is None:
            raise ValueError("ax_regime_subsignal needs `ou_result` (from calibrate_ou_half_life) for a sideways regime")
        if ou_result["half_life"] is None:
            return 0.0

        current_price = float(np.asarray(price, dtype=float)[-1])
        residuals = ou_result["residuals"]
        resid_scale = float(np.std(residuals)) if len(residuals) else 0.0
        if resid_scale == 0.0:
            return 0.0

        stationary_std = resid_scale / np.sqrt(2 * ou_result["theta"])
        z = (ou_result["mu"] - current_price) / stationary_std
        return float(np.tanh(z / reversion_scale))

    raise ValueError(f"unknown regime label {regime_label!r}, expected 'bull', 'bear', or 'sideways'")


def _walk_forward_folds(n, n_splits):
    """
    Yield (train_end, test_end) index pairs into an array of length `n`,
    splitting it into n_splits+1 contiguous blocks and using an expanding
    window: fold k trains on blocks[0:k] and tests on block k (the classic
    walk-forward / forward-chaining scheme, same idea as sklearn's
    TimeSeriesSplit). Never reorders anything -- every train index precedes
    every index in that fold's test block.
    """
    block = n // (n_splits + 1)
    if block < 20:
        raise ValueError(
            f"not enough data ({n} points) for {n_splits} walk-forward splits "
            "(each block would have fewer than 20 points)"
        )
    boundaries = [block * k for k in range(n_splits + 2)]
    boundaries[-1] = n  # last block absorbs any remainder from integer division
    for k in range(1, n_splits + 1):
        yield boundaries[k], boundaries[k + 1]


def validate_regime_edge(historical_bars, n_splits=5, n_permutations=DEFAULT_N_PERMUTATIONS,
                          ou_window=DEFAULT_OU_WINDOW, random_state=0, hmm_kwargs=None):
    """
    Walk-forward test of whether Program 1's decoded regimes actually carry
    the predictive content the Ax sub-signal design assumes: do sideways
    regimes precede mean-reversion toward a calibrated mu more often than a
    random-walk null predicts, and do bull/bear regimes precede continuation
    in their implied direction more often than chance?

    Method: `historical_bars` (a 1D price/level series) is split into
    `n_splits` walk-forward folds (`_walk_forward_folds`, never reordering
    time). For each fold, a fresh `StudentTHMM` is fit on only the data
    before the fold's test block (`hmm_kwargs` are passed through to
    `fit_and_label`, e.g. to use fewer random restarts for a faster test
    run), then regimes are decoded with the causal forward filter
    (`StudentTHMM.filter_states`) over train+test together, and sliced down
    to just the test block. Each test bar's label uses only returns up to
    that bar, and the filter carries the training data's belief state into
    the test block. Viterbi (`predict`) is not used: its path labels an
    early bar using later returns, which would leak future information into
    the edge estimate. For every test
    day with a next bar available: a bull/bear day's "success" is whether
    the next bar's return matches the regime's implied direction; a
    sideways day's "success" is whether it does, relative to a mu calibrated
    by `calibrate_ou_half_life` on a trailing `ou_window`-bar window ending
    at that day (recalibrated per day, not once per fold, since a single
    fold-long calibration can straddle more than one true regime and dilute
    a real reversion signal with unrelated trending stretches -- see this
    function's test suite and docs/writeups/05b_ax_regime_switch.md for the
    empirical comparison). A fold/day where that window's theta <= 0 (no
    valid mean-reverting structure) contributes no reversion sample --
    excluded, not counted as a failure.

    Statistical test: trend_edge and reversion_edge are each (success rate
    - 0.5) pooled across all folds -- the random-walk null is exactly 0.5
    for both (a coin flip on next-bar direction, or on which side of an
    already-historical mu the next bar lands). Significance is assessed by
    a permutation test, per the task spec: 500 shuffles (default) of which
    REGIME LABEL is attached to which (already time-ordered, never
    reshuffled) outcome, rebuilding the null distribution of each edge
    under "regime labels carry no information about what happens next."
    p-values are one-sided (is the real edge unusually high versus the
    shuffled null) with the standard +1 continuity correction.

    Returns a dict: trend_edge, trend_p_value, n_trend_samples,
    reversion_edge, reversion_p_value, n_reversion_samples.

    IMPORTANT FINDING -- read before trusting the reversion p-value. Across
    this module's test suite and the broader investigation in
    docs/writeups/05b_ax_regime_switch.md, the TREND edge comes through
    clearly and robustly: significant (p < 0.01) on data with real
    trend-continuation structure, non-significant on pure random walks.
    The REVERSION edge does not come through as reliably, even on synthetic
    data deliberately built with genuine Ornstein-Uhlenbeck mean-reversion
    during "sideways" stretches. The cause is the regime label's precision on
    this kind of mixed data: on one such fixture, the walk-forward decode
    labels days "sideways" with only 49% precision (a one-in-three chance
    rate), which dilutes a real reversion effect (~58-59% directional
    accuracy when measured against the TRUE block boundaries directly) down
    to statistical noise once it is measured against the HMM's own, noisier,
    boundaries. This is consistent with, and a direct consequence of, this
    project's own earlier finding that Program 1 is weak away from return
    shocks (docs/writeups/01_regime_detection.md). Treat a non-significant
    reversion p-value from this function as a real, informative result about
    THIS system's current regime detector, not evidence the OU half-life
    module or the reversion math themselves are broken.
    """
    from src.regime_detection_robust import fit_and_label
    from src.ou_half_life import calibrate_ou_half_life

    prices = np.asarray(historical_bars, dtype=float)
    rtn = np.diff(np.log(prices))
    hmm_kwargs = {} if hmm_kwargs is None else hmm_kwargs

    pooled_labels = []
    pooled_realized_up = []
    pooled_toward_mu = []

    for train_end, test_end in _walk_forward_folds(len(rtn), n_splits):
        model, state_labels, _ = fit_and_label(rtn[:train_end], **hmm_kwargs)

        hidden = model.filter_states(rtn[:test_end])
        decoded = np.array([state_labels[s] for s in hidden])
        test_labels = decoded[train_end:test_end]

        for offset, j in enumerate(range(train_end, test_end)):
            if j + 1 >= len(rtn):
                continue  # no next bar available to score against
            label = test_labels[offset]
            realized_up = bool(rtn[j + 1] > 0)

            toward_mu = None
            if label == "sideways":
                window_start = max(0, j + 1 - ou_window)
                ou = calibrate_ou_half_life(prices[window_start:j + 2])
                if ou["half_life"] is not None:
                    current_price = prices[j + 1]
                    toward_mu = bool((ou["mu"] - current_price > 0) == (rtn[j + 1] > 0))

            pooled_labels.append(label)
            pooled_realized_up.append(realized_up)
            pooled_toward_mu.append(toward_mu)

    pooled_labels = np.array(pooled_labels)
    pooled_realized_up = np.array(pooled_realized_up)

    def edges_for(labels):
        trend_vals, reversion_vals = [], []
        for lbl, up, tm in zip(labels, pooled_realized_up, pooled_toward_mu):
            if lbl == "bull":
                trend_vals.append(1.0 if up else 0.0)
            elif lbl == "bear":
                trend_vals.append(1.0 if not up else 0.0)
            elif lbl == "sideways" and tm is not None:
                reversion_vals.append(1.0 if tm else 0.0)
        trend_edge = (np.mean(trend_vals) - 0.5) if trend_vals else np.nan
        reversion_edge = (np.mean(reversion_vals) - 0.5) if reversion_vals else np.nan
        return trend_edge, reversion_edge, len(trend_vals), len(reversion_vals)

    observed_trend_edge, observed_reversion_edge, n_trend, n_reversion = edges_for(pooled_labels)

    rng = np.random.default_rng(random_state)
    null_trend_edges = np.empty(n_permutations)
    null_reversion_edges = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled_labels)
        t_edge, r_edge, _, _ = edges_for(shuffled)
        null_trend_edges[i] = t_edge
        null_reversion_edges[i] = r_edge

    def p_value(observed, null):
        valid = null[~np.isnan(null)]
        if np.isnan(observed) or len(valid) == 0:
            return float("nan")
        return float((1 + np.sum(valid >= observed)) / (1 + len(valid)))

    return {
        "trend_edge": float(observed_trend_edge) if not np.isnan(observed_trend_edge) else float("nan"),
        "trend_p_value": p_value(observed_trend_edge, null_trend_edges),
        "n_trend_samples": n_trend,
        "reversion_edge": float(observed_reversion_edge) if not np.isnan(observed_reversion_edge) else float("nan"),
        "reversion_p_value": p_value(observed_reversion_edge, null_reversion_edges),
        "n_reversion_samples": n_reversion,
    }
