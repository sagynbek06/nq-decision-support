"""
Rolling Hurst exponent via Detrended Fluctuation Analysis (DFA).

Estimates how persistent (trending) or anti-persistent (mean-reverting) a
return series currently is, following the standard DFA procedure used
across the long-memory literature (e.g. Asif & Frommel 2022, "Testing long
memory in exchange rates and its implications for the adaptive market
hypothesis," Physica A 593, applies this exact methodology to FX data):

1. Demean the returns and take their cumulative sum (the "integrated
   profile") -- this turns a stationary return series into a random-walk-
   like profile whose scaling behavior is what DFA actually measures.
2. Split the profile into non-overlapping segments at a range of box sizes
   (segmenting from both ends of the series to make use of all the data,
   not just the samples that divide evenly).
3. Fit a linear trend to each segment and compute its root-mean-square
   detrended residual.
4. Average those residuals (in RMS) at each box size to get the
   fluctuation function F(n).
5. The slope of log(F(n)) vs. log(n) is the Hurst exponent: ~0.5 for an
   uncorrelated (random-walk-like) series, significantly above 0.5 for
   persistent/trending series, significantly below for mean-reverting ones.

DESIGN NOTE -- why this is a threshold modulator, not a fourth directional
vote. src/consensus_engine.py already drew exactly this category line for
Program 4's Gamma Exposure: GEX describes how price is expected to *behave*
(dampened or amplified moves), not which way it's going, so it's surfaced
as separate context (`volatility_regime`) rather than folded into the 3-way
directional vote. The Hurst exponent is the same category of signal, for
the same reason: it describes how persistent or mean-reverting the price
process currently is -- genuinely useful information about whether there's
structure worth trusting a directional call on -- but it has no sign that
means "bullish" or "bearish." A market can be strongly trending (high
Hurst) while trending down, or strongly mean-reverting (low Hurst) while
centered above or below its recent range; Hurst alone doesn't say which.
Forcibly assigning it a directional vote would manufacture a claim this
signal doesn't make, exactly the mistake the GEX design note already
rejected. Instead, `compute_consensus_with_hurst` (below) uses Hurst to
modulate *how confident the engine needs to be* before calling a direction
at all -- widening the neutral band when there's little persistent
structure to trade on (Hurst near 0.5), narrowing it when there's a lot
(Hurst far from 0.5 in either direction) -- which is a statement about
signal quality, not a sign-carrying vote.

PERFORMANCE NOTE -- the per-segment linear detrending is done as one
vectorized OLS fit across all segments of a given box size (closed-form
slope/intercept, not a loop of `np.polyfit` calls). Profiling showed the
naive per-segment `np.polyfit` loop costs roughly 60x more than this for a
single DFA call -- a large enough gap to matter once it's run inside a 200-
resample bootstrap or a full rolling-window pass, not just a one-off
estimate.
"""

import numpy as np

import src.consensus_engine as consensus_engine

DEFAULT_MIN_BOX_SIZE = 10
DEFAULT_N_BOX_SIZES = 15
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ROLLING_WINDOW = 252  # one trading year


def _segment_detrended_mse(segments, t, sum_t, sum_t2):
    """
    Closed-form OLS linear detrend + per-segment MSE, vectorized across all
    rows (segments) of a given box size at once -- see PERFORMANCE NOTE.
    """
    box_size = segments.shape[1]
    sum_y = segments.sum(axis=1)
    sum_ty = segments @ t
    denom = box_size * sum_t2 - sum_t ** 2
    slope = (box_size * sum_ty - sum_t * sum_y) / denom
    intercept = (sum_y - slope * sum_t) / box_size
    trend = intercept[:, None] + slope[:, None] * t[None, :]
    residuals = segments - trend
    return np.mean(residuals ** 2, axis=1)


def compute_dfa_hurst(returns, min_box_size=DEFAULT_MIN_BOX_SIZE, max_box_size=None, n_box_sizes=DEFAULT_N_BOX_SIZES):
    """
    Detrended Fluctuation Analysis on a single window of returns.

    `max_box_size` defaults to len(returns) // 4 (the standard DFA rule of
    thumb -- box sizes need to be small enough, relative to the sample,
    that there are several segments to average over). Box sizes are
    logarithmically spaced between `min_box_size` and `max_box_size`.

    Returns (hurst, box_sizes, fluctuations, intercept): the Hurst exponent
    (the log-log regression slope), the box sizes actually used, the
    fluctuation F(n) at each one, and the regression intercept (for anyone
    who wants to inspect or plot the scaling relationship directly).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if max_box_size is None:
        max_box_size = n // 4

    profile = np.cumsum(returns - returns.mean())

    box_sizes = np.unique(
        np.round(np.logspace(np.log10(min_box_size), np.log10(max_box_size), n_box_sizes)).astype(int)
    )
    box_sizes = box_sizes[box_sizes >= 4]  # need enough points per segment to fit a line meaningfully

    fluctuations = np.empty(len(box_sizes))
    for i, box_size in enumerate(box_sizes):
        n_segments = n // box_size
        if n_segments < 2:
            fluctuations[i] = np.nan
            continue

        t = np.arange(box_size, dtype=float)
        sum_t = t.sum()
        sum_t2 = (t ** 2).sum()

        # Segment from both the start and the end of the series so data
        # that doesn't divide evenly by box_size still gets used.
        forward = profile[:n_segments * box_size].reshape(n_segments, box_size)
        backward = profile[n - n_segments * box_size:].reshape(n_segments, box_size)

        mse_forward = _segment_detrended_mse(forward, t, sum_t, sum_t2)
        mse_backward = _segment_detrended_mse(backward, t, sum_t, sum_t2)

        fluctuations[i] = np.sqrt(np.mean(np.concatenate([mse_forward, mse_backward])))

    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    log_sizes = np.log10(box_sizes[valid])
    log_fluctuations = np.log10(fluctuations[valid])

    hurst, intercept = np.polyfit(log_sizes, log_fluctuations, 1)
    return hurst, box_sizes[valid], fluctuations[valid], intercept


def bootstrap_hurst_confidence_interval(
    returns, n_resamples=DEFAULT_N_BOOTSTRAP, block_size=None,
    confidence=DEFAULT_CONFIDENCE, random_state=None, **dfa_kwargs
):
    """
    Moving block bootstrap confidence interval for the DFA Hurst estimate.

    A plain i.i.d. bootstrap (resampling individual return points) would
    destroy the serial dependence structure that DFA is specifically
    trying to measure, making the resulting "confidence interval" close to
    meaningless for a long-memory statistic. Block bootstrap (resampling
    contiguous blocks with replacement and concatenating them back to the
    original length -- the standard technique for bootstrapping time-series
    statistics, Kunsch 1989) at least approximately preserves short-range
    dependence within each block while still producing resampling
    variability.

    `block_size` defaults to `round(2 * n^(1/3))`, a common block-bootstrap
    scaling heuristic; pass your own if you have a specific reason to.

    Returns ((ci_lower, ci_upper), bootstrap_hursts) -- the interval at the
    requested `confidence` level (via percentiles of the bootstrap
    distribution) and the full array of resampled Hurst estimates, for
    anyone who wants to inspect the distribution directly rather than just
    its endpoints.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if block_size is None:
        block_size = max(10, int(round(2 * n ** (1 / 3))))
    block_size = min(block_size, n)

    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    bootstrap_hursts = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resampled = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        hurst, *_ = compute_dfa_hurst(resampled, **dfa_kwargs)
        bootstrap_hursts[i] = hurst

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_hursts, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_hursts, 100 * (1 - alpha / 2)))
    return (ci_lower, ci_upper), bootstrap_hursts


def rolling_dfa_hurst(returns, window=DEFAULT_ROLLING_WINDOW, step=1, **dfa_kwargs):
    """
    DFA Hurst over a trailing window, stepping through the series.

    Returns an array the same length as `returns`, with `np.nan` for the
    first `window - 1` positions (not enough trailing history yet) and,
    where computed, the Hurst estimate using `returns[i - window + 1 : i + 1]`
    placed at index `i` -- i.e. aligned to the *end* of its window, the way
    the rest of this project's rolling signals (order flow, kernel
    regression) are aligned, so it can be joined directly onto a price
    DataFrame by date.

    `step > 1` skips positions between estimates (e.g. `step=5` computes
    roughly every 5th day) to trade estimate density for speed; each
    individual DFA call is cheap after the vectorization described in this
    module's PERFORMANCE NOTE, but a full step=1 pass over a multi-year
    daily series is still hundreds of calls.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    result = np.full(n, np.nan)
    for end in range(window, n + 1, step):
        hurst, *_ = compute_dfa_hurst(returns[end - window:end], **dfa_kwargs)
        result[end - 1] = hurst
    return result


def hurst_to_threshold_multiplier(hurst, confidence_interval, max_multiplier=2.0, min_multiplier=0.5):
    """
    Map a Hurst estimate (and its confidence interval) to a multiplier on
    the consensus engine's NEUTRAL_BAND.

    If the confidence interval includes 0.5, the point estimate isn't
    statistically distinguishable from a random walk at this sample size --
    trusting it anyway would be worse than just acknowledging there's no
    reliable persistence signal here, so the multiplier is forced to
    `max_multiplier` (the widest band) regardless of where the point
    estimate happens to sit.

    Otherwise the multiplier decreases linearly in |hurst - 0.5|, from
    `max_multiplier` at distance 0 down to `min_multiplier` at distance 0.5
    (i.e. hurst at the extreme 0 or 1), so a market with no persistent
    structure requires a stronger consensus score before committing to a
    direction, and a market with strong trending or mean-reverting
    structure requires less.
    """
    ci_lower, ci_upper = confidence_interval
    if ci_lower <= 0.5 <= ci_upper:
        return max_multiplier

    distance = abs(hurst - 0.5)
    normalized_distance = min(distance / 0.5, 1.0)
    return max_multiplier - normalized_distance * (max_multiplier - min_multiplier)


def compute_consensus_with_hurst(
    regime_label, order_flow_signal, kernel_deviation, total_gex, returns,
    weights=None, n_bootstrap=DEFAULT_N_BOOTSTRAP, confidence=DEFAULT_CONFIDENCE,
    random_state=None, **dfa_kwargs
):
    """
    Thin wrapper around consensus_engine.compute_consensus that widens or
    narrows its neutral band based on the current rolling Hurst exponent,
    without changing compute_consensus's signature or internals.

    MECHANISM -- compute_consensus reads NEUTRAL_BAND as a module-level
    constant, not a function parameter (deliberately: that function's
    signature and behavior are unchanged here, per the consensus engine
    writeup's standard of not modifying existing tested code to bolt on a
    new feature). This wrapper computes the Hurst-adjusted band, temporarily
    reassigns `consensus_engine.NEUTRAL_BAND` to that value for the
    duration of one call, and restores the original value in a `finally`
    block regardless of whether the call succeeds or raises. This is NOT
    thread-safe -- concurrent calls to this wrapper, or to
    consensus_engine.compute_consensus directly, from different threads
    could observe the wrong NEUTRAL_BAND mid-call. That's an acceptable
    tradeoff for this single-threaded research/analysis codebase; revisit
    this mechanism (e.g. by threading an explicit band parameter through
    compute_consensus instead) before using it anywhere concurrent.

    `returns` is the return series DFA and its bootstrap CI are computed
    on -- typically a trailing window ending "today," the same day the
    other four arguments describe.

    Returns the same dict shape as compute_consensus, plus "hurst" (the
    point estimate) and "effective_neutral_band" (the actual band used).
    """
    returns = np.asarray(returns, dtype=float)
    hurst, *_ = compute_dfa_hurst(returns, **dfa_kwargs)
    confidence_interval, _ = bootstrap_hurst_confidence_interval(
        returns, n_resamples=n_bootstrap, confidence=confidence, random_state=random_state, **dfa_kwargs
    )
    multiplier = hurst_to_threshold_multiplier(hurst, confidence_interval)
    effective_neutral_band = consensus_engine.NEUTRAL_BAND * multiplier

    original_neutral_band = consensus_engine.NEUTRAL_BAND
    try:
        consensus_engine.NEUTRAL_BAND = effective_neutral_band
        result = consensus_engine.compute_consensus(
            regime_label, order_flow_signal, kernel_deviation, total_gex, weights=weights
        )
    finally:
        consensus_engine.NEUTRAL_BAND = original_neutral_band

    result["hurst"] = hurst
    result["effective_neutral_band"] = effective_neutral_band
    return result
