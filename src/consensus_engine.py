"""
Consensus Engine (Phase 5).

Combines standardized signals from Programs 1-3 into a single weighted
directional vote, with Program 4's Gamma Exposure (GEX) surfaced separately
as volatility-regime context rather than folded into that vote.

DESIGN NOTE -- why GEX isn't a fourth directional vote. Programs 1-3 each
produce something that's naturally a directional read: a regime label
(bull/bear/sideways), a signed order-flow pressure, a signed deviation from
a price trend. Program 4's GEX is not that kind of signal -- it describes
how dealer hedging flows are expected to *behave* (dampening moves when
dealers are net long gamma, amplifying them when net short), which is a
statement about expected volatility/mean-reversion-of-price-action, not
about direction. Treating a GEX sign as "bullish" or "bearish" the way the
other three are would be manufacturing a directional claim the signal
doesn't actually make. Instead, `compute_consensus` returns GEX as a
separate `volatility_regime` field alongside the Programs-1-3 directional
consensus -- still "a single decision-support view" per ROADMAP.md, just
one that's honest about which of its inputs says what.

This module is intentionally decoupled from Programs 1-4: it takes already-
computed signal values (a regime label, an order-flow signal, a kernel
deviation, a GEX total) rather than importing and calling those modules
itself. The caller runs each program and passes in its output; see
tests/test_consensus_engine.py's end-to-end test for what that looks like
wired together.

INTERPRETIVE CHOICE -- the kernel regression deviation is read as a
mean-reversion signal (price trading above its local trend votes bearish,
below votes bullish), the conventional reading for a distance-from-smoothed-
trend indicator (in the spirit of Bollinger-Bands-style mean reversion).
The opposite (momentum) reading is equally defensible in principle and
would just flip that vote's sign -- this module picks one and documents it
rather than pretending there's no choice being made. Whichever reading is
actually predictive is an empirical question for backtesting, not something
assumed here; Phase 7's walk-forward framework is where that would get
tested, along with tuning the vote weights below (also currently equal by
default, not fit to data).
"""

import numpy as np

REGIME_VOTES = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}

DEFAULT_WEIGHTS = {"regime": 1.0, "order_flow": 1.0, "kernel": 1.0}

NEUTRAL_BAND = 0.15  # |weighted score| below this is reported as "neutral"


def vote_from_regime(regime_label):
    """Program 1: bull -> +1, bear -> -1, sideways -> 0."""
    if regime_label not in REGIME_VOTES:
        raise ValueError(f"unknown regime label {regime_label!r}, expected one of {set(REGIME_VOTES)}")
    return REGIME_VOTES[regime_label]


def vote_from_regime_subsignal(sub_signal):
    """
    Program 1 (Ax-style upgrade, opt-in): a regime-conditioned trend/mean-
    reversion sub-signal from `src.ax_regime_switch.ax_regime_subsignal`,
    in place of the bare regime label `vote_from_regime` reads. That
    function already scales its output to [-1, 1] itself (see its
    docstring), so this is a pass-through clip, not a second squashing
    convention -- the same relationship `vote_from_regime` has to its own
    already-bounded `REGIME_VOTES` dict.
    """
    return float(np.clip(sub_signal, -1.0, 1.0))


def vote_from_order_flow(signal, scale=2.0):
    """
    Program 2: the MP-filtered composite order-flow signal (roughly
    standardized) read directly -- positive means net buying pressure.
    Squashed through tanh so a vote is always in [-1, 1] and large
    magnitudes saturate rather than dominating a weighted average outright.
    """
    return float(np.tanh(signal / scale))


def vote_from_kernel_deviation(deviation, scale=2.0):
    """
    Program 3: the standardized deviation of price from its kernel
    regression trend, read as mean-reversion -- price above trend votes
    bearish, below votes bullish (negative sign; see module docstring's
    INTERPRETIVE CHOICE). Same tanh squashing as the order-flow vote.
    """
    return float(-np.tanh(deviation / scale))


def volatility_regime_from_gex(total_gex):
    """
    Program 4: dealers net long gamma (GEX > 0) are expected to hedge in a
    way that dampens realized volatility (buying dips, selling rallies);
    net short gamma (GEX < 0) is expected to amplify moves (selling into
    drops, buying into rallies). GEX == 0 is reported as "dampening" (the
    boundary has to go somewhere; it's an arbitrary tie-break, not a claim
    that zero GEX has special dampening power).
    """
    return "dampening" if total_gex >= 0 else "amplifying"


def compute_consensus(regime_label, order_flow_signal, kernel_deviation, total_gex, weights=None,
                       regime_mode="label", regime_subsignal=None):
    """
    Combine Programs 1-3 into a weighted directional consensus, and attach
    Program 4's GEX as separate volatility-regime context (see module
    docstring's DESIGN NOTE for why it isn't folded into the vote).

    `weights` defaults to equal weighting across the three directional
    votes (DEFAULT_WEIGHTS); pass a dict with the same keys
    ("regime", "order_flow", "kernel") to override. Weights are normalized
    internally, so relative magnitudes are what matter, not their scale.

    `regime_mode` picks how Program 1 feeds the "regime" vote:
    - "label" (default): `vote_from_regime(regime_label)`, the original
      bare bull/bear/sideways -> +1/-1/0 mapping. Unchanged default
      behavior, so existing callers are unaffected.
    - "subsignal": `vote_from_regime_subsignal(regime_subsignal)` instead
      -- the Ax-style regime-conditioned trend/mean-reversion sub-signal
      from `src.ax_regime_switch.ax_regime_subsignal` (see
      docs/writeups/05b_ax_regime_switch.md), passed in by the caller via
      `regime_subsignal`. `regime_label` is still required but unused in
      this mode; ax_regime_subsignal needs it anyway to decide what to
      compute `regime_subsignal` from, so the caller has it regardless.

    Returns a dict with the per-program votes, the weighted consensus
    score (in [-1, 1]), a "bullish"/"bearish"/"neutral" label (score within
    +/-NEUTRAL_BAND of zero reads as neutral), and the GEX-derived
    volatility regime.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    missing = set(DEFAULT_WEIGHTS) - set(weights)
    if missing:
        raise ValueError(f"weights missing keys: {missing}")

    if regime_mode == "label":
        regime_vote = vote_from_regime(regime_label)
    elif regime_mode == "subsignal":
        if regime_subsignal is None:
            raise ValueError("compute_consensus: regime_mode='subsignal' requires regime_subsignal")
        regime_vote = vote_from_regime_subsignal(regime_subsignal)
    else:
        raise ValueError(f"unknown regime_mode {regime_mode!r}, expected 'label' or 'subsignal'")

    votes = {
        "regime": regime_vote,
        "order_flow": vote_from_order_flow(order_flow_signal),
        "kernel": vote_from_kernel_deviation(kernel_deviation),
    }

    total_weight = sum(weights[k] for k in votes)
    if total_weight <= 0:
        raise ValueError("sum of weights must be positive")
    score = sum(weights[k] * votes[k] for k in votes) / total_weight

    if score > NEUTRAL_BAND:
        label = "bullish"
    elif score < -NEUTRAL_BAND:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "votes": votes,
        "weights": dict(weights),
        "score": score,
        "label": label,
        "volatility_regime": volatility_regime_from_gex(total_gex),
        "total_gex": total_gex,
    }
