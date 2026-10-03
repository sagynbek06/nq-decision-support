"""
Weighted Surprise Score (WSS) -- "nq-surprise-regime".

A macro-event "surprise index" signal: when an economic release lands away
from consensus expectations, that surprise's market impact depends on more
than just its raw size. This module combines four terms into one score:

    WSS = S * W(event) * R(regime) * D(time_decay)

- S: the standardized surprise itself (how many standard deviations the
  actual reading landed from consensus, sign following the metric's own
  natural direction -- e.g. S=+2.0 for CPI means "inflation came in 2
  std devs hotter than expected," independent of what that means for
  markets).
- W(event): per-event-type weight, `EVENT_TYPE_PARAMS[event_type]`,
  combining an importance scalar with a *direction* sign -- see DESIGN
  NOTE below for why this carries direction at all.
- R(regime): a regime-conditioned amplifier/dampener (`regime_modulator`):
  a surprise whose implied direction agrees with the prevailing HMM regime
  gets amplified (confirms the trend), one that conflicts gets dampened
  (gets partially faded), and sideways regimes are neutral either way.
- D(time_decay): a per-event-type half-life decay (`time_decay`) -- a
  surprise's market impact fades over the days following the release, not
  instantly and not forever.

DESIGN NOTE -- why WSS is an opt-in fourth vote, not just context like
GEX/Hurst. `consensus_engine.py`'s own DESIGN NOTE excludes GEX from the
directional vote because GEX describes expected price *behavior*
(dampened/amplified moves), not a direction -- there's no sign that means
"bullish." The Hurst exponent (docs/writeups/05a_hurst_exponent.md) is
excluded for the identical reason: it describes trend persistence, not
which way the trend runs. A macro surprise is different in kind: "jobs
report beat consensus" or "inflation ran hotter than expected" are each
widely understood, in practice, to carry a conventional directional
market implication (this is exactly what `EVENT_TYPE_PARAMS`'s
`direction` field encodes). That's a real, documented, if simplified,
directional claim the signal itself makes -- unlike GEX or Hurst, where
forcing a direction would be manufacturing one. So WSS is BOTH: always
surfaced as `surprise_context` (the GEX/Hurst-style always-present
context field, via `compute_surprise_context`), AND available as an
opt-in fourth vote in `consensus_engine.compute_consensus`
(`include_surprise=True`), the same opt-in shape Phase 5b's `regime_mode`
uses -- never a silent default-on change, since `DEFAULT_WEIGHTS` and
`compute_consensus`'s required-weight-keys check are both left untouched.

A caveat on that directional claim, stated as plainly as the choice itself:
`EVENT_TYPE_PARAMS`'s `direction` values are ONE conventional reading (e.g.
"CPI beat = hotter inflation = bearish via tighter-policy expectations"),
not a law of markets. Real markets sometimes invert this (a "good news is
bad news" regime, where strong jobs data gets read as reducing the odds of
rate cuts and sells off) -- picking one documented convention and flagging
it, rather than pretending there's no judgment call, mirrors
`consensus_engine.py`'s own INTERPRETIVE CHOICE for the kernel-deviation
vote's mean-reversion-vs-momentum reading. Whether this convention is
actually predictive is the same kind of empirical, Phase-7 question as
everything else in `consensus_engine.py`'s vote weights.

Decoupling: like `consensus_engine.py` itself, this module does not import
or call `src.regime_detection_robust` -- it takes an already-decoded
`regime_label` (bull/bear/sideways) as a parameter. The caller is expected
to have sourced that from `regime_detection_robust.fit_and_label` /
`predict_regimes`, exactly as `consensus_engine.compute_consensus` already
expects for its own `regime_label` argument.
"""

import numpy as np
import pandas as pd

# direction: does a POSITIVE standardized surprise (metric above
# consensus) conventionally read as bullish (+1) or bearish (-1) for an
# equity index like NQ? importance: relative market-moving weight across
# event types. half_life_days: how fast the market's reaction to this
# event type typically fades. See module DESIGN NOTE for the caveat that
# `direction` is one documented convention, not a guaranteed market law.
EVENT_TYPE_PARAMS = {
    "NFP":  {"direction": +1.0, "importance": 1.0, "half_life_days": 3.0},
    "CPI":  {"direction": -1.0, "importance": 1.0, "half_life_days": 4.0},
    "FOMC": {"direction": -1.0, "importance": 1.2, "half_life_days": 7.0},
    "GDP":  {"direction": +1.0, "importance": 0.8, "half_life_days": 3.0},
}

REGIME_BIAS = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}
REGIME_ALIGNMENT_BONUS = 0.3  # R(regime) in [1-bonus, 1+bonus]; see regime_modulator

DEFAULT_DECAY_CUTOFF_HALF_LIVES = 5.0  # beyond this, D(time_decay) < 0.0313 -> treated as inactive
DEFAULT_WSS_VOTE_SCALE = 2.0  # matches vote_from_order_flow / vote_from_kernel_deviation's tanh scale

# A small, clearly illustrative set of annotated events -- NOT real
# historical economic data (this project has no live data feed; see
# ROADMAP.md Phase 6). Dates are deliberately recent/relative so the
# module's docstring examples and tests can exercise realistic age_days
# values without depending on wall-clock time when this file is read.
DEFAULT_EVENT_DATABASE = pd.DataFrame([
    {"event_date": pd.Timestamp("2024-01-05"), "event_type": "NFP", "surprise": 1.8},
    {"event_date": pd.Timestamp("2024-01-11"), "event_type": "CPI", "surprise": 0.9},
    {"event_date": pd.Timestamp("2024-01-31"), "event_type": "FOMC", "surprise": -0.6},
    {"event_date": pd.Timestamp("2024-02-29"), "event_type": "GDP", "surprise": 1.2},
])


def time_decay(age_days, half_life_days, cutoff_half_lives=DEFAULT_DECAY_CUTOFF_HALF_LIVES):
    """
    D(time_decay): standard half-life decay, 0.5**(age_days/half_life_days)
    -- exactly 1.0 at the moment of release, exactly 0.5 one half-life
    later. Returns 0.0 (event treated as no longer "active") for an event
    more than `cutoff_half_lives` half-lives in the past (by
    `cutoff_half_lives=5.0`, D has already decayed below 0.032), or for a
    negative age (a future event can't yet have a market impact).
    """
    if age_days < 0 or age_days > cutoff_half_lives * half_life_days:
        return 0.0
    return float(0.5 ** (age_days / half_life_days))


def regime_modulator(implied_direction_sign, regime_label):
    """
    R(regime): amplifies a surprise whose implied direction agrees with
    the prevailing regime's bias (REGIME_BIAS: bull=+1, bear=-1,
    sideways=0) by REGIME_ALIGNMENT_BONUS, dampens one that conflicts by
    the same amount, and is neutral (1.0) for a sideways regime (bias=0)
    regardless of the surprise's direction -- there's no trend for a
    surprise to confirm or fight in a sideways market. Always in
    [1-REGIME_ALIGNMENT_BONUS, 1+REGIME_ALIGNMENT_BONUS], so it only ever
    scales WSS's magnitude, never flips its sign.
    """
    if regime_label not in REGIME_BIAS:
        raise ValueError(f"unknown regime label {regime_label!r}, expected one of {set(REGIME_BIAS)}")
    alignment = implied_direction_sign * REGIME_BIAS[regime_label]
    return 1.0 + REGIME_ALIGNMENT_BONUS * alignment


def event_weight(event_type):
    """W(event): direction * importance for this event type."""
    if event_type not in EVENT_TYPE_PARAMS:
        raise ValueError(f"unknown event_type {event_type!r}, expected one of {set(EVENT_TYPE_PARAMS)}")
    params = EVENT_TYPE_PARAMS[event_type]
    return params["direction"] * params["importance"]


def compute_wss(surprise, event_type, age_days, regime_label):
    """WSS = S * W(event) * R(regime) * D(time_decay) for a single event."""
    w = event_weight(event_type)
    implied_direction = float(np.sign(surprise * w))
    r = regime_modulator(implied_direction, regime_label)
    d = time_decay(age_days, EVENT_TYPE_PARAMS[event_type]["half_life_days"])
    return float(surprise * w * r * d)


def compute_surprise_context(regime_label, current_time, event_database=None,
                              cutoff_half_lives=DEFAULT_DECAY_CUTOFF_HALF_LIVES):
    """
    Scan `event_database` for every event still inside its active
    time_decay window as of `current_time`, and combine them into the
    volatility_regime/Hurst-style always-present context dict:
    {"active": bool, "wss": float, "events": [...]}.

    If more than one event is concurrently active, their WSS contributions
    are summed -- a documented choice (a "surprise index" aggregating
    concurrent surprises), not picking only the single largest one.
    `events` lists every contributing event with its own age and WSS, for
    anyone inspecting why the aggregate came out the way it did.

    `event_database` defaults to `DEFAULT_EVENT_DATABASE` (illustrative
    example data, not a live feed -- see that constant's docstring);
    pass your own DataFrame (same columns: event_date, event_type,
    surprise) to use real annotated events.
    """
    event_database = DEFAULT_EVENT_DATABASE if event_database is None else event_database
    current_time = pd.Timestamp(current_time)

    active_events = []
    total_wss = 0.0
    for _, row in event_database.iterrows():
        event_type = row["event_type"]
        half_life_days = EVENT_TYPE_PARAMS[event_type]["half_life_days"]
        age_days = (current_time - pd.Timestamp(row["event_date"])) / pd.Timedelta(days=1)
        if not (0 <= age_days <= cutoff_half_lives * half_life_days):
            continue

        wss = compute_wss(row["surprise"], event_type, age_days, regime_label)
        active_events.append({
            "event_type": event_type,
            "event_date": pd.Timestamp(row["event_date"]),
            "surprise": float(row["surprise"]),
            "age_days": float(age_days),
            "wss": wss,
        })
        total_wss += wss

    return {
        "active": len(active_events) > 0,
        "wss": float(total_wss),
        "events": active_events,
    }


def vote_from_wss(wss, scale=DEFAULT_WSS_VOTE_SCALE):
    """
    Opt-in fourth vote: tanh-squash WSS to [-1, 1], matching
    `vote_from_order_flow` / `vote_from_kernel_deviation`'s scale (default
    `scale=2.0`, the same default those two use). An inactive/zero WSS
    (no event currently in its decay window) votes exactly 0.0 -- neutral,
    not a missing value -- so this always returns a well-formed vote.
    """
    return float(np.tanh(wss / scale))
