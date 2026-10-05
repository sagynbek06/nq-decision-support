"""
Dynamic per-vote accuracy weighting for the Consensus Engine.

`consensus_engine.DEFAULT_WEIGHTS` is a fixed equal weighting across
whichever directional votes are active -- deliberately not fit to data
(see that module's docstring). This module computes an ALTERNATIVE
weights dict from each vote's own recent track record instead: a vote
that's been right more often than a coin flip over a trailing window gets
more say; one that hasn't carries less, regardless of how often it's been
present. Feed its output straight into `compute_consensus(weights=...)` --
nothing in that function changes; this only computes a different dict to
hand it. A "surprise" entry in the returned dict is picked up automatically
by `compute_consensus(..., include_surprise=True)` without a separate
`surprise_weight` argument (see that function's docstring).

Motivation, concretely: Program 1's regime vote is documented to be weak
away from return shocks (docs/writeups/01_regime_detection.md: 42.4%
accuracy on a 3-way call even for the improved Student-t model;
docs/writeups/05b_ax_regime_switch.md: the Ax sideways/reversion
sub-signal doesn't reliably hold up walk-forward either). A fixed equal
weight gives that vote the same say as a more reliable one regardless of
this -- the only reason not to is that nothing in `compute_consensus`
otherwise looks at how any vote has actually been doing. This module is
that look, and `tests/test_dynamic_weights.py` checks directly that a
degraded regime vote's weight drops accordingly rather than coasting on
being present every bar.
"""

import numpy as np

DEFAULT_WINDOW = 100
MIN_WEIGHT_FLOOR = 0.01
CHANCE_ACCURACY = 0.5  # baseline for a binary next-bar-direction call


def resolved_accuracy(vote_values, realized_outcomes, window=DEFAULT_WINDOW):
    """
    Fraction of the trailing `window` (vote, outcome) pairs -- or all of
    them, if fewer than `window` are available -- where the vote's sign
    matched the realized outcome's sign. "Resolved" means both the vote
    and the outcome it's being checked against are already known: this
    never looks past the end of the arrays it's given, so rolling it
    forward bar by bar (as the companion notebook does) means calling it
    again with a slice that ends one bar later each time, not passing the
    whole future series at once. A vote of exactly 0.0 never counts as
    correct -- there's no direction in it to have gotten right.
    """
    vote_values = np.asarray(vote_values, dtype=float)
    realized_outcomes = np.asarray(realized_outcomes, dtype=float)
    if len(vote_values) != len(realized_outcomes):
        raise ValueError("resolved_accuracy: vote_values and realized_outcomes must be the same length")
    if len(vote_values) == 0:
        return float("nan")

    recent_votes = vote_values[-window:]
    recent_outcomes = realized_outcomes[-window:]
    vote_signs = np.sign(recent_votes)
    correct = (vote_signs != 0) & (vote_signs == np.sign(recent_outcomes))
    return float(np.mean(correct))


def compute_dynamic_weights(vote_histories, realized_outcomes, window=DEFAULT_WINDOW,
                             min_weight=MIN_WEIGHT_FLOOR):
    """
    `vote_histories`: {vote_name: array of that vote's past values}, for
    whichever votes are currently active -- only the keys present get a
    weight back, so an inactive vote (e.g. "surprise" when
    `include_surprise=False`) is simply omitted, not zeroed.
    `realized_outcomes`: the actual next-bar outcome (e.g. return),
    aligned 1:1 and the same length as every array in `vote_histories`.

    Each vote's weight is `max(accuracy - CHANCE_ACCURACY, 0) + min_weight`
    -- the excess of its trailing accuracy over a coin flip, floored at
    `min_weight` so the result can never sum to zero or go negative
    (`compute_consensus` already requires a positive weight sum) even if
    every vote's recent accuracy happens to be at or below chance.

    Returns a dict with the same keys as `vote_histories`, ready to pass
    directly as `compute_consensus`'s `weights=` argument.
    """
    weights = {}
    for name, values in vote_histories.items():
        accuracy = resolved_accuracy(values, realized_outcomes, window=window)
        weights[name] = max(accuracy - CHANCE_ACCURACY, 0.0) + min_weight
    return weights
