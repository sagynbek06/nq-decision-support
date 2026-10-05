"""Tests for dynamic per-vote accuracy weighting (src/dynamic_weights.py)."""

import numpy as np
import pytest

from src.dynamic_weights import (
    resolved_accuracy,
    compute_dynamic_weights,
    MIN_WEIGHT_FLOOR,
)


# ---------------------------------------------------------------------------
# resolved_accuracy
# ---------------------------------------------------------------------------

def test_resolved_accuracy_is_one_when_vote_always_matches_outcome_sign():
    votes = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    outcomes = np.array([0.5, -0.3, 0.2, 0.1, -0.4])
    assert resolved_accuracy(votes, outcomes) == 1.0


def test_resolved_accuracy_is_zero_when_vote_always_opposes_outcome_sign():
    votes = np.array([1.0, -1.0, 1.0])
    outcomes = np.array([-0.5, 0.3, -0.2])
    assert resolved_accuracy(votes, outcomes) == 0.0


def test_resolved_accuracy_uses_only_the_trailing_window():
    # first 5 entries: always wrong; last 5: always right
    votes = np.array([1.0] * 10)
    outcomes = np.array([-1.0] * 5 + [1.0] * 5)
    assert resolved_accuracy(votes, outcomes, window=5) == 1.0
    assert resolved_accuracy(votes, outcomes, window=10) == 0.5


def test_resolved_accuracy_never_credits_an_abstaining_vote():
    """A vote of exactly 0.0 abstains: it never counts as a hit, even when
    the realized outcome is also exactly zero."""
    votes = np.array([0.0, 0.0, 0.0])
    outcomes = np.array([0.0, 0.5, -0.5])
    assert resolved_accuracy(votes, outcomes) == 0.0


def test_resolved_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        resolved_accuracy([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# compute_dynamic_weights
# ---------------------------------------------------------------------------

def test_compute_dynamic_weights_only_returns_keys_for_active_votes():
    outcomes = np.array([1.0, -1.0] * 50)
    vote_histories = {"regime": outcomes.copy(), "order_flow": outcomes.copy()}
    weights = compute_dynamic_weights(vote_histories, outcomes)
    assert set(weights) == {"regime", "order_flow"}


def test_compute_dynamic_weights_gives_near_floor_weight_to_a_degraded_vote():
    """
    The explicit scenario requested: Program 1's regime vote is documented
    to be weak away from return shocks (docs/writeups/01_regime_detection.md,
    05b_ax_regime_switch.md). Simulate that with a regime vote
    uncorrelated with outcomes (an artificially degraded ~50% hit rate)
    alongside an order_flow vote that's genuinely predictive (~80% hit
    rate by construction), and confirm the degraded vote's computed
    weight sits near the floor while the predictive one gets substantially
    more -- i.e. it does NOT dominate purely by being present every bar.
    """
    rng = np.random.default_rng(0)
    n = 200
    outcomes = rng.choice([-1.0, 1.0], size=n)

    degraded_regime_vote = rng.choice([-1.0, 1.0], size=n)  # independent of outcomes
    agrees = rng.random(n) < 0.8
    predictive_order_flow_vote = np.where(agrees, outcomes, -outcomes)

    weights = compute_dynamic_weights(
        {"regime": degraded_regime_vote, "order_flow": predictive_order_flow_vote},
        outcomes, window=n,
    )

    assert weights["regime"] < 0.1  # near the floor (empirically ~0.055 at this seed)
    assert weights["order_flow"] > weights["regime"] * 3


def test_compute_dynamic_weights_tracks_a_recent_accuracy_change_not_the_full_history():
    """Confirms the 'rolling' part: a vote that WAS accurate for a long
    stretch but has recently gone consistently wrong should be
    down-weighted based on its RECENT (windowed) record, not a stale
    full-history average that would still look good."""
    n_good, n_bad = 150, 50
    outcomes = np.array([1.0] * (n_good + n_bad))
    vote = np.array([1.0] * n_good + [-1.0] * n_bad)  # flips to always-wrong at the end

    weights = compute_dynamic_weights({"regime": vote}, outcomes, window=50)
    assert weights["regime"] == pytest.approx(MIN_WEIGHT_FLOOR)  # last 50: all wrong


def test_compute_dynamic_weights_never_goes_to_zero_or_negative():
    outcomes = np.array([1.0, -1.0] * 50)
    always_wrong_vote = -outcomes
    weights = compute_dynamic_weights({"regime": always_wrong_vote}, outcomes)
    assert weights["regime"] > 0


def test_compute_dynamic_weights_output_is_directly_usable_by_compute_consensus():
    """Integration check: the dict this returns can be passed straight
    into compute_consensus's weights= parameter without modification."""
    from src.consensus_engine import compute_consensus

    outcomes = np.array([1.0, -1.0] * 50)
    vote_histories = {
        "regime": outcomes.copy(),
        "order_flow": -outcomes.copy(),
        "kernel": outcomes.copy(),
    }
    weights = compute_dynamic_weights(vote_histories, outcomes)
    result = compute_consensus("bull", 1.0, -1.0, total_gex=0.0, weights=weights)
    assert result["weights"] == weights


def test_compute_dynamic_weights_surprise_key_is_read_automatically_by_compute_consensus():
    """A single dynamically-computed weights dict covering all 4 possible
    votes can be passed straight through -- compute_consensus reads its
    'surprise' entry in place of a separate surprise_weight argument."""
    from src.consensus_engine import compute_consensus

    outcomes = np.array([1.0, -1.0] * 50)
    vote_histories = {
        "regime": outcomes.copy(), "order_flow": outcomes.copy(), "kernel": outcomes.copy(),
        "surprise": outcomes.copy(),
    }
    weights = compute_dynamic_weights(vote_histories, outcomes)
    context = {"active": True, "wss": 2.0, "events": []}

    result = compute_consensus(
        "bull", 1.0, -1.0, total_gex=0.0, weights=weights,
        surprise_context=context, include_surprise=True,
    )
    assert result["weights"]["surprise"] == pytest.approx(weights["surprise"])


def test_dynamic_weights_are_normalized_like_the_existing_weights_dict():
    """compute_consensus normalizes by the weight sum, so multiplying every
    dynamic weight by the same constant must leave the score unchanged --
    the same normalization contract the static DEFAULT_WEIGHTS already has."""
    from src.consensus_engine import compute_consensus

    outcomes = np.array([1.0, -1.0] * 50)
    weights = compute_dynamic_weights(
        {"regime": outcomes.copy(), "order_flow": -outcomes.copy(), "kernel": outcomes.copy()},
        outcomes,
    )
    scaled = {k: 10.0 * v for k, v in weights.items()}

    base = compute_consensus("bull", 1.0, -0.5, total_gex=0.0, weights=weights)
    rescaled = compute_consensus("bull", 1.0, -0.5, total_gex=0.0, weights=scaled)
    assert rescaled["score"] == pytest.approx(base["score"])
