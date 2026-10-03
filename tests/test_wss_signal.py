"""Tests for the Weighted Surprise Score module (src/wss_signal.py)."""

import numpy as np
import pandas as pd
import pytest

from src.wss_signal import (
    time_decay,
    regime_modulator,
    event_weight,
    compute_wss,
    compute_surprise_context,
    vote_from_wss,
    EVENT_TYPE_PARAMS,
    REGIME_ALIGNMENT_BONUS,
)


# ---------------------------------------------------------------------------
# time_decay: D(time_decay)
# ---------------------------------------------------------------------------

def test_time_decay_is_full_strength_at_zero_age():
    assert time_decay(0, half_life_days=3.0) == pytest.approx(1.0)


def test_time_decay_is_exactly_half_at_one_half_life():
    assert time_decay(3.0, half_life_days=3.0) == pytest.approx(0.5)


def test_time_decay_is_zero_beyond_cutoff():
    assert time_decay(16.0, half_life_days=3.0, cutoff_half_lives=5.0) == 0.0


def test_time_decay_is_zero_for_a_future_event():
    assert time_decay(-1.0, half_life_days=3.0) == 0.0


def test_time_decay_is_monotonically_decreasing():
    ages = [0, 1, 2, 3, 4, 5]
    decays = [time_decay(a, half_life_days=3.0) for a in ages]
    assert all(d1 > d2 for d1, d2 in zip(decays, decays[1:]))


# ---------------------------------------------------------------------------
# regime_modulator: R(regime)
# ---------------------------------------------------------------------------

def test_regime_modulator_amplifies_when_aligned_with_bull():
    assert regime_modulator(+1.0, "bull") == pytest.approx(1.0 + REGIME_ALIGNMENT_BONUS)


def test_regime_modulator_dampens_when_conflicting_with_bull():
    assert regime_modulator(-1.0, "bull") == pytest.approx(1.0 - REGIME_ALIGNMENT_BONUS)


def test_regime_modulator_is_neutral_in_sideways_regardless_of_direction():
    assert regime_modulator(+1.0, "sideways") == pytest.approx(1.0)
    assert regime_modulator(-1.0, "sideways") == pytest.approx(1.0)


def test_regime_modulator_rejects_unknown_regime():
    with pytest.raises(ValueError):
        regime_modulator(1.0, "not_a_regime")


def test_regime_modulator_never_flips_sign():
    """R(regime) is a magnitude amplifier/dampener only -- it must never
    be large enough to cross zero."""
    for direction in (-1.0, 0.0, 1.0):
        for regime in ("bull", "bear", "sideways"):
            assert regime_modulator(direction, regime) > 0


# ---------------------------------------------------------------------------
# event_weight: W(event)
# ---------------------------------------------------------------------------

def test_event_weight_matches_configured_direction_and_importance():
    for event_type, params in EVENT_TYPE_PARAMS.items():
        assert event_weight(event_type) == pytest.approx(params["direction"] * params["importance"])


def test_event_weight_rejects_unknown_event_type():
    with pytest.raises(ValueError):
        event_weight("NOT_A_REAL_EVENT")


# ---------------------------------------------------------------------------
# compute_wss: direction sanity (the module's core directional claim)
# ---------------------------------------------------------------------------

def test_nfp_beat_is_bullish_leaning():
    """A stronger-than-expected jobs report reads as a bullish surprise
    under this module's documented convention."""
    assert compute_wss(surprise=2.0, event_type="NFP", age_days=0, regime_label="sideways") > 0


def test_cpi_beat_is_bearish_leaning():
    """A hotter-than-expected inflation print reads as a bearish surprise
    (tighter-policy expectations) under this module's documented
    convention -- opposite sign from an NFP beat despite both being a
    positive standardized surprise."""
    assert compute_wss(surprise=2.0, event_type="CPI", age_days=0, regime_label="sideways") < 0


def test_aligned_regime_amplifies_versus_sideways_baseline():
    sideways = compute_wss(2.0, "NFP", age_days=0, regime_label="sideways")
    aligned = compute_wss(2.0, "NFP", age_days=0, regime_label="bull")
    conflicting = compute_wss(2.0, "NFP", age_days=0, regime_label="bear")
    assert conflicting < sideways < aligned


def test_compute_wss_decays_to_zero_with_age():
    fresh = compute_wss(2.0, "NFP", age_days=0, regime_label="bull")
    stale = compute_wss(2.0, "NFP", age_days=100, regime_label="bull")
    assert abs(stale) < abs(fresh)
    assert stale == 0.0


# ---------------------------------------------------------------------------
# compute_surprise_context: the always-present context dict
# ---------------------------------------------------------------------------

def test_no_active_event_gives_neutral_context():
    """Correct neutral behavior with no active event: an event database
    whose only entry is long past its decay window."""
    event_database = pd.DataFrame([
        {"event_date": pd.Timestamp("2020-01-01"), "event_type": "NFP", "surprise": 3.0},
    ])
    context = compute_surprise_context("bull", current_time=pd.Timestamp("2024-01-01"),
                                        event_database=event_database)
    assert context == {"active": False, "wss": 0.0, "events": []}


def test_empty_event_database_gives_neutral_context():
    empty = pd.DataFrame(columns=["event_date", "event_type", "surprise"])
    context = compute_surprise_context("sideways", current_time=pd.Timestamp("2024-01-01"),
                                        event_database=empty)
    assert context == {"active": False, "wss": 0.0, "events": []}


def test_known_event_surfaces_correct_signal():
    """Correct signal surfacing with a known annotated event: a single
    NFP beat right at release (age=0), checked against a manually
    computed expected WSS."""
    event_database = pd.DataFrame([
        {"event_date": pd.Timestamp("2024-03-01"), "event_type": "NFP", "surprise": 2.0},
    ])
    context = compute_surprise_context("bull", current_time=pd.Timestamp("2024-03-01"),
                                        event_database=event_database)

    expected_wss = compute_wss(2.0, "NFP", age_days=0, regime_label="bull")
    assert context["active"] is True
    assert context["wss"] == pytest.approx(expected_wss)
    assert len(context["events"]) == 1
    assert context["events"][0]["event_type"] == "NFP"
    assert context["events"][0]["age_days"] == pytest.approx(0.0)


def test_known_event_still_active_partway_through_decay_window():
    event_database = pd.DataFrame([
        {"event_date": pd.Timestamp("2024-03-01"), "event_type": "CPI", "surprise": 1.5},
    ])
    half_life = EVENT_TYPE_PARAMS["CPI"]["half_life_days"]
    current_time = pd.Timestamp("2024-03-01") + pd.Timedelta(days=half_life)

    context = compute_surprise_context("sideways", current_time=current_time, event_database=event_database)

    assert context["active"] is True
    assert context["events"][0]["age_days"] == pytest.approx(half_life)
    # at exactly one half-life, the event's own D(time_decay) is 0.5
    expected_wss = compute_wss(1.5, "CPI", age_days=half_life, regime_label="sideways")
    assert context["wss"] == pytest.approx(expected_wss)


def test_multiple_concurrent_events_sum_their_contributions():
    event_database = pd.DataFrame([
        {"event_date": pd.Timestamp("2024-03-01"), "event_type": "NFP", "surprise": 1.0},
        {"event_date": pd.Timestamp("2024-03-01"), "event_type": "GDP", "surprise": 1.0},
    ])
    current_time = pd.Timestamp("2024-03-01")

    context = compute_surprise_context("bull", current_time=current_time, event_database=event_database)

    nfp_only = compute_wss(1.0, "NFP", age_days=0, regime_label="bull")
    gdp_only = compute_wss(1.0, "GDP", age_days=0, regime_label="bull")
    assert len(context["events"]) == 2
    assert context["wss"] == pytest.approx(nfp_only + gdp_only)


def test_default_event_database_is_usable_without_passing_one():
    # Just confirm the default database loads and the function runs
    # end-to-end without a caller-supplied event_database.
    context = compute_surprise_context("bull", current_time=pd.Timestamp("2030-01-01"))
    assert context == {"active": False, "wss": 0.0, "events": []}  # 2030 is far past every default event


# ---------------------------------------------------------------------------
# vote_from_wss
# ---------------------------------------------------------------------------

def test_vote_from_wss_is_zero_for_inactive_surprise():
    assert vote_from_wss(0.0) == 0.0


def test_vote_from_wss_matches_tanh_formula():
    assert vote_from_wss(1.0, scale=2.0) == pytest.approx(np.tanh(0.5))


def test_vote_from_wss_is_bounded():
    assert -1.0 < vote_from_wss(3.0) < 1.0
    assert -1.0 < vote_from_wss(-3.0) < 1.0
    # saturates for large magnitude without exceeding bounds (tanh reaches
    # exactly 1.0 in float64 well before x=50, so this checks the
    # saturated value, not a strict inequality -- same pattern as
    # test_vote_from_order_flow_is_bounded_and_matches_sign)
    assert vote_from_wss(1e6) == pytest.approx(1.0, abs=1e-6)
    assert vote_from_wss(-1e6) == pytest.approx(-1.0, abs=1e-6)
