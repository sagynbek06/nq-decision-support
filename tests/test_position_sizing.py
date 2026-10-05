"""Tests for CDaR-throttled, confidence-scaled position sizing (src/risk/position_sizing.py)."""

import pytest

import src.consensus_engine as consensus_engine
from src.risk.position_sizing import position_size, throttle_from_cdar

BAND = 0.15
ALL_BULLISH = {"regime": 1.0, "order_flow": 1.0, "kernel": 1.0}
ALL_BEARISH = {"regime": -1.0, "order_flow": -1.0, "kernel": -1.0}


def _consensus(score, votes=None):
    return {"score": score, "votes": ALL_BULLISH if votes is None else votes}


# ---------------------------------------------------------------------------
# zero size inside the neutral band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", [0.0, 0.05, -0.05, 0.149, -0.149])
def test_size_is_exactly_zero_inside_the_neutral_band(score):
    assert position_size(_consensus(score), cdar_current=0.0, cdar_limit=0.1, neutral_band=BAND) == 0.0


@pytest.mark.parametrize("score", [0.15, -0.15])
def test_size_is_exactly_zero_on_the_band_edge(score):
    """compute_consensus calls |score| <= band neutral, so the edge itself is neutral."""
    assert position_size(_consensus(score), cdar_current=0.0, cdar_limit=0.1, neutral_band=BAND) == 0.0


def test_default_band_reads_the_consensus_engine_constant():
    edge = consensus_engine.NEUTRAL_BAND
    assert position_size(_consensus(edge), 0.0, 0.1) == 0.0
    assert position_size(_consensus(edge + 0.01), 0.0, 0.1) > 0.0


# ---------------------------------------------------------------------------
# direction, magnitude, agreement
# ---------------------------------------------------------------------------

def test_size_takes_the_sign_of_the_score():
    assert position_size(_consensus(0.6), 0.0, 0.1, neutral_band=BAND) > 0
    assert position_size(_consensus(-0.6, ALL_BEARISH), 0.0, 0.1, neutral_band=BAND) < 0


def test_size_is_continuous_at_the_threshold():
    just_above = position_size(_consensus(BAND + 1e-4), 0.0, 0.1, neutral_band=BAND)
    assert 0.0 < just_above < 1e-3


def test_size_grows_with_score_magnitude_above_the_band():
    sizes = [position_size(_consensus(s), 0.0, 0.1, neutral_band=BAND) for s in (0.2, 0.4, 0.6, 0.8)]
    assert all(b > a for a, b in zip(sizes, sizes[1:]))


def test_size_scales_with_vote_agreement():
    all_agree = position_size(_consensus(0.6, ALL_BULLISH), 0.0, 0.1, neutral_band=BAND)
    one_agrees = position_size(
        _consensus(0.6, {"regime": 1.0, "order_flow": 0.0, "kernel": -0.5}), 0.0, 0.1, neutral_band=BAND
    )
    assert one_agrees == pytest.approx(all_agree / 3)


# ---------------------------------------------------------------------------
# CDaR throttling
# ---------------------------------------------------------------------------

def test_throttle_is_full_with_no_drawdown_tail_and_zero_at_the_limit():
    assert throttle_from_cdar(0.0, 0.1) == pytest.approx(1.0)
    assert throttle_from_cdar(0.1, 0.1) == pytest.approx(0.0)


def test_throttle_is_linear_between_zero_and_the_limit():
    assert throttle_from_cdar(0.05, 0.1) == pytest.approx(0.5)


def test_throttle_is_clipped_past_the_limit():
    assert throttle_from_cdar(0.5, 0.1) == 0.0


def test_throttle_is_non_increasing_in_cdar():
    values = [throttle_from_cdar(c, 0.1) for c in (0.0, 0.02, 0.04, 0.08, 0.2)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_size_is_throttled_linearly_as_cdar_approaches_the_limit():
    full = position_size(_consensus(0.6), 0.0, 0.1, neutral_band=BAND)
    half = position_size(_consensus(0.6), 0.05, 0.1, neutral_band=BAND)
    assert half == pytest.approx(full / 2)


def test_size_is_exactly_zero_at_or_beyond_the_cdar_limit():
    assert position_size(_consensus(0.6), cdar_current=0.1, cdar_limit=0.1, neutral_band=BAND) == 0.0
    assert position_size(_consensus(0.6), cdar_current=0.3, cdar_limit=0.1, neutral_band=BAND) == 0.0


def test_throttle_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        throttle_from_cdar(0.0, 0.0)


def test_size_rejects_neutral_band_at_or_above_one():
    with pytest.raises(ValueError):
        position_size(_consensus(0.6), 0.0, 0.1, neutral_band=1.0)
