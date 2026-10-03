"""Tests for the Consensus Engine (Phase 5)."""

import numpy as np
import pandas as pd
import pytest

from src.consensus_engine import (
    vote_from_regime,
    vote_from_regime_subsignal,
    vote_from_surprise,
    vote_from_order_flow,
    vote_from_kernel_deviation,
    volatility_regime_from_gex,
    compute_consensus,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Per-program vote adapters
# ---------------------------------------------------------------------------

def test_vote_from_regime_mapping():
    assert vote_from_regime("bull") == 1.0
    assert vote_from_regime("bear") == -1.0
    assert vote_from_regime("sideways") == 0.0


def test_vote_from_regime_rejects_unknown_label():
    with pytest.raises(ValueError):
        vote_from_regime("not_a_regime")


def test_vote_from_order_flow_is_bounded_and_matches_sign():
    assert vote_from_order_flow(0.0) == pytest.approx(0.0)
    assert -1 < vote_from_order_flow(5.0) < 1
    assert -1 < vote_from_order_flow(-5.0) < 1
    assert vote_from_order_flow(3.0) > 0
    assert vote_from_order_flow(-3.0) < 0
    # saturates for large magnitude without exceeding bounds
    assert vote_from_order_flow(1e6) == pytest.approx(1.0, abs=1e-6)
    assert vote_from_order_flow(-1e6) == pytest.approx(-1.0, abs=1e-6)


def test_vote_from_order_flow_is_monotonically_increasing():
    xs = np.linspace(-10, 10, 50)
    votes = [vote_from_order_flow(x) for x in xs]
    assert all(b >= a for a, b in zip(votes, votes[1:]))


def test_vote_from_kernel_deviation_has_opposite_sign_of_input():
    """Mean-reversion reading: price above trend (positive deviation) votes
    bearish; below trend votes bullish."""
    assert vote_from_kernel_deviation(0.0) == pytest.approx(0.0)
    assert vote_from_kernel_deviation(3.0) < 0
    assert vote_from_kernel_deviation(-3.0) > 0


def test_vote_from_regime_subsignal_passes_through_already_scaled_value():
    assert vote_from_regime_subsignal(0.0) == pytest.approx(0.0)
    assert vote_from_regime_subsignal(0.5) == pytest.approx(0.5)
    assert vote_from_regime_subsignal(-0.5) == pytest.approx(-0.5)


def test_vote_from_regime_subsignal_clips_to_unit_interval():
    """Defensive clip, not a squashing transform -- ax_regime_subsignal
    already tanh-squashes its output, so this only guards against a
    caller-supplied value outside [-1, 1]."""
    assert vote_from_regime_subsignal(1.5) == pytest.approx(1.0)
    assert vote_from_regime_subsignal(-1.5) == pytest.approx(-1.0)


def test_volatility_regime_from_gex():
    assert volatility_regime_from_gex(100.0) == "dampening"
    assert volatility_regime_from_gex(-100.0) == "amplifying"
    assert volatility_regime_from_gex(0.0) == "dampening"  # documented tie-break


# ---------------------------------------------------------------------------
# compute_consensus
# ---------------------------------------------------------------------------

def test_consensus_is_bullish_when_all_three_inputs_agree_bullish():
    result = compute_consensus(
        regime_label="bull", order_flow_signal=2.0, kernel_deviation=-2.0, total_gex=0.0
    )
    assert result["label"] == "bullish"
    assert result["score"] > 0.5


def test_consensus_is_bearish_when_all_three_inputs_agree_bearish():
    result = compute_consensus(
        regime_label="bear", order_flow_signal=-2.0, kernel_deviation=2.0, total_gex=0.0
    )
    assert result["label"] == "bearish"
    assert result["score"] < -0.5


def test_consensus_is_neutral_for_all_zero_inputs():
    result = compute_consensus(
        regime_label="sideways", order_flow_signal=0.0, kernel_deviation=0.0, total_gex=0.0
    )
    assert result["score"] == pytest.approx(0.0)
    assert result["label"] == "neutral"


def test_consensus_score_matches_manual_weighted_average():
    """White-box check: the engine's weighted average must match an
    independently-computed one for arbitrary weights."""
    regime_label, order_flow_signal, kernel_deviation = "bull", 1.5, 0.5
    weights = {"regime": 2.0, "order_flow": 1.0, "kernel": 3.0}

    votes = {
        "regime": vote_from_regime(regime_label),
        "order_flow": vote_from_order_flow(order_flow_signal),
        "kernel": vote_from_kernel_deviation(kernel_deviation),
    }
    manual_score = sum(weights[k] * votes[k] for k in votes) / sum(weights.values())

    result = compute_consensus(regime_label, order_flow_signal, kernel_deviation, total_gex=10.0, weights=weights)
    assert result["score"] == pytest.approx(manual_score)


def test_consensus_weights_change_the_outcome():
    """Heavily overweighting one dissenting vote should be able to flip the label."""
    # regime says bull; order_flow and kernel both say bearish-leaning.
    base = compute_consensus("bull", order_flow_signal=-2.0, kernel_deviation=2.0, total_gex=0.0)
    assert base["label"] == "bearish"

    regime_favored = compute_consensus(
        "bull", order_flow_signal=-2.0, kernel_deviation=2.0, total_gex=0.0,
        weights={"regime": 10.0, "order_flow": 1.0, "kernel": 1.0},
    )
    assert regime_favored["label"] == "bullish"


def test_consensus_rejects_incomplete_weights():
    with pytest.raises(ValueError):
        compute_consensus("bull", 0.0, 0.0, 0.0, weights={"regime": 1.0})


def test_consensus_volatility_regime_is_independent_of_directional_label():
    """GEX should report its own regime regardless of which way the
    directional vote leans -- it's orthogonal information, not a 4th vote."""
    bullish_amplifying = compute_consensus("bull", 2.0, -2.0, total_gex=-50.0)
    bullish_dampening = compute_consensus("bull", 2.0, -2.0, total_gex=50.0)
    assert bullish_amplifying["label"] == bullish_dampening["label"] == "bullish"
    assert bullish_amplifying["volatility_regime"] == "amplifying"
    assert bullish_dampening["volatility_regime"] == "dampening"


def test_default_weights_are_equal():
    assert len(set(DEFAULT_WEIGHTS.values())) == 1


# ---------------------------------------------------------------------------
# regime_mode (Ax-style sub-signal opt-in)
# ---------------------------------------------------------------------------

def test_regime_mode_defaults_to_label_and_matches_original_behavior():
    with_default = compute_consensus("bull", 2.0, -2.0, total_gex=0.0)
    explicit_label_mode = compute_consensus("bull", 2.0, -2.0, total_gex=0.0, regime_mode="label")
    assert with_default == explicit_label_mode
    assert with_default["votes"]["regime"] == vote_from_regime("bull")


def test_regime_mode_subsignal_uses_subsignal_instead_of_label():
    result = compute_consensus(
        "sideways", order_flow_signal=0.0, kernel_deviation=0.0, total_gex=0.0,
        regime_mode="subsignal", regime_subsignal=0.6,
    )
    # regime_label "sideways" would vote 0.0 in "label" mode; "subsignal"
    # mode ignores the label and uses the passed-in float instead.
    assert result["votes"]["regime"] == pytest.approx(0.6)


def test_regime_mode_subsignal_requires_regime_subsignal():
    with pytest.raises(ValueError):
        compute_consensus("bull", 0.0, 0.0, 0.0, regime_mode="subsignal")


def test_compute_consensus_rejects_unknown_regime_mode():
    with pytest.raises(ValueError):
        compute_consensus("bull", 0.0, 0.0, 0.0, regime_mode="not_a_mode")


# ---------------------------------------------------------------------------
# surprise_context / include_surprise (WSS opt-in fourth vote)
# ---------------------------------------------------------------------------

def test_vote_from_surprise_matches_tanh_formula_and_is_zero_when_inactive():
    assert vote_from_surprise(0.0) == 0.0
    assert vote_from_surprise(1.0, scale=2.0) == pytest.approx(np.tanh(0.5))


def test_surprise_context_is_absent_by_default():
    result = compute_consensus("bull", 0.0, 0.0, 0.0)
    assert "surprise_context" not in result


def test_surprise_context_is_surfaced_but_not_voted_on_by_default():
    """Default integration: surprise_context is always echoed back when
    provided, exactly like volatility_regime is for GEX, WITHOUT changing
    the directional vote/score unless include_surprise=True is also set."""
    context = {"active": True, "wss": 5.0, "events": [{"event_type": "NFP"}]}

    without_context = compute_consensus("bull", 2.0, -2.0, total_gex=0.0)
    with_context = compute_consensus("bull", 2.0, -2.0, total_gex=0.0, surprise_context=context)

    assert with_context["surprise_context"] == context
    assert with_context["score"] == pytest.approx(without_context["score"])
    assert set(with_context["votes"]) == {"regime", "order_flow", "kernel"}


def test_include_surprise_adds_a_fourth_vote_with_default_weight():
    context = {"active": True, "wss": 5.0, "events": []}
    result = compute_consensus(
        "sideways", order_flow_signal=0.0, kernel_deviation=0.0, total_gex=0.0,
        surprise_context=context, include_surprise=True,
    )

    expected_surprise_vote = vote_from_surprise(5.0)
    manual_score = expected_surprise_vote / 4  # regime=0, order_flow=0, kernel=0, surprise weighted equally
    assert result["votes"]["surprise"] == pytest.approx(expected_surprise_vote)
    assert result["weights"]["surprise"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(manual_score)


def test_include_surprise_respects_custom_surprise_weight():
    context = {"active": True, "wss": 5.0, "events": []}
    result = compute_consensus(
        "sideways", 0.0, 0.0, 0.0,
        surprise_context=context, include_surprise=True, surprise_weight=0.1,
    )
    assert result["weights"]["surprise"] == pytest.approx(0.1)
    # a heavily down-weighted surprise vote should barely move the score
    assert abs(result["score"]) < abs(vote_from_surprise(5.0)) / 4


def test_include_surprise_requires_surprise_context():
    with pytest.raises(ValueError):
        compute_consensus("bull", 0.0, 0.0, 0.0, include_surprise=True)


def test_include_surprise_can_flip_the_label():
    """Demonstrates real effect, not just an extra field: a strong enough
    surprise vote can flip the label relative to the plain 3-vote
    consensus, same style as test_consensus_weights_change_the_outcome."""
    plain = compute_consensus("sideways", 0.0, 0.0, 0.0)
    assert plain["label"] == "neutral"

    bullish_surprise = {"active": True, "wss": 5.0, "events": []}
    with_surprise = compute_consensus(
        "sideways", 0.0, 0.0, 0.0,
        surprise_context=bullish_surprise, include_surprise=True,
    )
    assert with_surprise["label"] == "bullish"


# ---------------------------------------------------------------------------
# End-to-end: wire up the real Programs 1-4 on a self-contained series
# ---------------------------------------------------------------------------

def test_end_to_end_with_real_programs_1_through_4():
    """
    Not a claim that the resulting consensus is a *good* trading signal --
    it's a check that the whole system actually wires together: each
    program's real output, in its real shape, flows into compute_consensus
    without manual reshaping or silent type mismatches.
    """
    from src.regime_detection_robust import fit_and_label
    from src.order_flow import generate_synthetic_order_flow, compute_filtered_order_flow_signal
    from src.kernel_regression import fit_kernel_regression
    from src.greeks_dashboard import build_dashboard

    rng = np.random.default_rng(0)
    n = 500
    log_return = rng.standard_t(df=5, size=n) * 0.01
    dates = pd.bdate_range(start="2022-01-01", periods=n)
    price = 15_000 * np.exp(np.cumsum(log_return))
    df = pd.DataFrame({"log_return": log_return, "close": price}, index=dates)

    _, _, predicted_regimes = fit_and_label(df["log_return"].to_numpy())
    last_regime = predicted_regimes[-1]

    order_flow_df = generate_synthetic_order_flow(df)
    of_result = compute_filtered_order_flow_signal(order_flow_df)
    last_order_flow_signal = of_result["signal"][-1]

    kernel_result = fit_kernel_regression(df["close"].to_numpy())
    last_kernel_deviation = kernel_result["signal"][-1]

    dashboard = build_dashboard(spot=float(df["close"].iloc[-1]))
    total_gex = dashboard["total_gex"]

    consensus = compute_consensus(
        regime_label=last_regime,
        order_flow_signal=last_order_flow_signal,
        kernel_deviation=last_kernel_deviation,
        total_gex=total_gex,
    )

    assert consensus["label"] in {"bullish", "bearish", "neutral"}
    assert -1.0 <= consensus["score"] <= 1.0
    assert consensus["volatility_regime"] in {"dampening", "amplifying"}
    assert set(consensus["votes"]) == {"regime", "order_flow", "kernel"}
