"""Tests for the conformal abstention gate."""

import numpy as np
import pytest

from src.conformal_gate import ConformalGate, fit_scale, run_gate


def _series_with_edge(rng, n=4000, skill=0.5):
    """Scores carry a real but modest signal; returns have fat tails and a volatility shift midway."""
    scores = rng.uniform(-1, 1, n)
    vol = np.where(np.arange(n) < n // 2, 0.01, 0.02)
    noise = rng.standard_t(df=4, size=n) * vol
    returns = skill * 0.01 * scores + noise
    return scores, returns


def test_long_run_coverage_converges_to_target_despite_vol_shift():
    # The guarantee is marginal coverage without exchangeability: a mid-sample
    # volatility doubling must not break it.
    rng = np.random.default_rng(0)
    scores, returns = _series_with_edge(rng)
    out = run_gate(scores, returns, alpha=0.10)
    assert abs(out["coverage"] - 0.90) < 0.03


def test_no_skill_forecast_abstains_almost_always():
    # Pure noise: a gate that traded here would be manufacturing edge.
    rng = np.random.default_rng(1)
    n = 3000
    scores = rng.uniform(-1, 1, n)
    returns = rng.standard_t(df=4, size=n) * 0.01
    out = run_gate(scores, returns, alpha=0.10)
    assert out["abstain_rate"] > 0.95


def test_strong_signal_produces_directional_calls_that_are_mostly_right():
    rng = np.random.default_rng(2)
    n = 3000
    scores = rng.choice([-1.0, 1.0], size=n)
    returns = 0.01 * scores + rng.normal(0, 0.003, n)  # signal dwarfs noise
    out = run_gate(scores, returns, alpha=0.10)
    acted = out["action"] != "abstain"
    assert acted[100:].mean() > 0.5
    hit = np.sign(returns[acted]) == np.where(out["action"][acted] == "long", 1, -1)
    assert hit.mean() > 0.95


def test_interval_is_infinite_during_warmup():
    gate = ConformalGate(alpha=0.1, forecast_scale=1.0, warmup=10)
    for i in range(5):
        gate.update(0.5, 0.01)
    assert gate.decide(0.5)["action"] == "abstain"
    assert gate.interval(0.5) == (-np.inf, np.inf)


def test_decide_does_not_peek_at_the_current_return():
    # Two gates fed identical history must give identical intervals regardless of
    # what the NEXT return turns out to be.
    rng = np.random.default_rng(3)
    scores = rng.uniform(-1, 1, 200)
    returns = rng.normal(0, 0.01, 200)
    a = ConformalGate(alpha=0.1, forecast_scale=0.01, warmup=20)
    b = ConformalGate(alpha=0.1, forecast_scale=0.01, warmup=20)
    for t in range(150):
        a.update(scores[t], returns[t])
        b.update(scores[t], returns[t])
    assert a.interval(scores[150]) == b.interval(scores[150])
    a.update(scores[150], 100.0)  # an absurd outlier arrives afterwards
    assert b.interval(scores[150]) != a.interval(scores[150])


def test_fit_scale_recovers_known_slope_and_handles_degenerate_input():
    rng = np.random.default_rng(4)
    s = rng.normal(size=5000)
    r = 0.003 * s + rng.normal(0, 0.001, 5000)
    assert fit_scale(s, r) == pytest.approx(0.003, rel=0.05)
    assert fit_scale(np.zeros(10), np.ones(10)) == 0.0


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        ConformalGate(alpha=1.5)
    with pytest.raises(ValueError):
        ConformalGate(beta=0.4)
    with pytest.raises(ValueError):
        run_gate(np.zeros(10), np.zeros(10), warmup=60)
