"""Tests for Bayesian online changepoint detection."""

import numpy as np
import pytest

from src.bocpd import BOCPD, run_bocpd, regime_age_confidence


def test_run_length_posterior_is_a_distribution():
    rng = np.random.default_rng(0)
    model = BOCPD()
    for x in rng.normal(size=200):
        model.update(x)
    post = model.run_length_posterior
    assert post.sum() == pytest.approx(1.0)
    assert (post >= 0).all()


def test_detects_mean_shift_within_a_few_bars_with_no_early_false_alarms():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 300)])
    out = run_bocpd(x, hazard=1 / 200, k=5)
    # Quiet before the break (ignoring the burn-in while the prior washes out).
    assert out["change_prob"][50:295].max() < 0.5
    # Loud shortly after it.
    assert out["change_prob"][300:310].max() > 0.9
    # And the MAP run length collapses then regrows.
    assert out["map_run_length"][299] > 100
    assert out["map_run_length"][330] < 40


def test_detects_variance_shift_with_unchanged_mean():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(0, 3.0, 300)])
    out = run_bocpd(x, hazard=1 / 200, k=8)
    assert out["change_prob"][300:320].max() > 0.8


def test_moderate_outlier_is_absorbed_but_extreme_one_is_read_as_a_break():
    # Documents the measured behavior (see the module docstring): the t predictive
    # tolerates a 4 sigma spike, but a 7 sigma spike restarts the run length.
    # An honest limit, pinned so a future change to the prior shows up here.
    def map_run_after_spike(magnitude):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 400)
        x[250] = magnitude
        return run_bocpd(x, hazard=1 / 200, k=5)["map_run_length"][255]

    assert map_run_after_spike(4.0) > 200
    assert map_run_after_spike(7.0) < 20


def test_stationary_series_stays_quiet():
    rng = np.random.default_rng(4)
    out = run_bocpd(rng.normal(size=1000), hazard=1 / 200, k=5)
    assert (out["change_prob"][60:] > 0.5).mean() < 0.02


def test_update_is_causal():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    a = run_bocpd(x[:200])
    b = run_bocpd(np.concatenate([x[:200], 50 * np.ones(100)]))
    np.testing.assert_allclose(a["change_prob"], b["change_prob"][:200])


def test_truncation_keeps_posterior_normalized():
    rng = np.random.default_rng(6)
    model = BOCPD(max_run=50)
    for x in rng.normal(size=300):
        model.update(x)
    assert len(model.run_length_posterior) <= 50
    assert model.run_length_posterior.sum() == pytest.approx(1.0)


def test_regime_age_confidence_ramps_and_saturates():
    assert regime_age_confidence(0) == 0.0
    assert regime_age_confidence(15, saturation=30) == pytest.approx(0.5)
    assert regime_age_confidence(500, saturation=30) == 1.0


def test_invalid_hazard_rejected():
    with pytest.raises(ValueError):
        BOCPD(hazard=0.0)
