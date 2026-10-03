"""Tests for the Ornstein-Uhlenbeck half-life calibration module."""

import numpy as np
import pytest

from src.ou_half_life import calibrate_ou_half_life
from src.synthetic_data import generate_ornstein_uhlenbeck


# ---------------------------------------------------------------------------
# Recovery on genuine OU data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(1, 8))
def test_recovers_known_theta_and_half_life_from_ou_process(seed):
    true_theta = 0.15
    levels = generate_ornstein_uhlenbeck(2000, theta=true_theta, mu=100.0, sigma=1.0, x0=100.0, random_state=seed)
    result = calibrate_ou_half_life(levels)

    assert result["theta"] == pytest.approx(true_theta, abs=0.05)
    assert result["half_life"] == pytest.approx(np.log(2) / result["theta"])
    assert result["half_life"] == pytest.approx(np.log(2) / true_theta, abs=2.0)
    assert result["mu"] == pytest.approx(100.0, abs=1.0)


@pytest.mark.parametrize("true_theta", [0.02, 0.05, 0.15, 0.3, 0.5])
def test_theta_recovery_scales_correctly_across_reversion_speeds(true_theta):
    levels = generate_ornstein_uhlenbeck(2000, theta=true_theta, mu=100.0, sigma=1.0, x0=100.0, random_state=1)
    result = calibrate_ou_half_life(levels)
    assert result["theta"] == pytest.approx(true_theta, rel=0.2)


def test_residuals_have_expected_length_and_scale():
    levels = generate_ornstein_uhlenbeck(1000, theta=0.15, mu=100.0, sigma=1.0, x0=100.0, random_state=1)
    result = calibrate_ou_half_life(levels)
    assert len(result["residuals"]) == len(levels) - 1
    # OU was simulated with sigma=1.0 (dt=1), so residual scale should be close to 1
    assert np.std(result["residuals"]) == pytest.approx(1.0, abs=0.1)


def test_requires_at_least_three_points():
    with pytest.raises(ValueError):
        calibrate_ou_half_life([1.0, 2.0])


# ---------------------------------------------------------------------------
# No valid mean reversion: theta <= 0 -> half_life and mu are None
# ---------------------------------------------------------------------------

def test_trending_series_with_positive_feedback_gives_no_valid_half_life():
    """
    A price series with positive drift and multiplicative (GBM-like) noise
    has returns proportional to the *level*, i.e. a systematic positive
    relationship between p_{t-1} and the next increment -- the opposite of
    mean reversion. This should reliably clip theta to <= 0, not a random
    sign depending on noise.
    """
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0008, 0.01, size=2000)
    price = 100 * np.exp(np.cumsum(returns))

    result = calibrate_ou_half_life(price)

    assert result["theta"] <= 0
    assert result["half_life"] is None
    assert result["mu"] is None


def test_pure_random_walk_mean_reversion_signal_is_negligible_versus_genuine_ou():
    """
    The OLS estimator used here is exactly the (non-augmented) Dickey-Fuller
    unit-root test regression. Under a true random walk (theta=0 exactly),
    its finite-sample distribution is biased toward a spuriously *positive*
    theta_hat -- confirmed empirically (docs/writeups/05b_ax_regime_switch.md):
    46/50 random-walk seeds gave theta_hat > 0. That makes "theta > 0" alone
    an unreliable one-seed test for "this series really mean-reverts." What
    IS reliable is magnitude: across those 50 seeds, the largest spurious
    |theta| was 0.0084 -- more than an order of magnitude below a genuine
    OU signal's recovered theta (~0.15 here). This test checks that
    magnitude gap directly, rather than asserting a sign that isn't
    actually guaranteed.
    """
    rng = np.random.default_rng(1)
    levels = 100.0 + np.cumsum(rng.normal(0, 1.0, size=2000))
    random_walk_result = calibrate_ou_half_life(levels)

    genuine_ou = generate_ornstein_uhlenbeck(2000, theta=0.15, mu=100.0, sigma=1.0, x0=100.0, random_state=1)
    genuine_result = calibrate_ou_half_life(genuine_ou)

    assert abs(random_walk_result["theta"]) < abs(genuine_result["theta"]) / 10
