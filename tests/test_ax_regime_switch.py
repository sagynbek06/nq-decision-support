"""
Tests for the Ax-style regime-switching sub-signal (src/ax_regime_switch.py).

See that module's docstring and docs/writeups/05b_ax_regime_switch.md for
the full motivation and findings. The short version relevant to reading
these tests: the TREND half of this design (bull/bear -> follow Program
3's kernel slope) checks out cleanly and robustly. The REVERSION half
(sideways -> revert toward an OU-calibrated mu) is real and detectable when
isolated against ground truth, but `validate_regime_edge`'s honest,
walk-forward-realistic test of it does NOT reliably reach statistical
significance, because Program 1's "sideways" label has poor precision on
data that mixes trending and reverting stretches. `test_validate_regime_edge`
section below tests for exactly that (documented) asymmetry rather than
forcing a prettier result.
"""

import numpy as np
import pytest

from src.ax_regime_switch import (
    kernel_trend_slope,
    ax_regime_subsignal,
    validate_regime_edge,
)
from src.ou_half_life import calibrate_ou_half_life
from src.synthetic_data import generate_ornstein_uhlenbeck

FAST_HMM_KWARGS = {"n_init": 5, "n_iter": 50}


# ---------------------------------------------------------------------------
# kernel_trend_slope / ax_regime_subsignal: trend branch (bull/bear)
# ---------------------------------------------------------------------------

TRENDING_PARAMS = {"bull": (0.005, 0.010), "bear": (-0.005, 0.015)}


@pytest.mark.parametrize("seed", range(1, 8))
def test_trend_subsignal_direction_matches_trend_direction_bull(seed):
    rng = np.random.default_rng(seed)
    mu_r, sigma_r = TRENDING_PARAMS["bull"]
    returns = rng.normal(mu_r, sigma_r, size=300)
    price = 100 * np.exp(np.cumsum(returns))

    assert kernel_trend_slope(price) > 0
    assert ax_regime_subsignal("bull", price) > 0


@pytest.mark.parametrize("seed", range(1, 8))
def test_trend_subsignal_direction_matches_trend_direction_bear(seed):
    rng = np.random.default_rng(seed)
    mu_r, sigma_r = TRENDING_PARAMS["bear"]
    returns = rng.normal(mu_r, sigma_r, size=300)
    price = 100 * np.exp(np.cumsum(returns))

    assert kernel_trend_slope(price) < 0
    assert ax_regime_subsignal("bear", price) < 0


def test_trend_subsignal_is_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.02, 0.01, size=300)  # extreme drift
    price = 100 * np.exp(np.cumsum(returns))
    signal = ax_regime_subsignal("bull", price)
    assert -1.0 <= signal <= 1.0


def test_loocv_bandwidth_gives_unreliable_sign_versus_manual_bandwidth():
    """
    Documents the real reason this module doesn't reuse
    fit_kernel_regression's LOOCV-selected bandwidth (see
    kernel_trend_slope's docstring): on bandwidth=2 (what LOOCV picks on
    persistent price data per kernel_regression.py's own documented
    finding), the last-10-point slope is noisy enough to get the sign
    wrong a meaningful fraction of the time on a purely bear-trending
    series. The manually-chosen default (bandwidth=20) does not have this
    problem on the same seeds.
    """
    from src.kernel_regression import nadaraya_watson_regression

    n_wrong_at_bandwidth_2 = 0
    n_wrong_at_default = 0
    mu_r, sigma_r = TRENDING_PARAMS["bear"]
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        returns = rng.normal(mu_r, sigma_r, size=500)
        price = 100 * np.exp(np.cumsum(returns))

        x = np.arange(len(price), dtype=float)
        estimate_loocv_bandwidth = nadaraya_watson_regression(x, price, 2)
        slope = np.polyfit(np.arange(10), estimate_loocv_bandwidth[-10:], 1)[0]
        if slope >= 0:  # wrong sign for a bear trend
            n_wrong_at_bandwidth_2 += 1

        if kernel_trend_slope(price) >= 0:
            n_wrong_at_default += 1

    assert n_wrong_at_bandwidth_2 > 0  # the problem this module's default avoids is real
    assert n_wrong_at_default == 0


# ---------------------------------------------------------------------------
# ax_regime_subsignal: reversion branch (sideways)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(1, 8))
def test_reversion_subsignal_points_toward_mu_from_above(seed):
    levels = generate_ornstein_uhlenbeck(1000, theta=0.15, mu=100.0, sigma=1.0, x0=100.0, random_state=seed)
    ou_result = calibrate_ou_half_life(levels)
    price_above_mu = np.append(levels, ou_result["mu"] + 3.0)

    signal = ax_regime_subsignal("sideways", price_above_mu, ou_result=ou_result)
    assert signal < 0  # price above mu -> expect reversion down -> bearish sub-signal


@pytest.mark.parametrize("seed", range(1, 8))
def test_reversion_subsignal_points_toward_mu_from_below(seed):
    levels = generate_ornstein_uhlenbeck(1000, theta=0.15, mu=100.0, sigma=1.0, x0=100.0, random_state=seed)
    ou_result = calibrate_ou_half_life(levels)
    price_below_mu = np.append(levels, ou_result["mu"] - 3.0)

    signal = ax_regime_subsignal("sideways", price_below_mu, ou_result=ou_result)
    assert signal > 0  # price below mu -> expect reversion up -> bullish sub-signal


def test_reversion_subsignal_is_zero_when_no_valid_ou_structure():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0008, 0.01, size=2000)  # positive-feedback GBM; theta <= 0 (see test_ou_half_life.py)
    price = 100 * np.exp(np.cumsum(returns))
    ou_result = calibrate_ou_half_life(price)
    assert ou_result["half_life"] is None

    assert ax_regime_subsignal("sideways", price, ou_result=ou_result) == 0.0


def test_reversion_subsignal_grows_with_distance_from_mu():
    levels = generate_ornstein_uhlenbeck(1000, theta=0.15, mu=100.0, sigma=1.0, x0=100.0, random_state=1)
    ou_result = calibrate_ou_half_life(levels)

    small_gap = ax_regime_subsignal("sideways", np.append(levels, ou_result["mu"] - 0.5), ou_result=ou_result)
    large_gap = ax_regime_subsignal("sideways", np.append(levels, ou_result["mu"] - 5.0), ou_result=ou_result)
    assert 0 < small_gap < large_gap <= 1.0


def test_ax_regime_subsignal_rejects_unknown_regime_label():
    with pytest.raises(ValueError):
        ax_regime_subsignal("not_a_regime", np.array([1.0, 2.0, 3.0]))


def test_ax_regime_subsignal_requires_ou_result_for_sideways():
    with pytest.raises(ValueError):
        ax_regime_subsignal("sideways", np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# validate_regime_edge
# ---------------------------------------------------------------------------

def _build_mixed_trend_reversion_series(rng, n_cycles=4, block_len=120,
                                         bull=(0.004, 0.009), bear=(-0.004, 0.009),
                                         ou_theta=0.2, ou_sigma=1.0, start_price=100.0):
    """
    Stitches genuinely trending bull/bear blocks with genuinely
    mean-reverting OU blocks (each OU block anchored to, and reverting
    back to, the price level where it starts -- no discontinuous jumps at
    block boundaries). Built specifically for this test file rather than
    added to src/synthetic_data.py: `generate_well_separated_regime_returns`
    is NOT a substitute here, because its "sideways" block is just
    low-variance i.i.d. noise with no actual pull-back-to-a-level
    mechanism -- confirmed directly while developing this test (reversion
    p-value on that fixture is non-significant for the honest reason that
    there's genuinely no reversion structure in it, not because
    validate_regime_edge failed to find real structure). This generator
    exists to give the reversion edge a fixture that actually contains
    what it's being tested for.
    """
    price = [start_price]
    for _ in range(n_cycles):
        for kind in ("bull", "bear", "sideways"):
            if kind == "sideways":
                anchor = price[-1]
                levels = generate_ornstein_uhlenbeck(
                    block_len + 1, theta=ou_theta, mu=anchor, sigma=ou_sigma,
                    x0=anchor, random_state=int(rng.integers(0, 2**31)),
                )
                price.extend(levels[1:].tolist())
            else:
                mu_r, sigma_r = bull if kind == "bull" else bear
                r = rng.normal(mu_r, sigma_r, size=block_len)
                price.extend((price[-1] * np.exp(np.cumsum(r))).tolist())
    return np.array(price[1:])


def test_validate_regime_edge_finds_no_structure_on_pure_random_walk():
    rng = np.random.default_rng(11)
    price = 100 + np.cumsum(rng.normal(0, 1.0, size=1800))
    price = np.abs(price) + 50  # stay positive for log()

    result = validate_regime_edge(price, n_splits=5, hmm_kwargs=FAST_HMM_KWARGS)

    assert result["trend_p_value"] > 0.1
    assert result["reversion_p_value"] > 0.1


def test_validate_regime_edge_detects_real_trend_continuation_structure():
    rng = np.random.default_rng(5)
    price = _build_mixed_trend_reversion_series(rng, n_cycles=4, block_len=120)

    result = validate_regime_edge(price, n_splits=5, hmm_kwargs=FAST_HMM_KWARGS)

    assert result["trend_edge"] > 0
    assert result["trend_p_value"] < 0.05
    assert result["n_trend_samples"] > 0


def test_validate_regime_edge_reversion_detection_is_limited_by_regime_label_precision():
    """
    The documented, counterintuitive finding (see this module's own
    docstring and docs/writeups/05b_ax_regime_switch.md): even on data
    built with genuine OU mean-reversion during "sideways" stretches, the
    full walk-forward pipeline does not reliably detect it as significant,
    because Program 1's "sideways" label has poor precision against the
    true reverting periods once it has to be decoded (rather than known in
    advance) on data that also contains trending stretches. This is not
    asserting the machinery is broken -- `test_reversion_subsignal_points_*`
    above already confirms the reversion math itself works correctly in
    isolation -- it's asserting that this specific, harder, end-to-end
    question gets an honest "not reliably" answer rather than a
    manufactured "yes." Same seed and fixture as the trend test above,
    which DOES reach significance on its (different) question.
    """
    rng = np.random.default_rng(5)
    price = _build_mixed_trend_reversion_series(rng, n_cycles=4, block_len=120)

    result = validate_regime_edge(price, n_splits=5, hmm_kwargs=FAST_HMM_KWARGS)

    assert result["n_reversion_samples"] > 0
    assert 0.0 <= result["reversion_p_value"] <= 1.0
    assert result["reversion_p_value"] > 0.05


def test_validate_regime_edge_raises_for_too_many_splits_given_data_length():
    with pytest.raises(ValueError):
        validate_regime_edge(np.arange(1, 51, dtype=float), n_splits=5)
