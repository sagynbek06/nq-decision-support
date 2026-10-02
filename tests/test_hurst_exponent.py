"""Tests for the rolling Hurst exponent / DFA module."""

import warnings

import numpy as np
import pytest

import src.consensus_engine as consensus_engine
from src.hurst_exponent import (
    compute_dfa_hurst,
    bootstrap_hurst_confidence_interval,
    rolling_dfa_hurst,
    hurst_to_threshold_multiplier,
    compute_consensus_with_hurst,
)
from src.synthetic_data import generate_ornstein_uhlenbeck, generate_well_separated_regime_returns

# A moderate mean-separation override of the existing well-separated
# generator (same function as tests/test_regime_detection.py uses, smaller
# effect size) reliably gives DFA Hurst in the 0.75-0.91 range across
# random seeds -- the project's default (more separated) params instead
# give H > 1 (a legitimate DFA result for a series with large deterministic
# level shifts, but a less intuitive one for a test fixture; see this
# module's writeup for the full seed-by-seed robustness check behind this
# choice, and why the realistic regime-switching generator -- which has
# much smaller daily drift differences -- turned out NOT to reliably
# produce H significantly above 0.5 at all).
TRENDING_PARAMS = {"bull": (0.005, 0.010), "bear": (-0.005, 0.015), "sideways": (0.0, 0.008)}


# ---------------------------------------------------------------------------
# compute_dfa_hurst correctness
# ---------------------------------------------------------------------------

def test_white_noise_gives_hurst_near_half():
    rng = np.random.default_rng(0)
    white_noise = rng.normal(0, 1, size=3000)
    hurst, *_ = compute_dfa_hurst(white_noise)
    assert hurst == pytest.approx(0.5, abs=0.07)


@pytest.mark.parametrize("seed", range(1, 8))
def test_mean_reverting_ou_gives_hurst_well_below_half(seed):
    ou_level = generate_ornstein_uhlenbeck(2501, theta=0.15, random_state=seed)
    ou_returns = np.diff(ou_level)
    hurst, *_ = compute_dfa_hurst(ou_returns)
    assert hurst < 0.35


@pytest.mark.parametrize("seed", range(1, 8))
def test_trending_regime_series_gives_hurst_well_above_half(seed):
    rng = np.random.default_rng(seed)
    returns, _ = generate_well_separated_regime_returns(
        rng, segment_length=100, n_cycles=5, params=TRENDING_PARAMS
    )
    hurst, *_ = compute_dfa_hurst(returns)
    assert hurst > 0.65


def test_vectorized_detrending_matches_independent_polyfit_reference():
    """
    White-box check: the vectorized closed-form OLS detrending must match
    an independently-written, unvectorized np.polyfit-per-segment
    computation, for a single box size.
    """
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 1, size=500)
    profile = np.cumsum(returns - returns.mean())
    box_size = 25
    n_segments = len(profile) // box_size

    manual_mses = []
    for v in range(n_segments):
        segment = profile[v * box_size:(v + 1) * box_size]
        t = np.arange(box_size)
        coeffs = np.polyfit(t, segment, 1)
        trend = np.polyval(coeffs, t)
        manual_mses.append(np.mean((segment - trend) ** 2))
    manual_fluctuation = np.sqrt(np.mean(manual_mses))

    # A single box size makes the log-log slope fit degenerate (one point)
    # and numpy warns accordingly; this test only checks fluctuations[0],
    # not the resulting slope, so the warning is expected noise, not a
    # real conditioning problem -- silence it rather than restructure the
    # test around numpy's regression diagnostics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        hurst, box_sizes, fluctuations, _ = compute_dfa_hurst(
            returns, min_box_size=box_size, max_box_size=box_size, n_box_sizes=1
        )
    # fluctuations[0] averages forward AND backward segmentation; with
    # box_size=25 and n=500 (an exact multiple), forward and backward
    # segmentation produce the same set of segments, so this should match
    # the manual forward-only computation exactly.
    assert fluctuations[0] == pytest.approx(manual_fluctuation, rel=1e-10)


def test_dfa_requires_enough_data_for_at_least_one_box_size():
    rng = np.random.default_rng(6)
    returns = rng.normal(0, 1, size=200)
    hurst, box_sizes, fluctuations, _ = compute_dfa_hurst(returns)
    assert len(box_sizes) >= 2
    assert np.isfinite(hurst)


# ---------------------------------------------------------------------------
# bootstrap_hurst_confidence_interval
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_half_for_white_noise():
    rng = np.random.default_rng(0)
    white_noise = rng.normal(0, 1, size=1500)
    (ci_lower, ci_upper), bootstrap_hursts = bootstrap_hurst_confidence_interval(
        white_noise, n_resamples=200, random_state=1
    )
    assert ci_lower < 0.5 < ci_upper
    assert len(bootstrap_hursts) == 200


def test_bootstrap_ci_excludes_half_for_strongly_trending_series():
    rng = np.random.default_rng(3)
    returns, _ = generate_well_separated_regime_returns(
        rng, segment_length=100, n_cycles=5, params=TRENDING_PARAMS
    )
    (ci_lower, ci_upper), _ = bootstrap_hurst_confidence_interval(
        returns, n_resamples=200, random_state=2
    )
    assert ci_lower > 0.5


def test_bootstrap_ci_lower_bound_never_exceeds_upper_bound():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 1, size=600)
    (ci_lower, ci_upper), _ = bootstrap_hurst_confidence_interval(
        returns, n_resamples=50, random_state=8
    )
    assert ci_lower <= ci_upper


# ---------------------------------------------------------------------------
# hurst_to_threshold_multiplier
# ---------------------------------------------------------------------------

def test_multiplier_is_max_when_ci_contains_half():
    assert hurst_to_threshold_multiplier(0.5, (0.4, 0.6)) == 2.0
    assert hurst_to_threshold_multiplier(0.9, (0.3, 0.95)) == 2.0  # point far away, but CI still straddles 0.5


def test_multiplier_decreases_with_distance_from_half():
    m_close = hurst_to_threshold_multiplier(0.55, (0.51, 0.59))
    m_mid = hurst_to_threshold_multiplier(0.7, (0.6, 0.8))
    m_far = hurst_to_threshold_multiplier(0.9, (0.8, 0.95))
    assert m_close > m_mid > m_far


def test_multiplier_matches_manual_linear_interpolation():
    hurst, ci = 0.7, (0.6, 0.8)
    distance = abs(hurst - 0.5)
    normalized = min(distance / 0.5, 1.0)
    expected = 2.0 - normalized * (2.0 - 0.5)
    assert hurst_to_threshold_multiplier(hurst, ci) == pytest.approx(expected)


def test_multiplier_saturates_at_min_for_extreme_hurst():
    assert hurst_to_threshold_multiplier(1.0, (0.9, 1.1)) == pytest.approx(0.5)
    assert hurst_to_threshold_multiplier(0.0, (-0.1, 0.05)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rolling_dfa_hurst
# ---------------------------------------------------------------------------

def test_rolling_hurst_output_length_and_nan_padding():
    rng = np.random.default_rng(9)
    returns = rng.normal(0, 0.01, size=500)
    window = 252
    result = rolling_dfa_hurst(returns, window=window, step=1)

    assert len(result) == len(returns)
    assert np.all(np.isnan(result[:window - 1]))
    assert not np.isnan(result[window - 1])
    assert not np.any(np.isnan(result[window - 1:]))


def test_rolling_hurst_step_reduces_number_of_estimates():
    rng = np.random.default_rng(10)
    returns = rng.normal(0, 0.01, size=500)
    window = 252

    dense = rolling_dfa_hurst(returns, window=window, step=1)
    sparse = rolling_dfa_hurst(returns, window=window, step=5)

    n_dense = np.sum(~np.isnan(dense))
    n_sparse = np.sum(~np.isnan(sparse))
    assert n_sparse < n_dense
    # where sparse has an estimate, it should exactly match dense (same
    # underlying computation, just evaluated at fewer positions)
    valid_idx = ~np.isnan(sparse)
    np.testing.assert_allclose(sparse[valid_idx], dense[valid_idx])


# ---------------------------------------------------------------------------
# compute_consensus_with_hurst
# ---------------------------------------------------------------------------

def test_wrapper_returns_same_shape_as_compute_consensus_plus_two_fields():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=1500)
    result = compute_consensus_with_hurst(
        "bull", 0.3, -0.3, total_gex=0.0, returns=returns, random_state=1, n_bootstrap=50
    )
    plain_keys = set(consensus_engine.compute_consensus("bull", 0.3, -0.3, 0.0).keys())
    assert set(result.keys()) == plain_keys | {"hurst", "effective_neutral_band"}


def test_wrapper_restores_neutral_band_after_normal_call():
    original = consensus_engine.NEUTRAL_BAND
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=1500)

    compute_consensus_with_hurst("bull", 0.3, -0.3, 0.0, returns=returns, random_state=1, n_bootstrap=50)

    assert consensus_engine.NEUTRAL_BAND == original


def test_wrapper_restores_neutral_band_even_when_compute_consensus_raises():
    original = consensus_engine.NEUTRAL_BAND
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=1500)

    with pytest.raises(ValueError):
        compute_consensus_with_hurst(
            "not_a_real_regime", 0.0, 0.0, 0.0, returns=returns, random_state=1, n_bootstrap=50
        )

    assert consensus_engine.NEUTRAL_BAND == original


def test_wrapper_widens_band_for_white_noise_and_narrows_for_trending():
    rng = np.random.default_rng(0)
    white_noise_returns = rng.normal(0, 0.01, size=1500)
    result_noise = compute_consensus_with_hurst(
        "bull", 0.3, -0.3, 0.0, returns=white_noise_returns, random_state=1, n_bootstrap=50
    )

    rng2 = np.random.default_rng(3)
    trending_returns, _ = generate_well_separated_regime_returns(
        rng2, segment_length=100, n_cycles=5, params=TRENDING_PARAMS
    )
    result_trending = compute_consensus_with_hurst(
        "bull", 0.3, -0.3, 0.0, returns=trending_returns, random_state=2, n_bootstrap=50
    )

    assert result_noise["effective_neutral_band"] > consensus_engine.NEUTRAL_BAND
    assert result_trending["effective_neutral_band"] < result_noise["effective_neutral_band"]


def test_wrapper_can_change_the_label_relative_to_plain_compute_consensus():
    """
    Demonstrates the mechanism has real effect, not just cosmetic extra
    fields: a score that would read "bullish" under the default band can
    read "neutral" once Hurst (here, white noise) widens the effective band.
    """
    # order_flow=0.4 alone gives vote tanh(0.4/2)=0.1974; regime/kernel at 0
    # gives weighted score = 0.1974/3 = 0.0658 -- within default band 0.15
    # (neutral either way). Use a stronger order_flow signal instead so the
    # plain score clears the default band but not a doubled one.
    plain = consensus_engine.compute_consensus("sideways", 1.0, 0.0, 0.0)
    assert plain["label"] == "bullish"
    assert plain["score"] < 0.15 * 2  # would fall inside a doubled neutral band

    rng = np.random.default_rng(0)
    white_noise_returns = rng.normal(0, 0.01, size=1500)
    with_hurst = compute_consensus_with_hurst(
        "sideways", 1.0, 0.0, 0.0, returns=white_noise_returns, random_state=1, n_bootstrap=50
    )

    assert with_hurst["score"] == pytest.approx(plain["score"])  # same inputs, same vote math
    assert with_hurst["effective_neutral_band"] > 0.15
    assert with_hurst["label"] == "neutral"
    assert with_hurst["label"] != plain["label"]
