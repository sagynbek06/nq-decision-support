"""Tests for Program 4 (Black-Scholes Greeks, GEX, Vanna exposure)."""

import numpy as np
import pytest

from src.greeks_dashboard import (
    black_scholes_price,
    delta,
    gamma,
    vega,
    theta,
    rho,
    vanna,
    generate_synthetic_option_chain,
    compute_gamma_exposure,
    compute_vanna_exposure,
    build_dashboard,
)

S, K, T, R, SIGMA = 20_000.0, 20_500.0, 0.25, 0.05, 0.18


# ---------------------------------------------------------------------------
# Pricing identities
# ---------------------------------------------------------------------------

def test_put_call_parity():
    call = black_scholes_price(S, K, T, R, SIGMA, "call")
    put = black_scholes_price(S, K, T, R, SIGMA, "put")
    rhs = S - K * np.exp(-R * T)  # q=0
    assert (call - put) == pytest.approx(rhs, rel=1e-10)


def test_black_scholes_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        black_scholes_price(S, K, T, R, SIGMA, "straddle")


# ---------------------------------------------------------------------------
# Basic Greek properties
# ---------------------------------------------------------------------------

def test_delta_bounds():
    assert 0 <= delta(S, K, T, R, SIGMA, "call") <= 1
    assert -1 <= delta(S, K, T, R, SIGMA, "put") <= 0


def test_gamma_is_positive_and_identical_for_call_and_put():
    g = gamma(S, K, T, R, SIGMA)
    assert g > 0
    # gamma() doesn't take option_type -- this just documents *why* not:
    # Gamma_call == Gamma_put is a standard Black-Scholes identity.
    assert g == gamma(S, K, T, R, SIGMA)  # trivially same call, but see FD test below


def test_vanna_identical_for_call_and_put_via_put_call_parity():
    # Delta_put = Delta_call - exp(-qT) (put-call parity), a constant shift
    # w.r.t. sigma, so d(Delta)/d(sigma) must coincide for call and put.
    eps = 1e-6
    d_call_up = delta(S, K, T, R, SIGMA + eps, "call")
    d_call_dn = delta(S, K, T, R, SIGMA - eps, "call")
    d_put_up = delta(S, K, T, R, SIGMA + eps, "put")
    d_put_dn = delta(S, K, T, R, SIGMA - eps, "put")
    vanna_from_call_delta = (d_call_up - d_call_dn) / (2 * eps)
    vanna_from_put_delta = (d_put_up - d_put_dn) / (2 * eps)
    assert vanna_from_call_delta == pytest.approx(vanna_from_put_delta, abs=1e-8)
    assert vanna(S, K, T, R, SIGMA) == pytest.approx(vanna_from_call_delta, abs=1e-6)


# ---------------------------------------------------------------------------
# Every closed-form Greek vs. finite-difference of the price function itself
# ---------------------------------------------------------------------------

EPS_S = S * 1e-5
EPS_T = T * 1e-5
EPS_SIGMA = 1e-6
EPS_R = 1e-6


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    fd = (
        black_scholes_price(S + EPS_S, K, T, R, SIGMA, option_type)
        - black_scholes_price(S - EPS_S, K, T, R, SIGMA, option_type)
    ) / (2 * EPS_S)
    assert delta(S, K, T, R, SIGMA, option_type) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(option_type):
    fd = (
        black_scholes_price(S + EPS_S, K, T, R, SIGMA, option_type)
        - 2 * black_scholes_price(S, K, T, R, SIGMA, option_type)
        + black_scholes_price(S - EPS_S, K, T, R, SIGMA, option_type)
    ) / (EPS_S ** 2)
    assert gamma(S, K, T, R, SIGMA) == pytest.approx(fd, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    fd = (
        black_scholes_price(S, K, T, R, SIGMA + EPS_SIGMA, option_type)
        - black_scholes_price(S, K, T, R, SIGMA - EPS_SIGMA, option_type)
    ) / (2 * EPS_SIGMA)
    assert vega(S, K, T, R, SIGMA) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    # theta() returns -dPrice/dT (the conventional "decay" sign), so the
    # raw forward-time finite difference must be negated to compare.
    raw_fd = (
        black_scholes_price(S, K, T + EPS_T, R, SIGMA, option_type)
        - black_scholes_price(S, K, T - EPS_T, R, SIGMA, option_type)
    ) / (2 * EPS_T)
    assert theta(S, K, T, R, SIGMA, option_type) == pytest.approx(-raw_fd, rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    fd = (
        black_scholes_price(S, K, T, R + EPS_R, SIGMA, option_type)
        - black_scholes_price(S, K, T, R - EPS_R, SIGMA, option_type)
    ) / (2 * EPS_R)
    assert rho(S, K, T, R, SIGMA, option_type) == pytest.approx(fd, rel=1e-5)


def test_vanna_matches_finite_difference_of_vega_wrt_spot():
    fd = (
        vega(S + EPS_S, K, T, R, SIGMA) - vega(S - EPS_S, K, T, R, SIGMA)
    ) / (2 * EPS_S)
    assert vanna(S, K, T, R, SIGMA) == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Synthetic options chain
# ---------------------------------------------------------------------------

def test_synthetic_chain_has_positive_iv_and_nonnegative_oi():
    chain = generate_synthetic_option_chain(S, random_state=1)
    assert (chain["iv"] > 0).all()
    assert (chain["call_oi"] >= 0).all()
    assert (chain["put_oi"] >= 0).all()


def test_synthetic_chain_skew_is_downward_sloping():
    """Put-side (low-strike) IV should exceed call-side (high-strike) IV,
    per the documented skew assumption."""
    chain = generate_synthetic_option_chain(S, random_state=1)
    low_strike_iv = chain.loc[chain["strike"] < S, "iv"].mean()
    high_strike_iv = chain.loc[chain["strike"] > S, "iv"].mean()
    assert low_strike_iv > high_strike_iv


# ---------------------------------------------------------------------------
# GEX / Vanna exposure aggregation
# ---------------------------------------------------------------------------

def test_gamma_exposure_matches_manual_per_strike_sum():
    chain = generate_synthetic_option_chain(S, random_state=2)
    total_gex, gex_by_strike = compute_gamma_exposure(chain, S, T, R)

    manual_total = 0.0
    for _, row in chain.iterrows():
        g = gamma(S, row["strike"], T, R, row["iv"])
        manual_total += (row["call_oi"] - row["put_oi"]) * g * 20 * S ** 2 * 0.01

    assert total_gex == pytest.approx(manual_total, rel=1e-9)
    assert gex_by_strike.sum() == pytest.approx(total_gex, rel=1e-9)


def test_vanna_exposure_matches_manual_per_strike_sum():
    chain = generate_synthetic_option_chain(S, random_state=2)
    total_vex, vex_by_strike = compute_vanna_exposure(chain, S, T, R)

    manual_total = 0.0
    for _, row in chain.iterrows():
        v = vanna(S, row["strike"], T, R, row["iv"])
        manual_total += (row["call_oi"] - row["put_oi"]) * v * 20 * S ** 2 * 0.01

    assert total_vex == pytest.approx(manual_total, rel=1e-9)
    assert vex_by_strike.sum() == pytest.approx(total_vex, rel=1e-9)


def test_symmetric_oi_gives_near_zero_exposure():
    """If call and put OI are identical at every strike, the dealer
    long-calls/short-puts convention should net out to ~zero exposure."""
    chain = generate_synthetic_option_chain(S, random_state=3)
    chain["put_oi"] = chain["call_oi"]  # force symmetry
    total_gex, _ = compute_gamma_exposure(chain, S, T, R)
    assert total_gex == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_build_dashboard_end_to_end():
    result = build_dashboard(spot=19_500.0, dte_days=30, random_state=5)

    assert result["spot"] == 19_500.0
    assert len(result["chain"]) > 0
    assert np.isfinite(result["total_gex"])
    assert np.isfinite(result["total_vex"])
    assert len(result["gex_by_strike"]) == len(result["chain"])
    assert len(result["vex_by_strike"]) == len(result["chain"])
