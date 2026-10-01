"""
Program 4: Greeks Dashboard.

Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho, Vanna) for European
options, plus dealer Gamma Exposure (GEX) and Vanna Exposure aggregated
across an options chain, and the implied volatility skew.

There is no real options chain data yet (Phase 6: real data integration),
so -- same situation as Program 2's order flow -- this module generates a
synthetic chain (strikes, open interest, implied vols) around the current
underlying price. See `generate_synthetic_option_chain` for the explicit
assumptions behind it.

This is a SNAPSHOT dashboard, not a historical daily time series like
Programs 1-3: `build_dashboard` takes a single spot price and a single
expiration (days-to-expiry) and computes GEX/Vanna exposure and the skew
for that one moment, the way a real GEX dashboard shows "as of right now,"
not a multi-day backtest. Pass today's (or any day's) closing price from
Phase 1's data to get that day's snapshot; there's no date-indexed state
inside this module itself.

GEX/VEX SIGN CONVENTION -- dealers are assumed net long gamma from calls and
net short gamma from puts (equivalently: assumed to have sold puts and
bought/hedged calls on net). This is the standard simplifying convention
used by essentially every public GEX dashboard (SqueezeMetrics/SpotGamma
and the retail commentary that followed them), not something specific to
this project -- actual dealer positioning isn't public data anywhere, so
every such tool, real or synthetic, makes this same assumption rather than
observing it directly.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes pricing and Greeks (closed-form, per contract, European)
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under Black-Scholes-Merton (q = dividend yield)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, q))
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dS."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    T, q = np.asarray(T, dtype=float), np.asarray(q, dtype=float)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(S, K, T, r, sigma, q=0.0):
    """d^2 Price / dS^2. Identical for calls and puts at the same (S,K,T,sigma)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    S, T, q = (np.asarray(v, dtype=float) for v in (S, T, q))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """dPrice/dsigma, per unit (i.e. per 100% vol move). Identical for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    S, T, q = (np.asarray(v, dtype=float) for v in (S, T, q))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dT with T = time *to* expiry, negated so this reads as decay per unit time."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, q = (np.asarray(v, dtype=float) for v in (S, K, T, r, q))
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dr, per unit (i.e. per 100% rate move)."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    K, T, r = (np.asarray(v, dtype=float) for v in (K, T, r))
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def vanna(S, K, T, r, sigma, q=0.0):
    """
    d(Delta)/dsigma == d(Vega)/dS. Identical for calls and puts: Delta_put =
    Delta_call - exp(-qT) (put-call parity), a constant w.r.t. sigma, so
    their sigma-derivatives coincide.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    T, q = np.asarray(T, dtype=float), np.asarray(q, dtype=float)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


# ---------------------------------------------------------------------------
# Synthetic options chain (placeholder until Phase 6 real data integration)
# ---------------------------------------------------------------------------

STRIKE_SPACING = 50.0
N_STRIKES_EACH_SIDE = 40
DEFAULT_DTE_DAYS = 30
DEFAULT_RATE = 0.05
ATM_IV = 0.15
SKEW_SLOPE = -0.4
SKEW_CURVATURE = 0.35
OI_DECAY_SCALE = 0.05  # in log-moneyness units
BASE_OI = 5_000


def generate_synthetic_option_chain(spot, random_state=42):
    """
    Generate a synthetic single-expiration options chain around `spot`.

    Assumptions (explicit, since none of this is measured):

    1. Strikes are evenly spaced at STRIKE_SPACING around spot, spanning
       N_STRIKES_EACH_SIDE on either side -- a stand-in for a real listed
       strike grid, not calibrated to actual NQ option listings.
    2. Implied vol follows a downward-sloping quadratic skew in
       log-moneyness (ATM_IV + SKEW_SLOPE*m + SKEW_CURVATURE*m^2, m =
       log(K/S)): higher IV for low (put-side) strikes than high (call-side)
       ones. This is directionally realistic for equity-index options --
       the well-documented "volatility skew" driven by crash-hedging demand
       -- but the specific slope/curvature values are illustrative, not
       fit to real NQ option quotes.
    3. Open interest is concentrated near the money and decays with
       |log-moneyness| (a reasonable qualitative pattern -- real OI really
       does cluster near spot and at round strikes -- but the exact decay
       shape and magnitude are not calibrated to real positioning data,
       which in any case reflects actual market participants' choices that
       can't be synthesized from the price series alone).
    4. Call and put open interest are drawn independently (both decaying
       the same way with distance from the money), i.e. this does not
       encode any real skew in put-vs-call demand.

    None of these are calibrated to real NQ options; they produce a
    directionally sensible, non-degenerate chain for testing the Greeks /
    GEX / Vanna exposure logic below, nothing more.
    """
    rng = np.random.default_rng(random_state)
    offsets = np.arange(-N_STRIKES_EACH_SIDE, N_STRIKES_EACH_SIDE + 1)
    strikes = spot + offsets * STRIKE_SPACING
    strikes = strikes[strikes > 0]

    log_moneyness = np.log(strikes / spot)

    iv = ATM_IV + SKEW_SLOPE * log_moneyness + SKEW_CURVATURE * log_moneyness ** 2
    iv = iv + rng.normal(0, 0.005, size=len(strikes))
    iv = np.clip(iv, 0.03, None)

    oi_shape = np.exp(-0.5 * (log_moneyness / OI_DECAY_SCALE) ** 2)
    call_oi = np.clip(
        (BASE_OI * oi_shape * rng.lognormal(0, 0.3, size=len(strikes))).round(), 0, None
    )
    put_oi = np.clip(
        (BASE_OI * oi_shape * rng.lognormal(0, 0.3, size=len(strikes))).round(), 0, None
    )

    return pd.DataFrame({
        "strike": strikes,
        "iv": iv,
        "call_oi": call_oi,
        "put_oi": put_oi,
    })


# ---------------------------------------------------------------------------
# Dealer Gamma Exposure (GEX) and Vanna Exposure (VEX)
# ---------------------------------------------------------------------------

CONTRACT_MULTIPLIER = 20  # NQ-equivalent notional multiplier, illustrative


def compute_gamma_exposure(chain, spot, T, r=DEFAULT_RATE, contract_multiplier=CONTRACT_MULTIPLIER, q=0.0):
    """
    Dealer Gamma Exposure per strike and total, in dollars-of-gamma per 1%
    underlying move: GEX_k = (CallOI_k - PutOI_k) * Gamma_k * multiplier *
    S^2 * 0.01, using the long-calls/short-puts dealer convention documented
    in this module's docstring. Gamma is the same for calls and puts at a
    given strike under Black-Scholes, so only one Gamma per strike is needed.

    Returns (total_gex, per_strike_series).
    """
    g = gamma(spot, chain["strike"].to_numpy(), T, r, chain["iv"].to_numpy(), q)
    per_strike = (chain["call_oi"].to_numpy() - chain["put_oi"].to_numpy()) * g
    per_strike = per_strike * contract_multiplier * spot ** 2 * 0.01
    per_strike = pd.Series(per_strike, index=chain.index, name="gex")
    return float(per_strike.sum()), per_strike


def compute_vanna_exposure(chain, spot, T, r=DEFAULT_RATE, contract_multiplier=CONTRACT_MULTIPLIER, q=0.0):
    """
    Dealer Vanna Exposure per strike and total, in dollars-of-vanna per 1%
    underlying move per 1-point vol move: same long-calls/short-puts
    convention and per-strike formula as compute_gamma_exposure, using
    Vanna (also identical for calls/puts under Black-Scholes) in place of
    Gamma.

    Returns (total_vex, per_strike_series).
    """
    v = vanna(spot, chain["strike"].to_numpy(), T, r, chain["iv"].to_numpy(), q)
    per_strike = (chain["call_oi"].to_numpy() - chain["put_oi"].to_numpy()) * v
    per_strike = per_strike * contract_multiplier * spot ** 2 * 0.01
    per_strike = pd.Series(per_strike, index=chain.index, name="vex")
    return float(per_strike.sum()), per_strike


def build_dashboard(spot, dte_days=DEFAULT_DTE_DAYS, r=DEFAULT_RATE, random_state=42):
    """
    End-to-end: a spot price -> synthetic chain -> total/per-strike GEX and
    Vanna exposure -> the chain's IV skew (strike, iv columns -- ready to
    plot directly).
    """
    chain = generate_synthetic_option_chain(spot, random_state=random_state)
    T = dte_days / 365.0

    total_gex, gex_by_strike = compute_gamma_exposure(chain, spot, T, r)
    total_vex, vex_by_strike = compute_vanna_exposure(chain, spot, T, r)

    return {
        "spot": spot,
        "dte_days": dte_days,
        "chain": chain,
        "total_gex": total_gex,
        "gex_by_strike": gex_by_strike,
        "total_vex": total_vex,
        "vex_by_strike": vex_by_strike,
    }
