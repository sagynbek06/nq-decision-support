"""
Centralized synthetic regime-switching returns generation.

Two generators live here, and they're deliberately different, not
duplicates of each other:

- `generate_regime_switching_returns` is the realistic generator originally
  built in notebooks/01_synthetic_data.ipynb: Gamma-distributed regime
  durations, a transition structure where momentum regimes cool through
  sideways before reversing, and Student-t (fat-tailed) daily shocks. Its
  default parameters are the ones that notebook's markdown cells explain
  and justify (bull/bear/sideways drift, vol, and typical duration) -- see
  that notebook for the full rationale, not repeated here. This generator
  is intentionally *hard* to fit: bull and sideways drift differ by only a
  small fraction of daily volatility, which is realistic but genuinely low-
  power for a return-only HMM to separate (see
  docs/writeups/01_regime_detection.md).

- `generate_well_separated_regime_returns` is the generator originally
  hand-rolled in tests/test_regime_detection.py: fixed-length contiguous
  bull/bear/sideways cycles with no Gamma-distributed duration variability
  and no probabilistic transitions, using parameters deliberately more
  separated than the realistic ones above. It exists to validate that HMM
  fitting/labeling *code* is correct, independent of the realistic
  generator's genuinely hard separability problem -- a model that cannot
  recover these cleanly-separated regimes has an implementation bug, not
  just a hard dataset.

A third generator, `generate_ornstein_uhlenbeck`, produces a genuinely
mean-reverting *level* series (not a return series like the two above) --
added for the Hurst exponent module's test suite (src/hurst_exponent.py),
which needs a reliably mean-reverting case to confirm DFA correctly reports
Hurst well below 0.5. It's a classic, theoretically-grounded choice for
that (unlike relying on an emergent property of the regime-switching
generators, which turned out not to reliably produce a persistent/trending
signature at realistic parameters -- see hurst_exponent.py's test fixtures
for what does).

Not every synthetic-data generator in this project lives here. Order flow
(tests/test_order_flow.py), the kernel-regression trend-recovery check
(tests/test_kernel_regression.py), and the options chain generator
(src/greeks_dashboard.py's generate_synthetic_option_chain) each need
different, genuinely simpler things -- a plain i.i.d. fat-tailed return
series with no regime structure, a deterministic known trend to check
recovery against, and a strikes/IV/open-interest snapshot respectively --
and gain nothing from being forced through the regime-switching machinery
here. Consolidating them would be bundling unrelated data shapes into one
module for the appearance of DRY-ness rather than an actual reduction in
duplicated logic.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252

# annual_drift / annual_vol: approximate historical NQ behavior in each
# regime. mean_duration / duration_shape: parameters of the Gamma
# distribution used to sample how long (in trading days) a single visit to
# the regime lasts. See notebooks/01_synthetic_data.ipynb for the full
# justification of these specific values.
DEFAULT_REGIMES = {
    "bull": {
        "annual_drift": 0.20,
        "annual_vol": 0.15,
        "mean_duration": 90,
        "duration_shape": 3.0,
    },
    "bear": {
        "annual_drift": -0.30,
        "annual_vol": 0.30,
        "mean_duration": 45,
        "duration_shape": 2.5,
    },
    "sideways": {
        "annual_drift": 0.0,
        "annual_vol": 0.13,
        "mean_duration": 60,
        "duration_shape": 3.0,
    },
}

# A regime never "self-transitions" here because its dwell time is already
# modeled by the Gamma-distributed duration draw. Momentum regimes are more
# likely to cool into sideways consolidation before flipping to the opposite
# regime, mirroring how real market cycles usually don't reverse instantly.
DEFAULT_TRANSITION_PROBS = {
    "bull":     {"sideways": 0.65, "bear": 0.35},
    "bear":     {"sideways": 0.65, "bull": 0.35},
    "sideways": {"bull": 0.5, "bear": 0.5},
}

DEFAULT_N_DAYS = TRADING_DAYS_PER_YEAR * 5  # 5 years of daily bars
DEFAULT_RNG_SEED = 18

# Deliberately more separated than DEFAULT_REGIMES -- see module docstring.
DEFAULT_WELL_SEPARATED_PARAMS = {
    "bull": (0.020, 0.010),
    "bear": (-0.020, 0.015),
    "sideways": (0.0, 0.008),
}


def sample_duration(shape, mean, rng, min_days=5):
    """Gamma-distributed regime duration (right-skewed, always positive)."""
    scale = mean / shape
    duration = rng.gamma(shape, scale)
    return max(int(round(duration)), min_days)


def build_regime_path(n_days, rng, regimes=None, transition_probs=None, start_regime="sideways"):
    """Simulate a day-by-day regime path via Gamma-distributed dwell times
    and probabilistic transitions. See module docstring."""
    regimes = DEFAULT_REGIMES if regimes is None else regimes
    transition_probs = DEFAULT_TRANSITION_PROBS if transition_probs is None else transition_probs

    path = []
    current = start_regime
    while len(path) < n_days:
        duration = sample_duration(
            regimes[current]["duration_shape"],
            regimes[current]["mean_duration"],
            rng,
        )
        path.extend([current] * duration)
        next_options = list(transition_probs[current].keys())
        next_probs = list(transition_probs[current].values())
        current = rng.choice(next_options, p=next_probs)
    return path[:n_days]


def simulate_log_returns(regime_path, rng, regimes=None, trading_days_per_year=TRADING_DAYS_PER_YEAR, t_dof=5):
    """
    Draw daily log returns per regime using Student-t innovations instead of
    Gaussian ones, since NQ (and equity index futures generally) exhibit
    fatter tails and occasional large jumps than a normal distribution
    implies. t_dof=5 gives noticeably fat tails while keeping variance finite.
    """
    regimes = DEFAULT_REGIMES if regimes is None else regimes

    n = len(regime_path)
    regime_arr = np.array(regime_path)
    log_returns = np.empty(n)

    for regime_name, params in regimes.items():
        mask = regime_arr == regime_name
        n_r = mask.sum()
        if n_r == 0:
            continue
        daily_drift = params["annual_drift"] / trading_days_per_year
        daily_vol = params["annual_vol"] / np.sqrt(trading_days_per_year)
        # scale t-draws so their variance matches daily_vol**2
        t_scale = daily_vol * np.sqrt((t_dof - 2) / t_dof)
        shocks = rng.standard_t(t_dof, size=n_r) * t_scale
        log_returns[mask] = daily_drift + shocks

    return log_returns


def generate_regime_switching_returns(
    n_days=DEFAULT_N_DAYS,
    regimes=None,
    transition_probs=None,
    trading_days_per_year=TRADING_DAYS_PER_YEAR,
    t_dof=5,
    start_regime="sideways",
    rng=None,
    random_state=DEFAULT_RNG_SEED,
):
    """
    The realistic regime-switching return generator from
    notebooks/01_synthetic_data.ipynb: Gamma-distributed regime durations,
    momentum-cools-through-sideways transitions, Student-t daily shocks.
    All defaults match that notebook's parameters exactly.

    Pass your own `rng` (a numpy Generator) if subsequent code needs to
    continue the same draw sequence -- e.g. notebook 01 threads one `rng`
    through this function and then its own OHLC-bar synthesis, so bar noise
    continues from where the return simulation left off rather than
    restarting from a fresh seed. Without an explicit `rng`, one is created
    from `random_state`.

    Returns (log_returns, regime_path) as (np.ndarray[float], np.ndarray[str]).
    """
    if rng is None:
        rng = np.random.default_rng(random_state)

    regime_path = build_regime_path(n_days, rng, regimes, transition_probs, start_regime)
    log_returns = simulate_log_returns(regime_path, rng, regimes, trading_days_per_year, t_dof)
    return log_returns, np.array(regime_path)


def generate_well_separated_regime_returns(rng, segment_length=100, n_cycles=5, params=None):
    """
    The deliberately-easy regime generator originally hand-rolled in
    tests/test_regime_detection.py: fixed-length contiguous bull/bear/
    sideways cycles (no Gamma-distributed duration, no probabilistic
    transitions) with more separation between regimes than is realistic.
    See module docstring for why this exists alongside the realistic
    generator rather than replacing it.

    Returns (returns, true_labels) as (np.ndarray[float], np.ndarray[str]).
    """
    params = DEFAULT_WELL_SEPARATED_PARAMS if params is None else params

    order = ["bull", "bear", "sideways"] * n_cycles
    returns, true_labels = [], []
    for regime in order:
        mu, sigma = params[regime]
        returns.append(rng.normal(mu, sigma, size=segment_length))
        true_labels.extend([regime] * segment_length)
    return np.concatenate(returns), np.array(true_labels)


DEFAULT_OU_THETA = 0.15  # mean-reversion speed; see generate_ornstein_uhlenbeck
DEFAULT_OU_SIGMA = 1.0


def generate_ornstein_uhlenbeck(n, theta=DEFAULT_OU_THETA, mu=0.0, sigma=DEFAULT_OU_SIGMA, x0=None, dt=1.0, random_state=None):
    """
    Simulate a mean-reverting level series via the Ornstein-Uhlenbeck SDE,
    discretized with Euler-Maruyama:

        dX_t = theta * (mu - X_t) * dt + sigma * sqrt(dt) * dW_t

    Unlike `generate_regime_switching_returns` and
    `generate_well_separated_regime_returns` above (which produce *return*
    series with directional structure), this produces a *level* series that
    pulls back toward `mu` at rate `theta` -- the classic textbook
    mean-reverting process (used for modeling things like interest rates,
    spreads, or implied vol in the wider literature). To use it as a
    "returns" series (e.g. as DFA/Hurst input), take `np.diff(x)`: OU's
    increments are negatively autocorrelated by construction (after an
    up-move, reversion toward `mu` makes a down-move more likely), which is
    exactly the signature a correctly-implemented Hurst estimator should
    read as persistently mean-reverting (H well below 0.5).

    `theta` controls reversion strength: empirically (see
    tests/test_hurst_exponent.py), `theta=0.15` with `n=2500` reliably
    gives DFA Hurst around 0.17-0.19 across random seeds -- comfortably and
    consistently below 0.5 without needing extreme parameters.

    Returns the level series `x` (length `n`), not increments.
    """
    rng = np.random.default_rng(random_state)
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    noise = rng.normal(size=n - 1)
    sqrt_dt = np.sqrt(dt)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * sqrt_dt * noise[t - 1]
    return x
