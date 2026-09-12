"""
Program 2: Order Flow Monitor.

Computes Volume Delta and Order Book Imbalance, then denoises the resulting
order-flow signal via Marchenko-Pastur (random matrix theory) eigenvalue
filtering applied to a self-referential panel of smoothed views of that
single series -- a single-instrument adaptation of the standard
cross-sectional technique, not textbook RMT. See `mp_filter_signal`'s
docstring for exactly what that means and its limitations.

We don't have real order book / trade-tape data yet (that's Phase 6:
Bloomberg/Rithmic integration), so this module also generates a synthetic
bid/ask volume dataset correlated with the Phase 1 price series. See
`generate_synthetic_order_flow` for the explicit assumptions behind that
placeholder data. Every number this module produces is illustrative, not
tradable, until Phase 6 replaces the synthetic inputs with real order flow.
"""

import numpy as np
import pandas as pd

# --- synthetic order flow generation parameters (placeholder, undocumented
# real-market calibration -- see generate_synthetic_order_flow docstring) ---
BASE_DAILY_VOLUME = 200_000
BASE_BOOK_DEPTH = 500.0
VOLUME_VOL_SENSITIVITY = 0.8
LIQUIDITY_VOL_SENSITIVITY = 0.6
ORDER_FLOW_RETURN_SENSITIVITY = 0.08
DEPTH_IMBALANCE_RETURN_SENSITIVITY = 3.0

DEFAULT_SMOOTHING_WINDOWS = (3, 5, 10)


def generate_synthetic_order_flow(price_df, random_state=42):
    """
    Generate synthetic daily trade volume and resting order-book depth,
    correlated with `price_df`'s `log_return` column (from
    notebooks/01_synthetic_data.ipynb).

    Assumptions (explicit, since none of this is measured):

    1. Total traded volume scales up with the day's realized volatility
       proxy (|log_return|), consistent with the well-documented positive
       volume/volatility relationship in equity and futures markets (the
       "mixture of distributions hypothesis" -- Clark 1973).
    2. The aggressor split (buy- vs. sell-initiated volume) is correlated
       with the sign and magnitude of the day's return: up days see more
       buy-initiated volume, down days more sell-initiated volume. This
       mirrors the mechanical link between net signed order flow and price
       change in real markets (e.g. Kyle 1985's price-impact framework) --
       it is not a claim about the actual order flow on any historical day.
    3. Resting bid/ask depth (order-book liquidity) *shrinks* on
       higher-volatility days, consistent with liquidity providers reducing
       quoted size when short-term risk is elevated.
    4. The bid/ask depth *imbalance* (as opposed to its total level) is
       modeled as mostly independent noise with only a mild link to the
       day's return, since there is no principled way to simulate genuine
       limit-order-book dynamics (queue position, cancellations, iceberg
       orders) without real data.

    None of these coefficients are calibrated to real NQ order flow -- they
    produce directionally sensible, non-degenerate synthetic data for
    testing the Volume Delta / OBI / MP-filtering logic below, nothing more.
    """
    rng = np.random.default_rng(random_state)
    log_return = price_df["log_return"].to_numpy()
    n = len(log_return)
    abs_return = np.abs(log_return)
    typical_abs_return = np.mean(abs_return)
    return_scale = np.std(log_return)

    # --- total traded volume: scales with the realized-volatility proxy ---
    vol_ratio = abs_return / typical_abs_return
    volume_multiplier = np.clip(1 + VOLUME_VOL_SENSITIVITY * (vol_ratio - 1), 0.3, None)
    volume_noise = rng.lognormal(mean=0.0, sigma=0.25, size=n)
    total_volume = np.clip(BASE_DAILY_VOLUME * volume_multiplier * volume_noise, 1_000, None)

    # --- aggressor split: correlated with the day's return sign/magnitude ---
    buy_signal = ORDER_FLOW_RETURN_SENSITIVITY * log_return / return_scale
    buy_fraction = 0.5 + 0.4 * np.tanh(buy_signal) + rng.normal(0, 0.10, size=n)
    buy_fraction = np.clip(buy_fraction, 0.05, 0.95)
    buy_volume = total_volume * buy_fraction
    sell_volume = total_volume - buy_volume

    # --- resting book depth: shrinks with the realized-volatility proxy ---
    depth_multiplier = np.clip(1 - LIQUIDITY_VOL_SENSITIVITY * (vol_ratio - 1), 0.15, None)
    depth_noise = rng.lognormal(mean=0.0, sigma=0.2, size=n)
    baseline_depth = BASE_BOOK_DEPTH * depth_multiplier * depth_noise

    # --- depth imbalance: mostly noise, mild link to the day's return ---
    imbalance_signal = DEPTH_IMBALANCE_RETURN_SENSITIVITY * log_return / return_scale
    depth_imbalance = 0.05 * np.tanh(imbalance_signal) + rng.normal(0, 0.08, size=n)
    depth_imbalance = np.clip(depth_imbalance, -0.6, 0.6)
    bid_depth = np.clip(baseline_depth * (1 + depth_imbalance), 10, None)
    ask_depth = np.clip(baseline_depth * (1 - depth_imbalance), 10, None)

    out = price_df.copy()
    out["buy_volume"] = buy_volume
    out["sell_volume"] = sell_volume
    out["total_volume"] = total_volume
    out["bid_depth"] = bid_depth
    out["ask_depth"] = ask_depth
    return out


def compute_volume_delta(buy_volume, sell_volume):
    """Signed traded volume: positive means net aggressive buying."""
    return np.asarray(buy_volume) - np.asarray(sell_volume)


def compute_order_book_imbalance(bid_depth, ask_depth):
    """
    (bid - ask) / (bid + ask), bounded in [-1, 1]. Positive means more
    resting size on the bid (buy-side liquidity dominant); negative means
    more on the ask.
    """
    bid_depth = np.asarray(bid_depth, dtype=float)
    ask_depth = np.asarray(ask_depth, dtype=float)
    return (bid_depth - ask_depth) / (bid_depth + ask_depth)


def marchenko_pastur_upper_bound(n_observations, n_features, variance=1.0):
    """
    Upper edge of the Marchenko-Pastur eigenvalue distribution for the
    correlation matrix of an (n_observations x n_features) i.i.d. noise
    matrix. Sample-correlation eigenvalues above this are unlikely to be
    pure noise at this sample size.
    """
    q = n_features / n_observations
    return variance * (1 + np.sqrt(q)) ** 2


def marchenko_pastur_lower_bound(n_observations, n_features, variance=1.0):
    """Lower edge of the Marchenko-Pastur noise band (see upper_bound)."""
    q = n_features / n_observations
    return variance * (1 - np.sqrt(q)) ** 2


def mp_filter_signal(features, min_signal_components=1):
    """
    Denoise a matrix of correlated order-flow views via Marchenko-Pastur
    eigenvalue filtering, returning a single composite signal.

    `features` is a (T, N) array where each column is a differently-scaled
    or differently-smoothed view of the same underlying order-flow pressure
    (e.g. raw and rolling-mean-smoothed Volume Delta / OBI). Standardizes
    each column, eigendecomposes the correlation matrix, and treats
    eigenvalues above the Marchenko-Pastur upper bound as genuine shared
    signal -- the rest are statistically indistinguishable from
    random-matrix noise at this sample size. The composite signal is the
    projection of the standardized features onto the single largest such
    eigenvector -- idiosyncratic noise in any one view is discarded.

    If no eigenvalue clears the MP bound (too little data, or genuinely
    uncorrelated inputs), falls back to keeping the top
    `min_signal_components` eigenvector(s) anyway, so the function always
    returns a usable signal.

    IMPORTANT METHODOLOGICAL CAVEAT -- this is not textbook RMT filtering.

    The classic application (Laloux, Cizeau, Bouchaud & Potters 1999;
    Bouchaud & Potters' RMT-filtering of financial correlation matrices) is
    *cross-sectional*: N columns are N genuinely distinct instruments (e.g.
    different stocks) observed over the same T time periods, so under the
    null hypothesis of "no real co-movement" the columns are independent
    random variables and the Marchenko-Pastur distribution is the correct
    null spectrum to test eigenvalues against.

    We only have one instrument's order flow, so this function instead
    builds its panel *self-referentially*: several rolling-window-smoothed
    copies of the same one or two underlying series (see
    `compute_order_flow_features`). Those columns are not independent even
    under a genuine "no signal" null -- a 5-day and a 10-day rolling mean of
    the identical series share overlapping raw observations by
    construction, so they correlate mechanically regardless of whether the
    order flow contains any real information. That means the MP upper
    bound computed here is a useful, well-defined *threshold* (and the
    formula itself is verified correct against literal i.i.d. noise in
    `tests/test_order_flow.py`), but treating "eigenvalue exceeds the bound"
    as a rigorous statistical test of genuine signal is not justified the
    way it would be for a true cross-sectional panel: some of the
    "signal" this function finds may simply reflect the autocorrelation
    smoothing itself induces, not information content in the order flow.

    This is a deliberate, documented simplification to get an MP-style
    denoising step working end to end with a single instrument, not a
    claim of textbook rigor -- see docs/writeups for the fuller discussion
    and what a cross-sectional version would require (order flow across
    multiple correlated instruments, which we don't have until real data
    is integrated in Phase 6).
    """
    X = np.asarray(features, dtype=float)
    T, N = X.shape

    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - X.mean(axis=0)) / std

    corr = np.corrcoef(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    mp_upper = marchenko_pastur_upper_bound(T, N)
    n_signal = max(int(np.sum(eigvals > mp_upper)), min_signal_components)

    top_eigvec = eigvecs[:, 0]
    if np.dot(top_eigvec, np.ones(N)) < 0:
        top_eigvec = -top_eigvec  # sign convention: align with the simple average

    composite_signal = Z @ top_eigvec

    return {
        "signal": composite_signal,
        "eigenvalues": eigvals,
        "mp_upper_bound": mp_upper,
        "n_signal_components": n_signal,
    }


def compute_order_flow_features(df):
    """
    Build the panel of correlated order-flow views fed into MP filtering:
    raw Volume Delta and Order Book Imbalance, each plus a few rolling-mean
    smoothings, so there's genuine shared structure for the filter to find.

    This is the self-referential panel construction flagged in
    `mp_filter_signal`'s docstring: these columns are smoothed copies of the
    same one or two series, not independent instruments, so they share
    mechanical (window-overlap) correlation regardless of whether the
    underlying order flow carries real information. Read the caveat there
    before treating MP filtering's output as a rigorous significance test.
    """
    volume_delta = compute_volume_delta(df["buy_volume"], df["sell_volume"])
    obi = compute_order_book_imbalance(df["bid_depth"], df["ask_depth"])

    features = {"volume_delta": volume_delta, "obi": obi}
    for window in DEFAULT_SMOOTHING_WINDOWS:
        features[f"volume_delta_ma{window}"] = (
            pd.Series(volume_delta).rolling(window, min_periods=1).mean().to_numpy()
        )
        features[f"obi_ma{window}"] = (
            pd.Series(obi).rolling(window, min_periods=1).mean().to_numpy()
        )

    return pd.DataFrame(features, index=df.index)


def compute_filtered_order_flow_signal(df, min_signal_components=1):
    """
    End-to-end: order-flow columns in `df` (real or, for now,
    `generate_synthetic_order_flow`'s synthetic ones) -> Volume Delta / OBI
    feature panel -> Marchenko-Pastur-filtered composite signal.
    """
    features = compute_order_flow_features(df)
    result = mp_filter_signal(features.to_numpy(), min_signal_components=min_signal_components)
    result["volume_delta"] = features["volume_delta"].to_numpy()
    result["obi"] = features["obi"].to_numpy()
    return result
