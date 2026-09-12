"""Tests for Program 2 (order flow monitor)."""

import numpy as np
import pandas as pd
import pytest

from src.order_flow import (
    generate_synthetic_order_flow,
    compute_volume_delta,
    compute_order_book_imbalance,
    marchenko_pastur_upper_bound,
    marchenko_pastur_lower_bound,
    mp_filter_signal,
    compute_filtered_order_flow_signal,
)

# Self-contained synthetic price series (not the gitignored data/synthetic_nq.csv)
# so this test suite doesn't depend on notebook 01 having been run first.
N_DAYS = 1000


@pytest.fixture(scope="module")
def price_df():
    rng = np.random.default_rng(7)
    log_return = rng.standard_t(df=5, size=N_DAYS) * 0.01
    dates = pd.bdate_range(start="2020-01-01", periods=N_DAYS)
    return pd.DataFrame({"log_return": log_return}, index=dates)


@pytest.fixture(scope="module")
def order_flow_df(price_df):
    return generate_synthetic_order_flow(price_df, random_state=42)


def test_synthetic_order_flow_has_expected_columns(order_flow_df):
    for col in ["buy_volume", "sell_volume", "total_volume", "bid_depth", "ask_depth"]:
        assert col in order_flow_df.columns


def test_synthetic_order_flow_values_are_nonnegative(order_flow_df):
    cols = ["buy_volume", "sell_volume", "total_volume", "bid_depth", "ask_depth"]
    assert (order_flow_df[cols] >= 0).all().all()


def test_synthetic_buy_plus_sell_equals_total(order_flow_df):
    diff = (order_flow_df["buy_volume"] + order_flow_df["sell_volume"]) - order_flow_df["total_volume"]
    assert np.abs(diff).max() < 1e-6


def test_synthetic_order_flow_correlates_with_return_but_is_not_redundant(price_df, order_flow_df):
    """
    Volume Delta should meaningfully track the day's return (that's the
    documented assumption), but a correlation near 1.0 would mean it's
    just a relabeling of price rather than a complementary signal.
    """
    vd = compute_volume_delta(order_flow_df["buy_volume"], order_flow_df["sell_volume"])
    corr = np.corrcoef(vd, price_df["log_return"])[0, 1]
    assert 0.2 < corr < 0.85


def test_volume_delta_is_buy_minus_sell():
    buy = np.array([100.0, 50.0, 0.0])
    sell = np.array([40.0, 50.0, 30.0])
    np.testing.assert_allclose(compute_volume_delta(buy, sell), [60.0, 0.0, -30.0])


def test_order_book_imbalance_bounded_and_signed():
    bid = np.array([100.0, 50.0, 30.0])
    ask = np.array([100.0, 150.0, 70.0])
    obi = compute_order_book_imbalance(bid, ask)
    assert np.all(obi >= -1) and np.all(obi <= 1)
    assert obi[0] == pytest.approx(0.0)     # balanced book
    assert obi[1] < 0                        # more ask depth -> negative
    assert obi[2] < 0


def test_marchenko_pastur_bounds_match_random_matrix_theory():
    """Empirical eigenvalue spectrum of a pure-noise correlation matrix
    should sit close to the theoretical MP band."""
    rng = np.random.default_rng(0)
    T, N = 3000, 8
    X = rng.normal(size=(T, N))
    eigvals = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))

    mp_upper = marchenko_pastur_upper_bound(T, N)
    mp_lower = marchenko_pastur_lower_bound(T, N)

    # Finite-sample fluctuations (Tracy-Widom) mean the empirical edge can
    # sit slightly outside the asymptotic MP edge; allow a modest margin.
    assert eigvals.max() < mp_upper * 1.2
    assert eigvals.min() > mp_lower * 0.6


def test_mp_filter_recovers_an_injected_common_factor():
    rng = np.random.default_rng(1)
    T, N = 400, 8
    common_factor = rng.normal(size=T)
    loadings = rng.uniform(0.5, 1.5, size=N)
    noise = rng.normal(size=(T, N)) * 2.0  # noise dominates each individual column
    X = common_factor[:, None] * loadings[None, :] + noise

    result = mp_filter_signal(X)

    assert result["n_signal_components"] >= 1
    corr = abs(np.corrcoef(result["signal"], common_factor)[0, 1])
    assert corr > 0.7


def test_mp_filter_falls_back_gracefully_on_pure_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 8))
    result = mp_filter_signal(X, min_signal_components=1)

    assert result["n_signal_components"] == 1  # nothing clears the MP bound
    assert np.all(result["eigenvalues"] < result["mp_upper_bound"] * 1.05)
    assert len(result["signal"]) == 400
    assert not np.isnan(result["signal"]).any()


def test_filtered_signal_is_less_noisy_than_raw_volume_delta(order_flow_df):
    """MP filtering should smooth out idiosyncratic noise while retaining
    genuine shared signal, i.e. reduce day-to-day roughness."""

    def roughness(x):
        x = np.asarray(x, dtype=float)
        x = (x - x.mean()) / x.std()
        return np.mean(np.abs(np.diff(x)))

    result = compute_filtered_order_flow_signal(order_flow_df)

    assert len(result["signal"]) == len(order_flow_df)
    assert not np.isnan(result["signal"]).any()
    assert roughness(result["signal"]) < roughness(result["volume_delta"])
