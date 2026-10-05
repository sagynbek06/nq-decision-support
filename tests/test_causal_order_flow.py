"""Tests for the lookahead-free order-flow composite (src/order_flow.py)."""

import numpy as np
import pandas as pd
import pytest

from src.order_flow import causal_mp_filter_signal, generate_synthetic_order_flow, mp_filter_signal

VOLUME_COLUMNS = ["buy_volume", "sell_volume", "total_volume", "bid_depth", "ask_depth"]


def _returns_with_extreme_bar_at(index, size=300, seed=4):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, size=size)
    altered = returns.copy()
    altered[index] = 0.3
    return pd.DataFrame({"log_return": returns}), pd.DataFrame({"log_return": altered})


def test_causal_composite_is_nan_until_enough_rows_exist():
    rng = np.random.default_rng(0)
    signal = causal_mp_filter_signal(rng.normal(size=(120, 6)), min_periods=60)
    assert np.isnan(signal[:59]).all()
    assert np.isfinite(signal[59:]).all()


def test_causal_composite_equals_the_filter_on_the_expanding_window():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6))
    signal = causal_mp_filter_signal(X, min_periods=60)
    assert signal[100] == pytest.approx(mp_filter_signal(X[:101])["signal"][-1])


def test_causal_composite_does_not_change_when_the_future_changes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 6))
    altered = X.copy()
    altered[101:] = rng.normal(scale=10.0, size=(49, 6))

    base = causal_mp_filter_signal(X, min_periods=60)
    changed = causal_mp_filter_signal(altered, min_periods=60)
    np.testing.assert_allclose(base[:101], changed[:101], rtol=1e-12, equal_nan=True)
    assert not np.allclose(base[101:], changed[101:])


def test_causal_order_flow_does_not_change_earlier_bars_when_a_later_return_changes():
    before, after = _returns_with_extreme_bar_at(200)
    base = generate_synthetic_order_flow(before, causal_scales=True)
    changed = generate_synthetic_order_flow(after, causal_scales=True)
    for column in VOLUME_COLUMNS:
        np.testing.assert_array_equal(base[column].to_numpy()[:200], changed[column].to_numpy()[:200])


def test_full_sample_order_flow_scales_do_leak_later_returns_into_earlier_bars():
    """Non-vacuity: the default generator's full-sample scales must move bars before the change."""
    before, after = _returns_with_extreme_bar_at(200)
    base = generate_synthetic_order_flow(before)
    changed = generate_synthetic_order_flow(after)
    assert not np.array_equal(base["total_volume"].to_numpy()[:200], changed["total_volume"].to_numpy()[:200])
