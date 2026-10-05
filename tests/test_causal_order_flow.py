"""Tests for the lookahead-free order-flow composite (src/order_flow.py)."""

import numpy as np
import pytest

from src.order_flow import causal_mp_filter_signal, mp_filter_signal


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
