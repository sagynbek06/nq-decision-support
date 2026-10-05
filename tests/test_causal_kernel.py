"""Tests for the lookahead-free kernel deviation signal (src/kernel_regression.py)."""

import numpy as np
import pytest

from src.kernel_regression import (
    causal_deviation_signal,
    causal_nadaraya_watson,
    gaussian_kernel,
    nadaraya_watson_regression,
)


def _brute_force_one_sided(y, bandwidth):
    out = np.empty(len(y))
    for t in range(len(y)):
        i = np.arange(t + 1)
        w = gaussian_kernel((t - i) / bandwidth)
        out[t] = (w @ y[: t + 1]) / w.sum()
    return out


def test_causal_nadaraya_watson_matches_brute_force_one_sided_average():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=120))
    np.testing.assert_allclose(causal_nadaraya_watson(y, 4.0), _brute_force_one_sided(y, 4.0), rtol=1e-12)


def test_causal_estimate_does_not_change_when_the_future_changes():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=200))
    k = 120
    altered = y.copy()
    altered[k + 1:] += rng.normal(scale=50.0, size=len(y) - k - 1)

    np.testing.assert_allclose(
        causal_nadaraya_watson(y, 5.0)[: k + 1], causal_nadaraya_watson(altered, 5.0)[: k + 1], rtol=1e-12
    )
    two_sided_before = nadaraya_watson_regression(np.arange(200.0), y, 5.0)[: k + 1]
    two_sided_after = nadaraya_watson_regression(np.arange(200.0), altered, 5.0)[: k + 1]
    assert not np.allclose(two_sided_before, two_sided_after)


def test_causal_deviation_signal_does_not_change_when_the_future_changes():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=200))
    k = 120
    altered = y.copy()
    altered[k + 1:] += rng.normal(scale=50.0, size=len(y) - k - 1)

    base = causal_deviation_signal(y, bandwidth=5.0, standardization_window=40, min_periods=10)
    changed = causal_deviation_signal(altered, bandwidth=5.0, standardization_window=40, min_periods=10)
    np.testing.assert_allclose(base[: k + 1], changed[: k + 1], rtol=1e-12, equal_nan=True)
    assert np.isfinite(base[k])


def test_causal_deviation_signal_is_nan_until_enough_residuals_exist():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(size=100))
    signal = causal_deviation_signal(y, bandwidth=3.0, standardization_window=50, min_periods=20)
    assert np.isnan(signal[:19]).all()
    assert np.isfinite(signal[19:]).all()
