"""Tests for Conditional Drawdown at Risk (src/risk/cdar.py)."""

import numpy as np
import pytest

from src.risk.cdar import compute_cdar, drawdown_series, trailing_cdar


def test_drawdown_series_matches_hand_computed_values():
    equity = [100.0, 120.0, 90.0, 110.0, 130.0]
    expected = [0.0, 0.0, 30.0 / 120.0, 10.0 / 120.0, 0.0]
    np.testing.assert_allclose(drawdown_series(equity), expected)


def test_drawdown_series_is_empty_for_empty_equity():
    assert drawdown_series([]).size == 0


def test_cdar_at_alpha_one_is_the_mean_drawdown():
    d = np.array([0.0, 0.1, 0.2, 0.3])
    assert compute_cdar(d, alpha=1.0) == pytest.approx(0.15)


def test_cdar_averages_the_worst_alpha_fraction():
    d = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert compute_cdar(d, alpha=0.2) == pytest.approx((0.9 + 0.8) / 2)


def test_cdar_tail_uses_ceiling_of_alpha_times_n():
    """0.07 * 100 is 7.000000000000001 in float64; the tail must be exactly 7 observations, not 8."""
    d = np.arange(100, dtype=float)
    assert compute_cdar(d, alpha=0.07) == pytest.approx(sum(range(93, 100)) / 7)


def test_cdar_is_zero_for_no_drawdowns_and_for_empty_history():
    assert compute_cdar(np.zeros(50), alpha=0.05) == 0.0
    assert compute_cdar(np.array([]), alpha=0.05) == 0.0


def test_cdar_is_non_increasing_in_alpha_and_equals_mean_at_one():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 0.5, size=200)
    values = [compute_cdar(d, a) for a in (0.01, 0.05, 0.2, 0.5, 1.0)]
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(d.mean())


def test_cdar_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError):
        compute_cdar(np.array([0.1, 0.2]), alpha=0.0)
    with pytest.raises(ValueError):
        compute_cdar(np.array([0.1, 0.2]), alpha=1.5)


def test_trailing_cdar_measures_drawdowns_from_the_all_time_peak():
    equity = [100.0, 200.0, 100.0, 150.0]  # drawdowns from the 200 peak: 0, 0, 0.5, 0.25
    assert trailing_cdar(equity, alpha=1.0, window=2) == pytest.approx(0.375)
    assert trailing_cdar(equity, alpha=1.0, window=1) == pytest.approx(0.25)


def test_trailing_cdar_forgets_drawdowns_outside_the_window():
    equity = [100.0, 50.0, 100.0, 100.0, 100.0]  # one 50% drawdown, then full recovery
    assert trailing_cdar(equity, alpha=1.0, window=3) == 0.0
    assert trailing_cdar(equity, alpha=1.0, window=10) == pytest.approx(0.1)


def test_trailing_cdar_rejects_empty_window():
    with pytest.raises(ValueError):
        trailing_cdar([100.0, 90.0], alpha=0.5, window=0)
