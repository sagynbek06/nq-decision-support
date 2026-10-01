"""Tests for Program 3 (Nadaraya-Watson kernel regression)."""

import numpy as np
import pytest

from src.kernel_regression import (
    gaussian_kernel,
    nadaraya_watson_regression,
    loocv_select_bandwidth,
    compute_deviation_signal,
    fit_kernel_regression,
)


def test_gaussian_kernel_peak_and_symmetry():
    assert gaussian_kernel(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert gaussian_kernel(2.0) == pytest.approx(gaussian_kernel(-2.0))
    assert gaussian_kernel(3.0) < gaussian_kernel(1.0)  # decays with distance


def test_nadaraya_watson_recovers_a_known_smooth_function():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 300)
    true_y = np.sin(x) * 5 + 20
    noise_std = 0.3
    y = true_y + rng.normal(0, noise_std, size=len(x))

    estimate = nadaraya_watson_regression(x, y, bandwidth=0.4)

    mse = np.mean((estimate - true_y) ** 2)
    # The estimate should land close to the noise floor; a generous margin
    # keeps this from being flaky while still catching real regressions.
    assert mse < 4 * noise_std ** 2


def test_nadaraya_watson_huge_bandwidth_converges_to_global_mean():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + rng.normal(0, 0.1, size=len(x))

    estimate = nadaraya_watson_regression(x, y, bandwidth=1e6)

    np.testing.assert_allclose(estimate, np.full_like(estimate, y.mean()), atol=1e-6)


def test_nadaraya_watson_tiny_bandwidth_converges_to_training_values():
    x = np.linspace(0, 10, 50)
    y = np.sin(x) * 3 + 7

    estimate = nadaraya_watson_regression(x, y, bandwidth=1e-6, x_eval=x)

    np.testing.assert_allclose(estimate, y, atol=1e-6)


def test_loocv_select_bandwidth_matches_brute_force_recomputation():
    """White-box check: the vectorized implementation's scores must match
    an independently-written, unvectorized leave-one-out computation."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 60)
    y = np.sin(x) + rng.normal(0, 0.2, size=len(x))
    bandwidths = [0.2, 0.5, 1.0, 2.0, 5.0]

    best_h, best_score, scores = loocv_select_bandwidth(x, y, bandwidths)

    manual_scores = {}
    for h in bandwidths:
        preds = []
        for i in range(len(x)):
            mask = np.arange(len(x)) != i
            w = gaussian_kernel((x[i] - x[mask]) / h)
            preds.append(np.sum(w * y[mask]) / np.sum(w))
        manual_scores[h] = np.mean((y - np.array(preds)) ** 2)

    for h in bandwidths:
        assert scores[h] == pytest.approx(manual_scores[h], rel=1e-9)
    assert best_h == min(manual_scores, key=manual_scores.get)
    assert best_score == pytest.approx(scores[best_h])


def test_compute_deviation_signal_is_standardized():
    # Deterministic, exactly-symmetric residuals (not sampled noise, whose
    # sample mean would only be *approximately* zero) so mean/std are exact.
    y = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    estimate = np.full_like(y, 100.0)  # residuals: [-10, -5, 0, 5, 10]

    signal = compute_deviation_signal(y, estimate)

    assert signal.mean() == pytest.approx(0.0, abs=1e-10)
    assert signal.std() == pytest.approx(1.0, abs=1e-10)


def test_compute_deviation_signal_handles_zero_residual_std():
    y = np.full(10, 5.0)
    estimate = np.full(10, 5.0)  # residuals are all exactly zero

    signal = compute_deviation_signal(y, estimate)

    np.testing.assert_array_equal(signal, np.zeros(10))


def test_fit_kernel_regression_end_to_end_on_a_self_contained_series():
    """
    Self-contained synthetic price series (not data/synthetic_nq.csv, which
    is gitignored) with a known smooth trend plus noise, mirroring the
    project's established pattern of hermetic tests. Checks the full
    pipeline recovers that trend and returns well-formed outputs.
    """
    rng = np.random.default_rng(4)
    t = np.arange(400)
    trend = 15_000 + 50 * np.sin(t / 60) + t * 2
    price = trend + rng.normal(0, 40, size=len(t))

    result = fit_kernel_regression(price)

    assert len(result["estimate"]) == len(price)
    assert len(result["signal"]) == len(price)
    assert not np.isnan(result["estimate"]).any()
    assert not np.isnan(result["signal"]).any()
    assert result["bandwidth"] in result["loocv_scores"]

    corr = np.corrcoef(result["estimate"], trend)[0, 1]
    assert corr > 0.95
