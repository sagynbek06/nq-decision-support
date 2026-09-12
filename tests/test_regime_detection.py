"""Tests for Program 1 (HMM regime detection)."""

import numpy as np
import pytest

from src.regime_detection import fit_hmm, label_states, fit_and_label

# This test builds its own small, cleanly-separated synthetic dataset rather
# than depending on notebooks/01_synthetic_data.ipynb's output
# (data/synthetic_nq.csv is gitignored, so it may not exist on a fresh
# checkout or in CI). The mean/vol values below are deliberately more
# separated than the realistic ones used in that notebook: a Gaussian HMM
# fit on daily returns alone genuinely struggles to tell a low-drift bull
# regime apart from sideways chop when their means differ by only a small
# fraction of the daily volatility (as they realistically do) -- that's a
# real limitation of the Gaussian baseline, not a bug, and is exactly why
# ROADMAP.md lists a more robust skew-t emission model as a Phase 1
# follow-up. This test instead checks that the fitting/labeling code is
# *correct* by giving it regimes it should have no trouble separating.
PARAMS = {
    "bull": (0.020, 0.010),
    "bear": (-0.020, 0.015),
    "sideways": (0.0, 0.008),
}

SEGMENT_LENGTH = 100
N_CYCLES = 5  # repeat the bull/bear/sideways cycle so transitions aren't rare
ACCURACY_THRESHOLD = 0.90


def _generate_labeled_returns(rng, segment_length=SEGMENT_LENGTH, n_cycles=N_CYCLES):
    order = ["bull", "bear", "sideways"] * n_cycles
    returns, true_labels = [], []
    for regime in order:
        mu, sigma = PARAMS[regime]
        returns.append(rng.normal(mu, sigma, size=segment_length))
        true_labels.extend([regime] * segment_length)
    return np.concatenate(returns), np.array(true_labels)


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(1)
    return _generate_labeled_returns(rng)


@pytest.fixture(scope="module")
def fitted(synthetic_data):
    returns, _ = synthetic_data
    return fit_and_label(returns)


def test_fit_hmm_returns_requested_number_of_states(fitted):
    model, _, _ = fitted
    assert model.n_components == 3


def test_label_states_assigns_each_regime_exactly_once(fitted):
    _, state_labels, _ = fitted
    assert len(state_labels) == 3
    assert set(state_labels.values()) == {"bull", "bear", "sideways"}


def test_recovers_known_regimes_with_reasonable_accuracy(synthetic_data, fitted):
    _, true_labels = synthetic_data
    _, _, predicted = fitted
    accuracy = np.mean(predicted == true_labels)
    assert accuracy > ACCURACY_THRESHOLD, f"regime recovery accuracy too low: {accuracy:.2%}"


def test_label_states_rejects_wrong_number_of_states(synthetic_data):
    returns, _ = synthetic_data
    # n_init=1 / n_iter=10: fit quality is irrelevant here, only the
    # component-count validation in label_states is under test.
    model = fit_hmm(returns, n_states=4, n_iter=10, n_init=1)
    with pytest.raises(ValueError):
        label_states(model)
