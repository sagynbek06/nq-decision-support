"""Tests for Program 1 (HMM regime detection)."""

import numpy as np
import pytest

from src.regime_detection import fit_hmm, label_states, fit_and_label
from src.synthetic_data import generate_well_separated_regime_returns

# This test uses the well-separated regime generator from
# src/synthetic_data.py rather than notebooks/01_synthetic_data.ipynb's
# realistic one (data/synthetic_nq.csv is gitignored, so it may not exist
# on a fresh checkout or in CI anyway). A Gaussian HMM fit on daily returns
# alone genuinely struggles to tell a low-drift bull regime apart from
# sideways chop when their means differ by only a small fraction of daily
# volatility, as they realistically do -- a real limitation of the Gaussian
# baseline (see docs/writeups/01_regime_detection.md), not a bug, and not
# what this test is checking. This test instead verifies the fitting/
# labeling *code* is correct by giving it regimes it should have no trouble
# separating; see src/synthetic_data.py's module docstring for more.
SEGMENT_LENGTH = 100
N_CYCLES = 5  # repeat the bull/bear/sideways cycle so transitions aren't rare
ACCURACY_THRESHOLD = 0.90


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(1)
    return generate_well_separated_regime_returns(rng, segment_length=SEGMENT_LENGTH, n_cycles=N_CYCLES)


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
