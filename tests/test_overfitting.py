"""Tests for the overfitting gate (src/backtest/overfitting.py) around the deflated-sharpe package."""

import numpy as np
import pandas as pd
import pytest
from deflated_sharpe import deflated_sharpe_ratio
from scipy.stats import kurtosis, skew

from src.backtest.overfitting import deflated_sharpe_for_trial, per_period_sharpe


def _trial_log(n_trials):
    return pd.DataFrame({"trial_id": ["reported"] + [f"other_{i}" for i in range(n_trials - 1)]})


def _returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.001, 0.01, size=252)


def test_gate_passes_the_per_period_sharpe_and_daily_observation_count():
    returns = _returns()
    gate = deflated_sharpe_for_trial(trial_id="reported", trial_log=_trial_log(100), held_out_net_returns=returns)
    sr = returns.mean() / returns.std(ddof=1)
    expected, _ = deflated_sharpe_ratio(
        observed_sr=sr,
        num_trials=100,
        num_obs=len(returns),
        skewness=float(skew(returns)),
        kurtosis=float(kurtosis(returns, fisher=False)),
    )
    assert gate["observed_sr_per_period"] == pytest.approx(sr)
    assert gate["dsr"] == pytest.approx(expected)


def test_annualized_input_would_inflate_the_statistic_so_the_gate_must_not_use_it():
    """The package's docstring says 'annualized'; this documents why the gate passes per-period."""
    returns = _returns()
    gate = deflated_sharpe_for_trial(trial_id="reported", trial_log=_trial_log(100), held_out_net_returns=returns)
    sr = returns.mean() / returns.std(ddof=1)
    annualized_dsr, _ = deflated_sharpe_ratio(
        observed_sr=sr * np.sqrt(252),
        num_trials=100,
        num_obs=len(returns),
        skewness=float(skew(returns)),
        kurtosis=float(kurtosis(returns, fisher=False)),
    )
    assert annualized_dsr > gate["dsr"] + 5


def test_num_trials_counts_every_row_in_the_log():
    gate = deflated_sharpe_for_trial(trial_id="reported", trial_log=_trial_log(7), held_out_net_returns=_returns())
    assert gate["num_trials"] == 7


def test_the_reported_trial_must_be_named_explicitly():
    with pytest.raises(TypeError):
        deflated_sharpe_for_trial(trial_log=_trial_log(3), held_out_net_returns=_returns())


def test_an_unknown_trial_id_is_rejected():
    with pytest.raises(ValueError):
        deflated_sharpe_for_trial(trial_id="missing", trial_log=_trial_log(3), held_out_net_returns=_returns())


def test_per_period_sharpe_is_mean_over_sample_standard_deviation():
    values = np.array([0.01, -0.005, 0.02, 0.0])
    assert per_period_sharpe(values) == pytest.approx(values.mean() / values.std(ddof=1))
    assert per_period_sharpe(np.zeros(5)) == 0.0
