"""
Overfitting gate: the Deflated Sharpe Ratio from the `deflated-sharpe` package
(Apache-2.0, pinned in requirements.txt). The formulas are the package's; this
module only feeds it the right inputs.

Units. deflated_sharpe_ratio takes a PER-PERIOD Sharpe ratio with num_obs return
observations. Its docstring says "annualized", but its expected-maximum term
sqrt(1/(T-1)) is a per-period quantity. Passing an annualized Sharpe with daily
returns makes a random strategy look overwhelmingly significant (see
tests/test_overfitting.py). This module always passes the per-period Sharpe.

Trials. The caller names the reported trial explicitly. There is no default and no
maximum-selection here. num_trials is the number of rows in the trial log, so every
configuration evaluated in the search counts.
"""

import numpy as np
from deflated_sharpe import deflated_sharpe_ratio, min_backtest_length
from scipy.stats import kurtosis, skew


def per_period_sharpe(returns):
    values = np.asarray(returns, dtype=float)
    sd = values.std(ddof=1)
    return float(values.mean() / sd) if sd > 0 else 0.0


def deflated_sharpe_for_trial(*, trial_id, trial_log, held_out_net_returns):
    if trial_id not in set(trial_log["trial_id"]):
        raise ValueError(f"unknown trial_id {trial_id!r}")
    returns = np.asarray(held_out_net_returns, dtype=float)
    observed_sr = per_period_sharpe(returns)
    skewness = float(skew(returns))
    kurt = float(kurtosis(returns, fisher=False))
    num_trials = len(trial_log)
    dsr, p_value = deflated_sharpe_ratio(
        observed_sr=observed_sr,
        num_trials=num_trials,
        num_obs=len(returns),
        skewness=skewness,
        kurtosis=kurt,
    )
    min_obs = (
        min_backtest_length(target_sr=observed_sr, num_trials=num_trials, skewness=skewness, kurtosis=kurt)
        if observed_sr > 0
        else None
    )
    return {
        "trial_id": trial_id,
        "observed_sr_per_period": observed_sr,
        "num_trials": num_trials,
        "num_obs": len(returns),
        "skewness": skewness,
        "kurtosis_raw": kurt,
        "dsr": dsr,
        "p_value": p_value,
        "min_backtest_length": min_obs,
    }
