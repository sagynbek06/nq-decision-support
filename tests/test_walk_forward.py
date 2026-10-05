"""Tests for the walk-forward pipeline backtest (src/backtest/walk_forward.py)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import PipelineConfig, build_signals, run_config
from src.synthetic_data import generate_regime_switching_returns

N_BARS = 320
N_SPLITS = 2
FAST_HMM = {"n_init": 5, "n_iter": 50}
EVENTS = pd.DataFrame([
    {"event_date": pd.bdate_range("2020-01-01", periods=N_BARS)[200], "event_type": "NFP", "surprise": 2.0},
    {"event_date": pd.bdate_range("2020-01-01", periods=N_BARS)[250], "event_type": "CPI", "surprise": -1.5},
])
FULL = PipelineConfig(
    name="full",
    regime_mode="subsignal",
    weights="dynamic",
    recal_window=40,
    recal_every=10,
    wss_mode="vote",
    hurst=True,
    base_band=0.1,
)
BASELINE = PipelineConfig(name="baseline", weights="equal", wss_mode="off", hurst=False)


def _frame():
    returns, _ = generate_regime_switching_returns(n_days=N_BARS, random_state=18)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=N_BARS), "close": close, "log_return": returns})


def _with_spike_at(frame, k, factor=50.0):
    """Only bar k's close and the two returns it touches change; every other value stays bit-identical."""
    close = frame["close"].to_numpy().copy()
    log_return = frame["log_return"].to_numpy().copy()
    close[k] *= factor
    log_return[k] += np.log(factor)
    log_return[k + 1] -= np.log(factor)
    altered = frame.copy()
    altered["close"] = close
    altered["log_return"] = log_return
    return altered


@pytest.fixture(scope="module")
def frame():
    return _frame()


@pytest.fixture(scope="module")
def signals(frame):
    return build_signals(frame, n_splits=N_SPLITS, event_database=EVENTS, hmm_kwargs=FAST_HMM)


@pytest.mark.parametrize("config", [FULL, BASELINE], ids=["full", "baseline"])
@pytest.mark.parametrize("spike_bar", [200, 270])
def test_extreme_future_bar_does_not_change_earlier_decisions(frame, signals, config, spike_bar):
    base = run_config(signals, config)
    altered = build_signals(_with_spike_at(frame, spike_bar), n_splits=N_SPLITS,
                            event_database=EVENTS, hmm_kwargs=FAST_HMM)
    changed = run_config(altered, config)

    before = base["decisions"] < spike_bar
    assert before.sum() > 50, "the check must cover a meaningful number of earlier decisions"
    np.testing.assert_array_equal(base["scores"][before], changed["scores"][before])
    np.testing.assert_array_equal(base["positions"][before], changed["positions"][before])


def test_the_spike_does_change_decisions_at_and_after_the_outlier(frame, signals):
    """Non-vacuity: the same outlier must move at least one decision it can reach."""
    spike_bar = 200
    base = run_config(signals, FULL)
    altered = build_signals(_with_spike_at(frame, spike_bar), n_splits=N_SPLITS,
                            event_database=EVENTS, hmm_kwargs=FAST_HMM)
    changed = run_config(altered, FULL)
    after = base["decisions"] >= spike_bar
    assert not np.array_equal(base["scores"][after], changed["scores"][after])


def test_hurst_and_wss_context_do_not_change_positions_at_zero_base_band(signals):
    """At the calibrated band of 0.00, the Hurst multiplier scales nothing and WSS context is not a vote."""
    with_context = run_config(signals, PipelineConfig(name="a", hurst=True, wss_mode="context", base_band=0.0))
    without = run_config(signals, PipelineConfig(name="b", hurst=False, wss_mode="off", base_band=0.0))
    np.testing.assert_array_equal(with_context["positions"], without["positions"])


def test_signals_record_one_bandwidth_per_fold(signals):
    assert len(signals.fold_bandwidths) == N_SPLITS
    assert signals.start == N_BARS // (N_SPLITS + 1)


def test_pipeline_config_rejects_unknown_choices():
    with pytest.raises(ValueError):
        PipelineConfig(name="x", regime_mode="not_a_mode")
    with pytest.raises(ValueError):
        PipelineConfig(name="x", kernel_reading="sideways")
    with pytest.raises(ValueError):
        PipelineConfig(name="x", base_band=1.0)
