"""Tests for the calibration selection rule (src/risk/calibration.py)."""

import numpy as np
import pytest

from src.risk.calibration import select_setting, split_index, sweep


def _row(band, limit, train_sharpe, train_tpw, heldout_sharpe=0.0, heldout_tpw=0.0):
    return {
        "neutral_band": band,
        "cdar_limit": limit,
        "train_sharpe": train_sharpe,
        "train_trades_per_week": train_tpw,
        "heldout_sharpe": heldout_sharpe,
        "heldout_trades_per_week": heldout_tpw,
    }


def test_select_setting_picks_the_highest_train_sharpe_among_feasible_settings():
    rows = [_row(0.1, 0.2, 0.5, 2.0), _row(0.2, 0.2, 1.1, 3.0), _row(0.3, 0.2, 0.9, 4.0)]
    assert select_setting(rows)["neutral_band"] == 0.2


def test_select_setting_excludes_settings_outside_the_trade_target():
    rows = [_row(0.0, 0.2, 3.0, 6.0), _row(0.2, 0.2, 1.0, 2.0)]
    assert select_setting(rows)["neutral_band"] == 0.2


def test_select_setting_never_uses_held_out_results():
    rows = [_row(0.1, 0.2, 0.9, 2.0, heldout_sharpe=-5.0), _row(0.2, 0.2, 0.5, 2.0, heldout_sharpe=9.0)]
    assert select_setting(rows)["neutral_band"] == 0.1


def test_select_setting_raises_when_nothing_is_feasible():
    with pytest.raises(ValueError):
        select_setting([_row(0.0, 0.2, 2.0, 9.0), _row(0.5, 0.2, 2.0, 0.1)])


def test_split_index_rounds_and_rejects_bad_fractions():
    assert split_index(100, 0.6) == 60
    assert split_index(7, 0.6) == 4
    with pytest.raises(ValueError):
        split_index(100, 1.0)


def test_sweep_returns_one_row_per_setting_with_both_segments():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-1, 1, size=200)
    consensus = [
        {"score": s, "votes": {"regime": np.sign(s), "order_flow": np.sign(s), "kernel": np.sign(s)}}
        for s in scores
    ]
    returns = rng.normal(0, 0.01, size=200)

    rows = sweep(consensus, returns, bands=(0.0, 0.3), cdar_limits=(0.1, 1.0))

    assert {(r["neutral_band"], r["cdar_limit"]) for r in rows} == {(0.0, 0.1), (0.0, 1.0), (0.3, 0.1), (0.3, 1.0)}
    expected_keys = {"train_sharpe", "train_trades_per_week", "heldout_sharpe", "heldout_trades_per_week"}
    assert all(expected_keys <= set(r) for r in rows)
