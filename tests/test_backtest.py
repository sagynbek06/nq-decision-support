"""Tests for the causal sized backtest (src/risk/backtest.py)."""

import numpy as np
import pytest

from src.risk.backtest import run_sized_backtest, summarize


def _consensus(score):
    direction = np.sign(score)
    return {"score": score, "votes": {"regime": direction, "order_flow": direction, "kernel": direction}}


def test_positions_are_zero_inside_the_band_and_follow_the_score_sign_outside_it():
    scores = [0.1, 0.5, -0.5, 0.0]
    result = run_sized_backtest([_consensus(s) for s in scores], np.zeros(4), neutral_band=0.2, cdar_limit=1.0)
    assert result["positions"][0] == 0.0
    assert result["positions"][1] > 0
    assert result["positions"][2] < 0
    assert result["positions"][3] == 0.0


def test_equity_compounds_position_returns_less_turnover_costs():
    returns = np.array([0.01, 0.0, 0.02])  # never negative, so there is no drawdown to throttle on
    cost = 0.001
    result = run_sized_backtest([_consensus(0.5)] * 3, returns, neutral_band=0.2, cdar_limit=1.0,
                                cost_per_unit_turnover=cost)

    position = (0.5 - 0.2) / (1.0 - 0.2)
    assert result["positions"] == pytest.approx([position] * 3)
    equity, previous = 1.0, 0.0
    for r in returns:
        equity *= 1.0 + position * r - cost * abs(position - previous)
        previous = position
    assert result["equity"][-1] == pytest.approx(equity)


def test_trades_count_sign_changes_and_not_size_rebalances():
    scores = [0.5, 0.6, 0.7, -0.5, -0.6, 0.0]
    result = run_sized_backtest([_consensus(s) for s in scores], np.zeros(6), neutral_band=0.2, cdar_limit=1.0)
    assert result["is_trade"].tolist() == [True, False, False, True, False, True]


def test_future_changes_do_not_move_earlier_positions_or_equity():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-1, 1, size=120)
    returns = rng.normal(0, 0.01, size=120)
    k = 70
    base = run_sized_backtest([_consensus(s) for s in scores], returns, neutral_band=0.1, cdar_limit=0.2)

    scores_altered = scores.copy()
    scores_altered[k:] = rng.uniform(-1, 1, size=120 - k)
    returns_altered = returns.copy()
    returns_altered[k:] = rng.normal(0, 0.05, size=120 - k)
    changed = run_sized_backtest([_consensus(s) for s in scores_altered], returns_altered,
                                 neutral_band=0.1, cdar_limit=0.2)

    np.testing.assert_array_equal(base["positions"][:k], changed["positions"][:k])
    np.testing.assert_allclose(base["equity"][: k + 1], changed["equity"][: k + 1])
    assert not np.allclose(base["positions"][k:], changed["positions"][k:])


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        run_sized_backtest([_consensus(0.5)], np.zeros(2), neutral_band=0.1, cdar_limit=1.0)


def test_summarize_reports_annualized_sharpe_and_weekly_trade_rate():
    net = np.array([0.02, 0.0, 0.02, 0.0])
    is_trade = np.array([True, False, False, False])
    result = summarize(net, is_trade)
    assert result["sharpe"] == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252))
    assert result["trades_per_week"] == pytest.approx(1 / 4 * 5)
    assert result["bars"] == 4
    assert result["trades"] == 1


def test_summarize_rejects_an_empty_segment():
    with pytest.raises(ValueError):
        summarize(np.array([]), np.array([], dtype=bool))


def test_per_bar_neutral_band_changes_only_the_bars_it_touches():
    consensus = [_consensus(0.3)] * 3
    returns = np.zeros(3)
    scalar = run_sized_backtest(consensus, returns, neutral_band=0.0, cdar_limit=1.0, cost_per_unit_turnover=0.0)
    per_bar = run_sized_backtest(consensus, returns, neutral_band=np.array([0.0, 0.5, 0.0]), cdar_limit=1.0,
                                 cost_per_unit_turnover=0.0)
    assert per_bar["positions"][0] == scalar["positions"][0]
    assert per_bar["positions"][1] == 0.0
    assert per_bar["positions"][2] == scalar["positions"][2]
