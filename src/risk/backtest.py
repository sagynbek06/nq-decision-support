"""
Causal, sized backtest of consensus-driven position sizing with CDaR throttling.

Bars are processed in order. At each bar the trailing CDaR is measured on the
equity realized so far, position_size turns that bar's consensus result into a
signed position, and the position earns the next bar's return net of a cost on
the turnover it required. Nothing used at bar t reads a return after t.

A trade is a change of sign: entry from flat, exit to flat, or a reversal.
Changing size within one direction is a rebalance. It is charged the cost per
unit of turnover but not counted as a trade, which is what lets the neutral
band control trade frequency.
"""

import numpy as np

from src.risk.cdar import trailing_cdar
from src.risk.position_sizing import position_size

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_WEEK = 5
DEFAULT_CDAR_ALPHA = 0.05
DEFAULT_CDAR_WINDOW = 252
DEFAULT_COST_PER_UNIT_TURNOVER = 0.0002  # 2 bps per unit of turnover


def run_sized_backtest(
    consensus_results,
    next_returns,
    neutral_band,
    cdar_limit,
    cdar_alpha=DEFAULT_CDAR_ALPHA,
    cdar_window=DEFAULT_CDAR_WINDOW,
    cost_per_unit_turnover=DEFAULT_COST_PER_UNIT_TURNOVER,
):
    consensus_results = list(consensus_results)
    next_returns = np.asarray(next_returns, dtype=float)
    if len(consensus_results) != len(next_returns):
        raise ValueError("consensus_results and next_returns must be the same length")

    n = len(next_returns)
    bands = np.broadcast_to(np.asarray(neutral_band, dtype=float), (n,))
    equity = [1.0]
    positions = np.zeros(n)
    net_returns = np.zeros(n)
    is_trade = np.zeros(n, dtype=bool)
    previous = 0.0
    for t in range(n):
        cdar = trailing_cdar(equity, cdar_alpha, cdar_window)
        position = position_size(consensus_results[t], cdar, cdar_limit, neutral_band=bands[t])
        turnover = abs(position - previous)
        net = position * next_returns[t] - cost_per_unit_turnover * turnover
        equity.append(equity[-1] * (1.0 + net))
        positions[t] = position
        net_returns[t] = net
        is_trade[t] = np.sign(position) != np.sign(previous)
        previous = position

    return {
        "positions": positions,
        "net_returns": net_returns,
        "is_trade": is_trade,
        "equity": np.array(equity),
    }


def summarize(net_returns, is_trade):
    net_returns = np.asarray(net_returns, dtype=float)
    is_trade = np.asarray(is_trade, dtype=bool)
    n = len(net_returns)
    if n == 0:
        raise ValueError("cannot summarize an empty segment")
    sd = net_returns.std(ddof=1) if n > 1 else 0.0
    sharpe = float(net_returns.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0
    return {
        "bars": n,
        "trades": int(is_trade.sum()),
        "trades_per_week": float(is_trade.sum()) / n * TRADING_DAYS_PER_WEEK,
        "sharpe": sharpe,
    }
