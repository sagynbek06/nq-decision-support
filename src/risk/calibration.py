"""
Selection of the neutral band (threshold) and CDaR limit for position sizing.

Target: 1 to 5 trades per week. The selection rule is fixed before any held-out
result is looked at: among settings whose TRAIN trade rate is inside the target,
choose the one with the highest TRAIN Sharpe. Held-out results are reported but
never used to choose.

Under this module's counting (a reversal is one trade), a daily strategy has at
most one trade per bar, so 5 per week is the most it can reach.
"""

import numpy as np

from src.risk.backtest import DEFAULT_COST_PER_UNIT_TURNOVER, run_sized_backtest, summarize

MIN_TRADES_PER_WEEK = 1.0
MAX_TRADES_PER_WEEK = 5.0
TRAIN_FRACTION = 0.6
DEFAULT_BANDS = tuple(round(0.02 * k, 2) for k in range(26))  # 0.00 through 0.50
DEFAULT_CDAR_LIMITS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.00)  # 1.00 is effectively no limit


def split_index(n_decisions, train_fraction=TRAIN_FRACTION):
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    return int(round(n_decisions * train_fraction))


def select_setting(rows, min_trades_per_week=MIN_TRADES_PER_WEEK, max_trades_per_week=MAX_TRADES_PER_WEEK):
    feasible = [
        row for row in rows
        if min_trades_per_week <= row["train_trades_per_week"] <= max_trades_per_week
    ]
    if not feasible:
        raise ValueError("no setting has a train trade rate inside the target")
    return max(feasible, key=lambda row: row["train_sharpe"])


def sweep(
    consensus_results,
    next_returns,
    bands=DEFAULT_BANDS,
    cdar_limits=DEFAULT_CDAR_LIMITS,
    cost_per_unit_turnover=DEFAULT_COST_PER_UNIT_TURNOVER,
    train_fraction=TRAIN_FRACTION,
):
    split = split_index(len(next_returns), train_fraction)
    rows = []
    for band in bands:
        for limit in cdar_limits:
            backtest = run_sized_backtest(
                consensus_results, next_returns, band, limit,
                cost_per_unit_turnover=cost_per_unit_turnover,
            )
            train = summarize(backtest["net_returns"][:split], backtest["is_trade"][:split])
            heldout = summarize(backtest["net_returns"][split:], backtest["is_trade"][split:])
            rows.append({
                "neutral_band": band,
                "cdar_limit": limit,
                "train_sharpe": train["sharpe"],
                "train_trades_per_week": train["trades_per_week"],
                "heldout_sharpe": heldout["sharpe"],
                "heldout_trades_per_week": heldout["trades_per_week"],
            })
    return rows
