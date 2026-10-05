"""
Calibrates the neutral band and CDaR limit for consensus-driven position sizing,
and plots the sweep the choice is made from.

Usage:  python -m src.risk.calibrate_sizing
Output: reports/position_sizing_sweep.png and reports/position_sizing_sweep.txt

Every signal feeding the sweep is causal:
- Program 1: walk-forward Student-t HMM over 8 expanding folds, labelled with the
  filtered (forward-pass) state, not Viterbi, whose path uses later returns.
  Bars before the first fold's test block have no out-of-sample regime and are
  not decisions.
- Program 2: Marchenko-Pastur composite fit on an expanding window.
- Program 3: one-sided Nadaraya-Watson, bandwidth chosen by LOOCV on the training
  prices only, residuals standardized by a trailing 252-bar std.
- Consensus: equal weights over the regime, order-flow and kernel votes. The
  Hurst band, WSS and dynamic weights are left out, so the swept band is the only
  threshold.

The first 60% of decisions are the training split, used for selection. The last
40% is held out and only reported. Costs are 2 bps per unit of turnover in the
selection, and 0, 5 and 10 bps are reported for the chosen setting.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.consensus_engine import compute_consensus, vote_from_kernel_deviation
from src.kernel_regression import DEFAULT_BANDWIDTH_GRID, causal_deviation_signal, loocv_select_bandwidth
from src.order_flow import causal_mp_filter_signal, compute_order_flow_features, generate_synthetic_order_flow
from src.regime_detection_robust import fit_and_label
from src.risk.backtest import run_sized_backtest, summarize
from src.risk.calibration import (
    MAX_TRADES_PER_WEEK,
    MIN_TRADES_PER_WEEK,
    select_setting,
    split_index,
    sweep,
)
from src.risk.cdar import drawdown_series

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_nq.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOT_PATH = REPORTS_DIR / "position_sizing_sweep.png"
TXT_PATH = REPORTS_DIR / "position_sizing_sweep.txt"
N_REGIME_SPLITS = 8
COST_SENSITIVITY = (0.0, 0.0005, 0.001)
BASE_COST = 0.0002


def walk_forward_regime_labels(log_returns, n_splits=N_REGIME_SPLITS):
    n = len(log_returns)
    block = n // (n_splits + 1)
    boundaries = [block * k for k in range(n_splits + 2)]
    boundaries[-1] = n
    labels = np.full(n, None, dtype=object)
    for k in range(1, n_splits + 1):
        train_end, test_end = boundaries[k], boundaries[k + 1]
        model, state_labels, _ = fit_and_label(log_returns[:train_end])
        hidden = model.filter_states(log_returns[:test_end])
        decoded = np.array([state_labels[s] for s in hidden], dtype=object)
        labels[train_end:test_end] = decoded[train_end:test_end]
    return labels, block


def select_bandwidth(train_prices):
    x = np.arange(len(train_prices), dtype=float)
    best, _, _ = loocv_select_bandwidth(x, train_prices, DEFAULT_BANDWIDTH_GRID)
    return best


def consensus_for_bars(regimes, order_flow, kernel, bars):
    return [
        compute_consensus(regimes[t], float(order_flow[t]), float(kernel[t]), 0.0)
        for t in bars
    ]


def plot_sweep(rows, chosen):
    frame = pd.DataFrame(rows)
    curve = frame[frame["cdar_limit"] == chosen["cdar_limit"]].sort_values("neutral_band")

    fig, (ax_trades, ax_sharpe) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_trades.axhspan(MIN_TRADES_PER_WEEK, MAX_TRADES_PER_WEEK, color="#2ca02c", alpha=0.12,
                      label="target: 1 to 5 trades per week")
    ax_trades.plot(curve["neutral_band"], curve["train_trades_per_week"], marker="o", label="train (selection)")
    ax_trades.plot(curve["neutral_band"], curve["heldout_trades_per_week"], marker="s",
                   label="held-out (reported only)")
    ax_trades.set_ylabel("trades per week")
    ax_trades.legend(loc="upper right")

    ax_sharpe.plot(curve["neutral_band"], curve["train_sharpe"], marker="o", label="train (selection)")
    ax_sharpe.plot(curve["neutral_band"], curve["heldout_sharpe"], marker="s", label="held-out (reported only)")
    ax_sharpe.axhline(0.0, color="gray", linewidth=0.8)
    ax_sharpe.set_ylabel("annualized Sharpe, net of costs")
    ax_sharpe.set_xlabel("neutral band (threshold on the consensus score)")
    ax_sharpe.legend(loc="upper right")

    for ax in (ax_trades, ax_sharpe):
        ax.axvline(chosen["neutral_band"], color="black", linestyle="--", linewidth=1)

    fig.suptitle(
        f"Position sizing sweep at CDaR limit {chosen['cdar_limit']:.2f} "
        f"(chosen band {chosen['neutral_band']:.2f})"
    )
    fig.tight_layout()
    return fig


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run notebooks/01_synthetic_data.ipynb first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    log_returns = df["log_return"].to_numpy()
    price = df["close"].to_numpy()
    n = len(df)

    regimes, start = walk_forward_regime_labels(log_returns)
    order_flow = causal_mp_filter_signal(
        compute_order_flow_features(generate_synthetic_order_flow(df)).to_numpy()
    )

    decisions = np.arange(start, n - 1)
    split = split_index(len(decisions))
    bandwidth = select_bandwidth(price[: start + split])
    kernel = causal_deviation_signal(price, bandwidth)

    if not (np.isfinite(order_flow[decisions]).all() and np.isfinite(kernel[decisions]).all()):
        raise RuntimeError("causal signals are not finite over the decision range")

    results = consensus_for_bars(regimes, order_flow, kernel, decisions)
    next_returns = log_returns[decisions + 1]

    kernel_votes = np.array([vote_from_kernel_deviation(kernel[t]) for t in decisions])
    voted = kernel_votes != 0
    kernel_hit = float(np.mean(np.sign(kernel_votes[voted]) == np.sign(next_returns[voted])))

    rows = sweep(results, next_returns)
    chosen = select_setting(rows)

    base = run_sized_backtest(results, next_returns, chosen["neutral_band"], chosen["cdar_limit"])
    max_drawdown = float(drawdown_series(base["equity"]).max())

    sensitivity = []
    for cost in COST_SENSITIVITY:
        bt = run_sized_backtest(results, next_returns, chosen["neutral_band"], chosen["cdar_limit"],
                                cost_per_unit_turnover=cost)
        sensitivity.append({
            "cost_bps": round(cost * 1e4),
            "train_sharpe": summarize(bt["net_returns"][:split], bt["is_trade"][:split])["sharpe"],
            "heldout_sharpe": summarize(bt["net_returns"][split:], bt["is_trade"][split:])["sharpe"],
        })

    frame = pd.DataFrame(rows)
    frame["train_feasible"] = frame["train_trades_per_week"].between(MIN_TRADES_PER_WEEK, MAX_TRADES_PER_WEEK)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = frame.round(3).to_string(index=False)
    TXT_PATH.write_text(
        "Sweep over neutral band x CDaR limit. Selection uses the train columns only.\n\n" + table + "\n",
        encoding="utf-8",
    )
    fig = plot_sweep(rows, chosen)
    fig.savefig(PLOT_PATH, dpi=150)

    split_date = df["date"].iloc[decisions[split]].date()
    print(f"decisions: {len(decisions)}  train: {split}  held-out: {len(decisions) - split}  "
          f"(held-out starts {split_date})")
    print(f"causal kernel bandwidth (train LOOCV): {bandwidth}")
    print(f"causal kernel vote hit rate vs next bar: {kernel_hit:.3f} over {int(voted.sum())} bars")
    print()
    print(f"chosen: neutral band {chosen['neutral_band']:.2f}, CDaR limit {chosen['cdar_limit']:.2f}")
    print(f"  train:     Sharpe {chosen['train_sharpe']:+.2f}, {chosen['train_trades_per_week']:.2f} trades/week")
    print(f"  held-out:  Sharpe {chosen['heldout_sharpe']:+.2f}, {chosen['heldout_trades_per_week']:.2f} trades/week")
    print(f"  full-period max drawdown at {BASE_COST * 1e4:.0f} bps: {max_drawdown:.1%}")
    print()
    print("cost sensitivity for the chosen setting (selection used the base cost):")
    for row in sensitivity:
        print(f"  {row['cost_bps']:>3} bps  train Sharpe {row['train_sharpe']:+.2f}  "
              f"held-out Sharpe {row['heldout_sharpe']:+.2f}")
    print()
    print(f"Saved {PLOT_PATH}")
    print(f"Saved {TXT_PATH}")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
