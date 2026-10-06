"""
Does the conformal gate help or just trade less? A held-out comparison.

Usage:  python -m src.risk.conformal_experiment
Output: reports/conformal_gate_experiment.txt

Same causal signals as `src.risk.calibrate_sizing` (walk-forward Student-t HMM
regime labels, causal Marchenko-Pastur order-flow signal, one-sided kernel
deviation, equal-weight consensus), run over several synthetic seeds so one
lucky or unlucky path cannot decide the answer. For each seed:

- BASELINE: the repo's existing procedure, a neutral band picked on the first
  60% of decisions (`select_setting`) and then evaluated on the last 40%.
- GATE: a `ConformalGate` whose forecast scale is fit on that same first 60%
  and then frozen. It trades only when the next-return interval excludes
  zero, and is scored on the same last 40%.

Both pay the same 2 bps per unit of turnover. Only held-out numbers are
reported as results. Everything here is synthetic data, so the output says
whether the mechanism behaves, not whether it makes money in a real market.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.conformal_gate import ConformalGate, fit_scale
from src.kernel_regression import causal_deviation_signal
from src.order_flow import causal_mp_filter_signal, compute_order_flow_features, generate_synthetic_order_flow
from src.risk.backtest import run_sized_backtest, summarize
from src.risk.calibrate_sizing import consensus_for_bars, select_bandwidth, walk_forward_regime_labels
from src.risk.calibration import select_setting, split_index, sweep
from src.synthetic_data import generate_regime_switching_returns

SEEDS = (11, 12, 13, 14, 15, 16)
N_DAYS = 252 * 5
START_PRICE = 15_000.0
COST = 0.0002
ALPHAS = (0.10, 0.25)
REPORT_PATH = PROJECT_ROOT / "reports" / "conformal_gate_experiment.txt"


def build_seed(seed):
    log_returns, _ = generate_regime_switching_returns(n_days=N_DAYS, random_state=seed)
    price = START_PRICE * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({"log_return": log_returns, "close": price})
    regimes, start = walk_forward_regime_labels(log_returns)
    order_flow = causal_mp_filter_signal(compute_order_flow_features(generate_synthetic_order_flow(df)).to_numpy())
    decisions = np.arange(start, N_DAYS - 1)
    split = split_index(len(decisions))
    kernel = causal_deviation_signal(price, select_bandwidth(price[: start + split]))
    results = consensus_for_bars(regimes, order_flow, kernel, decisions)
    scores = np.array([r["score"] for r in results])
    return results, scores, log_returns[decisions + 1], split


def score_positions(positions, next_returns, split):
    previous = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - previous)
    net = positions * next_returns - COST * turnover
    is_trade = np.sign(positions) != np.sign(previous)
    return summarize(net[split:], is_trade[split:])


def gate_positions(scores, next_returns, split, alpha):
    scale = fit_scale(scores[:split], next_returns[:split])
    gate = ConformalGate(alpha=alpha, forecast_scale=scale, warmup=split)
    positions = np.zeros(len(scores))
    covered = []
    for t in range(len(scores)):
        d = gate.decide(scores[t])
        positions[t] = {"long": 1.0, "short": -1.0, "abstain": 0.0}[d["action"]]
        if t >= split:
            covered.append(d["lower"] <= next_returns[t] <= d["upper"])
        gate.update(scores[t], next_returns[t])
    return positions, float(np.mean(covered)), scale


def main():
    rows = []
    for seed in SEEDS:
        results, scores, next_returns, split = build_seed(seed)
        chosen = select_setting(sweep(results, next_returns))
        base = run_sized_backtest(results, next_returns, chosen["neutral_band"], chosen["cdar_limit"])
        base_held = summarize(base["net_returns"][split:], base["is_trade"][split:])
        row = {"seed": seed, "base_sharpe": base_held["sharpe"], "base_tpw": base_held["trades_per_week"]}
        for alpha in ALPHAS:
            positions, coverage, _ = gate_positions(scores, next_returns, split, alpha)
            held = score_positions(positions, next_returns, split)
            tag = f"a{int(alpha * 100)}"
            row[f"{tag}_sharpe"] = held["sharpe"]
            row[f"{tag}_tpw"] = held["trades_per_week"]
            row[f"{tag}_abstain"] = float(np.mean(positions[split:] == 0))
            row[f"{tag}_coverage"] = coverage
        rows.append(row)
        print(f"seed {seed} done", flush=True)

    frame = pd.DataFrame(rows).round(3)
    summary = frame.drop(columns="seed").agg(["mean", "median"]).round(3)
    text = (
        "Held-out comparison (last 40% of decisions), baseline neutral-band sizing vs conformal gate.\n"
        "Synthetic data, 2 bps per unit turnover, forecast scale and band both fit on the first 60% only.\n\n"
        + frame.to_string(index=False) + "\n\n" + summary.to_string() + "\n"
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
