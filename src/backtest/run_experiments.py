"""
Runs the walk-forward experiment, appends every trial to results/experiment_log.csv,
and applies the deflated-sharpe gate to the reported trial.

Reported trial: the highest TRAIN Sharpe among the full-pipeline configurations (the
original baseline is the reference and is not a candidate). The choice is made here
in code and passed to the gate by name. num_trials is every row in the log, including
the calibration sweep's 182 settings.

Usage:  python -m src.backtest.run_experiments
Output: results/experiment_log.csv (one append per experiment group)
"""

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtest.overfitting import deflated_sharpe_for_trial
from src.backtest.walk_forward import PipelineConfig, build_signals, run_config

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_nq.csv"
SWEEP_PATH = PROJECT_ROOT / "reports" / "position_sizing_sweep.txt"
LOG_PATH = PROJECT_ROOT / "results" / "experiment_log.csv"
WALK_FORWARD_GROUP = "walk_forward_v1"
SWEEP_GROUP = "calibration_sweep_v1"
BASELINE_NAME = "baseline_pre5a"


def default_configs():
    baseline = PipelineConfig(
        name=BASELINE_NAME,
        regime_mode="label",
        kernel_reading="mean_reversion",
        weights="equal",
        wss_mode="off",
        hurst=False,
    )
    full = PipelineConfig(name="full_meanrev")
    return [
        baseline,
        full,
        replace(full, name="full_momentum", kernel_reading="momentum"),
        replace(full, name="full_dynamic", weights="dynamic"),
        replace(full, name="full_wss_vote", wss_mode="vote"),
        replace(full, name="full_subsignal", regime_mode="subsignal"),
        replace(full, name="full_band010", base_band=0.10),
    ]


def _config_row(trial_id, group, config):
    return {
        "trial_id": trial_id,
        "group": group,
        "name": config.name,
        "regime_mode": config.regime_mode,
        "kernel_reading": config.kernel_reading,
        "weights": config.weights,
        "recal_window": config.recal_window if config.weights == "dynamic" else None,
        "recal_every": config.recal_every if config.weights == "dynamic" else None,
        "wss_mode": config.wss_mode,
        "hurst": config.hurst,
        "base_band": config.base_band,
        "cdar_limit": config.cdar_limit,
        "cdar_alpha": config.cdar_alpha,
        "cdar_window": config.cdar_window,
        "cost_bps": config.cost_per_unit_turnover * 1e4,
    }


def sweep_rows(path):
    frame = pd.read_csv(path, sep=r"\s+", skiprows=2)
    rows = []
    for i, r in enumerate(frame.itertuples(index=False)):
        rows.append({
            "trial_id": f"sweep_{i:03d}",
            "group": SWEEP_GROUP,
            "name": f"sweep_b{r.neutral_band:.2f}_c{r.cdar_limit:.2f}",
            "regime_mode": "label",
            "kernel_reading": "mean_reversion",
            "weights": "equal",
            "wss_mode": "off",
            "hurst": False,
            "base_band": r.neutral_band,
            "cdar_limit": r.cdar_limit,
            "cdar_alpha": 0.05,
            "cdar_window": 252,
            "cost_bps": 2.0,
            "train_sharpe_ann": r.train_sharpe,
            "train_trades_per_week": r.train_trades_per_week,
            "heldout_sharpe_ann": r.heldout_sharpe,
            "heldout_trades_per_week": r.heldout_trades_per_week,
            "reported": False,
        })
    return rows


def main():
    existing = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=["trial_id", "group"])
    if WALK_FORWARD_GROUP in set(existing["group"]):
        raise RuntimeError(f"{WALK_FORWARD_GROUP} is already in {LOG_PATH}; the log is append-only per group")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    signals = build_signals(df)

    results = {}
    rows = []
    for i, config in enumerate(default_configs(), start=1):
        trial_id = f"wf_{i:02d}_{config.name}"
        result = run_config(signals, config)
        results[trial_id] = result
        row = _config_row(trial_id, WALK_FORWARD_GROUP, config)
        row.update(result["metrics"])
        row["reported"] = False
        rows.append(row)

    candidates = [row for row in rows if row["name"] != BASELINE_NAME]
    reported_id = max(candidates, key=lambda row: row["train_sharpe_pp"])["trial_id"]
    for row in rows:
        row["reported"] = row["trial_id"] == reported_id

    new_rows = rows if SWEEP_GROUP in set(existing["group"]) else sweep_rows(SWEEP_PATH) + rows
    log = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)

    gate = deflated_sharpe_for_trial(
        trial_id=reported_id,
        trial_log=log,
        held_out_net_returns=results[reported_id]["heldout_net_returns"],
    )

    print(f"signals: {len(signals.fold_bandwidths)} folds, LOOCV bandwidths {signals.fold_bandwidths}, "
          f"decisions start at bar {signals.start}")
    print()
    print(f"{'trial':<28}{'train SR(ann)':>14}{'train tr/wk':>12}{'held SR(ann)':>13}{'held tr/wk':>11}")
    for row in rows:
        print(f"{row['trial_id']:<28}{row['train_sharpe_ann']:>14.2f}{row['train_trades_per_week']:>12.2f}"
              f"{row['heldout_sharpe_ann']:>13.2f}{row['heldout_trades_per_week']:>11.2f}")
    print()
    print(f"reported trial: {reported_id}")
    print(f"  held-out per-period SR {gate['observed_sr_per_period']:+.4f} over {gate['num_obs']} bars")
    print(f"  num_trials M = {gate['num_trials']} (every logged trial)")
    print(f"  DSR {gate['dsr']:+.4f}   p-value {gate['p_value']:.4f}")
    print(f"  min backtest length at this SR and M: {gate['min_backtest_length']}")
    print(f"Saved {LOG_PATH}")


if __name__ == "__main__":
    main()
