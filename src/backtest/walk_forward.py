"""
Walk-forward backtest of the full consensus pipeline (Phases 5a-5d), and of the
original equal-weight 3-vote consensus, through the same machinery.

build_signals computes every causal input once per data set:
- Program 1 regime: walk-forward Student-t HMM over expanding folds, labelled with
  the causal filter (StudentTHMM.filter_states).
- Program 2 order flow: Marchenko-Pastur composite on an expanding window.
- Program 3 kernel deviation: one-sided Nadaraya-Watson. Each fold's bandwidth is
  chosen by LOOCV on the prices before that fold's test block, so no later price
  reaches any decision.
- Program 4 GEX: the chain built from each bar's own spot. Context only.
- 5a Hurst: the band multiplier, refreshed every HURST_REFRESH bars from trailing
  returns.
- 5b Ax sub-signal: trailing price windows and an OU calibration.
- 5c WSS: surprise context at each bar's own date.

run_config applies one configuration: votes, weights (equal, or re-calibrated on the
resolved history every recal_every bars), consensus, position sizing and the sized
backtest. Nothing used at decision bar t reads data after t. The no-lookahead test in
tests/test_walk_forward.py checks this directly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.ax_regime_switch import ax_regime_subsignal
from src.consensus_engine import compute_consensus
from src.dynamic_weights import compute_dynamic_weights
from src.greeks_dashboard import build_dashboard
from src.hurst_exponent import (
    bootstrap_hurst_confidence_interval,
    compute_dfa_hurst,
    hurst_to_threshold_multiplier,
)
from src.kernel_regression import DEFAULT_BANDWIDTH_GRID, causal_deviation_signal, loocv_select_bandwidth
from src.order_flow import causal_mp_filter_signal, compute_order_flow_features, generate_synthetic_order_flow
from src.ou_half_life import calibrate_ou_half_life
from src.risk.backtest import (
    DEFAULT_CDAR_ALPHA,
    DEFAULT_CDAR_WINDOW,
    DEFAULT_COST_PER_UNIT_TURNOVER,
    run_sized_backtest,
)
from src.risk.calibrate_sizing import N_REGIME_SPLITS, walk_forward_regime_labels
from src.risk.calibration import split_index
from src.risk.cdar import drawdown_series
from src.wss_signal import DEFAULT_EVENT_DATABASE, compute_surprise_context

HURST_REFRESH = 20
HURST_WINDOW = 252
HURST_MIN_OBS = 40
HURST_BOOTSTRAP = 50
MAX_HURST_MULTIPLIER = 2.0
AX_TREND_WINDOW = 150
AX_OU_WINDOW = 100
MIN_HISTORY = 20
TRADING_DAYS_PER_YEAR = 252

REGIME_MODES = ("label", "subsignal")
KERNEL_READINGS = ("mean_reversion", "momentum")
WEIGHT_MODES = ("equal", "dynamic")
WSS_MODES = ("off", "context", "vote")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    regime_mode: str = "label"
    kernel_reading: str = "mean_reversion"
    weights: str = "equal"
    recal_window: int = 120
    recal_every: int = 20
    wss_mode: str = "context"
    hurst: bool = True
    base_band: float = 0.0
    cdar_limit: float = 1.0
    cdar_alpha: float = DEFAULT_CDAR_ALPHA
    cdar_window: int = DEFAULT_CDAR_WINDOW
    cost_per_unit_turnover: float = DEFAULT_COST_PER_UNIT_TURNOVER

    def __post_init__(self):
        for field_name, allowed in (
            ("regime_mode", REGIME_MODES),
            ("kernel_reading", KERNEL_READINGS),
            ("weights", WEIGHT_MODES),
            ("wss_mode", WSS_MODES),
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")
        if self.recal_window < 1 or self.recal_every < 1:
            raise ValueError("recal_window and recal_every must be positive")
        if not 0 <= self.base_band < 1:
            raise ValueError(f"base_band must be in [0, 1), got {self.base_band}")


@dataclass(frozen=True)
class Signals:
    dates: pd.DatetimeIndex
    prices: np.ndarray
    log_returns: np.ndarray
    start: int
    regimes: np.ndarray
    order_flow: np.ndarray
    kernel_dev: np.ndarray
    gex: np.ndarray
    hurst_multiplier: np.ndarray
    ax_subsignal: np.ndarray
    wss_contexts: list
    fold_bandwidths: tuple


def build_signals(df, n_splits=N_REGIME_SPLITS, event_database=None, hmm_kwargs=None):
    event_database = DEFAULT_EVENT_DATABASE if event_database is None else event_database
    hmm_kwargs = {} if hmm_kwargs is None else hmm_kwargs
    log_returns = df["log_return"].to_numpy(dtype=float)
    prices = df["close"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"] if "date" in df.columns else df.index))
    n = len(df)

    regimes, start = walk_forward_regime_labels(log_returns, n_splits, **hmm_kwargs)
    block = n // (n_splits + 1)
    boundaries = [block * k for k in range(n_splits + 2)]
    boundaries[-1] = n
    folds = [(boundaries[k], boundaries[k + 1]) for k in range(1, n_splits + 1)]

    kernel_dev = np.full(n, np.nan)
    bandwidths = []
    for train_end, test_end in folds:
        bandwidth, _, _ = loocv_select_bandwidth(
            np.arange(train_end, dtype=float), prices[:train_end], DEFAULT_BANDWIDTH_GRID
        )
        bandwidths.append(bandwidth)
        deviation = causal_deviation_signal(prices, bandwidth)
        kernel_dev[train_end:test_end] = deviation[train_end:test_end]

    order_flow = causal_mp_filter_signal(
        compute_order_flow_features(generate_synthetic_order_flow(df, causal_scales=True)).to_numpy()
    )

    gex = np.full(n, np.nan)
    hurst_multiplier = np.full(n, np.nan)
    ax_subsignal = np.full(n, np.nan)
    wss_contexts = [None] * n
    current_multiplier = MAX_HURST_MULTIPLIER
    for t in range(start, n - 1):
        gex[t] = build_dashboard(spot=float(prices[t]))["total_gex"]

        if (t - start) % HURST_REFRESH == 0:
            window = log_returns[max(0, t + 1 - HURST_WINDOW): t + 1]
            if len(window) >= HURST_MIN_OBS:
                hurst, *_ = compute_dfa_hurst(window)
                interval, _ = bootstrap_hurst_confidence_interval(
                    window, n_resamples=HURST_BOOTSTRAP, random_state=0
                )
                current_multiplier = hurst_to_threshold_multiplier(hurst, interval)
            else:
                current_multiplier = MAX_HURST_MULTIPLIER
        hurst_multiplier[t] = current_multiplier

        ou = calibrate_ou_half_life(prices[max(0, t + 1 - AX_OU_WINDOW): t + 1])
        ax_subsignal[t] = ax_regime_subsignal(
            regimes[t], prices[max(0, t + 1 - AX_TREND_WINDOW): t + 1], ou_result=ou
        )
        wss_contexts[t] = compute_surprise_context(regimes[t], dates[t], event_database=event_database)

    return Signals(
        dates=dates,
        prices=prices,
        log_returns=log_returns,
        start=start,
        regimes=regimes,
        order_flow=order_flow,
        kernel_dev=kernel_dev,
        gex=gex,
        hurst_multiplier=hurst_multiplier,
        ax_subsignal=ax_subsignal,
        wss_contexts=wss_contexts,
        fold_bandwidths=tuple(bandwidths),
    )


def _recalibrate_weights(vote_log, log_returns, decisions, k, config):
    lo = max(0, k - config.recal_window)
    outcomes = log_returns[decisions[lo:k] + 1]
    histories = {name: np.asarray(vote_log[name][lo:k], dtype=float) for name in ("regime", "order_flow", "kernel")}
    if vote_log["surprise"]:
        histories["surprise"] = np.asarray(vote_log["surprise"][lo:k], dtype=float)
    return compute_dynamic_weights(histories, outcomes, window=config.recal_window)


def _metrics(backtest, split):
    net = backtest["net_returns"]
    trades = backtest["is_trade"]
    train, heldout = net[:split], net[split:]
    n_trades_train = int(trades[:split].sum())
    n_trades_held = int(trades[split:].sum())

    def per_period(values):
        sd = values.std(ddof=1)
        return float(values.mean() / sd) if sd > 0 else 0.0

    return {
        "n_decisions": len(net),
        "split_index": split,
        "train_sharpe_pp": per_period(train),
        "train_sharpe_ann": per_period(train) * np.sqrt(TRADING_DAYS_PER_YEAR),
        "train_trades_per_week": n_trades_train / len(train) * 5,
        "heldout_sharpe_pp": per_period(heldout),
        "heldout_sharpe_ann": per_period(heldout) * np.sqrt(TRADING_DAYS_PER_YEAR),
        "heldout_trades_per_week": n_trades_held / len(heldout) * 5,
        "heldout_skew": float(skew(heldout)),
        "heldout_kurt_raw": float(kurtosis(heldout, fisher=False)),
        "heldout_total_return": float(np.prod(1.0 + heldout) - 1.0),
        "max_drawdown": float(drawdown_series(backtest["equity"]).max()),
    }


def run_config(signals, config):
    decisions = np.arange(signals.start, len(signals.log_returns) - 1)
    n = len(decisions)
    split = split_index(n)

    consensus_results = []
    scores = np.zeros(n)
    band = np.zeros(n)
    vote_log = {"regime": [], "order_flow": [], "kernel": [], "surprise": []}
    weights = None
    include_surprise = config.wss_mode == "vote"

    for k, t in enumerate(decisions):
        if config.weights == "dynamic" and k >= MIN_HISTORY and k % config.recal_every == 0:
            weights = _recalibrate_weights(vote_log, signals.log_returns, decisions, k, config)

        kernel_input = signals.kernel_dev[t] if config.kernel_reading == "mean_reversion" else -signals.kernel_dev[t]
        surprise = signals.wss_contexts[t] if config.wss_mode != "off" else None
        regime_kwargs = {}
        if config.regime_mode == "subsignal":
            regime_kwargs = {"regime_mode": "subsignal", "regime_subsignal": signals.ax_subsignal[t]}

        result = compute_consensus(
            signals.regimes[t],
            signals.order_flow[t],
            kernel_input,
            signals.gex[t],
            weights=weights,
            surprise_context=surprise,
            include_surprise=include_surprise,
            **regime_kwargs,
        )
        consensus_results.append(result)
        scores[k] = result["score"]
        vote_log["regime"].append(result["votes"]["regime"])
        vote_log["order_flow"].append(result["votes"]["order_flow"])
        vote_log["kernel"].append(result["votes"]["kernel"])
        if include_surprise:
            vote_log["surprise"].append(result["votes"]["surprise"])
        band[k] = config.base_band * signals.hurst_multiplier[t] if config.hurst else config.base_band

    next_returns = signals.log_returns[decisions + 1]
    backtest = run_sized_backtest(
        consensus_results,
        next_returns,
        neutral_band=band,
        cdar_limit=config.cdar_limit,
        cdar_alpha=config.cdar_alpha,
        cdar_window=config.cdar_window,
        cost_per_unit_turnover=config.cost_per_unit_turnover,
    )
    return {
        "config": config,
        "decisions": decisions,
        "scores": scores,
        "positions": backtest["positions"],
        "net_returns": backtest["net_returns"],
        "is_trade": backtest["is_trade"],
        "equity": backtest["equity"],
        "band": band,
        "metrics": _metrics(backtest, split),
        "heldout_net_returns": backtest["net_returns"][split:],
    }
