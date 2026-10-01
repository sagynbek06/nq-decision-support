"""
End-to-end pipeline visualization: Program 1 (regime detection) + Program 2
(order flow monitor) on the synthetic NQ dataset from Phase 1.

Pure visualization -- no new modeling. Loads data/synthetic_nq.csv, fits the
Student-t regime detection HMM (src/regime_detection_robust.py -- the Phase 1
outcome, shown to outperform the Gaussian baseline; see
docs/writeups/01_regime_detection.md) to get a decoded regime per day, runs
the Program 2 order-flow pipeline (src/order_flow.py) on synthetic bid/ask
data correlated with the same price series to get a denoised composite
order-flow signal per day, and plots both against a shared date axis.

Usage:  python src/visualize_pipeline.py
Output: reports/pipeline_overview.png (also shown in a window if run
        interactively).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.regime_detection_robust import fit_and_label
from src.order_flow import generate_synthetic_order_flow, compute_filtered_order_flow_signal

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_nq.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "pipeline_overview.png"

REGIME_COLORS = {"bull": "#2ca02c", "bear": "#d62728", "sideways": "#7f7f7f"}


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run notebooks/01_synthetic_data.ipynb first."
        )
    return pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")


def run_regime_detection(df):
    """Program 1: decode a bull/bear/sideways label for every day."""
    _, _, predicted = fit_and_label(df["log_return"].to_numpy())
    return predicted


def run_order_flow(df):
    """Program 2: synthetic order flow -> MP-filtered composite signal."""
    order_flow_df = generate_synthetic_order_flow(df)
    result = compute_filtered_order_flow_signal(order_flow_df)
    return result["signal"]


def plot_pipeline(df, regimes, order_flow_signal):
    fig, (ax_price, ax_flow) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    change_points = [i for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]]
    segment_starts = [0] + change_points
    segment_ends = change_points + [len(regimes)]
    for start, end in zip(segment_starts, segment_ends):
        ax_price.axvspan(
            df.index[start],
            df.index[min(end, len(df.index) - 1)],
            color=REGIME_COLORS[regimes[start]],
            alpha=0.15,
            linewidth=0,
        )

    ax_price.plot(df.index, df["close"], color="black", linewidth=1)
    ax_price.set_ylabel("Price")
    ax_price.set_title("NQ Price with Detected Regime (Program 1) and Order Flow Signal (Program 2)")
    ax_price.legend(
        handles=[Patch(facecolor=c, alpha=0.3, label=r) for r, c in REGIME_COLORS.items()],
        loc="upper left",
    )

    ax_flow.plot(df.index, order_flow_signal, color="steelblue", linewidth=0.8)
    ax_flow.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_flow.set_ylabel("Order Flow Signal\n(MP-filtered)")
    ax_flow.set_xlabel("Date")

    fig.tight_layout()
    return fig


def main():
    df = load_data()
    regimes = run_regime_detection(df)
    order_flow_signal = run_order_flow(df)

    fig = plot_pipeline(df, regimes, order_flow_signal)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")

    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
