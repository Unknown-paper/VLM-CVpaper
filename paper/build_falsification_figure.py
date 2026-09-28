"""Build a compact main-paper summary of the locked falsification analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"
TASKS = ["Recognition", "Binary", "Composition"]
TASK_COLORS = ["#4C78A8", "#F2B134", "#E45756"]
MODEL_COLORS = ["#6F4EA1", "#2A9D8F"]


def error_bars(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([values - low, high - values])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)

    # A: E17 breaking minus preserving, joint post-encoder movement.
    ax = axes[0]
    x = np.arange(2)
    width = 0.34
    topo = {
        "LLaVA": (np.array([0.61, 0.99]), np.array([-0.27, -0.26]), np.array([1.47, 2.25])),
        "Qwen": (np.array([-0.26, -1.22]), np.array([-1.24, -3.49]), np.array([0.69, 1.00])),
    }
    for idx, (model, (value, low, high)) in enumerate(topo.items()):
        pos = x + (idx - 0.5) * width
        ax.bar(pos, value, width, color=MODEL_COLORS[idx], label=model, zorder=2)
        ax.errorbar(pos, value, yerr=error_bars(value, low, high), fmt="none",
                    ecolor="#202020", capsize=2, linewidth=1, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, ["Binary", "Composition"])
    ax.set_ylabel("Accuracy change (pp)")
    ax.set_title("A  2D adjacency is insufficient", loc="left", weight="bold")
    ax.set_ylim(-4.2, 3.2)
    ax.legend(frameon=False, ncol=2, loc="lower left")

    # B: E18 five-view majority minus canonical S0.
    ax = axes[1]
    x = np.arange(3)
    ensemble = {
        "LLaVA": (
            np.array([0.0, -3.13, -0.52]),
            np.array([0.0, -5.73, -3.65]),
            np.array([0.0, -1.04, 3.13]),
        ),
        "Qwen": (
            np.array([0.0, -6.25, -5.73]),
            np.array([0.0, -9.90, -9.38]),
            np.array([0.0, -3.13, -2.60]),
        ),
    }
    for idx, (model, (value, low, high)) in enumerate(ensemble.items()):
        pos = x + (idx - 0.5) * width
        ax.bar(pos, value, width, color=MODEL_COLORS[idx], label=model, zorder=2)
        ax.errorbar(pos, value, yerr=error_bars(value, low, high), fmt="none",
                    ecolor="#202020", capsize=2, linewidth=1, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, ["Recognition", "Binary", "Composition"], rotation=12)
    ax.set_title("B  Five-view vote does not help", loc="left", weight="bold")
    ax.set_ylim(-11.0, 4.2)

    # C: 96-scene generic sequence-order control.
    ax = axes[2]
    x = np.arange(3)
    width = 0.34
    generic = {
        "Visual regions": (
            np.array([0.0, -25.00, -25.00]),
            np.array([0.0, -33.33, -35.42]),
            np.array([0.0, -16.67, -14.58]),
        ),
        "Language query": (
            np.array([0.0, -51.04, -69.79]),
            np.array([0.0, -61.48, -79.17]),
            np.array([0.0, -39.58, -60.42]),
        ),
    }
    generic_colors = ["#2F66E8", "#7A38EA"]
    for idx, (condition, (value, low, high)) in enumerate(generic.items()):
        pos = x + (idx - 0.5) * width
        ax.bar(pos, value, width, color=generic_colors[idx], label=condition, zorder=2)
        ax.errorbar(pos, value, yerr=error_bars(value, low, high), fmt="none",
                    ecolor="#202020", capsize=2, linewidth=1, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, ["Recognition", "Binary", "Composition"], rotation=12)
    ax.set_title("C  Generic order control", loc="left", weight="bold")
    ax.set_ylim(-83, 9)
    ax.legend(frameon=False, ncol=1, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#D8DDE5", linewidth=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(OUT / "falsification_main.pdf", bbox_inches="tight")
    fig.savefig(OUT / "falsification_main.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
