"""Build the manuscript cross-model figure from the locked E11 summary.

The values below are copied from artifacts_stage4/analysis_final/effects.json.
They are paired perturbed-minus-canonical accuracy changes in percentage points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures"

MODELS = ["LLaVA-\nOneVision-7B", "Qwen2.5-\nVL-7B", "InternVL3-8B"]
TASKS = ["Recognition", "Binary relation", "Composition"]
COLORS = ["#4C78A8", "#F2B134", "#E45756"]

# perturbed minus canonical, in percentage points
EFFECT = np.array(
    [
        [-0.2272727, -35.4545455, -26.5909091],
        [-13.1818182, -43.1818182, -56.3636364],
        [0.0, -20.2272727, -9.3181818],
    ]
)
LOW = np.array(
    [
        [-0.6818182, -40.0, -30.9090909],
        [-16.3636364, -47.7272727, -60.9090909],
        [0.0, -24.7727273, -12.9545455],
    ]
)
HIGH = np.array(
    [
        [0.0, -31.1363636, -22.2727273],
        [-10.0, -38.4090909, -51.5909091],
        [0.0, -15.6818182, -5.9090909],
    ]
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)
    x = np.arange(len(MODELS))
    width = 0.23

    for idx, (task, color) in enumerate(zip(TASKS, COLORS)):
        xpos = x + (idx - 1) * width
        y = EFFECT[:, idx]
        err = np.vstack([y - LOW[:, idx], HIGH[:, idx] - y])
        ax.bar(xpos, y, width=width, color=color, label=task, zorder=2)
        ax.errorbar(
            xpos,
            y,
            yerr=err,
            fmt="none",
            ecolor="#202020",
            elinewidth=1.1,
            capsize=2.2,
            capthick=1.1,
            zorder=3,
        )

    ax.axhline(0, color="#303030", linewidth=0.9)
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_xticks(x, MODELS)
    ax.set_ylim(-64, 8)
    ax.set_yticks(np.arange(-60, 1, 10))
    ax.grid(axis="y", color="#D8DDE5", linewidth=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(
        0.99,
        0.03,
        "n = 440 scenes/model; 95% scene-bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color="#4A4A4A",
    )

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"cross_model_main.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
