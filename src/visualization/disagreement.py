from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANELS = (
    ("LLaVA-OneVision-7B", "spatial_relation", "A  LLaVA — binary relation"),
    ("LLaVA-OneVision-7B", "compositional_relation", "B  LLaVA — compositional"),
    ("Qwen2.5-VL-7B", "spatial_relation", "C  Qwen — binary relation"),
    ("Qwen2.5-VL-7B", "compositional_relation", "D  Qwen — compositional"),
)


def errorbar(ax, frame: pd.DataFrame, column: str, low: str, high: str,
             label: str, color: str, marker: str, offset: float = 0.0) -> None:
    x = frame.serialization_disagreement.to_numpy(float) + offset
    y = 100 * frame[column].to_numpy(float)
    lower = y - 100 * frame[low].to_numpy(float)
    upper = 100 * frame[high].to_numpy(float) - y
    ax.errorbar(x, y, yerr=np.asarray([lower, upper]), marker=marker, linewidth=1.8,
                capsize=2.5, color=color, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.bins)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, (model, task, title) in zip(axes.flat, PANELS):
        frame = data[data.model.eq(model) & data.task.eq(task)].sort_values(
            "serialization_disagreement"
        )
        errorbar(ax, frame, "random_variant_accuracy", "random_variant_ci_low",
                 "random_variant_ci_high", "Random serialization", "#E45756", "o")
        errorbar(ax, frame, "ensemble_accuracy", "ensemble_ci_low", "ensemble_ci_high",
                 "Majority ensemble", "#F2A541", "s", offset=0.008)
        errorbar(ax, frame, "canonical_accuracy", "canonical_ci_low", "canonical_ci_high",
                 "Canonical S0", "#4C78A8", "^", offset=-0.008)
        for row in frame.itertuples(index=False):
            ax.text(row.serialization_disagreement, 22, f"n={row.n}", rotation=90,
                    ha="center", va="bottom", fontsize=7, color="#666666")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlim(-0.035, 0.835)
        ax.set_ylim(20, 104)
        ax.grid(color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Serialization disagreement (1 − pairwise agreement)")
    axes[1, 1].set_xlabel("Serialization disagreement (1 − pairwise agreement)")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Disagreement signals serialization instability, not universal canonical error",
                 weight="bold", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
