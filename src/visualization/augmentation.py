from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
TASK_LABELS = ("Recognition", "Binary relation", "Compositional")
COLORS = ("#4C78A8", "#F2A541", "#E45756")
MODELS = ("LLaVA-OneVision-7B", "Qwen2.5-VL-7B")
MODEL_TITLES = ("A  LLaVA-OneVision-7B", "B  Qwen2.5-VL-7B")
CONDITIONS = ("S0", "S1", "S2", "S3", "S4", "ensemble")
LABELS = ("S0\nCanonical", "S1\nReverse", "S2\nRotate 90°", "S3\nColumn major", "S4\nRandom break", "Majority\nensemble")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), constrained_layout=True, sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax, model, title in zip(axes, MODELS, MODEL_TITLES):
        for task, label, color in zip(TASKS, TASK_LABELS, COLORS):
            cells = result["models"][model]["tasks"][task]["conditions"]
            values = np.asarray([100 * cells[name]["accuracy"] for name in CONDITIONS])
            lower = values - np.asarray([100 * cells[name]["ci95"][0] for name in CONDITIONS])
            upper = np.asarray([100 * cells[name]["ci95"][1] for name in CONDITIONS]) - values
            ax.errorbar(x, values, yerr=np.asarray([lower, upper]), marker="o", linewidth=2,
                        capsize=2.5, color=color, label=label)
        ax.axvspan(4.55, 5.45, color="#eeeeee", zorder=-1)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(50, 102)
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.suptitle("Inference-time serialization augmentation does not improve canonical inference",
                 weight="bold", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
