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
CONDITIONS = ("canonical", "content_only", "position_only", "joint")
CONDITION_LABELS = ("Canonical", "Content only", "Position only", "Joint")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy_panel(ax, result: dict, title: str, lower: float) -> None:
    x = np.arange(len(CONDITIONS))
    for task, label, color in zip(TASKS, TASK_LABELS, COLORS):
        values = [100 * result["task_effects"][task]["accuracy"][c] for c in CONDITIONS]
        ax.plot(x, values, marker="o", linewidth=2.2, color=color, label=label)
    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_ylim(lower, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, loc="left", weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qwen = load(args.qwen)
    llava = load(args.llava)
    natural = load(args.natural)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)

    # A: compact cross-model summary of the relation-specific paired effects.
    ax = axes[0, 0]
    rows = []
    row_labels = []
    for model_name, result in (("Qwen", qwen), ("LLaVA", llava)):
        for task, task_label in zip(TASKS[1:], TASK_LABELS[1:]):
            rows.append([
                result["relation_specific_did"][f"{task}:{condition}"]["effect_pp"]
                for condition in CONDITIONS[1:]
            ])
            row_labels.append(f"{model_name} — {task_label}")
    matrix = np.asarray(rows)
    vmax = max(40.0, float(np.abs(matrix).max()))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(3), ("Content only", "Position only", "Joint"))
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("A  Relation-specific causal effect", loc="left", weight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:+.1f}", ha="center", va="center",
                    color="white" if abs(matrix[row, col]) > 0.48 * vmax else "black",
                    fontsize=9, weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Difference-in-differences (pp)")

    accuracy_panel(axes[0, 1], qwen, "B  Qwen: content-to-slot sensitivity", 70)
    accuracy_panel(axes[1, 0], llava, "C  LLaVA: content–position consistency", 40)

    # D: untouched controlled-natural test set after calibration-only locking.
    ax = axes[1, 1]
    natural_conditions = ("same", "cross", "cross_permuted")
    natural_labels = ("Same tile", "Cross tile", "Cross + permuted")
    x = np.arange(3)
    for task, label, color in zip(TASKS, TASK_LABELS, COLORS):
        same = 100 * natural["same_tile_gate"][task]["accuracy"]
        cross = same + natural["effects"][task]["cross_minus_same"]["effect_pp"]
        permuted = cross + natural["effects"][task]["permuted_minus_cross"]["effect_pp"]
        ax.plot(x, [same, cross, permuted], marker="o", linewidth=2.2,
                color=color, label=label)
    ax.axhline(85, color="#777777", linestyle="--", linewidth=1.1, label="85% gate")
    ax.set_xticks(x, natural_labels)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("D  Untouched controlled-natural test", loc="left", weight="bold")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
