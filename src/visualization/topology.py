from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from experiments.topology.build_manifest import GRID_EDGES


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
TASK_LABELS = ("Recognition", "Binary relation", "Compositional")
TASK_COLORS = ("#4C78A8", "#F2A541", "#E45756")
MODEL_LABELS = ("LLaVA-OneVision-7B", "Qwen2.5-VL-7B")
MODEL_SHORT = ("LLaVA", "Qwen")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_permutation(ax, permutation: tuple[int, ...], offset: float, title: str) -> None:
    centers = {}
    for destination, source in enumerate(permutation):
        row, col = divmod(destination, 2)
        x, y = offset + col, 1 - row
        centers[destination] = (x + 0.4, y + 0.4)
        ax.add_patch(Rectangle((x, y), 0.8, 0.8, facecolor="#E8EEF7", edgecolor="#555555"))
        ax.text(x + 0.4, y + 0.4, str(source + 1), ha="center", va="center",
                fontsize=12, weight="bold")
    for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
        source_edge = frozenset((permutation[left], permutation[right]))
        color = "#2F9E44" if source_edge in GRID_EDGES else "#D94841"
        x1, y1 = centers[left]
        x2, y2 = centers[right]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, zorder=0)
    preserved = sum(
        frozenset((permutation[left], permutation[right])) in GRID_EDGES
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3))
    )
    ax.text(offset + 0.9, 2.02, title, ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(offset + 0.9, -0.2, f"{preserved}/4 edges", ha="center", fontsize=9,
            color="#2F9E44" if preserved == 4 else "#D94841")


def named_panel(ax, result: dict, model: str, panel_title: str) -> None:
    keys = (
        "canonical",
        "reverse",
        "column_major",
        "rotate_90",
        "random_permutation",
        "topology_breaking_mean",
    )
    labels = ("Canonical", "Reverse", "Column\nmajor", "Rotate\n90°", "Random\nbreak", "All breaking\nmean")
    x = np.arange(len(keys))
    for task, label, color in zip(TASKS, TASK_LABELS, TASK_COLORS):
        named = result["models"][model]["tasks"][task]["named_accuracy"]
        ax.plot(x, [100 * named[key] for key in keys], marker="o", linewidth=2,
                color=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylim(35, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(panel_title, loc="left", weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load(args.analysis)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    draw_permutation(ax, (0, 1, 2, 3), 0.0, "Canonical")
    draw_permutation(ax, (2, 0, 3, 1), 2.4, "90° rotation")
    draw_permutation(ax, (0, 3, 1, 2), 4.8, "Topology breaking")
    ax.set_xlim(-0.2, 6.7)
    ax.set_ylim(-0.35, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A  Fixed topology taxonomy", loc="left", weight="bold")
    ax.text(3.25, -0.34, "Green: preserved neighbor   Red: broken neighbor",
            ha="center", fontsize=9)

    named_panel(axes[0, 1], result, MODEL_LABELS[0], "B  LLaVA serialization taxonomy")
    named_panel(axes[1, 0], result, MODEL_LABELS[1], "C  Qwen serialization taxonomy")

    ax = axes[1, 1]
    x = np.arange(len(TASKS))
    width = 0.28
    for model_index, (model, short, color) in enumerate(
        zip(MODEL_LABELS, MODEL_SHORT, ("#6B4C9A", "#2A9D8F"))
    ):
        values, lower, upper = [], [], []
        for task in TASKS:
            effect = result["models"][model]["tasks"][task]["taxonomy"]["effects"][
                "breaking_minus_preserving"
            ]
            values.append(effect["effect_pp"])
            lower.append(effect["effect_pp"] - effect["ci95_pp"][0])
            upper.append(effect["ci95_pp"][1] - effect["effect_pp"])
        position = x + (model_index - 0.5) * width
        ax.bar(position, values, width, color=color, label=short,
               yerr=np.asarray([lower, upper]), capsize=3)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, TASK_LABELS)
    ax.set_ylabel("Breaking − preserving (pp)")
    ax.set_title("D  Primary paired topology contrast", loc="left", weight="bold")
    ax.legend(frameon=False)

    for ax in axes.flat[1:]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
