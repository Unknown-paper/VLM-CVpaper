from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
TASK_LABELS = ("Recognition", "Binary relation", "Compositional")
COLORS = ("#5B8FF9", "#F6BD16", "#E8684A")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodel", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    multi = load(args.multimodel)
    factorial = load(args.factorial)
    causal = load(args.causal)
    natural = load(args.natural)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), constrained_layout=True)

    # A: architecture replication, using the cross-tile cell for explicit tile
    # models and the native-grid cell for Qwen.
    ax = axes[0, 0]
    models = list(multi)
    width = 0.22
    x = np.arange(len(models))
    for task_i, (task, label, color) in enumerate(zip(TASKS, TASK_LABELS, COLORS)):
        values = []
        errors = [[], []]
        for model in models:
            effects = multi[model]["permutation"]
            boundary = "native_grid" if f"{task}:native_grid" in effects else "cross_tile"
            item = effects[f"{task}:{boundary}"]
            values.append(item["effect_pp"])
            errors[0].append(item["effect_pp"] - item["ci95_pp"][0])
            errors[1].append(item["ci95_pp"][1] - item["effect_pp"])
        ax.bar(x + (task_i - 1) * width, values, width, yerr=np.asarray(errors),
               color=color, label=label, capsize=2.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, models)
    ax.set_ylabel("Permutation effect (percentage points)")
    ax.set_title("A  Cross-model serialization sensitivity", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3)

    # B: LLaVA 2x2x2 marginal factor effects.
    ax = axes[0, 1]
    factors = ("partition", "ordering", "thumbnail")
    factor_labels = ("Cross partition", "Permuted order", "No thumbnail")
    x = np.arange(len(factors))
    for task_i, (task, label, color) in enumerate(zip(TASKS, TASK_LABELS, COLORS)):
        values, errors = [], [[], []]
        for factor in factors:
            item = factorial["effects"][task]["main_effects"][factor]
            values.append(item["marginal_effect_pp"])
            errors[0].append(item["marginal_effect_pp"] - item["ci95_pp"][0])
            errors[1].append(item["ci95_pp"][1] - item["marginal_effect_pp"])
        ax.bar(x + (task_i - 1) * width, values, width, yerr=np.asarray(errors),
               color=color, label=label, capsize=2.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, factor_labels)
    ax.set_ylabel("Marginal effect (percentage points)")
    ax.set_title("B  Serialization factor decomposition", loc="left", weight="bold")

    # C: clean Qwen token-level content/position intervention.
    ax = axes[1, 0]
    conditions = ("canonical", "content_only", "position_only", "joint")
    condition_labels = ("Canonical", "Content only", "Position only", "Joint")
    x = np.arange(len(conditions))
    for task_i, (task, label, color) in enumerate(zip(TASKS, TASK_LABELS, COLORS)):
        accuracy = causal["task_effects"][task]["accuracy"]
        ax.plot(x, [100 * accuracy[c] for c in conditions], marker="o", lw=2,
                color=color, label=label)
    ax.set_xticks(x, condition_labels)
    ax.set_ylim(65, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C  Position causality: content dominates", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)

    # D: independent controlled-natural generation benchmark.
    ax = axes[1, 1]
    conditions = ("same", "cross", "cross_permuted")
    condition_labels = ("Same tile", "Cross tile", "Cross + permuted")
    x = np.arange(len(conditions))
    for task_i, (task, label, color) in enumerate(zip(TASKS, TASK_LABELS, COLORS)):
        same = natural["same_tile_gate"][task]["accuracy"] * 100
        cross = same + natural["effects"][task]["cross_minus_same"]["effect_pp"]
        perm = cross + natural["effects"][task]["permuted_minus_cross"]["effect_pp"]
        ax.plot(x, [same, cross, perm], marker="o", lw=2, color=color, label=label)
    ax.axhline(80, color="#888888", ls="--", lw=1, label="80% gate")
    ax.set_xticks(x, condition_labels)
    ax.set_ylim(45, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("D  Controlled-natural validation", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.6, alpha=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
