from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired(data: pd.DataFrame, left: str, right: str, seed: int, draws: int):
    pivot = data.pivot(index="scene_id", columns="boundary_status", values="correct").dropna(
        subset=[left, right]
    )
    diff = pivot[right].to_numpy(float) - pivot[left].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1) * 100
    improved = int(((pivot[left] == 0) & (pivot[right] == 1)).sum())
    worsened = int(((pivot[left] == 1) & (pivot[right] == 0)).sum())
    discordant = improved + worsened
    return {
        "n": int(len(pivot)),
        "left_accuracy": float(pivot[left].mean()),
        "right_accuracy": float(pivot[right].mean()),
        "effect_pp": float(diff.mean() * 100),
        "ci95_pp": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
        "improved": improved,
        "worsened": worsened,
        "exact_p": float(binomtest(min(improved, worsened), discordant).pvalue) if discordant else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--gate-threshold", type=float, default=0.8)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    accuracy = (
        data.groupby(["split", "stage2_task", "boundary_status"], as_index=False)
        .correct.agg(["count", "mean"])
        .reset_index()
    )
    same = accuracy[accuracy.boundary_status.eq("same")]
    gate = {
        row.stage2_task: {
            "n": int(row["count"]),
            "accuracy": float(row["mean"]),
            "passes_gate": bool(row["mean"] > args.gate_threshold),
        }
        for _, row in same.iterrows()
    }
    effects = {}
    for task_index, task in enumerate(sorted(data.stage2_task.unique())):
        subset = data[data.stage2_task.eq(task)]
        effects[task] = {
            "cross_minus_same": paired(
                subset, "same", "cross", args.seed + task_index * 10, args.draws
            ),
            "permuted_minus_cross": paired(
                subset,
                "cross",
                "cross_permuted",
                args.seed + task_index * 10 + 1,
                args.draws,
            ),
        }
    # Control multiplicity separately for the two preregistered paired contrast
    # families, each spanning the three task types.
    tasks = sorted(effects)
    for contrast in ("cross_minus_same", "permuted_minus_cross"):
        adjusted = holm_adjust([effects[task][contrast]["exact_p"] for task in tasks])
        for task, p_value in zip(tasks, adjusted):
            effects[task][contrast]["holm_p"] = p_value
    result = {
        "gate_threshold": args.gate_threshold,
        "gate_operator": ">",
        "same_tile_gate": gate,
        "effects": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(args.output.with_suffix(".accuracy.csv"), index=False)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
