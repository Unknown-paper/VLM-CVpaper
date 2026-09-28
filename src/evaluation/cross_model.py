from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def paired_effect(
    data: pd.DataFrame,
    condition: str,
    left: str,
    right: str,
    seed: int,
    draws: int,
) -> dict[str, object]:
    pivot = data.pivot(index="scene_id", columns=condition, values="correct")
    pivot = pivot.dropna(subset=[left, right])
    differences = pivot[right].to_numpy(float) - pivot[left].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    samples = differences[indices].mean(axis=1) * 100
    improved = int(((pivot[left] == 0) & (pivot[right] == 1)).sum())
    worsened = int(((pivot[left] == 1) & (pivot[right] == 0)).sum())
    discordant = improved + worsened
    p_value = (
        float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "n_scenes": int(len(pivot)),
        "left": left,
        "right": right,
        "left_accuracy": float(pivot[left].mean()),
        "right_accuracy": float(pivot[right].mean()),
        "effect_pp": float(differences.mean() * 100),
        "ci95_pp": _percentile_interval(samples),
        "improved": improved,
        "worsened": worsened,
        "exact_p": p_value,
    }


def difference_in_differences(
    data: pd.DataFrame,
    relation_task: str,
    intervention: str,
    seed: int,
    draws: int,
) -> dict[str, object]:
    subset = data[
        data.stage2_task.isin(["object_recognition", relation_task])
        & data.intervention.eq(intervention)
    ]
    pivot = subset.pivot(
        index="scene_id", columns=["stage2_task", "boundary_status"], values="correct"
    ).dropna()
    relation_delta = (
        pivot[(relation_task, "cross_tile")] - pivot[(relation_task, "same_tile")]
    )
    recognition_delta = (
        pivot[("object_recognition", "cross_tile")]
        - pivot[("object_recognition", "same_tile")]
    )
    did = (relation_delta - recognition_delta).to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(did), size=(draws, len(did)))
    samples = did[indices].mean(axis=1) * 100
    nonzero = did[did != 0]
    positives = int((nonzero > 0).sum())
    negatives = int((nonzero < 0).sum())
    p_value = (
        float(binomtest(min(positives, negatives), len(nonzero), 0.5).pvalue)
        if len(nonzero)
        else 1.0
    )
    return {
        "relation_task": relation_task,
        "intervention": intervention,
        "n_scenes": int(len(pivot)),
        "relation_cross_minus_same_pp": float(relation_delta.mean() * 100),
        "recognition_cross_minus_same_pp": float(recognition_delta.mean() * 100),
        "did_pp": float(did.mean() * 100),
        "ci95_pp": _percentile_interval(samples),
        "exact_sign_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/cross_model"))
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if len(args.predictions) != len(args.model_names):
        raise ValueError("--predictions and --model-names must have equal lengths")
    args.output.mkdir(parents=True, exist_ok=True)

    accuracy_rows: list[dict[str, object]] = []
    effects: dict[str, object] = {}
    for model_offset, (path, model_name) in enumerate(zip(args.predictions, args.model_names)):
        data = pd.read_csv(path)
        if data.example_id.duplicated().any():
            raise ValueError(f"Duplicate predictions in {path}")
        for keys, group in data.groupby(["stage2_task", "boundary_status", "intervention"]):
            task, boundary, intervention = keys
            accuracy_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "boundary_status": boundary,
                    "intervention": intervention,
                    "n": len(group),
                    "accuracy": float(group.correct.mean()),
                }
            )
        model_effects: dict[str, object] = {"boundary": {}, "permutation": {}, "did": {}}
        available_boundaries = set(data.boundary_status.astype(str))
        has_paired_boundary = {"same_tile", "cross_tile"}.issubset(available_boundaries)
        for task_index, task in enumerate(TASKS):
            if has_paired_boundary:
                for intervention_index, intervention in enumerate(("canonical", "perm_primary")):
                    subset = data[
                        data.stage2_task.eq(task) & data.intervention.eq(intervention)
                    ]
                    key = f"{task}:{intervention}"
                    model_effects["boundary"][key] = paired_effect(
                        subset,
                        "boundary_status",
                        "same_tile",
                        "cross_tile",
                        args.seed + model_offset * 100 + task_index * 10 + intervention_index,
                        args.draws,
                    )
            permutation_boundaries = (
                ("same_tile", "cross_tile") if has_paired_boundary else tuple(sorted(available_boundaries))
            )
            for boundary_index, boundary in enumerate(permutation_boundaries):
                subset = data[
                    data.stage2_task.eq(task) & data.boundary_status.eq(boundary)
                ]
                key = f"{task}:{boundary}"
                model_effects["permutation"][key] = paired_effect(
                    subset,
                    "intervention",
                    "canonical",
                    "perm_primary",
                    args.seed + model_offset * 100 + 30 + task_index * 10 + boundary_index,
                    args.draws,
                )
        if has_paired_boundary:
            for relation_index, relation_task in enumerate(TASKS[1:]):
                for intervention_index, intervention in enumerate(("canonical", "perm_primary")):
                    key = f"{relation_task}:{intervention}"
                    model_effects["did"][key] = difference_in_differences(
                        data,
                        relation_task,
                        intervention,
                        args.seed + model_offset * 100 + 70 + relation_index * 10 + intervention_index,
                        args.draws,
                    )
        effects[model_name] = model_effects

    pd.DataFrame(accuracy_rows).to_csv(args.output / "accuracy.csv", index=False)
    (args.output / "effects.json").write_text(
        json.dumps(effects, indent=2), encoding="utf-8"
    )
    print(json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
