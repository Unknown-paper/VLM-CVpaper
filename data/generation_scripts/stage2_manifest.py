from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic import _balanced_scenes


def _select_scene_ids(manifest: pd.DataFrame, n_scenes: int) -> list[str]:
    scene_table = manifest[["scene_id", "orientation"]].drop_duplicates()
    selected: list[str] = []
    per_orientation = n_scenes // 2
    for orientation in ("horizontal", "vertical"):
        ids = scene_table[scene_table.orientation.eq(orientation)].scene_id.tolist()
        indices = np.linspace(0, len(ids) - 1, per_orientation, dtype=int)
        selected.extend(ids[index] for index in indices)
    return selected


def build_stage2_manifest(
    gate1_manifest: Path,
    output_path: Path,
    n_scenes: int = 100,
    seed: int = 20260926,
) -> Path:
    source = pd.read_csv(gate1_manifest)
    selected_ids = _select_scene_ids(source, n_scenes)
    selected = source[source.scene_id.isin(selected_ids)].copy()
    base = selected[selected.task.isin(["relation_shape", "recognition_shape"])].copy()
    base["stage2_task"] = np.where(
        base.task.eq("relation_shape"), "compositional_relation", "object_recognition"
    )

    # Reconstruct the exact source scene table. The original pilot fixed this at
    # 200, which silently prevents a larger confirmatory dataset.
    scenes = {
        scene.scene_id: scene
        for scene in _balanced_scenes(source.scene_id.nunique(), seed)
    }
    truth_by_scene: dict[str, bool] = {}
    for orientation in ("horizontal", "vertical"):
        ids = sorted(
            selected[selected.orientation.eq(orientation)].scene_id.unique().tolist()
        )
        truth_by_scene.update({scene_id: index % 2 == 0 for index, scene_id in enumerate(ids)})
    spatial_rows = []
    relation_rows = selected[selected.task.eq("relation_shape")]
    for row in relation_rows.to_dict("records"):
        scene = scenes[row["scene_id"]]
        positive = truth_by_scene[scene.scene_id]
        if scene.orientation == "horizontal":
            relation = "left of" if positive else "right of"
        else:
            relation = "above" if positive else "below"
        row["example_id"] = row["example_id"].replace("_q0", "_q2")
        row["task"] = "spatial_relation"
        row["stage2_task"] = "spatial_relation"
        row["question"] = (
            "Answer with exactly yes or no. "
            f"Is the {scene.first.color} {scene.first.shape} {relation} "
            f"the {scene.second.color} {scene.second.shape}?"
        )
        row["answer"] = "yes" if positive else "no"
        spatial_rows.append(row)
    result = pd.concat([base, pd.DataFrame(spatial_rows)], ignore_index=True)
    result = result.sort_values(["scene_id", "phase", "task"]).reset_index(drop=True)
    if result.scene_id.nunique() != n_scenes or len(result) != n_scenes * 9 * 3:
        raise RuntimeError("Stage-2 manifest size invariant failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path
