from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .synthetic import build_gate1_dataset
from .stage2_manifest import build_stage2_manifest


PRIMARY_PERMUTATION = "2,0,3,1"
ROBUSTNESS_PERMUTATIONS = ("1,3,0,2", "3,2,1,0")
TASKS = ("object_recognition", "spatial_relation", "compositional_relation")


def _select_phase(scene_rows: pd.DataFrame, status: str) -> int:
    candidates = scene_rows[
        scene_rows.stage2_task.eq("compositional_relation")
        & scene_rows.boundary_status.eq(status)
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No {status} phase for scene {scene_rows.scene_id.iloc[0]}")
    candidates["abs_phase"] = candidates.phase.abs()
    return int(
        candidates.sort_values(
            ["boundary_clearance_px", "abs_phase"], ascending=[False, True]
        ).iloc[0].phase
    )


def build_multimodel_manifest(source: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_id in sorted(source.scene_id.unique()):
        scene_rows = source[source.scene_id.eq(scene_id)]
        for status in ("same_tile", "cross_tile"):
            phase = _select_phase(scene_rows, status)
            selected = scene_rows[
                scene_rows.phase.eq(phase) & scene_rows.stage2_task.isin(TASKS)
            ]
            if len(selected) != len(TASKS):
                raise RuntimeError(f"Incomplete task block for {scene_id}, {status}")
            for record in selected.to_dict("records"):
                for name, permutation in (
                    ("canonical", "0,1,2,3"),
                    ("perm_primary", PRIMARY_PERMUTATION),
                ):
                    item = dict(record)
                    item.update(
                        {
                            "example_id": f'{record["example_id"]}_s4_{name}',
                            "baseline_example_id": record["example_id"],
                            "intervention": name,
                            "global_thumbnail_mode": "normal",
                            "tile_permutation": permutation,
                        }
                    )
                    rows.append(item)
    frame = pd.DataFrame(rows).sort_values(
        ["scene_id", "boundary_status", "stage2_task", "intervention"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def build_factorial_manifest(source: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Create the input-level 2x2x2 factorial; position IDs are a model hook."""
    rows: list[dict[str, object]] = []
    for scene_id in sorted(source.scene_id.unique()):
        scene_rows = source[source.scene_id.eq(scene_id)]
        phase_by_partition = {
            "same": _select_phase(scene_rows, "same_tile"),
            "cross": _select_phase(scene_rows, "cross_tile"),
        }
        for partition, phase in phase_by_partition.items():
            selected = scene_rows[
                scene_rows.phase.eq(phase) & scene_rows.stage2_task.isin(TASKS)
            ]
            for record in selected.to_dict("records"):
                for ordering, permutation in (
                    ("canonical", "0,1,2,3"),
                    ("permuted", PRIMARY_PERMUTATION),
                ):
                    for thumbnail in ("normal", "objects_removed"):
                        item = dict(record)
                        item.update(
                            {
                                "example_id": (
                                    f'{record["example_id"]}_s4_fact_'
                                    f"{partition}_{ordering}_{thumbnail}"
                                ),
                                "baseline_example_id": record["example_id"],
                                "intervention": "factorial",
                                "partition_factor": partition,
                                "order_factor": ordering,
                                "thumbnail_factor": thumbnail,
                                "global_thumbnail_mode": thumbnail,
                                "tile_permutation": permutation,
                            }
                        )
                        rows.append(item)
    frame = pd.DataFrame(rows).sort_values(
        ["scene_id", "stage2_task", "partition_factor", "order_factor", "thumbnail_factor"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def build_qwen_native_manifest(primary: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Qwen has one native patch grid, not a movable explicit 2x2 crop origin.

    Keep one byte-identical image/question block per scene and compare canonical
    native patch order with the preregistered quadrant derangement. Calling the
    discarded LLaVA phase a Qwen tile boundary would be scientifically invalid.
    """
    frame = primary[primary.boundary_status.eq("same_tile")].copy()
    frame["boundary_status"] = "native_grid"
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated/synthetic"))
    # 440 is divisible by the 22 orientation/midpoint strata: 220 scenes per
    # orientation without duplicated or under-filled selection cells.
    parser.add_argument("--n-scenes", type=int, default=440)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    base_root = args.output / "synthetic_base"
    gate1 = build_gate1_dataset(base_root, n_scenes=args.n_scenes, seed=args.seed)
    stage2 = args.output / "synthetic_stage2_manifest.csv"
    build_stage2_manifest(gate1, stage2, n_scenes=args.n_scenes, seed=args.seed)
    source = pd.read_csv(stage2)
    source["image_path"] = source.image_path.map(
        lambda value: (Path("synthetic_base") / "dataset" / str(value)).as_posix()
    )
    source.to_csv(stage2, index=False)
    primary = build_multimodel_manifest(source, args.output / "multimodel_manifest.csv")
    factorial = build_factorial_manifest(source, args.output / "factorial_manifest.csv")
    qwen = build_qwen_native_manifest(primary, args.output / "qwen_native_manifest.csv")
    summary = {
        "seed": args.seed,
        "scenes": args.n_scenes,
        "scenes_per_orientation": args.n_scenes // 2,
        "primary_rows": len(primary),
        "factorial_rows": len(factorial),
        "qwen_native_rows": len(qwen),
        "primary_permutation": PRIMARY_PERMUTATION,
        "robustness_permutations": ROBUSTNESS_PERMUTATIONS,
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
