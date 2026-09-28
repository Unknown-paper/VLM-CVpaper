from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import pandas as pd


COORDS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
INDEX = {value: key for key, value in COORDS.items()}
GRID_EDGES = {frozenset(edge) for edge in ((0, 1), (0, 2), (1, 3), (2, 3))}


def transform_permutation(transform) -> tuple[int, int, int, int]:
    """Return destination->source assignment induced by a grid transform."""
    assignment = [0] * 4
    for source, coord in COORDS.items():
        destination = INDEX[transform(*coord)]
        assignment[destination] = source
    return tuple(assignment)


D4 = {
    "identity": transform_permutation(lambda r, c: (r, c)),
    "rotate_90": transform_permutation(lambda r, c: (c, 1 - r)),
    "rotate_180": transform_permutation(lambda r, c: (1 - r, 1 - c)),
    "rotate_270": transform_permutation(lambda r, c: (1 - c, r)),
    "flip_horizontal": transform_permutation(lambda r, c: (r, 1 - c)),
    "flip_vertical": transform_permutation(lambda r, c: (1 - r, c)),
    "column_major_transpose": transform_permutation(lambda r, c: (c, r)),
    "anti_diagonal_reflection": transform_permutation(lambda r, c: (1 - c, 1 - r)),
}
D4_BY_PERMUTATION = {value: key for key, value in D4.items()}
RANDOM_BREAKING = (0, 3, 1, 2)


def metrics(permutation: tuple[int, ...]) -> dict[str, float | int | str]:
    position = {source: destination for destination, source in enumerate(permutation)}
    preserved = sum(
        frozenset((permutation[left], permutation[right])) in GRID_EDGES
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3))
    )
    sequence_neighbors = sum(
        frozenset((permutation[index], permutation[index + 1])) in GRID_EDGES
        for index in range(3)
    )
    mean_neighbor_distance = sum(
        abs(position[left] - position[right])
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3))
    ) / 4
    average_displacement = sum(abs(position[node] - node) for node in range(4)) / 4
    d4_name = D4_BY_PERMUTATION.get(permutation, "")
    if permutation == D4["identity"]:
        taxonomy = "canonical"
    elif d4_name:
        taxonomy = "topology_preserving"
    else:
        taxonomy = "topology_breaking"
    named_condition = d4_name or (
        "random_permutation" if permutation == RANDOM_BREAKING else "topology_breaking"
    )
    return {
        "topology_taxonomy": taxonomy,
        "topology_transform": named_condition,
        "grid_edges_preserved": int(preserved),
        "grid_edge_preservation": preserved / 4,
        "sequence_neighbor_edges": int(sequence_neighbors),
        "sequence_neighbor_fraction": sequence_neighbors / 3,
        "mean_neighbor_sequence_distance": mean_neighbor_distance,
        "average_sequence_displacement": average_displacement,
    }


def stable_rank(scene_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{scene_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    base = pd.read_csv(args.base_manifest)
    base = base[base.position_intervention.eq("canonical")].copy()
    if base.groupby(["scene_id", "stage2_task"]).size().ne(1).any():
        raise ValueError("Base manifest must have one canonical row per scene/task")
    scene_meta = base.drop_duplicates("scene_id")[["scene_id", "orientation"]]
    orientations = sorted(scene_meta.orientation.unique())
    if args.scenes % len(orientations):
        raise ValueError("Requested scenes must divide evenly across orientations")
    per_orientation = args.scenes // len(orientations)
    selected: list[str] = []
    for orientation in orientations:
        group = scene_meta[scene_meta.orientation.eq(orientation)].copy()
        group["rank"] = group.scene_id.map(lambda value: stable_rank(str(value), args.seed))
        selected.extend(group.sort_values("rank").head(per_orientation).scene_id.tolist())
    base = base[base.scene_id.isin(selected)].copy()

    permutations = list(itertools.permutations(range(4)))
    rows = []
    for permutation_index, permutation in enumerate(permutations):
        metadata = metrics(permutation)
        permutation_text = ",".join(str(value) for value in permutation)
        for row in base.itertuples(index=False):
            record = row._asdict()
            record.update(metadata)
            record["position_intervention"] = "joint"
            record["merged_token_permutation"] = permutation_text
            record["topology_permutation_id"] = f"p{permutation_index:02d}"
            record["example_id"] = f"{row.example_id}_topo_p{permutation_index:02d}"
            rows.append(record)
    output = pd.DataFrame(rows)
    expected = args.scenes * 3 * 24
    if len(output) != expected or output.example_id.duplicated().any():
        raise ValueError("Topology manifest cardinality or uniqueness failure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = (
        output.drop_duplicates("topology_permutation_id")
        .groupby("topology_taxonomy")
        .size()
        .to_dict()
    )
    print(f"Wrote {len(output)} rows from {args.scenes} scenes: {summary}")


if __name__ == "__main__":
    main()
