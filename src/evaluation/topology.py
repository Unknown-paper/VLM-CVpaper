from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
METRICS = (
    "grid_edge_preservation",
    "sequence_neighbor_fraction",
    "mean_neighbor_sequence_distance",
    "average_sequence_displacement",
)


def bootstrap_mean(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(value) for value in np.percentile(sampled.mean(axis=1) * 100, [2.5, 97.5])]


def paired_effect(values: np.ndarray, seed: int, draws: int) -> dict[str, object]:
    nonzero = values[~np.isclose(values, 0)]
    if len(nonzero):
        positives = int((nonzero > 0).sum())
        p_value = float(binomtest(min(positives, len(nonzero) - positives), len(nonzero)).pvalue)
    else:
        p_value = 1.0
    return {
        "effect_pp": float(values.mean() * 100),
        "ci95_pp": bootstrap_mean(values, seed, draws),
        "exact_sign_p": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def taxonomy_analysis(data: pd.DataFrame, seed: int, draws: int) -> dict[str, object]:
    per_scene = (
        data.groupby(["scene_id", "topology_taxonomy"], as_index=False)
        .correct.mean()
        .pivot(index="scene_id", columns="topology_taxonomy", values="correct")
        .dropna(subset=["canonical", "topology_preserving", "topology_breaking"])
    )
    canonical = per_scene.canonical.to_numpy(float)
    preserving = per_scene.topology_preserving.to_numpy(float)
    breaking = per_scene.topology_breaking.to_numpy(float)
    return {
        "n_scenes": int(len(per_scene)),
        "accuracy": {
            "canonical": float(canonical.mean()),
            "topology_preserving": float(preserving.mean()),
            "topology_breaking": float(breaking.mean()),
        },
        "effects": {
            "preserving_minus_canonical": paired_effect(
                preserving - canonical, seed, draws
            ),
            "breaking_minus_canonical": paired_effect(
                breaking - canonical, seed + 1, draws
            ),
            "breaking_minus_preserving": paired_effect(
                breaking - preserving, seed + 2, draws
            ),
        },
    }


def _bootstrap_correlation(
    pivot: np.ndarray,
    metric: np.ndarray,
    method: str,
    seed: int,
    draws: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        accuracy = pivot[rng.integers(0, pivot.shape[0], size=pivot.shape[0])].mean(axis=0)
        correlation = (
            pearsonr(metric, accuracy).statistic
            if method == "pearson"
            else spearmanr(metric, accuracy).statistic
        )
        if np.isfinite(correlation):
            values.append(float(correlation))
    if not values:
        return [None, None]
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def correlation_analysis(
    data: pd.DataFrame, seed: int, draws: int
) -> dict[str, object]:
    permutation_meta = (
        data.drop_duplicates("topology_permutation_id")
        .set_index("topology_permutation_id")
        .sort_index()
    )
    pivot_frame = data.pivot(
        index="scene_id", columns="topology_permutation_id", values="correct"
    ).dropna()
    pivot_frame = pivot_frame[permutation_meta.index]
    pivot = pivot_frame.to_numpy(float)
    result: dict[str, object] = {
        "n_scenes": int(len(pivot_frame)),
        "n_permutations": int(len(permutation_meta)),
    }
    for metric_index, metric_name in enumerate(METRICS):
        metric = permutation_meta[metric_name].to_numpy(float)
        accuracy = pivot.mean(axis=0)
        pearson = pearsonr(metric, accuracy)
        spearman = spearmanr(metric, accuracy)
        result[metric_name] = {
            "pearson_r": finite_or_none(pearson.statistic),
            "pearson_p": finite_or_none(pearson.pvalue),
            "pearson_ci95": _bootstrap_correlation(
                pivot, metric, "pearson", seed + metric_index * 10, draws
            ),
            "spearman_rho": finite_or_none(spearman.statistic),
            "spearman_p": finite_or_none(spearman.pvalue),
            "spearman_ci95": _bootstrap_correlation(
                pivot, metric, "spearman", seed + metric_index * 10 + 1, draws
            ),
        }
    return result


def named_accuracy(data: pd.DataFrame) -> dict[str, float]:
    named = {
        "canonical": "identity",
        "random_permutation": "random_permutation",
        "reverse": "rotate_180",
        "column_major": "column_major_transpose",
        "rotate_90": "rotate_90",
    }
    result = {
        output_name: float(data[data.topology_transform.eq(transform)].correct.mean())
        for output_name, transform in named.items()
    }
    result["topology_preserving_mean"] = float(
        data[data.topology_taxonomy.eq("topology_preserving")].correct.mean()
    )
    result["topology_breaking_mean"] = float(
        data[data.topology_taxonomy.eq("topology_breaking")].correct.mean()
    )
    return result


def named_effects(data: pd.DataFrame, seed: int, draws: int) -> dict[str, object]:
    named = {
        "canonical": "identity",
        "random_permutation": "random_permutation",
        "reverse": "rotate_180",
        "column_major": "column_major_transpose",
        "rotate_90": "rotate_90",
    }
    pivot = data[data.topology_transform.isin(named.values())].pivot(
        index="scene_id", columns="topology_transform", values="correct"
    ).dropna(subset=list(named.values()))
    canonical = pivot[named["canonical"]].to_numpy(float)
    result = {}
    for offset, output_name in enumerate(
        ("random_permutation", "reverse", "column_major", "rotate_90")
    ):
        values = pivot[named[output_name]].to_numpy(float) - canonical
        result[f"{output_name}_minus_canonical"] = paired_effect(
            values, seed + offset, draws
        )
    return result


def validate(data: pd.DataFrame) -> None:
    if data.example_id.duplicated().any():
        raise ValueError("Duplicate example IDs")
    if set(data.topology_permutation_id.unique()) != {f"p{index:02d}" for index in range(24)}:
        raise ValueError("Expected all 24 permutations")
    counts = data.groupby(["scene_id", "stage2_task"]).topology_permutation_id.nunique()
    if not counts.eq(24).all():
        raise ValueError("Incomplete scene/task permutation blocks")
    if data.image_token_count.nunique() != 1:
        raise ValueError("Visual-token count drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--correlation-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    frames = []
    for label, path in (("LLaVA-OneVision-7B", args.llava), ("Qwen2.5-VL-7B", args.qwen)):
        frame = pd.read_csv(path)
        validate(frame)
        frame["model_label"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    result: dict[str, object] = {
        "design": {
            "scenes_per_model": int(frames[0].scene_id.nunique()),
            "permutations": 24,
            "topology_preserving_nonidentity": 7,
            "topology_breaking": 16,
        },
        "models": {},
    }
    primary_records = []
    cell_rows = []
    for model_index, (model_label, model_data) in enumerate(data.groupby("model_label")):
        model_result = {
            "visual_tokens": int(model_data.image_token_count.iloc[0]),
            "tasks": {},
        }
        for task_index, task in enumerate(TASKS):
            subset = model_data[model_data.stage2_task.eq(task)]
            taxonomy = taxonomy_analysis(
                subset, args.seed + model_index * 1000 + task_index * 100, args.draws
            )
            correlations = correlation_analysis(
                subset,
                args.seed + 5000 + model_index * 1000 + task_index * 100,
                args.correlation_draws,
            )
            model_result["tasks"][task] = {
                "named_accuracy": named_accuracy(subset),
                "named_effects": named_effects(
                    subset,
                    args.seed + 13000 + model_index * 1000 + task_index * 100,
                    args.draws,
                ),
                "taxonomy": taxonomy,
                "correlations": correlations,
                "correlations_nonidentity": correlation_analysis(
                    subset[~subset.topology_permutation_id.eq("p00")],
                    args.seed + 9000 + model_index * 1000 + task_index * 100,
                    args.correlation_draws,
                ),
            }
            primary_records.append(taxonomy["effects"]["breaking_minus_preserving"])
            grouped = (
                subset.groupby(
                    [
                        "topology_permutation_id",
                        "merged_token_permutation",
                        "topology_taxonomy",
                        "topology_transform",
                        *METRICS,
                    ],
                    as_index=False,
                )
                .correct.agg(["count", "mean"])
                .reset_index()
            )
            grouped.insert(0, "task", task)
            grouped.insert(0, "model", model_label)
            cell_rows.append(grouped)
        result["models"][model_label] = model_result
    for record, adjusted in zip(
        primary_records,
        holm_adjust([record["exact_sign_p"] for record in primary_records]),
    ):
        record["holm_p"] = adjusted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat(cell_rows, ignore_index=True).to_csv(
        args.output.with_suffix(".permutation_cells.csv"), index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
