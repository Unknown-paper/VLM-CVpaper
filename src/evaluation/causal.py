from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
CONDITIONS = ("canonical", "content_only", "position_only", "joint")


def bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(x) for x in np.percentile(sampled.mean(axis=1) * 100, [2.5, 97.5])]


def sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if not len(nonzero):
        return 1.0
    positives = int((nonzero > 0).sum())
    return float(binomtest(min(positives, len(nonzero) - positives), len(nonzero)).pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def effect(values: np.ndarray, seed: int, draws: int) -> dict[str, object]:
    return {
        "effect_pp": float(values.mean() * 100),
        "ci95_pp": bootstrap_interval(values, seed, draws),
        "exact_sign_p": sign_p(values),
    }


def task_effects(data: pd.DataFrame, task: str, seed: int, draws: int) -> dict[str, object]:
    pivot = data[data.stage2_task.eq(task)].pivot(
        index="scene_id", columns="position_intervention", values="correct"
    ).dropna(subset=list(CONDITIONS))
    result: dict[str, object] = {
        "n_scenes": int(len(pivot)),
        "accuracy": {condition: float(pivot[condition].mean()) for condition in CONDITIONS},
        "contrasts": {},
    }
    canonical = pivot["canonical"].to_numpy(float)
    for offset, condition in enumerate(CONDITIONS[1:]):
        result["contrasts"][f"{condition}_minus_canonical"] = effect(
            pivot[condition].to_numpy(float) - canonical, seed + offset, draws
        )
    interaction = (
        pivot["joint"] - pivot["content_only"] - pivot["position_only"] + pivot["canonical"]
    ).to_numpy(float)
    result["factorial_interaction"] = effect(interaction, seed + 10, draws)
    return result


def relation_specific_did(
    data: pd.DataFrame, relation_task: str, condition: str, seed: int, draws: int
) -> dict[str, object]:
    subset = data[data.stage2_task.isin(["object_recognition", relation_task])]
    pivot = subset.pivot(
        index="scene_id", columns=["stage2_task", "position_intervention"], values="correct"
    ).dropna()
    relation_change = (
        pivot[(relation_task, condition)] - pivot[(relation_task, "canonical")]
    )
    recognition_change = (
        pivot[("object_recognition", condition)]
        - pivot[("object_recognition", "canonical")]
    )
    did = (relation_change - recognition_change).to_numpy(float)
    return {
        "n_scenes": int(len(pivot)),
        "relation_change_pp": float(relation_change.mean() * 100),
        "recognition_change_pp": float(recognition_change.mean() * 100),
        **effect(did, seed, draws),
    }


def fit_gee(data: pd.DataFrame) -> dict[str, object]:
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    frame = data.copy()
    frame["content_moved"] = frame.position_intervention.isin(["content_only", "joint"]).astype(int)
    frame["position_moved"] = frame.position_intervention.isin(["position_only", "joint"]).astype(int)
    fit = GEE.from_formula(
        "correct ~ C(stage2_task) * content_moved * position_moved",
        groups="scene_id",
        cov_struct=Exchangeable(),
        family=Binomial(),
        data=frame,
    ).fit(maxiter=200)
    return {
        "converged": bool(fit.converged),
        "terms": [
            {
                "term": name,
                "log_odds": float(fit.params[name]),
                "std_error": float(fit.bse[name]),
                "p_value": float(fit.pvalues[name]),
                "odds_ratio": float(np.exp(np.clip(fit.params[name], -50, 50))),
            }
            for name in fit.params.index
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--expected-visual-tokens", type=int)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    if data.example_id.duplicated().any():
        raise ValueError("Duplicate example IDs")
    if set(data.position_intervention) != set(CONDITIONS):
        raise ValueError("Incomplete causal conditions")
    token_counts = set(data.image_token_count.unique())
    if len(token_counts) != 1:
        raise ValueError(f"Visual token count drift: {sorted(data.image_token_count.unique())}")
    visual_tokens = int(next(iter(token_counts)))
    if args.expected_visual_tokens is not None and visual_tokens != args.expected_visual_tokens:
        raise ValueError(
            f"Expected {args.expected_visual_tokens} visual tokens, got {visual_tokens}"
        )

    result: dict[str, object] = {
        "rows": int(len(data)),
        "scenes": int(data.scene_id.nunique()),
        "visual_tokens": visual_tokens,
        "intervened_tokens": int(data.intervened_visual_tokens.max()),
        "task_effects": {},
        "relation_specific_did": {},
    }
    for index, task in enumerate(TASKS):
        result["task_effects"][task] = task_effects(
            data, task, args.seed + index * 100, args.draws
        )
    for task_index, task in enumerate(TASKS[1:]):
        for condition_index, condition in enumerate(CONDITIONS[1:]):
            result["relation_specific_did"][f"{task}:{condition}"] = relation_specific_did(
                data,
                task,
                condition,
                args.seed + 500 + task_index * 10 + condition_index,
                args.draws,
            )
    primary_records = []
    for task in TASKS:
        primary_records.extend(result["task_effects"][task]["contrasts"].values())
        primary_records.append(result["task_effects"][task]["factorial_interaction"])
    primary_records.extend(result["relation_specific_did"].values())
    for record, adjusted in zip(
        primary_records,
        holm_adjust([record["exact_sign_p"] for record in primary_records]),
    ):
        record["holm_p"] = adjusted
    try:
        result["gee"] = fit_gee(data)
    except Exception as error:
        result["gee"] = {"converged": False, "error": f"{type(error).__name__}: {error}"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cells = (
        data.groupby(["stage2_task", "position_intervention"], as_index=False)
        .correct.agg(["count", "mean"])
        .reset_index()
    )
    cells.to_csv(args.output.with_suffix(".cells.csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
