from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


FACTOR_COLUMNS = {
    "partition": "partition_cross",
    "ordering": "order_permuted",
    "thumbnail": "thumbnail_removed",
}


def _bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return [float(x) for x in np.percentile(values[indices].mean(axis=1) * 100, [2.5, 97.5])]


def _sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if not len(nonzero):
        return 1.0
    positive = int((nonzero > 0).sum())
    return float(binomtest(min(positive, len(nonzero) - positive), len(nonzero), 0.5).pvalue)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def main_effect(data: pd.DataFrame, factor: str, seed: int, draws: int):
    column = FACTOR_COLUMNS[factor]
    by_scene = data.groupby(["scene_id", column], as_index=False).correct.mean()
    pivot = by_scene.pivot(index="scene_id", columns=column, values="correct").dropna()
    diff = pivot[1].to_numpy(float) - pivot[0].to_numpy(float)
    return {
        "n_scenes": int(len(pivot)),
        "factor_0_accuracy": float(pivot[0].mean()),
        "factor_1_accuracy": float(pivot[1].mean()),
        "marginal_effect_pp": float(diff.mean() * 100),
        "ci95_pp": _bootstrap(diff, seed, draws),
        "exact_sign_p": _sign_p(diff),
    }


def interaction_effect(data: pd.DataFrame, first: str, second: str, seed: int, draws: int):
    a, b = FACTOR_COLUMNS[first], FACTOR_COLUMNS[second]
    by_scene = data.groupby(["scene_id", a, b], as_index=False).correct.mean()
    pivot = by_scene.pivot(index="scene_id", columns=[a, b], values="correct").dropna()
    did = (
        pivot[(1, 1)] - pivot[(1, 0)] - pivot[(0, 1)] + pivot[(0, 0)]
    ).to_numpy(float)
    return {
        "n_scenes": int(len(pivot)),
        "interaction_pp": float(did.mean() * 100),
        "ci95_pp": _bootstrap(did, seed, draws),
        "exact_sign_p": _sign_p(did),
    }


def fit_gee(data: pd.DataFrame):
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    formula = (
        "correct ~ C(stage2_task) * partition_cross * order_permuted * thumbnail_removed"
    )
    result = GEE.from_formula(
        formula,
        groups="scene_id",
        cov_struct=Exchangeable(),
        family=Binomial(),
        data=data,
    ).fit(maxiter=200)
    table = []
    for name in result.params.index:
        table.append(
            {
                "term": name,
                "log_odds": float(result.params[name]),
                "std_error": float(result.bse[name]),
                "p_value": float(result.pvalues[name]),
                "odds_ratio": float(np.exp(np.clip(result.params[name], -50, 50))),
            }
        )
    return {"converged": bool(result.converged), "terms": table}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    if data.example_id.duplicated().any():
        raise ValueError("Duplicate example ids")
    if set(data.image_token_count.unique()) != {3699}:
        raise ValueError(f"Visual token count drift: {sorted(data.image_token_count.unique())}")
    data = data.copy()
    data["partition_cross"] = data.partition_factor.eq("cross").astype(int)
    data["order_permuted"] = data.order_factor.eq("permuted").astype(int)
    data["thumbnail_removed"] = data.thumbnail_factor.eq("objects_removed").astype(int)

    cells = (
        data.groupby(
            ["stage2_task", "partition_factor", "order_factor", "thumbnail_factor"],
            as_index=False,
        )
        .correct.agg(["count", "mean"])
        .reset_index()
    )
    effects: dict[str, dict[str, object]] = {}
    records: list[tuple[dict[str, object], str]] = []
    for task_index, task in enumerate(sorted(data.stage2_task.unique())):
        subset = data[data.stage2_task.eq(task)]
        task_result = {"main_effects": {}, "interactions": {}}
        for factor_index, factor in enumerate(FACTOR_COLUMNS):
            value = main_effect(
                subset, factor, args.seed + task_index * 100 + factor_index, args.draws
            )
            task_result["main_effects"][factor] = value
            records.append((value, "exact_sign_p"))
        pairs = (("partition", "ordering"), ("partition", "thumbnail"), ("ordering", "thumbnail"))
        for pair_index, (first, second) in enumerate(pairs):
            value = interaction_effect(
                subset,
                first,
                second,
                args.seed + task_index * 100 + 20 + pair_index,
                args.draws,
            )
            task_result["interactions"][f"{first}:{second}"] = value
            records.append((value, "exact_sign_p"))
        effects[task] = task_result
    adjusted = _holm([record[key] for record, key in records])
    for (record, _), p_adjusted in zip(records, adjusted):
        record["holm_p"] = p_adjusted

    try:
        gee = fit_gee(data)
    except Exception as error:
        gee = {"converged": False, "error": f"{type(error).__name__}: {error}"}
    result = {
        "rows": int(len(data)),
        "scenes": int(data.scene_id.nunique()),
        "visual_token_count": 3699,
        "effects": effects,
        "gee": gee,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output.with_suffix(".cells.csv"), index=False)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
