from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr


TASKS = ("object_recognition", "spatial_relation", "compositional_relation")
SERIALIZATIONS = {
    "S0": "identity",
    "S1": "rotate_180",
    "S2": "rotate_90",
    "S3": "column_major_transpose",
    "S4": "random_permutation",
}


def bootstrap_mean(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(value) for value in np.percentile(sampled.mean(axis=1) * 100, [2.5, 97.5])]


def accuracy(values: np.ndarray, seed: int, draws: int) -> dict[str, object]:
    return {
        "accuracy": float(values.mean()),
        "ci95": [value / 100 for value in bootstrap_mean(values, seed, draws)],
        "n": int(len(values)),
    }


def paired_effect(values: np.ndarray, seed: int, draws: int) -> dict[str, object]:
    nonzero = values[values != 0]
    if len(nonzero):
        positives = int((nonzero > 0).sum())
        p_value = float(binomtest(min(positives, len(nonzero) - positives), len(nonzero)).pvalue)
    else:
        p_value = 1.0
    return {
        "effect_pp": float(values.mean() * 100),
        "ci95_pp": bootstrap_mean(values, seed, draws),
        "exact_sign_p": p_value,
        "improved": int((values > 0).sum()),
        "worsened": int((values < 0).sum()),
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


def parse_candidates(question: str) -> list[str]:
    if "exactly yes or no" in question.lower():
        return ["yes", "no"]
    match = re.search(r"Answer with exactly one of:\s*(.*?)\.\s*", question, re.I)
    if not match:
        raise ValueError(f"Cannot parse answer candidates: {question}")
    return [item.strip().lower() for item in match.group(1).split(",")]


def majority_vote(predictions: list[str]) -> tuple[str, bool]:
    counts = Counter(predictions)
    largest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == largest]
    if len(winners) == 1:
        return winners[0], False
    # Locked conservative tie policy: fall back to canonical S0 without using
    # the ground-truth label.
    return predictions[0], True


def pairwise_agreement(predictions: list[str]) -> float:
    counts = Counter(predictions)
    pairs = sum(count * (count - 1) // 2 for count in counts.values())
    total = len(predictions) * (len(predictions) - 1) // 2
    return pairs / total


def entropy_bits(predictions: list[str]) -> float:
    counts = Counter(predictions)
    total = len(predictions)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def prepare(path: Path, model_label: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    transforms = set(SERIALIZATIONS.values())
    data = data[data.topology_transform.isin(transforms)].copy()
    reverse = {transform: name for name, transform in SERIALIZATIONS.items()}
    data["serialization"] = data.topology_transform.map(reverse)
    index = ["scene_id", "stage2_task", "answer", "question"]
    predictions = data.pivot(index=index, columns="serialization", values="prediction")
    predictions = predictions[list(SERIALIZATIONS)].reset_index()
    if len(predictions) != data.scene_id.nunique() * len(TASKS):
        raise ValueError("Incomplete scene/task serialization matrix")
    rows = []
    for row in predictions.itertuples(index=False):
        candidates = parse_candidates(row.question)
        votes = [str(getattr(row, name)) for name in SERIALIZATIONS]
        if any(vote not in candidates for vote in votes):
            raise ValueError(f"Out-of-set prediction in {row.scene_id}/{row.stage2_task}")
        ensemble, tied = majority_vote(votes)
        counts = Counter(votes)
        largest_count = max(counts.values())
        tied_winners = [answer for answer, count in counts.items() if count == largest_count]
        oracle_tie_correct = int(
            ensemble == str(row.answer)
            or (tied and str(row.answer) in tied_winners)
        )
        random_tie_expected_correct = (
            (1 / len(tied_winners) if str(row.answer) in tied_winners else 0.0)
            if tied
            else float(ensemble == str(row.answer))
        )
        modal = max(counts.values()) / len(votes)
        agreement = pairwise_agreement(votes)
        entropy = entropy_bits(votes)
        alt_votes = votes[1:]
        record = {
            "model": model_label,
            "scene_id": row.scene_id,
            "stage2_task": row.stage2_task,
            "answer": str(row.answer),
            "question": row.question,
            "candidate_count": len(candidates),
            "ensemble_prediction": ensemble,
            "ensemble_correct": int(ensemble == str(row.answer)),
            "canonical_correct": int(votes[0] == str(row.answer)),
            "tie_fallback": int(tied),
            "oracle_tie_correct": oracle_tie_correct,
            "random_tie_expected_correct": random_tie_expected_correct,
            "modal_agreement": modal,
            "pairwise_agreement": agreement,
            "serialization_disagreement": 1 - agreement,
            "entropy_bits": entropy,
            "normalized_entropy": entropy / math.log2(min(len(votes), len(candidates))),
            "unanimous": int(len(counts) == 1),
            "alternative_disagreement": 1 - pairwise_agreement(alt_votes),
            "random_variant_accuracy": np.mean(
                [vote == str(row.answer) for vote in votes]
            ),
        }
        for name, vote in zip(SERIALIZATIONS, votes):
            record[f"{name}_prediction"] = vote
            record[f"{name}_correct"] = int(vote == str(row.answer))
        rows.append(record)
    return pd.DataFrame(rows)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def auc_error(disagreement: np.ndarray, correct: np.ndarray) -> float | None:
    error = 1 - correct
    positives = int(error.sum())
    negatives = len(error) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(disagreement, method="average")
    return float((ranks[error == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def disagreement_diagnostic(
    disagreement: np.ndarray,
    correct: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, object]:
    if np.unique(correct).size < 2 or np.unique(disagreement).size < 2:
        correlation = None
        correlation_ci = [None, None]
    else:
        correlation = finite_or_none(spearmanr(disagreement, correct).statistic)
        rng = np.random.default_rng(seed)
        sampled = []
        for _ in range(draws):
            index = rng.integers(0, len(correct), size=len(correct))
            if np.unique(correct[index]).size < 2 or np.unique(disagreement[index]).size < 2:
                continue
            value = spearmanr(disagreement[index], correct[index]).statistic
            if np.isfinite(value):
                sampled.append(value)
        correlation_ci = (
            [float(value) for value in np.percentile(sampled, [2.5, 97.5])]
            if sampled
            else [None, None]
        )
    failed = disagreement[correct == 0]
    passed = disagreement[correct == 1]
    if len(failed) and len(passed):
        difference = float(failed.mean() - passed.mean())
        rng = np.random.default_rng(seed + 1)
        boot = []
        for _ in range(draws):
            boot.append(
                failed[rng.integers(0, len(failed), size=len(failed))].mean()
                - passed[rng.integers(0, len(passed), size=len(passed))].mean()
            )
        difference_ci = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
    else:
        difference = None
        difference_ci = [None, None]
    return {
        "spearman_correctness": correlation,
        "spearman_ci95": correlation_ci,
        "error_auc": auc_error(disagreement, correct),
        "mean_disagreement_correct": float(passed.mean()) if len(passed) else None,
        "mean_disagreement_failed": float(failed.mean()) if len(failed) else None,
        "failed_minus_correct_disagreement": difference,
        "failed_minus_correct_ci95": difference_ci,
        "n_correct": int(len(passed)),
        "n_failed": int(len(failed)),
    }


def continuous_accuracy_correlation(
    disagreement: np.ndarray,
    outcome: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, object]:
    if np.unique(disagreement).size < 2 or np.unique(outcome).size < 2:
        return {"spearman_rho": None, "spearman_ci95": [None, None]}
    point = spearmanr(disagreement, outcome).statistic
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(draws):
        index = rng.integers(0, len(outcome), size=len(outcome))
        if np.unique(disagreement[index]).size < 2 or np.unique(outcome[index]).size < 2:
            continue
        value = spearmanr(disagreement[index], outcome[index]).statistic
        if np.isfinite(value):
            sampled.append(value)
    return {
        "spearman_rho": finite_or_none(point),
        "spearman_ci95": (
            [float(value) for value in np.percentile(sampled, [2.5, 97.5])]
            if sampled
            else [None, None]
        ),
    }


def wilson(successes: float, count: int) -> list[float]:
    if not count:
        return [None, None]
    proportion = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def analyze_task(data: pd.DataFrame, seed: int, draws: int) -> tuple[dict[str, object], list[dict]]:
    cells = {}
    for offset, name in enumerate(SERIALIZATIONS):
        cells[name] = accuracy(data[f"{name}_correct"].to_numpy(float), seed + offset, draws)
    ensemble_correct = data.ensemble_correct.to_numpy(float)
    canonical_correct = data.canonical_correct.to_numpy(float)
    cells["ensemble"] = accuracy(ensemble_correct, seed + 10, draws)
    best_name = max(SERIALIZATIONS, key=lambda name: cells[name]["accuracy"])
    effect = paired_effect(ensemble_correct - canonical_correct, seed + 20, draws)
    result = {
        "conditions": cells,
        "best_fixed_serialization": best_name,
        "best_fixed_accuracy": cells[best_name]["accuracy"],
        "ensemble_minus_canonical": effect,
        "confidence_weighted_ensemble": "skipped_no_comparable_answer_probabilities",
        "controls": {
            "canonical_repeated_five_accuracy": float(canonical_correct.mean()),
            "random_variant_selector_expected_accuracy": float(data.random_variant_accuracy.mean()),
            "random_answer_five_vote_expected_accuracy": float((1 / data.candidate_count).mean()),
            "forward_passes": {"canonical": 1, "ensemble": 5, "matched_canonical_repeat": 5},
            "oracle_any_serialization_accuracy": float(
                data[[f"{name}_correct" for name in SERIALIZATIONS]].max(axis=1).mean()
            ),
            "canonical_errors_rescued_by_any_alternative": int(
                (
                    data.canonical_correct.eq(0)
                    & data[[f"S{index}_correct" for index in range(1, 5)]].max(axis=1).eq(1)
                ).sum()
            ),
            "majority_oracle_tie_accuracy": float(data.oracle_tie_correct.mean()),
            "majority_random_tie_expected_accuracy": float(
                data.random_tie_expected_correct.mean()
            ),
        },
        "agreement": {
            "mean_pairwise_agreement": float(data.pairwise_agreement.mean()),
            "mean_modal_agreement": float(data.modal_agreement.mean()),
            "unanimous_rate": float(data.unanimous.mean()),
            "mean_entropy_bits": float(data.entropy_bits.mean()),
            "mean_normalized_entropy": float(data.normalized_entropy.mean()),
            "tie_fallback_rate": float(data.tie_fallback.mean()),
        },
        "canonical_failure_diagnostic": disagreement_diagnostic(
            data.serialization_disagreement.to_numpy(float),
            canonical_correct,
            seed + 30,
            min(draws, 5_000),
        ),
        "ensemble_failure_diagnostic": disagreement_diagnostic(
            data.serialization_disagreement.to_numpy(float),
            ensemble_correct,
            seed + 35,
            min(draws, 5_000),
        ),
        "serialization_conditional_accuracy_correlation": continuous_accuracy_correlation(
            data.serialization_disagreement.to_numpy(float),
            data.random_variant_accuracy.to_numpy(float),
            seed + 37,
            min(draws, 5_000),
        ),
        "leave_s0_out_failure_diagnostic": disagreement_diagnostic(
            data.alternative_disagreement.to_numpy(float),
            canonical_correct,
            seed + 40,
            min(draws, 5_000),
        ),
    }
    bin_rows = []
    for bin_index, (disagreement, group) in enumerate(data.groupby("serialization_disagreement")):
        n = len(group)
        random_variant_ci = [
            value / 100
            for value in bootstrap_mean(
                group.random_variant_accuracy.to_numpy(float),
                seed + 50 + bin_index,
                draws,
            )
        ]
        row = {
            "serialization_disagreement": float(disagreement),
            "n": n,
            "canonical_accuracy": float(group.canonical_correct.mean()),
            "ensemble_accuracy": float(group.ensemble_correct.mean()),
            "random_variant_accuracy": float(group.random_variant_accuracy.mean()),
            "random_variant_ci_low": random_variant_ci[0],
            "random_variant_ci_high": random_variant_ci[1],
            "canonical_ci_low": wilson(float(group.canonical_correct.sum()), n)[0],
            "canonical_ci_high": wilson(float(group.canonical_correct.sum()), n)[1],
            "ensemble_ci_low": wilson(float(group.ensemble_correct.sum()), n)[0],
            "ensemble_ci_high": wilson(float(group.ensemble_correct.sum()), n)[1],
        }
        bin_rows.append(row)
    return result, bin_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    prepared = pd.concat(
        [
            prepare(args.llava, "LLaVA-OneVision-7B"),
            prepare(args.qwen, "Qwen2.5-VL-7B"),
        ],
        ignore_index=True,
    )
    result: dict[str, object] = {
        "design": {
            "scenes_per_model": 192,
            "serializations": SERIALIZATIONS,
            "majority_tie_policy": "canonical_S0_fallback",
        },
        "models": {},
    }
    primary = []
    bin_rows = []
    for model_index, (model, model_data) in enumerate(prepared.groupby("model")):
        model_result = {"tasks": {}}
        for task_index, task in enumerate(TASKS):
            subset = model_data[model_data.stage2_task.eq(task)].copy()
            task_result, task_bins = analyze_task(
                subset, args.seed + model_index * 1000 + task_index * 100, args.draws
            )
            model_result["tasks"][task] = task_result
            primary.append(task_result["ensemble_minus_canonical"])
            for row in task_bins:
                row.update({"model": model, "task": task})
                bin_rows.append(row)
        result["models"][model] = model_result
    for record, adjusted in zip(
        primary, holm_adjust([record["exact_sign_p"] for record in primary])
    ):
        record["holm_p"] = adjusted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    prepared.to_csv(args.output.with_suffix(".samples.csv"), index=False)
    pd.DataFrame(bin_rows).to_csv(args.output.with_suffix(".disagreement_bins.csv"), index=False)
    main_rows = []
    agreement_rows = []
    for model, model_result in result["models"].items():
        for task, task_result in model_result["tasks"].items():
            effect = task_result["ensemble_minus_canonical"]
            main_rows.append(
                {
                    "model": model,
                    "task": task,
                    "canonical_accuracy": task_result["conditions"]["S0"]["accuracy"],
                    "best_fixed_serialization": task_result["best_fixed_serialization"],
                    "best_fixed_accuracy": task_result["best_fixed_accuracy"],
                    "ensemble_accuracy": task_result["conditions"]["ensemble"]["accuracy"],
                    "ensemble_minus_canonical_pp": effect["effect_pp"],
                    "ci95_low_pp": effect["ci95_pp"][0],
                    "ci95_high_pp": effect["ci95_pp"][1],
                    "holm_p": effect["holm_p"],
                }
            )
            agreement_rows.append(
                {
                    "model": model,
                    "task": task,
                    **task_result["agreement"],
                    **task_result["canonical_failure_diagnostic"],
                }
            )
    pd.DataFrame(main_rows).to_csv(args.output.with_suffix(".main_table.csv"), index=False)
    pd.DataFrame(agreement_rows).to_csv(
        args.output.with_suffix(".agreement_table.csv"), index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
