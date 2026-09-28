from __future__ import annotations

import itertools

import pandas as pd

from evaluation.topology import correlation_analysis, taxonomy_analysis
from experiments.topology.build_manifest import metrics


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for scene in ("a", "b", "c"):
        for index, permutation in enumerate(itertools.permutations(range(4))):
            metadata = metrics(permutation)
            rows.append(
                {
                    "scene_id": scene,
                    "topology_permutation_id": f"p{index:02d}",
                    "correct": int(metadata["grid_edge_preservation"] == 1.0),
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def test_taxonomy_effect_recovers_constructed_gap() -> None:
    result = taxonomy_analysis(synthetic_frame(), seed=1, draws=100)
    assert result["accuracy"]["topology_preserving"] == 1.0
    assert result["accuracy"]["topology_breaking"] == 0.0
    assert result["effects"]["breaking_minus_preserving"]["effect_pp"] == -100.0


def test_graph_correlation_recovers_constructed_relationship() -> None:
    result = correlation_analysis(synthetic_frame(), seed=1, draws=20)
    assert result["grid_edge_preservation"]["pearson_r"] == 1.0
    assert result["grid_edge_preservation"]["spearman_rho"] == 1.0
