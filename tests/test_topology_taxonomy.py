from __future__ import annotations

import itertools

from experiments.topology.build_manifest import D4, RANDOM_BREAKING, metrics


def test_d4_has_eight_unique_grid_automorphisms() -> None:
    assert len(D4) == 8
    assert len(set(D4.values())) == 8
    for permutation in D4.values():
        item = metrics(permutation)
        assert item["grid_edges_preserved"] == 4
        assert item["grid_edge_preservation"] == 1.0


def test_all_other_permutations_break_grid_adjacency() -> None:
    d4 = set(D4.values())
    breaking = [p for p in itertools.permutations(range(4)) if p not in d4]
    assert len(breaking) == 16
    assert all(metrics(p)["grid_edges_preserved"] < 4 for p in breaking)
    assert metrics(RANDOM_BREAKING)["topology_transform"] == "random_permutation"


def test_requested_named_controls() -> None:
    assert D4["identity"] == (0, 1, 2, 3)
    assert D4["rotate_180"] == (3, 2, 1, 0)
    assert D4["column_major_transpose"] == (0, 2, 1, 3)
