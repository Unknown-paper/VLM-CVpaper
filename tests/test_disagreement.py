from __future__ import annotations

import pytest

from evaluation.disagreement import (
    entropy_bits,
    majority_vote,
    pairwise_agreement,
    parse_candidates,
)


def test_majority_vote_and_locked_canonical_tie_break() -> None:
    assert majority_vote(["a", "a", "a", "b", "c"]) == ("a", False)
    assert majority_vote(["a", "b", "b", "a", "c"]) == ("a", True)


def test_agreement_and_entropy() -> None:
    assert pairwise_agreement(["a"] * 5) == 1.0
    assert pairwise_agreement(["a", "a", "a", "a", "b"]) == 0.6
    assert entropy_bits(["a"] * 5) == 0.0


def test_candidate_parser() -> None:
    assert parse_candidates("Answer with exactly yes or no. Is it left?") == ["yes", "no"]
    assert parse_candidates(
        "Answer with exactly one of: square, circle, triangle, diamond. What shape?"
    ) == ["square", "circle", "triangle", "diamond"]
    with pytest.raises(ValueError):
        parse_candidates("What is this?")
