import torch

from models.qwen import (
    permute_merged_corner_quadrants,
    permute_patch_quadrants,
)
from interventions.llava_causal import permute_llava_packed_tiles


def test_qwen_patch_quadrant_permutation_preserves_shape_and_moves_content():
    values = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    grid = torch.tensor([[1, 4, 4]])
    output = permute_patch_quadrants(values, grid, (2, 0, 3, 1)).reshape(4, 4)
    expected = torch.tensor(
        [
            [8, 9, 0, 1],
            [12, 13, 4, 5],
            [10, 11, 2, 3],
            [14, 15, 6, 7],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(output, expected)


def test_qwen_identity_permutation_returns_same_tensor():
    values = torch.randn(16, 8)
    assert permute_patch_quadrants(values, torch.tensor([[1, 4, 4]]), (0, 1, 2, 3)) is values


def test_merged_odd_grid_permutation_leaves_center_seams_fixed():
    values = torch.arange(25, dtype=torch.float32).reshape(25, 1)
    output = permute_merged_corner_quadrants(values, (1, 5, 5), (2, 0, 3, 1)).reshape(5, 5)
    expected = torch.tensor(
        [
            [15, 16, 2, 0, 1],
            [20, 21, 7, 5, 6],
            [10, 11, 12, 13, 14],
            [18, 19, 17, 3, 4],
            [23, 24, 22, 8, 9],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(output, expected)


def test_llava_packed_permutation_moves_tiles_but_keeps_thumbnail_and_newlines():
    values = torch.arange(3699, dtype=torch.float32).reshape(3699, 1)
    output = permute_llava_packed_tiles(values, (2, 0, 3, 1)).squeeze(-1)
    assert torch.equal(output[:729], values[:729, 0])
    original = values[729:, 0].reshape(54, 55)
    moved = output[729:].reshape(54, 55)
    assert torch.equal(moved[:, 54], original[:, 54])
    assert torch.equal(moved[:27, :27], original[27:, :27])
    assert torch.equal(moved[:27, 27:54], original[:27, :27])
    assert torch.equal(moved[27:, :27], original[27:, 27:54])
    assert torch.equal(moved[27:, 27:54], original[:27, 27:54])
