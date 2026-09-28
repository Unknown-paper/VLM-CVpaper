from pathlib import Path

import pandas as pd
import pytest

from models.common import resolve_image_path, select_manifest_shard


def test_manifest_shards_are_disjoint_and_scene_complete():
    manifest = pd.DataFrame(
        {
            "scene_id": ["a", "a", "b", "b", "c", "c"],
            "example_id": [f"e{i}" for i in range(6)],
        }
    )
    shards = [select_manifest_shard(manifest, index, 2) for index in range(2)]
    assert set(shards[0].example_id).isdisjoint(set(shards[1].example_id))
    assert set(pd.concat(shards).example_id) == set(manifest.example_id)
    for scene_id in manifest.scene_id.unique():
        assert sum(scene_id in set(shard.scene_id) for shard in shards) == 1


def test_invalid_shard_index_rejected():
    with pytest.raises(ValueError):
        select_manifest_shard(pd.DataFrame({"example_id": ["a"]}), 2, 2)


def test_image_root_rebases_relocated_manifest_path(tmp_path: Path):
    image = tmp_path / "example.png"
    image.write_bytes(b"x")
    assert resolve_image_path("old/dataset/example.png", tmp_path) == image


def test_image_root_preserves_portable_relative_subdirectories(tmp_path: Path):
    image = tmp_path / "images" / "example.png"
    image.parent.mkdir()
    image.write_bytes(b"x")
    assert resolve_image_path("images/example.png", tmp_path) == image
