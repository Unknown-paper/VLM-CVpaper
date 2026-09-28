from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


NATURAL_CATEGORY_EQUIVALENTS = (
    {"tv", "television"},
    {"couch", "sofa"},
    {"bicycle", "bike"},
    {"motorcycle", "motorbike"},
    {"airplane", "plane"},
    {"cell phone", "phone"},
    {"dining table", "table"},
    {"person", "man", "woman", "boy", "girl", "people"},
)


def select_manifest_shard(
    manifest: pd.DataFrame,
    shard_index: int = 0,
    num_shards: int = 1,
) -> pd.DataFrame:
    """Assign complete scenes to disjoint workers."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}); got {shard_index}")
    if num_shards == 1:
        return manifest.reset_index(drop=True)
    unit_column = "scene_id" if "scene_id" in manifest.columns else "example_id"
    units = sorted(manifest[unit_column].astype(str).unique())
    assigned = {unit for index, unit in enumerate(units) if index % num_shards == shard_index}
    return manifest[manifest[unit_column].astype(str).isin(assigned)].reset_index(drop=True)


def resolve_image_path(value: str, image_root: Path | None = None) -> Path:
    """Resolve a manifest path locally or rebase it onto an explicit image root."""
    original = Path(str(value))
    if original.exists():
        return original
    if image_root is not None:
        normalized = str(value).replace("\\", "/")
        candidate = image_root / normalized
        if candidate.exists():
            return candidate
        candidate = image_root / normalized.rsplit("/", 1)[-1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {value!r}; image_root={image_root}")


def parse_tile_permutation(value, crop_count: int = 4) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple(range(crop_count))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return tuple(range(crop_count))
    return tuple(int(item.strip()) for item in text.split(","))


# Compatibility name used by the experiment runners.
_parse_tile_permutation = parse_tile_permutation


def normalize_answer(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower().strip())
    valid = {
        "square", "circle", "triangle", "diamond",
        "red", "blue", "green", "yellow", "yes", "no",
    }
    for token in tokens:
        if token in valid:
            return token
    return tokens[0] if tokens else ""


def normalize_expected_text(text: str, expected: str) -> str:
    if str(expected).lower() in {"yes", "no"}:
        return normalize_answer(text)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    normalized = " ".join(tokens)
    expected_tokens = re.findall(r"[a-z0-9]+", str(expected).lower())
    while expected_tokens and expected_tokens[0] in {"a", "an", "the"}:
        expected_tokens.pop(0)
    expected_normalized = " ".join(expected_tokens)
    for equivalents in NATURAL_CATEGORY_EQUIVALENTS:
        if expected_normalized in equivalents and normalized in equivalents:
            return expected_normalized
    return normalized
