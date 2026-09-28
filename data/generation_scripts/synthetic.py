from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .base_synthetic import BACKGROUND, COLORS, ObjectSpec, SceneSpec, _draw_shape, boundary_status


GATE1_PHASES = (-128, -96, -64, -32, 0, 32, 64, 96, 128)
SHAPES = ("square", "circle", "triangle", "diamond")
SHAPE_PAIRS = tuple(itertools.permutations(SHAPES, 2))
COLOR_PAIRS = tuple(itertools.permutations(COLORS, 2))
MIDPOINT_OFFSETS = (-160, -128, -96, -64, -32, 0, 32, 64, 96, 128, 160)
ORTHOGONAL_OFFSETS = (-144, -96, 96, 144)


def _balanced_scenes(n_scenes: int, seed: int = 20260926) -> list[SceneSpec]:
    """Stratify by orientation/midpoint, then sample unique matched controls."""
    rng = random.Random(seed)
    strata = list(itertools.product(("horizontal", "vertical"), MIDPOINT_OFFSETS))
    rng.shuffle(strata)
    quotas = {stratum: n_scenes // len(strata) for stratum in strata}
    for stratum in strata[: n_scenes % len(strata)]:
        quotas[stratum] += 1
    scenes = []
    seen_single_targets: set[tuple[object, ...]] = set()
    half_distance = 84
    for orientation, midpoint_offset in strata:
        if quotas[(orientation, midpoint_offset)] == 0:
            continue
        candidates = list(itertools.product(ORTHOGONAL_OFFSETS, SHAPE_PAIRS, COLOR_PAIRS))
        rng.shuffle(candidates)
        selected = 0
        for orthogonal_offset, shapes, colors in candidates:
            midpoint = 384 + midpoint_offset
            orthogonal = 384 + orthogonal_offset
            if orientation == "horizontal":
                p1 = (midpoint - half_distance, orthogonal)
                p2 = (midpoint + half_distance, orthogonal)
            else:
                p1 = (orthogonal, midpoint - half_distance)
                p2 = (orthogonal, midpoint + half_distance)
            single_signature = (shapes[0], colors[0], p1)
            if single_signature in seen_single_targets:
                continue
            seen_single_targets.add(single_signature)
            idx = len(scenes)
            scenes.append(
                SceneSpec(
                    scene_id=f"gate1_{idx:04d}",
                    orientation=orientation,
                    first=ObjectSpec(shapes[0], colors[0], *p1),
                    second=ObjectSpec(shapes[1], colors[1], *p2),
                )
            )
            selected += 1
            if selected == quotas[(orientation, midpoint_offset)]:
                break
        if selected != quotas[(orientation, midpoint_offset)]:
            raise RuntimeError(f"Could not fill stratum {(orientation, midpoint_offset)}")
    if len(scenes) != n_scenes:
        raise RuntimeError(f"Could only construct {len(scenes)} unique-control scenes")
    return scenes


def _distractors(scene: SceneSpec) -> tuple[ObjectSpec, ObjectSpec]:
    """Create a second aligned pair so the named reference must be bound."""
    remaining_shapes = [shape for shape in SHAPES if shape not in {scene.first.shape, scene.second.shape}]
    remaining_colors = [color for color in COLORS if color not in {scene.first.color, scene.second.color}]
    if scene.orientation == "horizontal":
        other_axis = 768 - scene.first.y
        positions = ((scene.first.x, other_axis), (scene.second.x, other_axis))
    else:
        other_axis = 768 - scene.first.x
        positions = ((other_axis, scene.first.y), (other_axis, scene.second.y))
    return (
        ObjectSpec(remaining_shapes[0], remaining_colors[0], *positions[0]),
        ObjectSpec(remaining_shapes[1], remaining_colors[1], *positions[1]),
    )


def _render(scene: SceneSpec, path: Path, single: bool) -> str:
    image = Image.new("RGB", (768, 768), BACKGROUND)
    draw = ImageDraw.Draw(image)
    if single:
        _draw_shape(draw, scene.first)
    else:
        _draw_shape(draw, scene.first)
        _draw_shape(draw, scene.second)
        for distractor in _distractors(scene):
            _draw_shape(draw, distractor)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _questions(scene: SceneSpec) -> tuple[dict[str, str], dict[str, str]]:
    relation = "left of" if scene.orientation == "horizontal" else "above"
    choices = "square, circle, triangle, diamond"
    return (
        {
            "task": "relation_shape",
            "question": (
                f"Answer with exactly one of: {choices}. "
                f"What shape is {relation} the {scene.second.color} {scene.second.shape}?"
            ),
            "answer": scene.first.shape,
        },
        {
            "task": "recognition_shape",
            "question": f"Answer with exactly one of: {choices}. What shape is the object?",
            "answer": scene.first.shape,
        },
    )


def build_gate1_dataset(root: Path, n_scenes: int = 200, seed: int = 20260926) -> Path:
    dataset_dir = root / "dataset"
    image_dir = dataset_dir / "images"
    rows: list[dict[str, object]] = []
    scenes = _balanced_scenes(n_scenes, seed)
    for scene in scenes:
        pair_path = image_dir / f"{scene.scene_id}_pair.png"
        single_path = image_dir / f"{scene.scene_id}_single.png"
        pair_hash = _render(scene, pair_path, single=False)
        single_hash = _render(scene, single_path, single=True)
        distractors = _distractors(scene)
        for phase in GATE1_PHASES:
            status, clearance = boundary_status(scene, phase)
            dx = phase if scene.orientation == "horizontal" else 0
            dy = phase if scene.orientation == "vertical" else 0
            for q_idx, question in enumerate(_questions(scene)):
                recognition = question["task"].startswith("recognition")
                rows.append(
                    {
                        "example_id": f"{scene.scene_id}_p{phase:+04d}_q{q_idx}",
                        "scene_id": scene.scene_id,
                        "orientation": scene.orientation,
                        "phase": phase,
                        "phase_x": dx,
                        "phase_y": dy,
                        "boundary_status": status,
                        "boundary_clearance_px": clearance,
                        "intervention": "phase",
                        "overlap_x": 0,
                        "overlap_y": 0,
                        "task": question["task"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "image_kind": "single_object" if recognition else "object_pair",
                        "image_path": (
                            (single_path if recognition else pair_path)
                            .relative_to(dataset_dir)
                            .as_posix()
                        ),
                        "image_sha256": single_hash if recognition else pair_hash,
                        "image_width": 768,
                        "image_height": 768,
                        "object_size": 40,
                        "object_distance": 168,
                        "distractor_shapes": "+".join(obj.shape for obj in distractors),
                        "distractor_colors": "+".join(obj.color for obj in distractors),
                    }
                )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = dataset_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "stage": "gate1-kill-screen",
        "independent_scenes": n_scenes,
        "seed": seed,
        "phases": GATE1_PHASES,
        "tasks": ["relation_shape", "recognition_shape"],
        "confirmatory": False,
        "reason": "Local 4-bit inference is exploratory; confirmatory runs require bf16/fp16.",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest
