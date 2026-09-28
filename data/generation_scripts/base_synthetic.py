from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


CANVAS_SIZE = 768
TILE_SIZE = 384
OBJECT_SIZE = 40
OBJECT_DISTANCE = 168
BACKGROUND = (245, 245, 242)
CHOICES = ("square", "circle", "triangle")
COLORS = {
    "red": (220, 42, 42),
    "blue": (36, 92, 210),
    "green": (35, 155, 75),
    "yellow": (232, 181, 25),
}


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    color: str
    x: int
    y: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        r = OBJECT_SIZE // 2
        return self.x - r, self.y - r, self.x + r, self.y + r


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    orientation: str
    first: ObjectSpec
    second: ObjectSpec


def _draw_shape(draw: ImageDraw.ImageDraw, obj: ObjectSpec) -> None:
    box = obj.box
    fill = COLORS[obj.color]
    outline = (25, 25, 25)
    if obj.shape == "square":
        draw.rectangle(box, fill=fill, outline=outline, width=3)
    elif obj.shape == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=3)
    elif obj.shape == "triangle":
        x0, y0, x1, y1 = box
        draw.polygon([(obj.x, y0), (x1, y1), (x0, y1)], fill=fill, outline=outline)
        draw.line([(obj.x, y0), (x1, y1), (x0, y1), (obj.x, y0)], fill=outline, width=3)
    elif obj.shape == "diamond":
        x0, y0, x1, y1 = box
        draw.polygon([(obj.x, y0), (x1, obj.y), (obj.x, y1), (x0, obj.y)], fill=fill, outline=outline)
        draw.line([(obj.x, y0), (x1, obj.y), (obj.x, y1), (x0, obj.y), (obj.x, y0)], fill=outline, width=3)
    else:
        raise ValueError(f"Unknown shape: {obj.shape}")


def make_scenes(n_scenes: int = 12) -> list[SceneSpec]:
    """Create deterministic, balanced horizontal and vertical two-object scenes."""
    shape_pairs = [
        ("square", "circle"),
        ("triangle", "square"),
        ("circle", "triangle"),
        ("circle", "square"),
        ("square", "triangle"),
        ("triangle", "circle"),
    ]
    color_pairs = [
        ("red", "blue"),
        ("green", "yellow"),
        ("blue", "red"),
        ("yellow", "green"),
        ("red", "green"),
        ("blue", "yellow"),
    ]
    scenes: list[SceneSpec] = []
    half_d = OBJECT_DISTANCE // 2
    for idx in range(n_scenes):
        orientation = "horizontal" if idx % 2 == 0 else "vertical"
        shapes = shape_pairs[(idx // 2) % len(shape_pairs)]
        colors = color_pairs[(idx // 2 + idx % 2) % len(color_pairs)]
        # Counterbalance the pair midpoint along the manipulated axis. This
        # prevents boundary status from being synonymous with large |phase|.
        midpoint_offset = (-64, -32, 32, 64)[(idx // 2) % 4]
        # Counterbalance the orthogonal location while keeping every object well
        # away from the unchanged orthogonal tile boundary at 384 px.
        jitter = (-96, 96)[(idx // 2) % 2]
        if orientation == "horizontal":
            p1 = (TILE_SIZE + midpoint_offset - half_d, TILE_SIZE + jitter)
            p2 = (TILE_SIZE + midpoint_offset + half_d, TILE_SIZE + jitter)
        else:
            p1 = (TILE_SIZE + jitter, TILE_SIZE + midpoint_offset - half_d)
            p2 = (TILE_SIZE + jitter, TILE_SIZE + midpoint_offset + half_d)
        scenes.append(
            SceneSpec(
                scene_id=f"scene_{idx:03d}",
                orientation=orientation,
                first=ObjectSpec(shapes[0], colors[0], *p1),
                second=ObjectSpec(shapes[1], colors[1], *p2),
            )
        )
    return scenes


def render_scene(scene: SceneSpec, path: Path) -> str:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, scene.first)
    _draw_shape(draw, scene.second)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_single_object(scene: SceneSpec, path: Path) -> str:
    """Render the target object alone for the matched recognition control."""
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, scene.second)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boundary_status(scene: SceneSpec, phase: int) -> tuple[str, int]:
    """Return tile relationship and signed minimum clearance from object edges."""
    boundary = TILE_SIZE + phase
    axis = "x" if scene.orientation == "horizontal" else "y"
    a = scene.first
    b = scene.second
    ar = OBJECT_SIZE // 2
    av = getattr(a, axis)
    bv = getattr(b, axis)
    low, high = sorted((av, bv))
    if low - ar < boundary < low + ar or high - ar < boundary < high + ar:
        clearance = -min(abs(boundary - low), abs(boundary - high))
        return "object_cut", clearance
    if low + ar <= boundary <= high - ar:
        clearance = min(boundary - (low + ar), (high - ar) - boundary)
        return "cross_tile", clearance
    nearest_edge = min(abs(boundary - (low - ar)), abs(boundary - (high + ar)))
    return "same_tile", nearest_edge


def questions(scene: SceneSpec) -> list[dict[str, str]]:
    relation_word = "left of" if scene.orientation == "horizontal" else "above"
    first, second = scene.first, scene.second
    shape_options = ", ".join(CHOICES)
    color_options = ", ".join(COLORS)
    return [
        {
            "task": "relation_shape",
            "question": (
                f"Answer with exactly one of: {shape_options}. "
                f"What shape is {relation_word} the {second.color} {second.shape}?"
            ),
            "answer": first.shape,
        },
        {
            "task": "relation_color",
            "question": (
                f"Answer with exactly one of: {color_options}. "
                f"What color is the object {relation_word} the {second.color} {second.shape}?"
            ),
            "answer": first.color,
        },
        {
            "task": "recognition_color",
            "question": (
                f"Answer with exactly one of: {color_options}. "
                "What color is the object?"
            ),
            "answer": second.color,
        },
        {
            "task": "recognition_shape",
            "question": (
                f"Answer with exactly one of: {shape_options}. "
                "What shape is the object?"
            ),
            "answer": second.shape,
        },
    ]


def build_dataset(
    root: Path,
    phases: Iterable[int],
    n_scenes: int = 12,
    include_overlap_rescue: bool = True,
) -> Path:
    dataset_dir = root / "dataset"
    image_dir = dataset_dir / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    scenes = make_scenes(n_scenes)
    for scene in scenes:
        image_path = image_dir / f"{scene.scene_id}.png"
        sha256 = render_scene(scene, image_path)
        single_image_path = image_dir / f"{scene.scene_id}_single.png"
        single_sha256 = render_single_object(scene, single_image_path)
        for phase in phases:
            status, clearance = boundary_status(scene, phase)
            dx = phase if scene.orientation == "horizontal" else 0
            dy = phase if scene.orientation == "vertical" else 0
            for q_idx, question in enumerate(questions(scene)):
                is_recognition = question["task"].startswith("recognition")
                task_image_path = single_image_path if is_recognition else image_path
                task_sha256 = single_sha256 if is_recognition else sha256
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
                        "image_kind": "single_object" if is_recognition else "object_pair",
                        "image_path": task_image_path.relative_to(dataset_dir).as_posix(),
                        "image_sha256": task_sha256,
                        "image_width": CANVAS_SIZE,
                        "image_height": CANVAS_SIZE,
                        "object_size": OBJECT_SIZE,
                        "object_distance": OBJECT_DISTANCE,
                    }
                )
        if include_overlap_rescue:
            # At native phase, 200 px overlap places both objects together in
            # one high-resolution crop while keeping the 2x2 crop count fixed.
            for q_idx, question in enumerate(questions(scene)):
                is_recognition = question["task"].startswith("recognition")
                task_image_path = single_image_path if is_recognition else image_path
                task_sha256 = single_sha256 if is_recognition else sha256
                rows.append(
                    {
                        "example_id": f"{scene.scene_id}_overlap200_q{q_idx}",
                        "scene_id": scene.scene_id,
                        "orientation": scene.orientation,
                        "phase": 0,
                        "phase_x": 0,
                        "phase_y": 0,
                        "boundary_status": "overlap_rescue",
                        "boundary_clearance_px": 0,
                        "intervention": "overlap200",
                        "overlap_x": 200 if scene.orientation == "horizontal" else 0,
                        "overlap_y": 200 if scene.orientation == "vertical" else 0,
                        "task": question["task"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "image_kind": "single_object" if is_recognition else "object_pair",
                        "image_path": task_image_path.relative_to(dataset_dir).as_posix(),
                        "image_sha256": task_sha256,
                        "image_width": CANVAS_SIZE,
                        "image_height": CANVAS_SIZE,
                        "object_size": OBJECT_SIZE,
                        "object_distance": OBJECT_DISTANCE,
                    }
                )
    manifest = dataset_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "canvas_size": CANVAS_SIZE,
        "tile_size": TILE_SIZE,
        "object_size": OBJECT_SIZE,
        "object_distance": OBJECT_DISTANCE,
        "background_rgb": BACKGROUND,
        "phases": list(phases),
        "n_scenes": n_scenes,
        "include_overlap_rescue": include_overlap_rescue,
        "source_images_reused_across_phases": True,
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest
