from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .controlled_coco_source import INVERSE, PRIMARY_PERMUTATION, choose_phase


CANVAS_SIZE = 768
PANEL_SIZE = 160
HALF_DISTANCE = 100
BACKGROUND = (238, 238, 238)


def _resolve_local(path: str, source_image_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    normalized = str(path).replace("\\", "/")
    rebased = source_image_dir / normalized
    if rebased.exists():
        return rebased
    rebased = source_image_dir / normalized.rsplit("/", 1)[-1]
    if rebased.exists():
        return rebased
    raise FileNotFoundError(path)


def _object_crop(image: Image.Image, box) -> Image.Image:
    left, top, right, bottom = (float(x) for x in box)
    width, height = right - left, bottom - top
    side = max(width, height) * 1.18
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    crop_box = (
        round(max(0, center_x - side / 2)),
        round(max(0, center_y - side / 2)),
        round(min(image.width, center_x + side / 2)),
        round(min(image.height, center_y + side / 2)),
    )
    crop = image.crop(crop_box)
    return ImageOps.pad(crop, (PANEL_SIZE, PANEL_SIZE), method=Image.Resampling.LANCZOS, color=BACKGROUND)


def _positions(relation: str):
    if relation == "left of":
        return (384 - HALF_DISTANCE, 384), (384 + HALF_DISTANCE, 384)
    if relation == "right of":
        return (384 + HALF_DISTANCE, 384), (384 - HALF_DISTANCE, 384)
    if relation == "above":
        return (384, 384 - HALF_DISTANCE), (384, 384 + HALF_DISTANCE)
    if relation == "below":
        return (384, 384 + HALF_DISTANCE), (384, 384 - HALF_DISTANCE)
    raise ValueError(relation)


def _paste(
    canvas: Image.Image,
    crop: Image.Image,
    center,
    outline=(80, 80, 80),
) -> tuple[float, float, float, float]:
    left, top = center[0] - PANEL_SIZE // 2, center[1] - PANEL_SIZE // 2
    canvas.paste(crop, (left, top))
    ImageDraw.Draw(canvas).rectangle(
        (left, top, left + PANEL_SIZE, top + PANEL_SIZE), outline=outline, width=6
    )
    return float(left), float(top), float(left + PANEL_SIZE), float(top + PANEL_SIZE)


def build(args) -> pd.DataFrame:
    source = pd.read_csv(args.source_manifest)
    scenes = source[source.boundary_status.eq("same")].drop_duplicates("scene_id")
    rows = []
    image_dir = args.output.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for record in scenes.to_dict("records"):
        image = Image.open(_resolve_local(record["image_path"], args.source_image_dir)).convert("RGB")
        first_crop = _object_crop(image, json.loads(record["first_box"]))
        second_crop = _object_crop(image, json.loads(record["second_box"]))
        first_center, second_center = _positions(record["relation"])

        pair = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
        first_outline = (220, 30, 30) if args.marker_reference else (80, 80, 80)
        second_outline = (20, 90, 230) if args.marker_reference else (80, 80, 80)
        first_panel = _paste(pair, first_crop, first_center, first_outline)
        second_panel = _paste(pair, second_crop, second_center, second_outline)
        pair_path = image_dir / f'{record["scene_id"]}_pair.jpg'
        pair.save(pair_path, quality=95)

        single = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
        _paste(single, first_crop, first_center, first_outline)
        single_path = image_dir / f'{record["scene_id"]}_single.jpg'
        single.save(single_path, quality=95)

        orientation = record["orientation"]
        same_phase = choose_phase(first_panel, second_panel, orientation, "same_tile")
        cross_phase = choose_phase(first_panel, second_panel, orientation, "cross_tile")
        other = "vertical" if orientation == "horizontal" else "horizontal"
        orthogonal_phase = choose_phase(first_panel, second_panel, other, "same_tile")
        if same_phase is None or cross_phase is None or orthogonal_phase is None:
            raise RuntimeError(f'No phase pair for {record["scene_id"]}')
        scene_number = int(str(record["scene_id"]).rsplit("_", 1)[-1])
        queried_relation = record["relation"] if scene_number % 2 == 0 else INVERSE[record["relation"]]
        binary_answer = "yes" if scene_number % 2 == 0 else "no"
        conditions = (
            ("same", same_phase, "0,1,2,3"),
            ("cross", cross_phase, "0,1,2,3"),
            ("cross_permuted", cross_phase, PRIMARY_PERMUTATION),
        )
        for condition, phase, permutation in conditions:
            phase_x = phase if orientation == "horizontal" else orthogonal_phase
            phase_y = phase if orientation == "vertical" else orthogonal_phase
            base = {
                "scene_id": record["scene_id"],
                "source_image_id": record["source_image_id"],
                "split": args.force_split or record["split"],
                "orientation": orientation,
                "relation": record["relation"],
                "boundary_status": condition,
                "phase": phase,
                "phase_x": phase_x,
                "phase_y": phase_y,
                "overlap_x": 0,
                "overlap_y": 0,
                "first_category": record["first_category"],
                "second_category": record["second_category"],
                "first_box": json.dumps(first_panel),
                "second_box": json.dumps(second_panel),
                "global_thumbnail_mode": "normal",
                "tile_permutation": permutation,
                "image_width": CANVAS_SIZE,
                "image_height": CANVAS_SIZE,
                "intervention": "perm_primary" if condition == "cross_permuted" else "canonical",
            }
            questions = (
                (
                    "object_recognition",
                    (
                        f'Answer with exactly one of: {record["first_category"]}, '
                        f'{record["second_category"]}. What object is inside the red frame?'
                        if args.marker_reference
                        else f'Answer with exactly one of: {record["first_category"]}, '
                        f'{record["second_category"]}. What object is shown?'
                        if args.forced_choice
                        else "Answer with exactly one object category. What object is shown?"
                    ),
                    record["first_category"],
                    single_path,
                ),
                (
                    "spatial_relation",
                    (
                        f'Answer exactly yes or no. Is the red-framed object '
                        f'{queried_relation} the blue-framed object?'
                        if args.marker_reference
                        else f'Answer exactly yes or no. Is the {record["first_category"]} '
                        f'{queried_relation} the {record["second_category"]}?'
                    ),
                    binary_answer,
                    pair_path,
                ),
                (
                    "compositional_relation",
                    (
                        f'Answer with exactly one of: {record["first_category"]}, '
                        f'{record["second_category"]}. Which object is {record["relation"]} '
                        f'the blue-framed object?'
                        if args.marker_reference
                        else f'Answer with exactly one of: {record["first_category"]}, '
                        f'{record["second_category"]}. Which object is {record["relation"]}: '
                        f'{record["first_category"]} or {record["second_category"]}?'
                        if args.forced_choice
                        else f'Answer with exactly one object category. What object is '
                        f'{record["relation"]} the {record["second_category"]}?'
                    ),
                    record["first_category"],
                    pair_path,
                ),
            )
            for task, question, answer, image_path in questions:
                rows.append(
                    base
                    | {
                        "example_id": f'{record["scene_id"]}_{condition}_{task}',
                        "task": task,
                        "stage2_task": task,
                        "question": question,
                        "answer": answer,
                        "image_path": image_path.relative_to(args.output.parent).as_posix(),
                        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    }
                )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    frame[frame.split.eq("calibration")].to_csv(args.output.parent / "calibration_manifest.csv", index=False)
    frame[frame.split.eq("test")].to_csv(args.output.parent / "test_manifest.csv", index=False)
    metadata = {
        "source": "COCO 2017 photographic object crops selected before test inference",
        "scenes": int(frame.scene_id.nunique()),
        "primary_questions": int(frame[frame.boundary_status.isin(["same", "cross"])].shape[0]),
        "panel_size": PANEL_SIZE,
        "object_center_distance": HALF_DISTANCE * 2,
        "relation_balance": frame.drop_duplicates("scene_id").relation.value_counts().to_dict(),
        "note": "Controlled naturalistic bridge benchmark; not a full-scene natural benchmark.",
        "answer_format": "locked two-alternative forced choice" if args.forced_choice else "open category",
        "reference_marker": (
            "red target frame and blue reference frame"
            if args.marker_reference
            else "none"
        ),
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/generated/controlled_coco/source_manifest.csv")
    )
    parser.add_argument(
        "--source-image-dir", type=Path, default=Path("data/generated/controlled_coco/source_images")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/controlled_coco/manifest.csv")
    )
    parser.add_argument("--force-split", choices=("calibration", "test"))
    parser.add_argument("--forced-choice", action="store_true")
    parser.add_argument("--marker-reference", action="store_true")
    args = parser.parse_args()
    frame = build(args)
    print(frame.groupby(["split", "boundary_status", "stage2_task"]).size())


if __name__ == "__main__":
    main()
