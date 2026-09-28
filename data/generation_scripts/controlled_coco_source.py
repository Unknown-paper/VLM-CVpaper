from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = 768
CONTENT_SIZE = 720
PHASES = tuple(range(-320, 321, 32))
PRIMARY_PERMUTATION = "2,0,3,1"
INVERSE = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}
ORIENTATION = {"left of": "horizontal", "right of": "horizontal", "above": "vertical", "below": "vertical"}


def letterbox_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    scale = min(CONTENT_SIZE / width, CONTENT_SIZE / height)
    pad_x = (CANVAS_SIZE - width * scale) / 2
    pad_y = (CANVAS_SIZE - height * scale) / 2
    x, y, w, h = box
    return pad_x + x * scale, pad_y + y * scale, pad_x + (x + w) * scale, pad_y + (y + h) * scale


def phase_status(first, second, phase: int, orientation: str) -> tuple[str, float]:
    axis = 0 if orientation == "horizontal" else 1
    low_indices = (0, 1)
    high_indices = (2, 3)
    lows = (first[low_indices[axis]], second[low_indices[axis]])
    highs = (first[high_indices[axis]], second[high_indices[axis]])
    if min(lows) < phase or max(highs) > phase + CANVAS_SIZE:
        return "invalid", -1.0
    seam = phase + CANVAS_SIZE / 2
    slots, clearances = [], []
    for low, high in zip(lows, highs):
        if low < seam < high:
            return "object_cut", -min(seam - low, high - seam)
        slots.append(0 if high <= seam else 1)
        clearances.append(min(abs(low - seam), abs(high - seam)))
    return ("same_tile" if slots[0] == slots[1] else "cross_tile"), min(clearances)


def choose_phase(first, second, orientation: str, target: str) -> int | None:
    options = []
    for phase in PHASES:
        status, clearance = phase_status(first, second, phase, orientation)
        if status == target:
            options.append((clearance, -abs(phase), phase))
    return int(max(options)[2]) if options else None


def relation_candidates(first, second, margin: float = 16.0):
    ax, ay = (first[0] + first[2]) / 2, (first[1] + first[3]) / 2
    bx, by = (second[0] + second[2]) / 2, (second[1] + second[3]) / 2
    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    vertical_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    min_width = min(first[2] - first[0], second[2] - second[0])
    min_height = min(first[3] - first[1], second[3] - second[1])
    relations = []
    if first[2] + margin < second[0] and abs(by - ay) < 260 and bx - ax < 380:
        relations.append("left of")
    if second[2] + margin < first[0] and abs(by - ay) < 260 and ax - bx < 380:
        relations.append("right of")
    if first[3] + margin < second[1] and abs(bx - ax) < 260 and by - ay < 380:
        relations.append("above")
    if second[3] + margin < first[1] and abs(bx - ax) < 260 and ay - by < 380:
        relations.append("below")
    return relations


def render_image(source: Path, output: Path, first_box, second_box, overlay: bool = True) -> None:
    image = Image.open(source).convert("RGB")
    scale = min(CONTENT_SIZE / image.width, CONTENT_SIZE / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), image.getpixel((0, 0)))
    canvas.paste(resized, ((CANVAS_SIZE - resized.width) // 2, (CANVAS_SIZE - resized.height) // 2))
    if overlay:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        for label, box, color in (("A", first_box, (220, 30, 30)), ("B", second_box, (20, 90, 230))):
            xy = tuple(round(value) for value in box)
            draw.rectangle(xy, outline=color, width=6)
            label_x, label_y = max(0, xy[0]), max(0, xy[1] - 34)
            draw.rectangle((label_x, label_y, label_x + 34, label_y + 34), fill=color)
            draw.text((label_x + 7, label_y + 1), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=95)


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "serialization-fragility/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if not temporary.stat().st_size:
                raise RuntimeError("downloaded file is empty")
            temporary.replace(output)
            return
        except Exception as error:  # network errors differ across Python versions
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def build(args) -> pd.DataFrame:
    raw = json.loads(args.instances.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in raw["categories"]}
    image_info = {item["id"]: item for item in raw["images"]}
    annotations = defaultdict(list)
    for annotation in raw["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[annotation["image_id"]].append(annotation)

    pools = defaultdict(list)
    diagnostics = Counter()
    allowed_targets = (
        {item.strip() for item in args.allowed_targets.split(",") if item.strip()}
        if args.allowed_targets
        else None
    )
    excluded_image_ids: set[int] = set()
    for excluded_manifest in args.exclude_manifest or []:
        excluded = pd.read_csv(excluded_manifest)
        excluded_image_ids.update(excluded.source_image_id.astype(int).unique())
    for image_id in sorted(image_info):
        if image_id in excluded_image_ids:
            diagnostics["excluded_previous_test"] += 1
            continue
        info = image_info[image_id]
        anns = annotations[image_id]
        counts = Counter(annotation["category_id"] for annotation in anns)
        valid = []
        for annotation in anns:
            if counts[annotation["category_id"]] != 1:
                continue
            box = letterbox_box(annotation["bbox"], info["width"], info["height"])
            width, height = box[2] - box[0], box[3] - box[1]
            area_ratio = annotation["area"] / (info["width"] * info["height"])
            if min(width, height) < 50 or not 0.008 <= area_ratio <= 0.35:
                continue
            valid.append((annotation, box))
        found = False
        for first_index, (first_ann, first_box) in enumerate(valid):
            for second_ann, second_box in valid[first_index + 1 :]:
                if first_ann["category_id"] == second_ann["category_id"]:
                    continue
                if allowed_targets is not None and categories[first_ann["category_id"]] not in allowed_targets:
                    continue
                relations = relation_candidates(first_box, second_box)
                if not relations:
                    continue
                relation = sorted(
                    relations,
                    key=lambda value: hashlib.sha256(f"{image_id}:{value}".encode()).hexdigest(),
                )[0]
                orientation = ORIENTATION[relation]
                same_phase = choose_phase(first_box, second_box, orientation, "same_tile")
                cross_phase = choose_phase(first_box, second_box, orientation, "cross_tile")
                other_orientation = "vertical" if orientation == "horizontal" else "horizontal"
                orthogonal_phase = choose_phase(first_box, second_box, other_orientation, "same_tile")
                if same_phase is None or cross_phase is None or orthogonal_phase is None:
                    diagnostics["no_phase_pair"] += 1
                    continue
                target = categories[first_ann["category_id"]]
                reference = categories[second_ann["category_id"]]
                record = {
                    "image_id": image_id,
                    "info": info,
                    "first_box": first_box,
                    "second_box": second_box,
                    "target": target,
                    "reference": reference,
                    "relation": relation,
                    "orientation": orientation,
                    "same_phase": same_phase,
                    "cross_phase": cross_phase,
                    "orthogonal_phase": orthogonal_phase,
                    "present_categories": {categories[a["category_id"]] for a in anns},
                }
                pools[relation].append(record)
                found = True
                break
            if found:
                break
        if not found:
            diagnostics["no_eligible_pair"] += 1

    for relation in pools:
        pools[relation].sort(
            key=lambda item: hashlib.sha256(
                f'{args.selection_salt}:{item["image_id"]}'.encode()
            ).hexdigest()
        )
    selected, category_counts = [], Counter()
    while len(selected) < args.n_scenes:
        advanced = False
        for relation in ("left of", "right of", "above", "below"):
            index = next(
                (
                    i
                    for i, item in enumerate(pools[relation])
                    if category_counts[item["target"]] < args.max_per_answer
                ),
                None,
            )
            if index is None:
                continue
            item = pools[relation].pop(index)
            selected.append(item)
            category_counts[item["target"]] += 1
            advanced = True
            if len(selected) == args.n_scenes:
                break
        if not advanced:
            break
    if len(selected) < args.n_scenes:
        pool_sizes = {key: len(value) for key, value in pools.items()}
        raise RuntimeError(
            f"Only selected {len(selected)}/{args.n_scenes}; diagnostics={dict(diagnostics)}; "
            f"pool_sizes={pool_sizes}"
        )

    all_category_names = sorted(categories.values())
    rows = []
    raw_dir = args.raw_dir
    rendered_dir = args.output.parent / "images"
    for index, item in enumerate(selected):
        info = item["info"]
        source = raw_dir / info["file_name"]
        if not source.exists() or not source.stat().st_size:
            download(info["coco_url"], source)
        rendered = rendered_dir / f"coco_{index:04d}.jpg"
        render_image(
            source,
            rendered,
            item["first_box"],
            item["second_box"],
            overlay=not args.plain,
        )
        split = "calibration" if index % 5 == 0 else "test"
        target, reference, relation = item["target"], item["reference"], item["relation"]
        queried_relation = relation if index % 2 == 0 else INVERSE[relation]
        binary_answer = "yes" if index % 2 == 0 else "no"
        conditions = (
            ("same", item["same_phase"], "0,1,2,3"),
            ("cross", item["cross_phase"], "0,1,2,3"),
            ("cross_permuted", item["cross_phase"], PRIMARY_PERMUTATION),
        )
        for condition, phase, permutation in conditions:
            phase_x = phase if item["orientation"] == "horizontal" else item["orthogonal_phase"]
            phase_y = phase if item["orientation"] == "vertical" else item["orthogonal_phase"]
            base = {
                "scene_id": f"coco_{index:04d}",
                "source_image_id": item["image_id"],
                "split": split,
                "image_path": rendered.relative_to(args.output.parent).as_posix(),
                "image_sha256": hashlib.sha256(rendered.read_bytes()).hexdigest(),
                "orientation": item["orientation"],
                "relation": relation,
                "boundary_status": condition,
                "phase": phase,
                "phase_x": phase_x,
                "phase_y": phase_y,
                "overlap_x": 0,
                "overlap_y": 0,
                "first_category": target,
                "second_category": reference,
                "first_box": json.dumps(item["first_box"]),
                "second_box": json.dumps(item["second_box"]),
                "global_thumbnail_mode": "normal",
                "tile_permutation": permutation,
                "image_width": CANVAS_SIZE,
                "image_height": CANVAS_SIZE,
            }
            questions = (
                (
                    "object_recognition",
                    "Two relevant objects are outlined and labeled A and B. "
                    "Answer with exactly one object category. What object category is inside box A?",
                    target,
                ),
                (
                    "spatial_relation",
                    f"Two relevant objects are outlined and labeled A and B. "
                    f"Answer exactly yes or no. Is box A {queried_relation} box B?",
                    binary_answer,
                ),
                (
                    "compositional_relation",
                    "Two relevant objects are outlined and labeled A and B. "
                    f"Answer with exactly one object category. What object category is {relation} box B?",
                    target,
                ),
            )
            for task, question, answer in questions:
                rows.append(
                    base
                    | {
                        "example_id": f"coco_{index:04d}_{condition}_{task}",
                        "task": task,
                        "stage2_task": task,
                        "question": question,
                        "answer": answer,
                        "intervention": "perm_primary" if condition == "cross_permuted" else "canonical",
                    }
                )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    frame[frame.split.eq("calibration")].to_csv(
        args.output.parent / "calibration_manifest.csv", index=False
    )
    frame[frame.split.eq("test")].to_csv(
        args.output.parent / "test_manifest.csv", index=False
    )
    metadata = {
        "source": args.instances.name,
        "seed_rule": f"SHA256 stable ordering with locked salt={args.selection_salt}; no new-test predictions used",
        "scenes": len(selected),
        "calibration_scenes": int(frame[frame.split.eq("calibration")].scene_id.nunique()),
        "test_scenes": int(frame[frame.split.eq("test")].scene_id.nunique()),
        "content_size": CONTENT_SIZE,
        "annotation_overlay": (
            "none" if args.plain else "6-pixel red A and blue B boxes from COCO ground-truth bounding boxes"
        ),
        "allowed_targets": sorted(allowed_targets) if allowed_targets else None,
        "excluded_previous_image_ids": len(excluded_image_ids),
        "selection": "unique category instances, >=50 px rendered box sides, 0.8%-35% area, clear edge separation, intact in same/cross phases",
        "diagnostics": dict(diagnostics),
    }
    (args.output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("data/coco/annotations/instances_val2017.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/controlled_coco/manifest.csv")
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/coco/val2017")
    )
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--allowed-targets", default="")
    parser.add_argument("--exclude-manifest", type=Path, nargs="*")
    parser.add_argument("--selection-salt", default="stage4-primary")
    # 104 scenes balance all four relations exactly (26 each), yielding 624
    # same/cross primary questions and 936 rows including cross-permutation.
    parser.add_argument("--n-scenes", type=int, default=104)
    parser.add_argument("--max-per-answer", type=int, default=20)
    args = parser.parse_args()
    frame = build(args)
    print(frame.groupby(["split", "boundary_status", "stage2_task"]).size())


if __name__ == "__main__":
    main()
