from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import pandas as pd
import torch
from PIL import Image

from models.common import (
    _parse_tile_permutation,
    normalize_answer,
    normalize_expected_text,
    resolve_image_path,
    select_manifest_shard,
)


def permute_patch_quadrants(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    permutation: tuple[int, ...],
) -> torch.Tensor:
    """Permute native Qwen patch-grid quadrants without changing token count."""
    if permutation == (0, 1, 2, 3):
        return pixel_values
    if sorted(permutation) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid quadrant permutation: {permutation}")
    if image_grid_thw.shape[0] != 1:
        raise ValueError("Stage-4 Qwen intervention currently supports exactly one image")
    temporal, height, width = (int(x) for x in image_grid_thw[0].tolist())
    if height % 2 or width % 2:
        raise ValueError(f"Patch grid must be even for quadrant intervention; got {height}x{width}")
    if pixel_values.shape[0] != temporal * height * width:
        raise ValueError("Qwen pixel_values length does not match image_grid_thw")
    grid = pixel_values.reshape(temporal, height, width, -1)
    half_h, half_w = height // 2, width // 2
    source = (
        grid[:, :half_h, :half_w],
        grid[:, :half_h, half_w:],
        grid[:, half_h:, :half_w],
        grid[:, half_h:, half_w:],
    )
    output = torch.empty_like(grid)
    destinations = (
        (slice(None), slice(0, half_h), slice(0, half_w)),
        (slice(None), slice(0, half_h), slice(half_w, width)),
        (slice(None), slice(half_h, height), slice(0, half_w)),
        (slice(None), slice(half_h, height), slice(half_w, width)),
    )
    for destination, source_index in zip(destinations, permutation):
        output[destination] = source[source_index]
    return output.reshape_as(pixel_values)


def permute_merged_corner_quadrants(
    values: torch.Tensor,
    merged_grid_thw: tuple[int, int, int],
    permutation: tuple[int, ...],
) -> torch.Tensor:
    """Permute equal corner blocks of an odd merged grid, keeping seam tokens fixed.

    Qwen spatially merges 2x2 vision patches.  A 768px image produces a 27x27
    LLM-token grid, so a clean four-way permutation cannot split every token
    equally.  We therefore permute the four 13x13 corner blocks and leave the
    central row and column untouched.  This moves 676/729 visual tokens while
    preserving tensor shape and gives content and M-RoPE interventions the exact
    same token-level permutation.
    """
    if permutation == (0, 1, 2, 3):
        return values
    if sorted(permutation) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid quadrant permutation: {permutation}")
    temporal, height, width = merged_grid_thw
    if temporal != 1:
        raise ValueError("Stage-4 merged-token intervention supports one image frame")
    if values.shape[0] != temporal * height * width:
        raise ValueError("Merged visual-token length does not match grid")
    grid = values.reshape(temporal, height, width, *values.shape[1:])
    half_h, half_w = height // 2, width // 2
    row_hi = height - half_h
    col_hi = width - half_w
    source = (
        grid[:, :half_h, :half_w].clone(),
        grid[:, :half_h, col_hi:].clone(),
        grid[:, row_hi:, :half_w].clone(),
        grid[:, row_hi:, col_hi:].clone(),
    )
    output = grid.clone()
    destinations = (
        (slice(None), slice(0, half_h), slice(0, half_w)),
        (slice(None), slice(0, half_h), slice(col_hi, width)),
        (slice(None), slice(row_hi, height), slice(0, half_w)),
        (slice(None), slice(row_hi, height), slice(col_hi, width)),
    )
    for destination, source_index in zip(destinations, permutation):
        output[destination] = source[source_index]
    return output.reshape_as(values)


def _merged_grid(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int]:
    if image_grid_thw.shape[0] != 1:
        raise ValueError("Stage-4 causal intervention supports exactly one image")
    temporal, height, width = (int(x) for x in image_grid_thw[0].tolist())
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("Vision grid is not divisible by spatial_merge_size")
    return temporal, height // spatial_merge_size, width // spatial_merge_size


@contextmanager
def qwen_token_causal_intervention(
    model,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    intervention: str,
    permutation: tuple[int, ...],
):
    """Intervene on post-encoder content and/or 3D M-RoPE coordinates.

    Conditions implement a 2x2 design at the exact LLM visual-token interface:
    canonical=(content 0, position 0), content_only=(1,0),
    position_only=(0,1), joint=(1,1).  In the joint condition a content token
    carries its original coordinate to its new serialized slot.
    """
    valid = {"canonical", "content_only", "position_only", "joint"}
    if intervention not in valid:
        raise ValueError(f"Unknown Qwen causal intervention: {intervention}")
    change_content = intervention in {"content_only", "joint"}
    change_position = intervention in {"position_only", "joint"}
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    merged_grid = _merged_grid(image_grid_thw, spatial_merge_size)
    image_token_id = int(model.config.image_token_id)
    expected_tokens = merged_grid[0] * merged_grid[1] * merged_grid[2]
    visual_indices = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if visual_indices.numel() != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} image placeholders, got {visual_indices.numel()}"
        )

    qwen_model = model.model
    original_features = qwen_model.get_image_features
    original_rope = qwen_model.get_rope_index

    if change_content:
        def patched_features(this, pixel_values, grid_thw):
            features = original_features(pixel_values, grid_thw)
            if len(features) != 1:
                raise ValueError("Expected exactly one image feature tensor")
            return (permute_merged_corner_quadrants(features[0], merged_grid, permutation),)

        qwen_model.get_image_features = MethodType(patched_features, qwen_model)

    if change_position:
        def patched_rope(this, *args, **kwargs):
            positions, deltas = original_rope(*args, **kwargs)
            positions = positions.clone()
            current = positions[:, 0, visual_indices].transpose(0, 1)
            moved = permute_merged_corner_quadrants(current, merged_grid, permutation)
            positions[:, 0, visual_indices] = moved.transpose(0, 1)
            return positions, deltas

        qwen_model.get_rope_index = MethodType(patched_rope, qwen_model)

    try:
        yield {
            "merged_grid_thw": "x".join(str(x) for x in merged_grid),
            "intervened_visual_tokens": 4 * (merged_grid[1] // 2) * (merged_grid[2] // 2),
        }
    finally:
        qwen_model.get_image_features = original_features
        qwen_model.get_rope_index = original_rope


def load_qwen(model_id: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    return model, processor


def run_qwen_manifest(
    manifest_path: Path,
    output_path: Path,
    model_id: str,
    resume: bool = True,
    max_examples: int | None = None,
    answer_mode: str = "closed_set",
    generation_max_new_tokens: int = 8,
    shard_index: int = 0,
    num_shards: int = 1,
    image_root: Path | None = None,
) -> pd.DataFrame:
    image_root = image_root or manifest_path.parent
    manifest = select_manifest_shard(
        pd.read_csv(manifest_path), shard_index=shard_index, num_shards=num_shards
    )
    completed: dict[str, dict] = {}
    if resume and output_path.exists():
        old = pd.read_csv(output_path)
        completed = {str(row.example_id): row._asdict() for row in old.itertuples(index=False)}
    model, processor = load_qwen(model_id)
    results = list(completed.values())
    done = set(completed)
    new_examples = 0
    for row in manifest.itertuples(index=False):
        if row.example_id in done:
            continue
        if max_examples is not None and new_examples >= max_examples:
            break
        image = Image.open(resolve_image_path(row.image_path, image_root)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row.question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        permutation = _parse_tile_permutation(getattr(row, "tile_permutation", None))
        causal_intervention = getattr(row, "position_intervention", None)
        if causal_intervention is None:
            inputs["pixel_values"] = permute_patch_quadrants(
                inputs["pixel_values"], inputs["image_grid_thw"], permutation
            )
        image_token_id = int(model.config.image_token_id)
        image_token_count = int((inputs["input_ids"] == image_token_id).sum().item())
        patch_grid = "x".join(str(int(x)) for x in inputs["image_grid_thw"][0].tolist())
        inputs = {
            key: (
                value.to("cuda", dtype=torch.bfloat16)
                if key == "pixel_values"
                else value.to("cuda")
            )
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        start = time.perf_counter()
        model.model.rope_deltas = None
        if causal_intervention is None:
            causal_context = qwen_token_causal_intervention(
                model, inputs["input_ids"], inputs["image_grid_thw"],
                "canonical", (0, 1, 2, 3),
            )
        else:
            causal_permutation = _parse_tile_permutation(
                getattr(row, "merged_token_permutation", "2,0,3,1")
            )
            causal_context = qwen_token_causal_intervention(
                model, inputs["input_ids"], inputs["image_grid_thw"],
                str(causal_intervention), causal_permutation,
            )
        with causal_context as causal_metadata, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        prediction = (
            normalize_expected_text(raw, str(row.answer))
            if answer_mode == "expected_text"
            else normalize_answer(raw)
        )
        expected = (
            normalize_expected_text(str(row.answer), str(row.answer))
            if answer_mode == "expected_text"
            else str(row.answer).lower()
        )
        result = row._asdict()
        result.update(
            {
                "model": model_id,
                "quantization": "none-bf16",
                "raw_prediction": raw,
                "prediction": prediction,
                "correct": int(prediction == expected),
                "image_token_count": image_token_count,
                "native_patch_grid_thw": patch_grid,
                **causal_metadata,
                "latency_s": time.perf_counter() - start,
                "shard_index": shard_index,
                "num_shards": num_shards,
            }
        )
        results.append(result)
        new_examples += 1
        if new_examples % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
        print(
            f"[{len(results):04d}/{len(manifest):04d}] {row.example_id} "
            f"{prediction!r}/{row.answer!r} correct={result['correct']} tokens={image_token_count}",
            flush=True,
        )
    frame = pd.DataFrame(results)
    if new_examples:
        frame.to_csv(output_path, index=False)
    return frame
