from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import pandas as pd
import torch
from PIL import Image

from models.llava import (
    _parse_tile_permutation,
    load_model,
    normalize_answer,
    resolve_image_path,
    select_manifest_shard,
)
from interventions.llava_anyres import phased_anyres_grid


def permute_llava_packed_tiles(
    values: torch.Tensor, permutation: tuple[int, ...]
) -> torch.Tensor:
    """Permute four 27x27 high-resolution blocks in a 3,699-token image sequence.

    The first 729 tokens are the global thumbnail. The remaining 2,970 tokens
    are a 54x54 high-resolution mosaic with one newline token after each row
    (54x55). Thumbnail and newline tokens remain fixed, while all 2,916 tile
    content tokens are reassigned with the same four-block permutation.
    """
    if permutation == (0, 1, 2, 3):
        return values
    if sorted(permutation) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid tile permutation: {permutation}")
    if values.shape[0] != 3699:
        raise ValueError(f"Expected 3699 packed visual tokens, got {values.shape[0]}")
    output = values.clone()
    mosaic = output[729:].reshape(54, 55, *values.shape[1:])
    content = mosaic[:, :54]
    source = (
        content[:27, :27].clone(),
        content[:27, 27:].clone(),
        content[27:, :27].clone(),
        content[27:, 27:].clone(),
    )
    destinations = (
        (slice(0, 27), slice(0, 27)),
        (slice(0, 27), slice(27, 54)),
        (slice(27, 54), slice(0, 27)),
        (slice(27, 54), slice(27, 54)),
    )
    for destination, source_index in zip(destinations, permutation):
        content[destination] = source[source_index]
    return output


@contextmanager
def llava_token_causal_intervention(
    model,
    input_ids: torch.Tensor,
    intervention: str,
    permutation: tuple[int, ...],
):
    valid = {"canonical", "content_only", "position_only", "joint"}
    if intervention not in valid:
        raise ValueError(f"Unknown LLaVA causal intervention: {intervention}")
    change_content = intervention in {"content_only", "joint"}
    change_position = intervention in {"position_only", "joint"}
    image_token_id = int(model.config.image_token_index)
    visual_indices = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if visual_indices.numel() != 3699:
        raise ValueError(f"Expected 3699 image placeholders, got {visual_indices.numel()}")

    core = model.model
    original_features = core.get_image_features
    language_model = core.language_model
    original_lm_forward = language_model.forward

    if change_content:
        def patched_features(this, *args, **kwargs):
            features = original_features(*args, **kwargs)
            if len(features) != 1:
                raise ValueError("Expected exactly one packed image feature tensor")
            moved = permute_llava_packed_tiles(features[0], permutation)
            return (moved,) if isinstance(features, tuple) else [moved]

        core.get_image_features = MethodType(patched_features, core)

    if change_position:
        prompt_length = int(input_ids.shape[1])

        def patched_lm_forward(this, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is not None and inputs_embeds.shape[1] == prompt_length:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    attention_mask = kwargs.get("attention_mask")
                    if attention_mask is None:
                        position_ids = torch.arange(
                            prompt_length, device=inputs_embeds.device
                        ).unsqueeze(0)
                    else:
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 0)
                else:
                    position_ids = position_ids.clone()
                current = position_ids[0, visual_indices].unsqueeze(-1)
                moved = permute_llava_packed_tiles(current, permutation)
                position_ids[0, visual_indices] = moved.squeeze(-1)
                kwargs["position_ids"] = position_ids
            return original_lm_forward(*args, **kwargs)

        language_model.forward = MethodType(patched_lm_forward, language_model)

    try:
        yield {"intervened_visual_tokens": 2916, "packed_visual_layout": "729+54x55"}
    finally:
        core.get_image_features = original_features
        language_model.forward = original_lm_forward


def run_llava_causal_manifest(
    manifest_path: Path,
    output_path: Path,
    model_id: str,
    resume: bool = True,
    max_examples: int | None = None,
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
    model, processor = load_model(model_id, load_in_4bit=False)
    device = model.device
    results = list(completed.values())
    done = set(completed)
    new_examples = 0
    for row in manifest.itertuples(index=False):
        if row.example_id in done:
            continue
        if max_examples is not None and new_examples >= max_examples:
            break
        image = Image.open(resolve_image_path(row.image_path, image_root)).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row.question}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        with phased_anyres_grid(
            processor.image_processor,
            int(row.phase_x),
            int(row.phase_y),
            int(row.overlap_x),
            int(row.overlap_y),
            str(getattr(row, "global_thumbnail_mode", "normal")),
            (0, 1, 2, 3),
        ):
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        image_token_count = int(
            (inputs["input_ids"] == int(model.config.image_token_index)).sum().item()
        )
        inputs = {
            key: (
                value.to(device, dtype=torch.float16)
                if key == "pixel_values"
                else value.to(device)
            )
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        intervention = str(row.position_intervention)
        permutation = _parse_tile_permutation(row.merged_token_permutation)
        start = time.perf_counter()
        with llava_token_causal_intervention(
            model, inputs["input_ids"], intervention, permutation
        ) as metadata, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        prediction = normalize_answer(raw)
        result = row._asdict()
        result.update(
            {
                "model": model_id,
                "quantization": "none-fp16",
                "raw_prediction": raw,
                "prediction": prediction,
                "correct": int(prediction == str(row.answer).lower()),
                "image_token_count": image_token_count,
                **metadata,
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
            f"{prediction!r}/{row.answer!r} correct={result['correct']}",
            flush=True,
        )
    frame = pd.DataFrame(results)
    if new_examples:
        frame.to_csv(output_path, index=False)
    return frame
