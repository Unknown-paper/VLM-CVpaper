from __future__ import annotations

import re
import time
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from interventions.llava_anyres import phased_anyres_grid


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
    """Split by scene so every paired intervention stays on one worker."""
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
    """Resolve local paths after moving a manifest between Windows and Linux."""
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


def _parse_tile_permutation(value, crop_count: int = 4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple(range(crop_count))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return tuple(range(crop_count))
    return tuple(int(item.strip()) for item in text.split(","))


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    tokens = re.findall(r"[a-z]+", text)
    valid = {
        "square",
        "circle",
        "triangle",
        "diamond",
        "red",
        "blue",
        "green",
        "yellow",
        "yes",
        "no",
    }
    for token in tokens:
        if token in valid:
            return token
    return tokens[0] if tokens else ""


def normalize_expected_text(text: str, expected: str) -> str:
    """Strict normalization for open-vocabulary category answers in the natural control."""
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


def load_model(model_id: str, load_in_4bit: bool = False):
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration

    # The intervention hooks the explicit Python crop routine, so do not select
    # a fused fast processor whose implementation can vary across versions.
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model_kwargs = {"dtype": torch.float16, "low_cpu_mem_usage": True}
    if load_in_4bit:
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
                "device_map": "auto",
                "max_memory": {0: "7GiB", "cpu": "48GiB"},
            }
        )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if not load_in_4bit:
        model = model.to("cuda")
    model.eval()
    return model, processor


def score_answer_candidates(model, processor, inputs: dict, candidates: list[str]):
    """Choose a closed-set answer by mean conditional token log-likelihood."""
    base_length = inputs["input_ids"].shape[1]
    scores: dict[str, float] = {}
    for candidate in candidates:
        candidate_ids = processor.tokenizer(
            str(candidate), add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(inputs["input_ids"].device)
        if candidate_ids.shape[1] == 0:
            raise ValueError(f"Candidate tokenized to an empty sequence: {candidate!r}")
        candidate_inputs = dict(inputs)
        candidate_inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], candidate_ids], dim=1
        )
        extra_mask = torch.ones_like(candidate_ids, dtype=inputs["attention_mask"].dtype)
        candidate_inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], extra_mask], dim=1
        )
        output = model(**candidate_inputs, use_cache=False, return_dict=True)
        logits = output.logits[:, base_length - 1 : base_length + candidate_ids.shape[1] - 1]
        token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, candidate_ids.unsqueeze(-1)
        ).squeeze(-1)
        scores[str(candidate)] = float(token_log_probs.mean().item())
    prediction = max(candidates, key=lambda value: scores[str(value)])
    return str(prediction), scores


def run_manifest(
    manifest_path: Path,
    output_path: Path,
    model_id: str,
    resume: bool = True,
    load_in_4bit: bool = False,
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
        completed = {str(row["example_id"]): row.to_dict() for _, row in old.iterrows()}

    model, processor = load_model(model_id, load_in_4bit=load_in_4bit)
    input_device = model.device
    results = list(completed.values())
    done = set(completed)
    new_examples = 0
    for index, row in manifest.iterrows():
        if row.example_id in done:
            continue
        if max_examples is not None and new_examples >= max_examples:
            break
        image = Image.open(resolve_image_path(row.image_path, image_root)).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row.question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        start = time.perf_counter()
        with phased_anyres_grid(
            processor.image_processor,
            int(row.phase_x),
            int(row.phase_y),
            int(row.overlap_x),
            int(row.overlap_y),
            str(row.get("global_thumbnail_mode", "normal")),
            _parse_tile_permutation(row.get("tile_permutation", None)),
        ):
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        crop_count = int(inputs["pixel_values"].shape[1])
        image_token_id = int(model.config.image_token_index)
        image_token_count = int((inputs["input_ids"] == image_token_id).sum().item())
        inputs = {
            k: (
                v.to(input_device, dtype=torch.float16)
                if k == "pixel_values"
                else v.to(input_device)
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }
        candidate_scores = None
        with torch.inference_mode():
            if answer_mode == "candidate_score":
                candidates = (
                    ["yes", "no"]
                    if str(row.stage2_task) == "spatial_relation"
                    else [str(row.first_category), str(row.second_category)]
                )
                raw, candidate_scores = score_answer_candidates(
                    model, processor, inputs, candidates
                )
                prediction = normalize_expected_text(raw, str(row.answer))
                expected_normalized = normalize_expected_text(
                    str(row.answer), str(row.answer)
                )
            else:
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
                expected_normalized = (
                    normalize_expected_text(str(row.answer), str(row.answer))
                    if answer_mode == "expected_text"
                    else str(row.answer).lower()
                )
        result = row.to_dict()
        result.update(
            {
                "model": model_id,
                "quantization": "nf4-4bit" if load_in_4bit else "none",
                "raw_prediction": raw,
                "candidate_scores": (
                    json.dumps(candidate_scores, sort_keys=True)
                    if candidate_scores is not None
                    else None
                ),
                "prediction": prediction,
                "correct": int(prediction == expected_normalized),
                "crop_count": crop_count,
                "image_token_count": image_token_count,
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
    result_frame = pd.DataFrame(results)
    if new_examples:
        result_frame.to_csv(output_path, index=False)
    return result_frame
