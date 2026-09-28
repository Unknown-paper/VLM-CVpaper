from __future__ import annotations

import time
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from models.common import normalize_answer, resolve_image_path, select_manifest_shard


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_crop(image: Image.Image, left: int, top: int, size: int = 384) -> Image.Image:
    fill = image.getpixel((0, 0))
    output = Image.new("RGB", (size, size), fill)
    source = (
        max(0, left),
        max(0, top),
        min(image.width, left + size),
        min(image.height, top + size),
    )
    if source[2] > source[0] and source[3] > source[1]:
        region = image.crop(source)
        output.paste(region, (source[0] - left, source[1] - top))
    return output


def internvl_tiles(
    image: Image.Image,
    phase_x: int,
    phase_y: int,
    use_thumbnail: bool = True,
    tile_permutation: tuple[int, ...] | None = None,
) -> list[Image.Image]:
    tiles = [
        _pad_crop(image, col * 384 + phase_x, row * 384 + phase_y)
        for row in range(2)
        for col in range(2)
    ]
    if tile_permutation is not None:
        permutation = tuple(int(index) for index in tile_permutation)
        if sorted(permutation) != list(range(len(tiles))):
            raise ValueError(f"Invalid tile permutation: {permutation}")
        tiles = [tiles[index] for index in permutation]
    if use_thumbnail:
        tiles.append(image.copy())
    return tiles


def build_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_internvl(model_path: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float16 if load_in_4bit else torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one quantization mode")
    if load_in_4bit or load_in_8bit:
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            if load_in_4bit
            else BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        )
        kwargs.update(
            {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "max_memory": {0: "7GiB", "cpu": "48GiB"},
            }
        )
    model = AutoModel.from_pretrained(model_path, **kwargs).eval()
    if not load_in_4bit and not load_in_8bit:
        model = model.cuda()
    return model, tokenizer


def run_internvl_manifest(
    manifest_path: Path,
    output_path: Path,
    model_path: str,
    resume: bool = True,
    max_examples: int | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
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
    model, tokenizer = load_internvl(
        model_path, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
    )
    transform = build_transform()
    results = list(completed.values())
    done = set(completed)
    new_examples = 0
    for row in manifest.itertuples(index=False):
        if row.example_id in done:
            continue
        if max_examples is not None and new_examples >= max_examples:
            break
        image = Image.open(resolve_image_path(row.image_path, image_root)).convert("RGB")
        permutation_value = getattr(row, "tile_permutation", "0,1,2,3")
        permutation = tuple(int(item) for item in str(permutation_value).split(","))
        tiles = internvl_tiles(
            image,
            int(row.phase_x),
            int(row.phase_y),
            use_thumbnail=True,
            tile_permutation=permutation,
        )
        model_device = next(model.vision_model.parameters()).device
        pixel_values = torch.stack([transform(tile) for tile in tiles]).to(
            device=model_device,
            dtype=torch.float16 if load_in_4bit else torch.bfloat16,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            raw = model.chat(
                tokenizer,
                pixel_values,
                row.question,
                {
                    "max_new_tokens": 8,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                },
                num_patches_list=[len(tiles)],
            ).strip()
        prediction = normalize_answer(raw)
        result = row._asdict()
        result.update(
            {
                "model": model_path,
                "quantization": (
                    "nf4-4bit" if load_in_4bit else "bnb-int8" if load_in_8bit else "none-bf16"
                ),
                "raw_prediction": raw,
                "prediction": prediction,
                "correct": int(prediction == str(row.answer).lower()),
                "crop_count": len(tiles),
                "image_token_count": int(model.num_image_token * len(tiles)),
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
