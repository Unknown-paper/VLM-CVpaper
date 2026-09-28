"""Run the cross-model manifest with Qwen2.5-VL."""

import argparse
from pathlib import Path

from models.qwen import run_qwen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--answer-mode", choices=("closed_set", "expected_text"), default="closed_set")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_qwen_manifest(
        args.manifest, args.output, args.model,
        resume=not args.no_resume, max_examples=args.max_examples,
        answer_mode=args.answer_mode, generation_max_new_tokens=args.max_new_tokens,
        shard_index=args.shard_index, num_shards=args.num_shards,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
