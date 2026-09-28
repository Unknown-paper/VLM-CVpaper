#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 {llava|qwen|internvl} MANIFEST OUTPUT [IMAGE_ROOT]" >&2
  exit 2
fi

model_name="$1"
manifest="$2"
output="$3"
image_root="${4:-}"
common=(--manifest "$manifest" --output "$output")
if [[ -n "$image_root" ]]; then common+=(--image-root "$image_root"); fi

case "$model_name" in
  llava)
    : "${LLAVA_MODEL:?Set LLAVA_MODEL to a Hugging Face ID or checkpoint path}"
    python -m experiments.cross_model.run_llava "${common[@]}" --model "$LLAVA_MODEL"
    ;;
  qwen)
    : "${QWEN_MODEL:?Set QWEN_MODEL to a Hugging Face ID or checkpoint path}"
    python -m experiments.cross_model.run_qwen "${common[@]}" --model "$QWEN_MODEL"
    ;;
  internvl)
    : "${INTERNVL_MODEL:?Set INTERNVL_MODEL to a Hugging Face ID or checkpoint path}"
    python -m experiments.cross_model.run_internvl "${common[@]}" --model "$INTERNVL_MODEL"
    ;;
  *)
    echo "Unknown model: $model_name" >&2
    exit 2
    ;;
esac
