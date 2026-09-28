#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 LLAVA_CSV QWEN_CSV INTERNVL_CSV OUTPUT_DIR" >&2
  exit 2
fi

output_dir="$4"
mkdir -p "$output_dir"
python -m evaluation.cross_model \
  --predictions "$1" "$2" "$3" \
  --model-names LLaVA-OneVision-7B Qwen2.5-VL-7B InternVL3-8B \
  --output "$output_dir/cross_model"
