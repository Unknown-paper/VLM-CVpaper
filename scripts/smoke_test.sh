#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

python -m compileall -q data experiments src tests
python -m pytest -q
python -m data.generation_scripts.build_synthetic_benchmark \
  --n-scenes 22 --seed 20260927 --output outputs/smoke/synthetic
python -m experiments.causal_intervention.build_llava_manifest \
  --input outputs/smoke/synthetic/multimodel_manifest.csv \
  --output outputs/smoke/llava_causal.csv
python -m experiments.causal_intervention.build_qwen_manifest \
  --input outputs/smoke/synthetic/qwen_native_manifest.csv \
  --output outputs/smoke/qwen_causal.csv
python -m experiments.topology.build_manifest \
  --base-manifest outputs/smoke/qwen_causal.csv \
  --output outputs/smoke/topology.csv --scenes 22 --seed 20260927
python -m experiments.cross_model.run_llava --help >/dev/null
python -m experiments.cross_model.run_qwen --help >/dev/null
python -m experiments.cross_model.run_internvl --help >/dev/null
echo "CPU smoke tests passed. GPU model inference was not run."
