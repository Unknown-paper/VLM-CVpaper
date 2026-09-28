# Spatial Serialization Fragility in Dynamic-Resolution Vision-Language Models

Anonymous review repository for **Spatial Serialization Fragility in Dynamic-Resolution Vision-Language Models**.

## Claim

This project characterizes evidence-preserving serialization invariance as a reliability property of VLM visual-language interfaces. Across the evaluated LLaVA-OneVision, Qwen2.5-VL, and InternVL3 settings, relational predictions are substantially more sensitive to controlled reserialization than object recognition. We do **not** claim that dynamic resolution itself causes the observed degradation, that tile-boundary crossing is a universal condition, or that the tested architectures share one internal process.

## Repository map

```text
serialization-fragility-vlm/
├── configs/                       # documented experiment defaults
├── data/generation_scripts/       # synthetic and controlled-COCO builders
├── experiments/
│   ├── cross_model/               # LLaVA, Qwen, and InternVL inference CLIs
│   ├── causal_intervention/       # manifest builders and causal runners
│   ├── topology/                  # exhaustive 2x2 permutation manifest
│   ├── natural_validation/        # protocol documentation
│   └── disagreement/              # protocol documentation
├── src/
│   ├── models/                    # model adapters
│   ├── interventions/             # processor and post-encoder interventions
│   ├── evaluation/                # paired bootstrap/statistical analyses
│   └── visualization/             # paper figure scripts
├── scripts/                       # end-to-end shell entry points
├── tests/                         # CPU unit tests
├── REPRODUCIBILITY.md
└── REPO_AUDIT.md
```

Generated images, predictions, model weights, and paper-internal notes are intentionally excluded.

## Installation

Python 3.10 or 3.11, PyTorch with CUDA, and a recent NVIDIA GPU are required for inference. The reported runs used four independent RTX 4090 24 GB workers; each worker handled one scene-level shard.

```bash
git clone https://github.com/Unknown-paper/VLM-CVpaper.git
cd VLM-CVpaper
conda env create -f environment.yml
conda activate serialization-fragility
pip install -e .
```

For optional NF4/int8 pilot reproduction, install `pip install -e '.[quantization]'`. The main causal results use FP16 or BF16, not quantization.

## Supported models

- LLaVA-OneVision-7B
- Qwen2.5-VL-7B
- InternVL3-8B

## 1. Generate the controlled synthetic benchmark

The default command creates the 440-scene benchmark and manifests using seed `20260927`:

```bash
python -m data.generation_scripts.build_synthetic_benchmark \
  --output data/generated/synthetic \
  --n-scenes 440 \
  --seed 20260927
```

The three tasks are object recognition, binary spatial relation, and compositional relation. Images are 768×768; answer-critical geometry, question, answer, and visual-token count are paired within a model.

## 2. Run cross-model inference

Set model identifiers to local checkpoint directories or Hugging Face IDs:

```bash
export LLAVA_MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
export QWEN_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export INTERNVL_MODEL=/path/to/internvl3-8b

bash scripts/run_main_experiment.sh llava \
  data/generated/synthetic/multimodel_manifest.csv outputs/llava.csv
bash scripts/run_main_experiment.sh qwen \
  data/generated/synthetic/qwen_native_manifest.csv outputs/qwen.csv
bash scripts/run_main_experiment.sh internvl \
  data/generated/synthetic/multimodel_manifest.csv outputs/internvl.csv
```

To distribute a manifest across GPUs, invoke the Python entry point once per GPU with `--num-shards 4 --shard-index 0..3`. Sharding is by scene, so paired conditions never cross workers. Merge only after all shards finish:

```bash
python scripts/merge_prediction_shards.py \
  --inputs outputs/llava_shard*.csv \
  --manifest data/generated/synthetic/multimodel_manifest.csv \
  --output outputs/llava.csv
```

## 3. Run content/position causal interventions

Create a four-condition manifest, then run it. These interventions act after the vision encoder and keep source pixels, prompts, answers, tensor shape, and the visual-token multiset fixed.

```bash
python -m experiments.causal_intervention.build_qwen_manifest \
  --input data/generated/synthetic/qwen_native_manifest.csv \
  --output data/generated/synthetic/qwen_causal_manifest.csv
bash scripts/run_causal.sh qwen \
  data/generated/synthetic/qwen_causal_manifest.csv outputs/qwen_causal.csv

python -m experiments.causal_intervention.build_llava_manifest \
  --input data/generated/synthetic/multimodel_manifest.csv \
  --output data/generated/synthetic/llava_causal_manifest.csv
bash scripts/run_causal.sh llava \
  data/generated/synthetic/llava_causal_manifest.csv outputs/llava_causal.csv
```

Conditions are canonical, content-only, position-only, and joint content+position movement. See [the intervention protocol](experiments/causal_intervention/README.md) for exact invariants.

## 4. Analyze results

```bash
bash scripts/evaluate.sh outputs/llava.csv outputs/qwen.csv outputs/internvl.csv outputs/analysis

python -m evaluation.causal \
  --predictions outputs/qwen_causal.csv \
  --output outputs/analysis/qwen_causal.json
```

Analysis uses scene-paired bootstrap confidence intervals, exact paired tests where applicable, and Holm correction for prespecified families. Every evaluator validates key invariants before reporting results.

## Recreate analysis figures

After producing the corresponding analysis JSON/CSV files:

```bash
python -m visualization.topology \
  --analysis outputs/analysis/topology.json \
  --output figures/topology.png

python -m visualization.disagreement \
  --bins outputs/analysis/disagreement.disagreement_bins.csv \
  --output figures/disagreement.png

python -m visualization.final_validation \
  --qwen outputs/analysis/qwen_causal.json \
  --llava outputs/analysis/llava_causal.json \
  --natural outputs/analysis/natural.json \
  --output figures/final_validation.png
```

Each visualization also writes a PDF companion where implemented.

## Controlled-natural validation

Download COCO 2017 validation images and `instances_val2017.json` into `data/coco/`, then use the builders in `data/generation_scripts/`. The final test composes real COCO object crops on controlled canvases with colored role frames. It is a controlled-natural bridge, not untouched full-scene ecological validation; the calibration/test split must remain disjoint by scene and COCO source image.

## Main evidence at a glance

- Full-precision cross-model experiments show relation-selective sensitivity to coherent region reserialization.
- Post-encoder interventions rule out reduced pixel resolution, fewer tokens, and loss of object evidence for the main causal contrasts.
- Qwen is predominantly sensitive to content-to-slot assignment; LLaVA is sensitive to isolated content and position movement and recovers substantially when both move jointly.
- Exhaustive 2×2 permutations reject preservation of the tile-adjacency graph as a general explanation.
- Controlled COCO compositions reproduce order sensitivity, while simple five-view majority voting does not mitigate it.
- Cross-serialization disagreement is an instability indicator, not calibrated confidence for canonical answers.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact scope, known unknowns, and experiment-to-command mapping.

## Tests

```bash
bash scripts/smoke_test.sh
```

This checks imports, unit tests, CLI entry points, and a small dataset build. It does not download checkpoints or run GPU inference.

## Citation

Replace this review-period placeholder with the archival citation upon release:

```bibtex
@misc{spatial_serialization_fragility_2026,
  title  = {Spatial Serialization Fragility in Dynamic-Resolution Vision-Language Models},
  author = {Authors withheld during review},
  year   = {2026},
  note   = {Citation placeholder; cite the repository URL and commit hash}
}
```

## License

Code is released under the [MIT License](LICENSE). COCO images, model checkpoints, and third-party model code retain their original licenses and are not redistributed here.
