# Reproducibility Guide

## Scope

This repository contains the code needed to regenerate manifests, run model inference and interventions, compute statistics, and recreate figures. It intentionally excludes checkpoints, COCO assets, generated images, prediction CSVs, logs, and internal paper notes.

## Fixed settings

| Item | Value |
|---|---|
| Random seed | `20260927` unless overridden |
| Synthetic image | 768×768 RGB |
| Object size | 40 px |
| Critical-object distance | 168 px |
| Tasks | recognition, binary spatial relation, compositional relation |
| Decoding | greedy, `do_sample=False`, at most 8 new tokens |
| Bootstrap | scene-level, 20,000 draws, 95% percentile CI |
| Tests | paired sign/McNemar-style tests as implemented; Holm family correction |
| Hardware used | 4× NVIDIA RTX 4090, 24 GB each; one independent shard per GPU |

Inference batch size is one. Stopping uses the model tokenizer's EOS behavior plus `max_new_tokens=8`. Temperature is not used because sampling is disabled.

## Model configurations

| Model | Precision | Interface | Expected visual tokens |
|---|---:|---|---:|
| LLaVA-OneVision-7B | FP16 | Transformers 4.57.1, slow processor | 3,699 |
| Qwen2.5-VL-7B | BF16 | Transformers 4.57.1, SDPA | 729 |
| InternVL3-8B | BF16 | Transformers remote model code | 1,280 |

The exact immutable checkpoint revisions were not logged in the completed run and are therefore **UNKNOWN**. Model IDs in `configs/models.yaml` are public defaults, not claims about immutable revisions. Pin revisions and record them before archival reproduction. InternVL's exact public checkpoint identifier is also marked `UNKNOWN` rather than guessed.

## Experiment map

| Experiment | Entry points | Input | Primary output |
|---|---|---|---|
| E11 cross-model | `experiments.cross_model.run_{llava,qwen,internvl}` | synthetic multimodel/native manifests | prediction CSV per model |
| E12 LLaVA factorial | LLaVA runner + `evaluation.factorial` | factorial manifest | effects JSON/tables |
| E13 Qwen causal | `build_qwen_manifest`, `run_qwen`, `evaluation.causal` | Qwen native manifest | causal predictions/results |
| E14 LLaVA causal | `build_llava_manifest`, `run_llava`, `evaluation.causal` | multimodel manifest | causal predictions/results |
| E16 controlled natural | COCO builders + LLaVA runner + `evaluation.natural` | disjoint calibration/test manifests | natural-test results |
| E17 topology | `topology.build_manifest`, causal runners, `evaluation.topology` | 192-scene locked subset | 24-permutation results |
| E18 augmentation | `evaluation.disagreement` | selected E17 predictions | five-view aggregation results |
| E19 disagreement | `evaluation.disagreement`, `visualization.disagreement` | selected E17 predictions | metrics, bins, figure |

Run each Python module with `--help` for its full arguments.

## Causal invariants

### Qwen

The native vision grid is spatially merged before the LLM. On the 27×27 merged grid, four 13×13 corner blocks are permuted and the center row/column remains fixed (676/729 tokens moved).

- `content_only`: changes post-encoder feature assignment to LLM slots; position IDs remain canonical.
- `position_only`: changes M-RoPE coordinate assignment; feature content remains canonical.
- `joint`: moves the same content blocks and their coordinates together.
- Fixed in all conditions: source image, vision-encoder input, prompt, answer, tensor shape, token count, and feature multiset.

### LLaVA

The packed representation contains a 729-token global thumbnail followed by a 54×54 high-resolution mosaic with a fixed newline token after every row (`729 + 54×55 = 3,699`). Four 27×27 high-resolution blocks are permuted (2,916 tokens).

- `content_only`: changes packed high-resolution feature assignment; language-model position IDs stay canonical.
- `position_only`: changes position-ID assignment; packed feature content stays canonical.
- `joint`: changes both using the same block permutation.
- Fixed in all conditions: source image, vision features as a multiset, global-thumbnail tokens, newline tokens, prompt, answer, tensor shape, and token count.

## Four-GPU execution

Use one process per GPU and the same `--num-shards 4` value. Example for LLaVA:

```bash
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank python -m experiments.cross_model.run_llava \
    --manifest data/generated/synthetic/multimodel_manifest.csv \
    --output outputs/llava_shard${rank}.csv \
    --model "$LLAVA_MODEL" --num-shards 4 --shard-index $rank &
done
wait
```

Then use `scripts/merge_prediction_shards.py`, which rejects duplicate IDs and incomplete coverage.

## Controlled COCO protocol

COCO files are not redistributed. The builder uses COCO instance annotations and source images to create 768×768 controlled compositions. The final protocol uses colored role frames and conditional candidate scoring. Calibration and test must be disjoint at both scene and COCO source-image levels. The strongest evidence remains controlled object-crop composition; untouched natural full scenes were not established.

## Verification checklist

1. Record Git commit, Python, CUDA, driver, PyTorch, Transformers, and checkpoint revisions.
2. Run `bash scripts/smoke_test.sh`.
3. Verify each model's `image_token_count` is constant within a paired experiment.
4. Verify unique `example_id`, complete shard coverage, and expected rows before analysis.
5. Preserve scene pairing during bootstrap and statistical tests.
6. Keep calibration categories, prompts, and gates locked before inspecting the natural test.
7. Retain raw predictions and manifests with checksums outside Git.

## Known reproducibility limitations

- Exact checkpoint commits and the complete package lockfile from the original server are unavailable.
- InternVL uses `trust_remote_code=True`; upstream code changes can affect results unless a revision is pinned.
- No model weights or prediction artifacts are included.
- Natural validation is controlled-natural rather than unaltered full-scene evaluation.
- Results were obtained with deterministic decoding, but low-level GPU kernels may still vary across hardware/software stacks.
