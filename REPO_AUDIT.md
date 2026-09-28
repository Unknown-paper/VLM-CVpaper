# Repository Audit

Audit date: 2026-09-27. Source workspace: the parent project directory. Public target: this clean repository.

## Current source tree

```text
title 2/
├── .conda/                       # local environment; exclude
├── models/                       # 32 GB checkpoints; exclude
├── external/                     # COCO/third-party assets; exclude
├── artifacts*/                   # generated data, predictions, logs; exclude
├── boundarybench/                # original dataset/inference package; curate
├── scripts/                      # server launchers; replace with portable CLIs
├── tests/                        # relevant tests; curate
├── *.py                          # builders, inference, analysis, plots; curate
├── *.md                          # internal reports/protocols; exclude from code release
├── *.zip, *.tar.gz               # duplicate bundles; exclude
└── serialization-fragility-vlm/ # clean public repository
```

## Source workspace inventory

| Source area | Approx. size | Files | Public-repo decision |
|---|---:|---:|---|
| `models/` | 32,294.2 MiB | 54 | Excluded: checkpoint weights |
| `.conda/` | 4,929.8 MiB | 42,835 | Excluded: machine-specific environment |
| `external/` | 1,108.1 MiB | 972 | Excluded: third-party COCO/model assets |
| `artifacts_stage4/` | 323.2 MiB | 3,395 | Excluded: generated images, predictions, logs |
| earlier `artifacts*` | 33.4 MiB | 633 | Excluded: pilots and generated output |
| `boundarybench/`, root scripts, `tests/` | <1 MiB code | 50+ | Curated and reorganized |
| archives (`*.zip`, `*.tar.gz`) | multiple bundles | — | Excluded: duplicates and generated deliverables |
| internal reports/fact sheets | text only | — | Excluded: paper-internal notes, not executable research code |

## Included code

- Dataset generators and manifest builders required for E11–E19.
- Model adapters for LLaVA-OneVision, Qwen2.5-VL, and InternVL.
- Exact LLaVA and Qwen post-encoder interventions.
- Cross-model, factorial, causal, natural, topology, augmentation, and disagreement analyses.
- Figure scripts, CPU unit tests, clean shell entry points, and public documentation.

## Excluded material

- All model weights and tokenizer caches.
- COCO images/annotations and all third-party datasets.
- Generated images, manifests, predictions, logs, plots, and archived bundles.
- Local environments, caches, editor state, and machine-specific launchers.
- Server credentials, private endpoints, absolute local/server paths, and internal paper-development reports.

## Security and portability review

- The target remote was confirmed empty before preparation.
- No credential is required by repository code.
- Model and data locations are CLI arguments or environment variables.
- `.gitignore` blocks checkpoints, generated data, logs, caches, and `.env` files.
- No automatic Git commit or push is performed.

## Remaining release checks

- Pin exact checkpoint revisions and publish checksums when known.
- Decide whether prediction tables and small benchmark manifests will be released separately.
- Run GPU smoke inference on each public checkpoint ID after the release environment is frozen.
- Add paper citation and archival DOI when available.
