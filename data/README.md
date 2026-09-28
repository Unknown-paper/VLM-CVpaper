# Data

`generation_scripts/` contains deterministic builders for the 768×768 synthetic benchmark and the controlled-natural COCO compositions.

Generated data is written under `data/generated/` and ignored by Git. COCO 2017 validation images and `instances_val2017.json` should be placed under `data/coco/`; their original license and terms apply.

The synthetic default is 440 scenes, seed `20260927`, 40 px objects, and 168 px critical-object distance. Do not alter geometry between paired serialization conditions.
