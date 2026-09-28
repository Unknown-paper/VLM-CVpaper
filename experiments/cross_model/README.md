# Cross-model benchmark (E11)

Run the same controlled synthetic benchmark on LLaVA-OneVision, Qwen2.5-VL, and InternVL. Use `run_llava.py`, `run_qwen.py`, or `run_internvl.py`; all support scene-level sharding and resumable CSV output.

Within each model, paired rows keep resolution, geometry, question, answer, and visual-token count fixed. Visual-token counts differ across architectures and are not compared as though they were identical.
