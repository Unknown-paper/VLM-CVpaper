# Content/position causal interventions (E13–E14)

The manifest builders expand canonical rows into four conditions: `canonical`, `content_only`, `position_only`, and `joint`.

For Qwen, content blocks and/or their M-RoPE coordinate assignments are permuted at the merged LLM-token interface. For LLaVA, high-resolution packed features and/or language-model position IDs are permuted. The same coherent four-block permutation is used; source pixels, prompts, answers, tensor shapes, feature multisets, and token counts remain fixed. LLaVA global-thumbnail and newline tokens remain fixed. Qwen's odd-grid center row and column remain fixed.

These interventions establish causal sensitivity to assignments, not a universal neural mechanism.
