# Paper writing notes

This directory contains the first evidence-constrained CVPR manuscript draft for:

**Spatial Serialization Fragility in Dynamic-Resolution Vision-Language Models**

The paper is positioned as a controlled empirical characterization of the visual-token/LLM interface. It is not written as a hidden-binding, attention-mechanism, or universal position-encoding paper.

## Files

- `main.tex`: complete main-paper draft with the required eight sections.
- `references.bib`: bibliography entries checked against primary paper pages or arXiv records.
- `supplementary.tex`: structured supplementary outline with verified values and required caveats.
- `figures/`: manuscript-local copies of existing figures plus the cross-model plot generated from the final E11 records.
- `build_cross_model_figure.py`: regenerates the cross-model plot from the recorded E11 values.
- `build_falsification_figure.py`: regenerates the compact topology/voting/generic-order summary.

## Evidence policy used in the draft

The central claims use:

- E11 for cross-model behavioral replication;
- E13/E14 for post-encoder content/position intervention;
- the 96-scene matched four-object recognition control;
- E16 only as controlled-natural COCO-crop evidence;
- E17/E18 and the generic order control as falsification/negative results.

The manuscript deliberately does not use the following as headline support:

- attention or hidden-state probes;
- prompt-coordinate “rescue” results;
- bridge-overlap probes;
- the 60-scene VSR subset, whose canonical accuracies are near chance;
- early NF4 or small-model pilots superseded by the final full-precision runs.

## Claim guardrails

Use:

> In the evaluated dynamic-resolution VLMs, relational predictions are sensitive to how spatial-region evidence is serialized. The clean post-encoder profile differs between Qwen2.5-VL and LLaVA-OneVision.

Do not use:

- “tile boundaries cause relational binding failure”;
- “position encoding is the mechanism”;
- “all dynamic-resolution VLMs fail”;
- “natural images prove the effect”;
- “we discover the internal mechanism”;
- “visual permutation is uniquely more harmful than generic order perturbation.”

## Building

Add the official CVPR 2027 author-kit files (`cvpr.sty`, `ieeenat_fullname.bst`, and any required support files) to this directory before submission formatting.

The draft has a two-column fallback when `cvpr.sty` is absent. With the author kit installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The supplementary outline can be checked independently:

```bash
pdflatex supplementary.tex
pdflatex supplementary.tex
```

## Figures and caption caveats

- `method_overview.pdf`: main Figure 1.
- `cross_model_main.pdf`: main Figure 2; state that Qwen E11 is pre-encoder while LLaVA/InternVL use crop-order perturbations.
- `final_validation.pdf`: main Figure 3. The manuscript-local artwork has been regenerated with “Held-out controlled-natural test.” The source is a composed COCO-crop benchmark, not unmodified photographs.
- `matched_recognition_control.pdf`: main Figure 4.
- `falsification_main.pdf`: main Figure 5; summarizes topology, five-view voting, and the generic-order control.
- `topology_main.pdf` plus `serialization_augmentation.pdf`: full versions for the supplement.
- `generic_order_control.pdf`: supplementary negative control.
- `real_spatial_validation.pdf`: supplementary pilot only, clearly marked inconclusive.

## Remaining writing tasks

1. Install the official CVPR 2027 template and measure the actual page count.
2. Replace anonymous author metadata only after the review version is frozen.
3. Run a final title/abstract prior-art search close to submission, including concurrent 2026–2027 work.
4. Decide whether controlled-natural E16 stays in the main paper or moves to the supplement if space is tight; do not remove its limitations.
5. Add exact code/result release URLs and the frozen repository commit.
6. Recheck the final-validation figure's wording and sign conventions after any analysis update.
7. Check that all confidence intervals and adjusted p-values match the released CSV/JSON tables after any analysis-code cleanup.
8. Perform the CVPR anonymity and supplementary-link audit.

## Final self-review: remaining reviewer risks

### Reviewer 1: “This is generic token-order sensitivity.”

**Risk: high.** The generic language-query permutation is more destructive than visual reserialization, not less. It is not semantics preserving, so it does not settle the strongest version of the objection. The paper should argue for a narrower contribution: task-selective fragility of a specific visual-language interface under evidence-preserving reserialization. It must not claim uniqueness among transformer order effects.

### Reviewer 2: “Where is the mechanism?”

**Risk: medium-high.** E13/E14 provide a causal decomposition of interface variables, not a circuit-level mechanism. Qwen and LLaVA have different profiles. The paper should present architecture dependence as a result and state explicitly that the learned origin and internal mediator remain unknown.

### Reviewer 3: “Recognition is a ceiling control.”

**Risk: medium.** The matched four-object control addresses image complexity and distractors but is still 100% accurate. E16 provides below-ceiling recognition with a zero permutation effect, but it is a controlled crop composition. Avoid claiming universal recognition invariance.

### Reviewer 4: “The natural-image evidence is weak.”

**Risk: high.** E16 uses COCO crops on a controlled canvas with colored role markers, and the composition gate passes by one example. The VSR subset is inconclusive. Natural generalization must remain a limitation, not a headline claim.

### Reviewer 5: “The interventions are not identical across models.”

**Risk: medium.** This objection is valid. E11 is behavioral and architecture specific; only Qwen and LLaVA have clean post-encoder decompositions, with different position systems and fixed token subsets. The text must not imply a numerically identical intervention across all three families.

### Reviewer 6: “Dynamic resolution itself has not been isolated.”

**Risk: medium-high.** No matched fixed-resolution architecture/control with the same weights was run. The paper studies tested dynamic-resolution pipelines but does not establish that dynamic resolution is necessary for the effect.

### Reviewer 7: “The topology analysis is too small.”

**Risk: medium.** Exhausting all 24 four-region permutations is strong for the 2×2 graph, but it does not cover larger grids, multiscale tiling, or continuous geometry. Keep the conclusion local: 2×2 adjacency preservation is not sufficient.

### Reviewer 8: “There is no practical mitigation.”

**Risk: medium.** The unweighted five-view vote fails and exploratory rescues are unstable. The paper can motivate reliability tests and future training-time robustness, but it should not promise a solution.

### Overall internal assessment

The draft is viable as a careful CVPR analysis paper if the causal-interface framing remains disciplined. Acceptance risk is driven less by missing effect size than by novelty framing, natural-image scope, and the unresolved generic-order objection. The strongest paper is the one that makes these boundaries explicit and treats negative results as part of the contribution.
