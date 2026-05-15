# Relevance To ISLES'26

This file translates the literature review into a preparation plan for our ISLES'26 entry.

## ISLES'26 Task Interpretation

The current ISLES'26 challenge is a direct successor to ISLES'22-ATLAS:

- input: native-space T1w MRI,
- metadata: `DAYS_POST_STROKE`, `CHRONICITY`, `CENTER`,
- output: binary infarct mask,
- scope: acute, sub-acute, and chronic lesions,
- scale: about 2,000 annotated scans from 60+ centers.

Compared with ATLAS v2.0, the big changes are scale, center diversity, native-space emphasis, and explicit phase metadata. Compared with ISLES'22 multimodal MRI, the big change is that T1w is much less lesion-specific than DWI/ADC/FLAIR.

## Baseline Strategy

The first target should be a faithful, auditable nnU-Net baseline:

1. Implement the evaluation pipeline first.
2. Match challenge metrics locally: Dice, lesion-wise F1, lesion count error, volume difference.
3. Add stratified reports by:
   - center,
   - chronicity,
   - days post stroke bins,
   - lesion volume bins,
   - connected-component count.
4. Train nnU-Net with center/chronicity/lesion-size aware validation.
5. Reproduce the ATLAS CTRL/MAPPING setup on ATLAS data if access permits.

## Model Development Strategy

After nnU-Net is running:

1. Add MAPPING-style model diversity:
   - default loss,
   - TopK or focal/Tversky-like lesion-sensitive loss,
   - residual encoder variant,
   - self-training only if public unlabeled data is allowed by challenge rules.
2. Add small-lesion specific supervision:
   - MSL labels by lesion component size,
   - DBL labels by boundary/interior distance,
   - lesion-wise oversampling.
3. Add T1w robustness:
   - strong bias-field and intensity augmentation,
   - SynthSeg-like contrast randomization,
   - resolution and spacing perturbation,
   - mild artifact simulation.
4. Add metadata cautiously:
   - metadata-conditioned validation first,
   - optional model conditioning only if it improves held-out center/chronicity.
5. Distill:
   - train a small deployment model from the best ensemble,
   - then optimize for browser runtime.

## Evaluation Must-Haves

Reports should include:

- overall mean and median metrics,
- per-case rank aggregation,
- lesion-size binned metrics,
- lesion-wise precision/recall/F1,
- center-wise worst-case metrics,
- chronicity-wise metrics,
- calibration of connected-component probability thresholds,
- pre/post-processing comparison.

The most important dashboard is not "best Dice". It is: can we detect tiny/scattered lesions without unacceptable false positives, across centers and stroke phases?

## Immediate Work Items

1. Build metric code and unit tests against known small synthetic masks.
2. Download ATLAS v2.0 and CTRL/MAPPING weights where permitted.
3. Run CTRL/MAPPING inference on a small subset to validate IO and metric code.
4. Create ISLES'26 data inventory and metadata summary.
5. Implement stratified splits.
6. Train nnU-Net baseline.
7. Add MSL/DBL branch.
8. Add SynthSeg-like augmentation branch.
9. Distill best teacher into a smaller model for browser feasibility.

## Algorithm Branches To Track

See [stroke_segmentation_algorithms.md](stroke_segmentation_algorithms.md) for the broader survey. The practical order is:

1. nnU-Net / residual nnU-Net.
2. MAPPING-style averaging and component post-processing.
3. MSL/DBL and cross-scale attention for small lesions.
4. SynthSeg-style contrast randomization and center/chronicity robustness.
5. 2.5D or distilled compact model for browser deployment.
6. SAM-like foundation model experiments only after the baseline is stable.

## Risk Register

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Small lesions missed | Hurts lesion-wise F1 and clinical utility | MSL/DBL, oversampling, lesion-wise metrics |
| Center shift | ISLES'26 has 60+ centers | grouped validation, robust augmentation |
| Phase shift | T1w appearance changes by chronicity | metadata-stratified reporting |
| Public leaderboard overfit | ATLAS explicitly downweighted public ranking | internal holdout, limit leaderboard tuning |
| Post-processing deletes true positives | Tiny lesions look like false positives | threshold sweeps by lesion size |
| Browser goal reduces accuracy too early | Small models may underfit subtle T1w lesions | train strong teacher first, distill later |

## Sources

- ISLES'26 page: https://isles-26.grand-challenge.org/
- ATLAS v2.0 paper: https://www.nature.com/articles/s41597-022-01401-7
- MAPPING winner report: https://arxiv.org/abs/2211.15486
- MAPPING code: https://github.com/King-HAW/ATLAS-R2-Docker-Submission
- MSL/DBL paper: https://arxiv.org/abs/2408.02929
