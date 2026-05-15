# Literature Review — ISLES'22 & ATLAS R2.0 Challenges

Background research to inform our ISLES'26 entry. ISLES'26 inputs are
**native-space T1w MRI** (acute, sub-acute, and chronic lesions) with
binary infarct masks as output — i.e. the direct successor of the
**MICCAI 2022 ATLAS R2.0** task. The contemporaneous **ISLES'22**
challenge used DWI/ADC/FLAIR; we cover it for cross-modal insights.

## Files

| File | What's in it |
|------|--------------|
| [atlas_r2_challenge.md](atlas_r2_challenge.md) | ATLAS R2.0 (T1w): dataset, metrics, top teams, winner (MAPPING) |
| [isles22_challenge.md](isles22_challenge.md) | ISLES'22 (DWI/ADC/FLAIR): adjacent lessons from DeepISLES |
| [winning_methods.md](winning_methods.md) | Deep dive on architectures, loss, post-processing, ensembling |
| [stroke_segmentation_algorithms.md](stroke_segmentation_algorithms.md) | Broader survey of existing stroke segmentation algorithms, tools, and reusable patterns |
| [problems_and_lessons.md](problems_and_lessons.md) | Failure modes, metric traps, evaluation pitfalls |
| [relevance_to_isles26.md](relevance_to_isles26.md) | Implications and first experiment ladder for ISLES'26 |

## TL;DR

- **Winners use nnU-Net**, often plus customizations. MAPPING (ATLAS'22 winner)
  is essentially `nnUNet 3d_fullres` with size-balanced 5-fold CV and probability-map
  averaging plus connected-component post-processing.
- **Ensembling diverse architectures beats a single big model** — DeepISLES
  combines nnU-Net (SEALS), MONAI Auto3DSeg (NVAUTO), and Factorizer (SWAN) by
  majority voting and ships as the post-challenge reference algorithm.
- **Small lesions are systematically missed.** Dice is dominated by big lesions
  (~100k vox) so a network that misses every <100-vox lesion barely loses Dice
  but tanks lesion-wise F1 and clinical utility.
- **Lesion-wise F1 is the metric to optimise** alongside Dice — it's the
  closest proxy for "did we find the right lesions" and ATLAS aggregates by
  mean rank across {Dice, F1, SLC, VD}.
- **Domain randomization (SynthSeg-style) is a promising route** to a
  model that generalizes beyond T1w to other contrasts, consistent with the
  ISLES'26 stretch goal of segmenting other contrasts.
- **Broader literature still points back to nnU-Net first.** Attention,
  transformer, SAM, and domain-agnostic methods are worth experiments, but
  should be compared against nnU-Net/MAPPING under identical splits.

## Sources

Every claim below is sourced from one of:

- ATLAS dataset paper: Liew et al. 2022, *Scientific Data*
  ([Nature](https://www.nature.com/articles/s41597-022-01401-7))
- ISLES'22 dataset paper: de la Rosa et al. 2022, *Scientific Data*
  ([Nature](https://www.nature.com/articles/s41597-022-01875-5))
- MAPPING (ATLAS'22 winner): Huo, Chen et al. 2022
  ([arXiv 2211.15486](https://arxiv.org/abs/2211.15486))
- DeepISLES ensemble paper: de la Rosa et al. 2025, *Nature Comm.*
  ([Nature](https://www.nature.com/articles/s41467-025-62373-x) /
  [arXiv 2403.19425](https://arxiv.org/abs/2403.19425))
- DeepISLES code: [github.com/ezequieldlrosa/DeepIsles](https://github.com/ezequieldlrosa/DeepIsles)
- Small-lesion (MSL/DBL) follow-up: Liu et al. 2024
  ([arXiv 2408.02929](https://arxiv.org/abs/2408.02929))
- ATLAS challenge: [atlas.grand-challenge.org](https://atlas.grand-challenge.org/)
- ISLES challenge: [isles22.grand-challenge.org](https://isles22.grand-challenge.org/)
- SynthSeg: Billot et al. 2023, *Med. Image Anal.*
  ([arXiv 2107.09559](https://arxiv.org/abs/2107.09559))
- Broader algorithm survey and source map:
  [stroke_segmentation_algorithms.md](stroke_segmentation_algorithms.md)
