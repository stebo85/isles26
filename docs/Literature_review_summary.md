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
  ISLES'26 stretch goal of segmenting other contrasts. The 2025 successor
  **qSynth** (physics-constrained qMRI synthesis) is now the stronger
  default — it beats plain SynthSeg on out-of-domain stroke data.
- **Broader literature still points back to nnU-Net first.** Attention,
  transformer, SAM, and domain-agnostic methods are worth experiments, but
  should be compared against nnU-Net/MAPPING under identical splits.
- **Preprocessing is not a utility script.** HD-BET (DKFZ, 2019) and
  deepbet (Münster, 2024) are the modern deep-learning brain-extraction
  baselines and are robust to stroke pathology where FSL BET and ROBEX
  fail. Run one of them first and audit the masks before training.
- **Post-MAPPING ATLAS work (2024–2025) is worth tracking.**
  Neuro-TransUNet (SwinUNETR + U-Net hybrid), BrainSegFounder
  (foundation model pretrained on UK Biobank), and qATLAS/qSynth
  all target weaknesses of the CTRL/MAPPING baseline — long-range context,
  data efficiency, and scanner-shift robustness respectively.
- **StrokeSeg (2025) is a worked example of clinical deployment** —
  decoupled BIDS preprocessing, ONNX Float16 inference, ~equivalent
  Dice to the PyTorch teacher. Useful template once a teacher exists.
- **SAMRI (Wang et al. 2025, in-house authorship)** is the strongest
  current SAM-family model for MRI: trained on 1.1M slices across 47
  targets, +42.4% DSC on small structures vs MedSAM. Not a stand-alone
  automatic submission (it is prompted), but a strong candidate for
  annotation assistance, pseudo-label generation, and as a distillation
  teacher targeting the ATLAS small-lesion failure mode.

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
- HD-BET: Isensee et al. 2019, *Human Brain Mapping*
  ([arXiv 1901.11341](https://arxiv.org/abs/1901.11341))
- deepbet: Fisch et al. 2024, *Comput. Biol. Med.*
  ([arXiv 2308.07003](https://arxiv.org/abs/2308.07003))
- Neuro-TransUNet: Nouman et al. 2024
  ([arXiv 2406.06017](https://arxiv.org/abs/2406.06017))
- BrainSegFounder: Cox/Liu et al. 2024, *Med. Image Anal.*
  ([arXiv 2406.10395](https://arxiv.org/abs/2406.10395))
- qATLAS / qSynth: Chalcroft et al., MICCAI 2025
  ([arXiv 2412.03318](https://arxiv.org/abs/2412.03318))
- StrokeSeg deployment framework, 2025
  ([arXiv 2510.24378](https://arxiv.org/abs/2510.24378))
- SAMRI: Wang, Dai, Dao, Bollmann, Sun, Engstrom, Chandra 2025
  ([arXiv 2510.26635](https://arxiv.org/abs/2510.26635))
- Broader algorithm survey and source map:
  [stroke_segmentation_algorithms.md](stroke_segmentation_algorithms.md)
