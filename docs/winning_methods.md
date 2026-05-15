# Winning Methods And Reusable Patterns

This file extracts the algorithmic lessons from ISLES'22-ATLAS and adjacent ISLES'22 work.

## ATLAS Winner: CTRL / MAPPING

The first-place ATLAS method, MAPPING, is the most relevant reference for ISLES'26 because it was trained on T1w MRI and released code plus weights.

### Pipeline

MAPPING is built around nnU-Net:

1. Convert ATLAS images to nnU-Net task format.
2. Split the 655 public training cases into 5 folds stratified by lesion size.
3. Train four model families:
   - default nnU-Net,
   - nnU-Net with TopK10 loss,
   - residual encoder U-Net,
   - self-training variant trained with pseudo-labels generated on the public test set.
4. Average softmax probability maps across model families and folds.
5. Apply connected-component post-processing.
6. Export as a Grand Challenge Docker algorithm.

### Why It Worked

The solution worked because it aligned the engineering with the evaluation:

- Size-balanced folds reduced validation noise from the long-tailed lesion-size distribution.
- Ensembling improved lesion-wise F1 and Hausdorff distance more than Dice.
- Multiple training schemes gave useful diversity without relying on fragile architecture novelty.
- Connected-component post-processing targeted false positives and lesion-count metrics.
- Docker submission and open implementation made the approach reproducible.

### Limits

The authors explicitly note failure on very small lesions and multi-component lesions. Their qualitative failures are exactly the cases that matter clinically: subtle lesions, artifacts, poor contrast, and scattered disconnected infarcts.

## Related ISLES'22 Multimodal Winner Patterns

The DWI/ADC/FLAIR ISLES'22 task is not the same problem as ISLES'26, but the post-challenge DeepISLES paper is useful because it analyzes why top challenge submissions worked.

DeepISLES was built from the top ISLES'22 submissions and found that:

- Top approaches were diverse, including nnU-Net, MONAI Auto3DSeg-style pipelines, and Factorizer/SWAN-like approaches.
- The winning SEALS algorithm led detection-oriented metrics such as lesion-wise F1 and absolute lesion count difference.
- NVAUTO led segmentation-derived metrics such as Dice and absolute volume difference.
- Ensembling top diverse methods produced stronger all-around performance than any single challenge solution.
- Dice is biased toward large lesions, so lesion-wise metrics are needed to assess clinical detection.

The practical lesson is not to copy DWI methods directly into T1w. The lesson is to keep algorithmic diversity in the candidate pool and evaluate each model by clinical subgroup and metric family.

## Small-Lesion Follow-Up: MSL / DBL

The strongest ATLAS-specific follow-up I found is "Segmenting Small Stroke Lesions with Novel Labeling Strategies" by Shang et al. It proposes two label transformations that can be plugged into U-Net/nnU-Net-like models:

- Multi-Size Labeling (MSL): transform binary masks into classes based on connected-component volume.
- Distance-Based Labeling (DBL): separate lesion boundary and lesion interior voxels.

The method converts binary segmentation into a multi-class auxiliary representation, then collapses foreground probabilities back into a binary lesion mask. It adds very little model cost and is directly relevant to ISLES'26 because small lesions are a known ATLAS failure mode.

Reported improvements over the ATLAS winner baseline:

| Dataset | Metric | Reported Gain |
|---|---|---:|
| Mini-lesion subset | Recall | +3.6% |
| Mini-lesion subset | F1 | +2.4% |
| Mini-lesion subset | Dice | +1.3% |
| Entire dataset | Recall | +3.7% |
| Entire dataset | F1 | +1.5% |
| Entire dataset | Dice | +0.0% |

This is a good candidate for our second-line experiment after reproducing nnU-Net and MAPPING metrics.

## Recommended Experiment Ladder

1. Baseline: nnU-Net 3D full-resolution on ISLES'26 training data.
2. Validation discipline: center-, chronicity-, and lesion-size-stratified folds.
3. ATLAS replication: run CTRL/MAPPING on ATLAS v2.0 and verify published metrics if data access permits.
4. Small-lesion branch: add MSL and DBL labels to a compact nnU-Net-compatible model.
5. Metadata branch: condition normalization or sampling on `CENTER`, `CHRONICITY`, and `DAYS_POST_STROKE`.
6. Augmentation branch: SynthSeg-like intensity and contrast randomization for cross-contrast robustness.
7. Compression branch: distill ensemble into a smaller browser-capable model only after a high-quality teacher exists.

## Sources

- Huo et al. 2022, MAPPING: https://arxiv.org/abs/2211.15486
- CTRL/MAPPING code: https://github.com/King-HAW/ATLAS-R2-Docker-Submission
- de la Rosa et al. 2025, DeepISLES: https://www.nature.com/articles/s41467-025-62373-x
- Shang et al. 2024, MSL/DBL: https://arxiv.org/abs/2408.02929
