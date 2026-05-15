# ATLAS R2.0 / ISLES'22-ATLAS Challenge Review

This file reviews the T1-weighted MRI ATLAS R2.0 task that ran as Task 2 of ISLES'22. It is the closest predecessor of ISLES'26, which also asks for ischemic stroke infarct segmentation from T1w MRI, now at larger scale and with acute, sub-acute, and chronic coverage.

## Why This Challenge Matters For ISLES'26

ISLES'26 explicitly targets the generalization gaps left by ISLES'22-ATLAS: T1w infarct segmentation from native-space MRI, with metadata such as `DAYS_POST_STROKE`, `CHRONICITY`, and `CENTER`, and approximately 2,000 annotated scans from 60+ centers. The official ISLES'26 page describes it as a "multi-phase infarct segmentation" benchmark on T1w MRI with a more than 2-fold expansion in training data and about 40% more diversity than the prior attempt.

ATLAS v2.0 was a major step beyond ATLAS v1.2 because it added hidden test and generalizability sets. Liew et al. argue this was necessary because ATLAS v1.2 had no partitioned hidden test set, making overfitting and inflated validation plausible. They also note that many prior ATLAS v1.2 papers did not release code, limiting practical reuse.

## Dataset

ATLAS v2.0 contains 1,271 T1w MRIs with manually segmented stroke lesion masks from 44 cohorts across 11 countries. The challenge split was:

| Split | Size | Public? | Purpose |
|---|---:|---|---|
| Training | 655 | T1w + masks public | Development, validation, tuning |
| Public test | 300 | T1w public, masks hidden | Prediction submission leaderboard |
| Generalizability / hidden test | 316 | Fully hidden | Docker-based algorithm evaluation |

The ATLAS v2.0 paper states that training and public-test data were from similar cohort distributions, while the hidden generalizability set came from separate cohorts. That design is important for ISLES'26: do not use only random cross-validation as the final estimate. Use held-out center and held-out chronicity splits.

All ATLAS v2.0 images were distributed in BIDS-like naming, with native-space and MNI-space variants. The public challenge emphasized T1w MRI alone, unlike the contemporaneous ISLES'22 DWI/ADC/FLAIR task.

## Evaluation

ATLAS followed the ISLES-style philosophy that voxel overlap is not enough. The winner paper and challenge materials report four main metrics:

| Metric | Direction | What It Rewards | Failure It Exposes |
|---|---:|---|---|
| Dice | Higher | Voxel overlap, especially for large lesions | Boundary quality and large-lesion segmentation |
| Lesion-wise F1 | Higher | Detecting separate lesions | Missing small or scattered lesions |
| Simple Lesion Count | Lower | Matching number of connected components | Over-splitting or merging lesions |
| Volume Difference | Lower | Matching lesion burden | Over/under-segmentation |

The final ATLAS-MICCAI score used rank aggregation across public and hidden results. Huo et al. report that hidden-test ranking was weighted 4x higher than public-test ranking to reduce public-leaderboard overfitting.

## Leaderboard

The winner report gives the final ATLAS-MICCAI ranking:

| Rank | Team | ATLAS-MICCAI Score |
|---:|---|---:|
| 1 | CTRL | 3.317 |
| 2 | POBOTRI | 3.739 |
| 3 | MIRC | 4.937 |
| 4 | Yileinus | 5.053 |
| 5 | PLORAS | 5.304 |
| 6 | BIA | 6.190 |
| 7 | AICONS | 8.080 |

The public test score reported for the first-place method was Dice 0.6667, lesion-wise F1 0.5643, Simple Lesion Count 4.5367, and Volume Difference 8804.9102.

## What Worked

The dominant pattern was not a novel architecture. It was a careful nnU-Net pipeline with controlled validation, model diversity, ensembling, and metric-aware post-processing.

The CTRL/MAPPING solution used:

- nnU-Net 3D full-resolution as the base.
- 5-fold cross-validation stratified by lesion size.
- Four training schemes to create diversity:
  - default nnU-Net with Dice + cross-entropy,
  - TopK10 loss variant,
  - residual U-Net variant,
  - self-training with pseudo-labels on the public test set.
- Probability-map averaging across the four schemes.
- Post-processing based on connected components and probability thresholds.

Key numbers from the winner paper:

| Method | Public Dice | Public Lesion F1 | Public SLC | Public VD |
|---|---:|---:|---:|---:|
| Default | 0.661 | not reported | 4.670 | 8921 |
| DTK10 | 0.660 | not reported | 4.617 | 8891 |
| Res U-Net | 0.659 | 0.541 | 4.547 | 8779 |
| Self-training | 0.661 | 0.545 | 4.680 | 9119 |
| Ensemble | 0.664 | 0.555 | 4.653 | 8891 |
| Ensemble + PP | 0.667 | 0.564 | 4.537 | 8805 |

The absolute Dice gain from ensembling and post-processing was small, but the lesion-wise F1 and lesion-count metrics improved more clearly. That matters because Grand Challenge rank aggregation rewards balanced performance, not a single metric.

## What Did Not Work Well

The biggest failure mode was small lesions. The winner paper shows failure cases where:

- very small lesions were missed entirely,
- artifacts made lesions hard to distinguish,
- low contrast made lesion intensity similar to surrounding tissue,
- separated lesion regions were merged into one continuous prediction,
- gray matter was confused with lesion tissue because of similar T1w intensity.

ATLAS v2.0 has severe lesion-size imbalance. The later MSL/DBL paper reports that in the 655-image ATLAS training set, 517 lesions had volume under 100 voxels, together accounting for about 50,000 voxels, while a single large lesion can exceed 100,000 voxels. A pure Dice objective can therefore ignore many tiny lesions with little penalty.

## Practical Takeaways

For ISLES'26, the baseline should be boring and strong before becoming novel:

1. Reproduce ATLAS-style metrics exactly: Dice, lesion-wise F1, lesion count error, and volume difference.
2. Train nnU-Net as the first baseline, with 3D full-resolution if memory allows.
3. Use stratified folds by lesion volume, chronicity, and center.
4. Track separate reports for tiny, small, medium, and large connected components.
5. Add post-processing only after measuring metric tradeoffs; thresholding can improve false positives but may damage recall.
6. Treat public leaderboard results as weak evidence. Validate against held-out center/chronicity splits before trusting a model.

## Sources

- ISLES'26 Grand Challenge page: https://isles-26.grand-challenge.org/
- ATLAS R2.0 challenge page: https://atlas.grand-challenge.org/ATLAS/
- Liew et al. 2022, "A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms": https://www.nature.com/articles/s41597-022-01401-7
- Huo et al. 2022, "MAPPING: Model Average with Post-processing for Stroke Lesion Segmentation": https://arxiv.org/abs/2211.15486
- CTRL/MAPPING code: https://github.com/King-HAW/ATLAS-R2-Docker-Submission
- Shang et al. 2024, "Segmenting Small Stroke Lesions with Novel Labeling Strategies": https://arxiv.org/abs/2408.02929
