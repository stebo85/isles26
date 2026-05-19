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

## Post-MAPPING Advances (2024–2025)

Three directions have emerged since the MICCAI'22 challenge closed. All three are ATLAS-targeted, all three have released code, and none of them has yet been compared head-to-head on the hidden generalizability set, so leaderboard claims should be read with care.

### Transformer hybrids: Neuro-TransUNet

Nouman et al. (arXiv 2406.06017) couple a SwinUNETR transformer backbone (Mix Vision Transformer encoder) with a U-Net decoder, processed 2.5D, and lean on an "ecological augmentation" scheme that inserts real lesion topologies into healthy brains during training. Reported to outperform nnU-Net and vanilla SwinUNETR on the ATLAS v2.0 training partition. The trade-off is the usual transformer one — more data hunger, less inductive bias, harder to train than nnU-Net out of the box.

### Foundation models: BrainSegFounder

Cox/Liu et al. (Med. Image Anal. 2024; arXiv 2406.10395) pretrain a 3D vision transformer with two-stage self-supervised learning on 41,400 multimodal MRI scans from the UK Biobank, then fine-tune on ATLAS v2.0 and BraTS. The thesis is that a model that already knows healthy neuroanatomy needs far fewer pathological examples to spot a lesion. Reportedly matches the MAPPING ensemble with a structurally simpler architecture and is directly relevant to the ISLES'26 cross-contrast stretch goal because the same encoder transfers to other modalities. Code: `lab-smile/BrainSegFounder`.

### Physics-constrained synthetic data: qATLAS / qSynth

Chalcroft et al. (arXiv 2412.03318, MICCAI 2025) replace ad-hoc augmentation with physics-constrained synthesis of qMRI maps — qATLAS estimates them from MPRAGE, qSynth generates them from tissue labels using MRI signal equations. Networks trained on this synthetic data learn lesion physiology rather than scanner-specific noise. Reported to beat both a real-data UNet baseline and earlier SynthSeg-style augmentation on out-of-domain datasets. Code: `liamchalcroft/qsynth`.

### Deployment: StrokeSeg

Published October 2025 (arXiv 2510.24378), StrokeSeg decouples preprocessing (Anima/BIDS), inference (ONNX Float16, ~50% size reduction), and post-processing into independent modules. On 300 held-out sub-acute/chronic stroke MRIs, the quantized ONNX model matched the PyTorch teacher (Dice difference < 10⁻³). A useful reference template once a teacher model exists.

### Cross-MRI promptable segmentation: SAMRI

Wang, Dai, Dao, Bollmann, Sun, Engstrom, Chandra (arXiv 2510.26635, October 2025) fine-tune only the SAM mask decoder on 1.1 million MRI slice-mask pairs covering 47 targets and 10+ imaging protocols, freezing the encoders to keep the surface small (96% fewer trainable params, 94% less training time vs full retraining). Reports mean DSC 0.87 ± 0.11 across all targets vs MedSAM 0.74 ± 0.24, with the headline number for our use case being a **+42.4% gain on small structures** (and +26.9% on medium structures) — directly relevant to the ATLAS small-lesion failure mode. Still prompted, so not an automatic challenge submission on its own, but a strong candidate for annotation assistance, pseudo-label generation, or as a teacher in a distillation pipeline. The author list overlaps with this project.

## Recommended Experiment Ladder

1. Preprocessing: HD-BET or deepbet brain extraction, audited on a stratified sample before any training.
2. Baseline: nnU-Net 3D full-resolution on ISLES'26 training data.
3. Validation discipline: center-, chronicity-, and lesion-size-stratified folds.
4. ATLAS replication: run CTRL/MAPPING on ATLAS v2.0 and verify published metrics if data access permits.
5. Small-lesion branch: add MSL and DBL labels to a compact nnU-Net-compatible model.
6. Metadata branch: condition normalization or sampling on `CENTER`, `CHRONICITY`, and `DAYS_POST_STROKE`.
7. Augmentation branch: qSynth physics-constrained synthesis (preferred over plain SynthSeg) for center-shift robustness.
8. Foundation-model branch: fine-tune the BrainSegFounder encoder against the nnU-Net baseline on identical folds.
9. Transformer-hybrid branch: Neuro-TransUNet on identical folds.
10. Annotation / pseudo-label branch: SAMRI as a labeling assistant and pseudo-label generator on uncertain cases, with expert review before any downstream training.
11. Compression branch: StrokeSeg-style ONNX Float16 deployment after a strong teacher exists.

## Sources

- Huo et al. 2022, MAPPING: https://arxiv.org/abs/2211.15486
- CTRL/MAPPING code: https://github.com/King-HAW/ATLAS-R2-Docker-Submission
- de la Rosa et al. 2025, DeepISLES: https://www.nature.com/articles/s41467-025-62373-x
- Shang et al. 2024, MSL/DBL: https://arxiv.org/abs/2408.02929
- Nouman et al. 2024, Neuro-TransUNet: https://arxiv.org/abs/2406.06017
- Cox/Liu et al. 2024, BrainSegFounder: https://arxiv.org/abs/2406.10395
- Chalcroft et al. 2025, qATLAS/qSynth: https://arxiv.org/abs/2412.03318
- StrokeSeg deployment framework, 2025: https://arxiv.org/abs/2510.24378
- HD-BET (Isensee et al. 2019): https://arxiv.org/abs/1901.11341
- deepbet (Fisch et al. 2024): https://arxiv.org/abs/2308.07003
- SAMRI (Wang, Dai, Dao, Bollmann, Sun, Engstrom, Chandra 2025): https://arxiv.org/abs/2510.26635
