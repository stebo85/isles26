# Existing Stroke Segmentation Algorithms

This document surveys stroke lesion segmentation algorithms and tools relevant to ISLES'26 preparation. The emphasis is on methods that could inform a T1w MRI infarct segmentation entry, but the broader stroke literature is included where it gives reusable lessons about architecture, preprocessing, generalization, or deployment.

## Executive Summary

The field has converged on a practical recipe:

1. Use a U-Net family model, usually nnU-Net or a close derivative, as the first serious baseline.
2. Spend as much care on preprocessing, validation splits, and post-processing as on architecture.
3. Evaluate lesion detection separately from voxel overlap; small lesions are the repeated failure case.
4. Use ensembling for challenge performance, then distill or compress for deployment.
5. Treat transformers, SAM-like foundation models, and domain-agnostic models as experiment branches, not replacements for nnU-Net until they beat it under identical splits.

For ISLES'26 specifically, the highest-priority candidates are:

- nnU-Net / residual nnU-Net baseline,
- MAPPING-style model averaging plus component post-processing,
- small-lesion label transformations such as MSL/DBL,
- attention modules aimed at cross-scale lesion representation,
- strong preprocessing and intensity normalization,
- SynthSeg-style contrast randomization for cross-contrast robustness,
- teacher-student distillation for browser deployment.

## Algorithm Families

| Family | Typical Inputs | Strengths | Weaknesses | ISLES'26 Relevance |
|---|---|---|---|---|
| Classical / generative / random forest | Multispectral MRI, hand-crafted features | Interpretable, useful historical baselines | Usually outperformed by CNNs; heavy feature engineering | Low, except as sanity baselines |
| 2D U-Net | Slices from T1, DWI, ADC, FLAIR, CT | Simple, fast, browser-friendly | Loses 3D context; slice inconsistency | Good for compact deployment experiments |
| 2.5D U-Net / multi-plane CNN | Orthogonal slices or neighboring slices | More context than 2D with lower memory than 3D | Engineering complexity; still not fully volumetric | Strong candidate for small/browser model |
| 3D U-Net / DeepMedic | Volumetric MRI or CT patches | Captures 3D lesion shape and context | Memory hungry; patch sampling matters | Useful baseline if nnU-Net unavailable |
| nnU-Net | Any sufficiently standardized 3D segmentation task | Self-configuring, hard to beat, reproducible | Not tiny; may need dataset-specific preprocessing | Required first baseline |
| Residual nnU-Net / Auto3DSeg | MRI or CT | Strong challenge performance | Heavier models; more moving parts | Good second baseline |
| Attention U-Net / cross-scale attention | T1w or multimodal MRI | Better small/variable lesion representation | Gains can be dataset-specific | Good branch after baseline |
| Transformers / ViT hybrids | DWI or T1w MRI | Long-range context, potentially robust | Data-hungry; often heavy | Research branch, not first baseline |
| SAM / MedSAM / Brain-SAM | Prompted 2D/3D MRI segmentation | Interactive workflows, pretraining | Not naturally automatic binary lesion segmentation; prompt burden | Explore for annotation or weak supervision |
| Domain-agnostic / synthetic-contrast models | Multi-contrast MRI | Robust to modality/contrast shifts | Less mature for stroke | Directly aligned with cross-contrast stretch goal |

## Challenge-Era Methods

### ISLES 2015

ISLES 2015 benchmarked sub-acute ischemic stroke lesion segmentation and acute stroke outcome/penumbra estimation from multispectral MRI. The Medical Image Analysis benchmark paper and challenge materials describe a field that still mixed classical machine learning, probabilistic modeling, and early CNNs.

Important lessons:

- Challenge organizers already identified lesion-size and morphology variability as core difficulties.
- Random forest and feature-based methods were competitive historically, but CNNs quickly became the dominant direction.
- Multispectral inputs were useful because stroke appearance changes across DWI, FLAIR, T1, T2, and perfusion maps.

Reusable point for ISLES'26: even when the final input is single-channel T1w, the model should be stress-tested against size, morphology, and contrast variation.

### ISLES 2016 / 2017

ISLES 2016 and 2017 focused on lesion outcome prediction from acute MRI, including DWI, ADC, and perfusion maps, with follow-up lesions as targets. The 2017 page reports that the training data contained 43 patients and the test data 32 cases. Winning and high-ranking approaches included CNN ensembles and voxel-wise methods.

Important lessons:

- Outcome prediction is harder than visible lesion segmentation because the target can represent future tissue fate.
- Small datasets made overfitting and validation instability severe.
- Ensembles of 3D CNNs appeared early because single models were brittle.

Reusable point for ISLES'26: hidden generalization should be simulated locally. Random folds alone are not enough.

### ISLES 2018

ISLES 2018 moved to acute CT perfusion. The official challenge page reports 63 training patients and 40 test stroke cases, with CBF, MTT, CBV, Tmax, CTP source data, and DWI-derived ground truth. Finalists included Tao Song, Liu Pengbo, Yu Chen, Xiaojun Hu, and Albert Clerigues Garcia.

Important lessons:

- CT/CTP segmentation can depend strongly on intensity windowing, perfusion map calculation, and preprocessing.
- The imaging modality may not fully show the final infarct target, making the task partly prognostic.

Reusable point for ISLES'26: if T1w signal is weak for acute/sub-acute lesions, metadata and robust priors may matter, but they must not replace direct segmentation evidence.

### ISLES 2022 / DeepISLES

DeepISLES combined top ISLES'22 approaches:

- SEALS: nnU-Net-based.
- NVAUTO: MONAI Auto3DSeg-based.
- SWAN: Factorizer-based.

DeepISLES uses majority voting and supports DWI, ADC, and FLAIR inputs. The released repository states that it can run in native image space and can save individual team outputs. The DeepISLES paper reports that the ensemble was motivated by method diversity and validated beyond the original challenge setting.

Reusable point for ISLES'26: keep diverse high-performing models around until ensemble complementarity is measured. A compact browser model should be distilled after the ensemble teacher exists.

### ISLES 2024

ISLES'24 targeted final infarct prediction from pre-treatment acute CT/CTA/CTP plus clinical data. The winning write-up is unusually direct: the title says they won "by preprocessing." Their solution combined deep-learning skull stripping, custom CT intensity windowing, and a large residual nnU-Net, reporting mean test Dice 28.5 with standard deviation 21.27.

Reusable point for ISLES'26: preprocessing can dominate architecture. For T1w, this argues for careful brain extraction, bias-field correction or augmentation, intensity normalization, and consistent orientation/spacing before architecture experiments.

## T1w / ATLAS-Relevant Algorithms

### MAPPING / CTRL

MAPPING is the strongest directly relevant released ATLAS method. It uses nnU-Net 3D full-resolution, lesion-size-balanced five-fold validation, multiple training variants, probability averaging, and connected-component post-processing.

Why it matters:

- It won the ATLAS-MICCAI ranking.
- It is close to a reproducible baseline, with code released.
- Its limitations are clear: tiny lesions, artifacts, low contrast, and component merging.

Recommended use:

- Reproduce it on ATLAS v2.0 if data access permits.
- Treat its metrics as the first target to beat.

### Multi-Path 2.5D CNN

Pustina et al. proposed a multi-path 2.5D CNN for stroke lesion segmentation from MRI. The system used nine 2D U-Nets across three anatomical planes and three normalizations, then concatenated outputs into a 3D volume processed by a 3D CNN. It was trained and tested across Medical College of Wisconsin, Kessler Foundation, and ATLAS data.

Why it matters:

- It addresses one weakness of 2D models without the full cost of 3D training.
- It was designed around T1/FLAIR stroke survivor MRI, closer to chronic and subacute lesion mapping than acute DWI-only work.
- This is a useful design pattern for browser deployment: run lightweight 2D/2.5D inference and fuse context.

Limitations:

- More complex inference pipeline.
- Not as standardized as nnU-Net.
- Reported performance depends heavily on preprocessing, normalization, and dataset scope.

### 3D Multi-Path CNN / DeepMedic Comparisons

Several ATLAS-era papers compare 3D CNNs such as DeepMedic, 3D U-Net, AnatomyNet-like models, and multi-path feature-fusion CNNs. A 2019 arXiv paper reports a fully 3D multi-path network with feature fusion and weighting, claiming statistically significant gains over DeepMedic, 3D U-Net, and AnatomyNet on an ATLAS benchmark subset.

Why it matters:

- Multi-path fusion is a recurring response to lesions of variable size and location.
- These methods are useful conceptually, but nnU-Net has largely absorbed the practical advantage of hand-designed 3D U-Net variants.

Recommended use:

- Do not start here.
- Borrow ideas only if nnU-Net failure analysis shows missing global context or weak multi-scale representation.

### MSL / DBL Small-Lesion Labeling

Shang et al. proposed Multi-Size Labeling and Distance-Based Labeling for ATLAS v2.0. The core idea is to convert binary labels into richer multi-class supervision:

- MSL separates lesions by connected-component size.
- DBL separates lesion interiors and boundaries.

Why it matters:

- It directly targets the main ATLAS failure mode: small lesions contribute little voxel mass but dominate detection failures.
- It can be bolted onto U-Net-like architectures with limited cost.

Recommended use:

- High-priority second experiment after nnU-Net baseline.
- Report lesion-wise recall by lesion-size bin to determine whether it actually helps ISLES'26.

### Multi-Stage Cross-Scale Attention

The 2025 MSCSA paper applies a multi-stage cross-scale attention mechanism to U-Net family models on ATLAS v2.0. The abstract reports stronger Dice and F1 on a small-lesion subset while remaining competitive on the full dataset, with best full-dataset and small-lesion results from an ensemble including MSCSA.

Why it matters:

- It is a targeted attempt to solve scale variation, not generic attention decoration.
- It aligns with the small-lesion weakness identified by ATLAS and DeepISLES.

Recommended use:

- Treat as an experiment branch after reproducing baseline metrics.
- Compare against MSL/DBL and oversampling under the same folds.

### nnU-Net External Validation Across Acute And Chronic MRI

A 2026 arXiv preprint systematically evaluated nnU-Net across DWI, FLAIR, and T1w stroke MRI datasets. The abstract reports robust generalization across stroke stages, better acute performance from DWI than FLAIR, limited gains from multimodal combinations in acute stroke, improved chronic performance with larger training sets, and strong lesion-volume dependence.

Why it matters:

- It supports using nnU-Net as the common yardstick across modalities and stages.
- It reinforces that lesion volume and image quality drive performance.
- It suggests more data helps chronic T1w segmentation, but with diminishing returns after several hundred cases.

Recommended use:

- Use the paper's open code/models if accessible.
- Mirror its external-validation mindset locally with center/chronicity splits.

## Acute DWI / ADC / FLAIR Algorithms

### EDD Net + MUSCLE Net

Chen, Bentley, and Rueckert proposed a two-CNN framework for acute DWI lesion segmentation:

- EDD Net: ensemble of DeconvNets to detect lesions.
- MUSCLE Net: multi-scale label evaluation network to remove false positives.

They validated on 741 DWI subjects and reported mean Dice 0.67 overall, Dice 0.61 for small-lesion subjects, Dice 0.83 for large-lesion subjects, and lesion detection rate 0.94.

Why it matters:

- This is an early example of a detector plus false-positive-reduction cascade.
- It documents the same size effect later seen in ATLAS.

Recommended use:

- Consider a second-stage false-positive reducer only after measuring lesion-wise precision/recall.

### LambdaUNet

LambdaUNet is a 2.5D architecture for DWI stroke segmentation. It was motivated by discontinuous slice structure and compared against U-Net, Attention U-Net, and TransUNet-style baselines.

Why it matters:

- It is relevant to compact 2.5D inference.
- Less directly relevant to T1w than ATLAS methods because DWI has much clearer acute lesion contrast.

### ISLA

ISLA, a 2026 arXiv preprint, describes a U-Net-based acute ischemic stroke segmentation model with deep supervision, attention, domain adaptation, and ensembling, trained on more than 1,500 diffusion MRI participants from three multicenter databases.

Why it matters:

- It combines several patterns that recur in winning solutions: U-Net base, supervision tricks, domain adaptation, and ensemble learning.
- The input domain is DWI, so it should inform engineering style more than direct transfer.

## CT / CTP Algorithms

CT stroke segmentation is not the ISLES'26 target, but it is useful for two reasons: preprocessing lessons and prognostic-target lessons.

### ISLES'24 Winner

The ISLES'24 winning write-up argues that standard nnU-Net preprocessing for CT was insufficient. The winning pipeline used:

- SynthStrip-style deep-learning skull stripping,
- custom CT intensity windowing,
- large residual nnU-Net.

Lesson for ISLES'26:

- Before architecture tuning, inspect intensity distributions and scanner/site effects.
- T1w preprocessing should be treated as a model component, not a utility script.

### Adversarial Learning And CT Transformer Papers

Several CT/CTP papers add adversarial learning, residual deformable transformers, or convolutional attention mechanisms. These methods are interesting but less transferable because CT/CTP intensity physics and target visibility differ strongly from T1w MRI.

Recommended use:

- Track them only for ideas about domain adaptation, not as first-line model choices.

## Foundation Models And SAM-Like Methods

SAM, MedSAM, SAM-Med3D, Brain-SAM, and similar models are attractive because they offer promptable segmentation and large-scale pretraining. However, stroke lesion segmentation has properties that make direct SAM use risky:

- lesions are small, subtle, and often low contrast,
- T1w lesions may not have crisp visual boundaries,
- automatic challenge submission cannot rely on human prompts,
- 2D promptable masks can be inconsistent across slices,
- generic medical SAMs may be trained across organs whose structure differs sharply from brain MRI.

Evidence to track:

- A 2023 SAM brain tumor paper reported strong prompted glioma segmentation in MRI, but performance depended on prompt selection and was weak on peripheral slices with few tumor voxels.
- A 2026 Brain-SAM preprint argues that brain MRI requires specialized adaptation because mixed-organ SAM training does not capture fine-grained brain lesion structure well.

Recommended use for ISLES'26:

- Use SAM-like models for interactive annotation support or pseudo-label review first.
- Do not rely on zero-shot SAM as an automatic challenge model.
- If using Shakes SAM or another SAM base, frame it as a fine-tuning/distillation branch and compare to nnU-Net under identical metrics.

## Open-Source Tools And Repositories

| Tool / Repo | Domain | What It Offers | Use For ISLES'26 |
|---|---|---|---|
| `King-HAW/ATLAS-R2-Docker-Submission` | ATLAS T1w | MAPPING winner code and Docker submission | Reproduce ATLAS baseline |
| `ezequieldlrosa/DeepIsles` | ISLES'22 DWI/ADC/FLAIR | Ensemble of SEALS, NVAUTO, SWAN | Study ensemble design; not direct T1w model |
| `Tabrisrei/ISLES22_Ensemble` | ISLES'22 DWI/ADC/FLAIR | Earlier DeepISLES-style ensemble repo | Alternative reference implementation |
| `LounesMD/MMStrokeNet` | ATLAS T1w + private T1/FLAIR | nnU-Net baseline and T1/FLAIR fine-tuning scripts | Inspect adaptation from single to dual modality |
| `ezequieldlrosa/isles22` | ISLES'22 dataset utilities | Data loading and metric starter code | Useful evaluation patterns |

## What Looks Most Promising For Our Plan

### First-Line

- nnU-Net v2 3D full-resolution.
- Residual nnU-Net if memory allows.
- MAPPING reproduction.
- Strict evaluation with lesion-wise F1, lesion-size bins, center splits, and chronicity splits.

### Second-Line

- MSL/DBL label transforms.
- Multi-stage cross-scale attention.
- Lesion-aware patch sampling.
- MAPPING-style probability ensembling.
- Component post-processing with lesion-size-specific audits.

### Robustness Branch

- SynthSeg-style intensity and contrast randomization.
- Bias-field, resolution, orientation, and artifact augmentation.
- Center-stratified normalization experiments.
- Metadata-aware sampling or conditioning.

### Deployment Branch

- 2.5D U-Net or multi-plane fusion model.
- Distillation from the best 3D ensemble.
- Quantization-aware or ONNX/WebGPU path only after accuracy is competitive.

## Negative Findings

These are ideas that should not distract the first phase:

- Pure architecture novelty without a clean nnU-Net comparison.
- Dice-only model selection.
- Generic SAM zero-shot segmentation as the main challenge entry.
- CT/CTP methods transferred directly to T1w MRI.
- Post-processing that deletes small components before lesion-wise recall is measured.
- Public leaderboard tuning without center/chronicity holdout confirmation.

## Source Map

- ISLES 2015 challenge: https://www.isles-challenge.org/ISLES2015/
- ISLES 2015 benchmark paper: https://pubmed.ncbi.nlm.nih.gov/27475911/
- ISLES 2017 challenge: https://www.isles-challenge.org/ISLES2017/
- ISLES 2018 challenge: https://www.isles-challenge.org/ISLES2018/
- ISLES 2024 challenge: https://isles-24.grand-challenge.org/
- ISLES'24 winner write-up: https://arxiv.org/abs/2505.18424
- DeepISLES paper: https://www.nature.com/articles/s41467-025-62373-x
- DeepISLES code: https://github.com/ezequieldlrosa/DeepIsles
- ISLES22 ensemble code: https://github.com/Tabrisrei/ISLES22_Ensemble
- MAPPING winner report: https://arxiv.org/abs/2211.15486
- MAPPING code: https://github.com/King-HAW/ATLAS-R2-Docker-Submission
- ATLAS v2.0 paper: https://www.nature.com/articles/s41597-022-01401-7
- Multi-path 2.5D CNN: https://arxiv.org/abs/1905.10835
- EDD Net + MUSCLE Net: https://pmc.ncbi.nlm.nih.gov/articles/PMC5480013/
- MSL/DBL small-lesion paper: https://arxiv.org/abs/2408.02929
- MSCSA cross-scale attention: https://arxiv.org/abs/2501.15423
- nnU-Net external validation preprint: https://arxiv.org/abs/2601.08701
- SAM brain MRI tumor paper: https://arxiv.org/abs/2304.07875
- Brain-SAM preprint: https://www.medrxiv.org/content/10.64898/2026.01.30.26345164v1.full-text
- 2024 deep learning stroke segmentation survey: https://doi.org/10.1016/j.compbiomed.2024.108509
- Stroke lesion segmentation deep learning review: https://www.mdpi.com/2306-5354/11/1/86
- MMStrokeNet code: https://github.com/LounesMD/MMStrokeNet
