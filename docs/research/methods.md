---
name: Stroke segmentation methods
tags: [research, methods, literature]
description: Canonical literature survey of relevant preprocessing, model, robustness, and deployment methods.
---

# Stroke Segmentation Methods

This is the canonical literature survey for the project. It records external
evidence and experiment options; [the model status](../models/status.md) owns
current project decisions and results.

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

## Preprocessing: Skull Stripping On Pathological Brains

Brain extraction is the dominant upstream failure mode for chronic-T1w stroke pipelines. Traditional gradient-based methods (FSL BET, ROBEX) were designed on healthy cohorts and frequently leak into the skull or carve out cortical infarcts where the lesion alters the expected GM/CSF boundary. The two deep-learning replacements currently considered state of the art are HD-BET and deepbet.

### HD-BET

Isensee et al. (Human Brain Mapping, 2019) trained a multi-sequence brain extraction network on a multicentric clinical trial including patients with tumors and post-surgical cavities, explicitly so that the model would not collapse on pathological anatomy. Reported median Dice improvements of +1.16 to +2.50 points over six public algorithms (including FSL BET and ROBEX), with parallel improvements on Hausdorff distance. Supports pre/post-contrast T1w, T2w, and FLAIR. Two community forks are actively maintained: `MIC-DKFZ/HD-BET` and `BrainLesion/HD-BET`.

### deepbet

Fisch et al. (Computers in Biology and Medicine, 2024) replaced the standard U-Net with a two-stage LinkNet pipeline trained on 7,837 T1w images from 191 OpenNeuro datasets. Reports median DSC 99.0% on held-out validation (vs ~97.8–97.9% for prior SOTA), worst-case >96.9% on pathological samples, and ~2 s/volume inference on commodity hardware. Code: `wwu-mmll/deepbet`.

### Why This Matters For ISLES'26

ISLES'26 spans acute, sub-acute, and chronic T1w lesions across 60+ centers. A non-trivial fraction of failure cases at the model level will trace back to brain-extraction errors that crop or merge cortical lesions. The recommendation is to run HD-BET or deepbet as the first preprocessing step and audit the brain masks on a stratified sample (chronicity × center × lesion size) before training any segmentation model.

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

### Neuro-TransUNet

Nouman, Mabrok, and Rashed (arXiv 2406.06017, June 2024) propose a hybrid architecture for ATLAS v2.0 that integrates a SwinUNETR transformer backbone (with a Mix Vision Transformer encoder) into a U-Net framework, processed in a 2.5D fashion to keep memory tractable. The pipeline depends heavily on careful preprocessing (resampling, bias-field correction, skull stripping) and on an "ecological data augmentation" scheme that inserts real lesion topologies into healthy brains. The authors report outperforming nnU-Net and a vanilla SwinUNETR on the ATLAS v2.0 training partition, though the numbers should be read with the usual caveat that they are not from the hidden Grand Challenge set.

Why it matters:

- It is the strongest published transformer-hybrid result on ATLAS T1w.
- The 2.5D + ecological-augmentation pattern is a viable middle ground between full 3D nnU-Net and 2D browser models.

Recommended use:

- Treat as a research branch alongside nnU-Net + MSL/DBL.
- Compare on identical folds and report hidden-test performance if a submission slot is available.

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

## Foundation Models And Physics-Constrained Synthetic Data

### BrainSegFounder

Cox, Liu et al. (Medical Image Analysis, 2024; arXiv 2406.10395) build a domain-specific 3D foundation model for neuroimaging instead of starting from ImageNet weights. A vision-transformer encoder is pretrained in two stages on 41,400 multimodal MRI scans from the UK Biobank (mostly healthy), then fine-tuned downstream on ATLAS v2.0 and BraTS. The argument is that a model that has already learned healthy neuroanatomy needs far fewer pathological examples to recognize a stroke void. Code: `lab-smile/BrainSegFounder`.

Why it matters:

- It is the first credible "brain MRI foundation model" with ATLAS evaluation.
- It is structurally simpler than the CTRL/MAPPING ensemble while reportedly competitive with it.
- It directly relevant to ISLES'26's cross-contrast stretch goal because the same pretrained encoder can be fine-tuned for other modalities.

Recommended use:

- Treat as a research branch — clone the released weights and fine-tune on ISLES'26 once a strong nnU-Net baseline is established.
- Use only with a like-for-like comparison against nnU-Net on identical folds.

### qATLAS / qSynth (Physics-Constrained Synthetic Data)

Chalcroft et al. (arXiv 2412.03318, MICCAI 2025) introduce two physics-constrained synthetic-data approaches to immunize stroke segmentation models against scanner shift. qATLAS trains a network to estimate quantitative MRI (qMRI) maps from standard MPRAGE images; qSynth synthesizes qMRI maps directly from tissue labels using label-conditioned Gaussian mixture models combined with MRI signal equations (T1 relaxation, proton density, sequence timing). Both can simulate arbitrary acquisition parameters at training time, forcing the segmentation network to learn lesion physiology rather than scanner-specific noise. The paper reports that both methods outperform a baseline UNet on out-of-domain datasets, with qSynth notably surpassing prior synthetic-data approaches (including SynthSeg-style augmentation). Code: `liamchalcroft/qsynth`.

Why it matters:

- Directly addresses the ISLES'26 60+-center generalization problem.
- A successor to SynthSeg-style randomization, not a replacement for nnU-Net.
- The synthetic data is independent of the segmentation backbone, so it stacks with nnU-Net or any of the architectures above.

Recommended use:

- High-priority robustness branch. Train an nnU-Net baseline twice — once with real data only, once with qSynth-augmented data — and compare on the held-out center splits.

### SAMRI

Wang, Dai, Dao, Bollmann, Sun, Engstrom, and Chandra (arXiv 2510.26635, October 2025) adapt SAM specifically to MRI by fine-tuning only the mask decoder while freezing both encoders. The reduced surface — 96% fewer trainable parameters, 94% less training time vs full retraining — lets them train on a large-scale MRI corpus of 1.1 million 2D slice-mask pairs from 30 datasets across 47 segmentation targets, spanning T1/T2/FLAIR/DWI and whole-body anatomy. Reported mean DSC 0.87 ± 0.11 across all 47 targets vs MedSAM 0.74 ± 0.24, with the largest relative gains on **small structures (+42.4%)** and **medium structures (+26.9%)**. Zero-shot DSC on unseen datasets is reported at 0.85.

Why it matters for ISLES'26:

- The +42.4% small-structure gain is the most relevant feature. Small lesions are the dominant ATLAS failure mode, and SAMRI is one of the few recent SAM-family models that explicitly targets that regime.
- The cross-contrast training (T1/T2/FLAIR/DWI in a single model) aligns directly with the ISLES'26 cross-contrast stretch goal.
- It is still a prompted model, so it cannot be the primary automatic challenge submission without a prompt-generation stage. The natural deployments are: (a) interactive annotation / label review, (b) pseudo-label generation to seed self-training of an automatic model, (c) teacher in a distillation pipeline where prompts are derived from a coarse automatic detector.

Recommended use:

- Treat as the SAM-family baseline of choice (replacing MedSAM/Brain-SAM in earlier docs).
- Evaluate first as a labeling assistant on a held-out subset of ISLES'26.
- If used for pseudo-labels, require expert review before training a downstream automatic model.

### Other SAM-Like Methods

MedSAM, SAM-Med3D, Brain-SAM, and similar models are attractive because they offer promptable segmentation and large-scale pretraining. However, stroke lesion segmentation has properties that make direct SAM use risky:

- lesions are small, subtle, and often low contrast,
- T1w lesions may not have crisp visual boundaries,
- automatic challenge submission cannot rely on human prompts,
- 2D promptable masks can be inconsistent across slices,
- generic medical SAMs may be trained across organs whose structure differs sharply from brain MRI.

Evidence to track:

- A 2023 SAM brain tumor paper reported strong prompted glioma segmentation in MRI, but performance depended on prompt selection and was weak on peripheral slices with few tumor voxels.
- A 2026 Brain-SAM preprint argues that brain MRI requires specialized adaptation because mixed-organ SAM training does not capture fine-grained brain lesion structure well.

Recommended use for ISLES'26:

- Use SAM-like models (SAMRI preferred) for interactive annotation support or pseudo-label review first.
- Do not rely on zero-shot SAM as an automatic challenge model.
- If using SAMRI or another SAM base, frame it as a fine-tuning/distillation branch and compare to nnU-Net under identical metrics.

## Clinical Deployment Frameworks

### StrokeSeg

Published October 2025 (arXiv 2510.24378), StrokeSeg is a deployment-oriented restructuring of a research nnU-Net stroke pipeline. The architecture decouples preprocessing (Anima toolbox, BIDS-compliant), inference (ONNX Runtime with Float16 quantization, ~50% size reduction), and post-processing into independent modules. On a held-out set of 300 sub-acute and chronic stroke MRIs, the quantized ONNX model matched the PyTorch teacher with a Dice difference below 10⁻³. Distributed as Python scripts and as a standalone Windows executable with both GUI and CLI front-ends.

Why it matters:

- It is a concrete reference for ISLES'26's deployment branch — i.e. a worked example of taking a research-grade nnU-Net to a clinically packageable tool.
- The decoupled-module pattern (BIDS preprocessing → ONNX inference → post-processing) is directly transferable.

Recommended use:

- Use as the deployment template only after a strong teacher is trained.
- Do not start here — start with nnU-Net accuracy first.

## Open-Source Tools And Repositories

| Tool / Repo | Domain | What It Offers | Use For ISLES'26 |
|---|---|---|---|
| `MIC-DKFZ/HD-BET` (and `BrainLesion/HD-BET` fork) | T1/T2/FLAIR brain extraction | DL skull stripping robust to pathology | First preprocessing step |
| `wwu-mmll/deepbet` | T1w brain extraction | LinkNet-based, 99.0% DSC, ~2 s/volume | Alternative preprocessing, fast |
| `King-HAW/ATLAS-R2-Docker-Submission` | ATLAS T1w | MAPPING winner code and Docker submission | Reproduce ATLAS baseline |
| `ezequieldlrosa/DeepIsles` | ISLES'22 DWI/ADC/FLAIR | Ensemble of SEALS, NVAUTO, SWAN | Study ensemble design; not direct T1w model |
| `Tabrisrei/ISLES22_Ensemble` | ISLES'22 DWI/ADC/FLAIR | Earlier DeepISLES-style ensemble repo | Alternative reference implementation |
| `LounesMD/MMStrokeNet` | ATLAS T1w + private T1/FLAIR | nnU-Net baseline and T1/FLAIR fine-tuning scripts | Inspect adaptation from single to dual modality |
| `ezequieldlrosa/isles22` | ISLES'22 dataset utilities | Data loading and metric starter code | Useful evaluation patterns |
| `lab-smile/BrainSegFounder` | UKB pretraining → ATLAS/BraTS | 3D foundation model with two-stage SSL pretraining | Fine-tuning branch, cross-modal stretch goal |
| `liamchalcroft/qsynth` | Physics-constrained MRI synthesis | qATLAS and qSynth augmentation pipelines | Robustness branch for center shift |
| SAMRI (Wang et al. 2025, arXiv 2510.26635) | Cross-MRI promptable segmentation | SAM mask-decoder fine-tune trained on 1.1M slices, 47 targets, +42.4% small-structure DSC vs MedSAM | Annotation assist, pseudo-label generation, distillation seed |

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

- qSynth-style physics-constrained synthetic data (preferred over plain SynthSeg).
- Bias-field, resolution, orientation, and artifact augmentation.
- Center-stratified normalization experiments.
- Metadata-aware sampling or conditioning.
- BrainSegFounder pretrained encoder as an alternative initialization.

### Deployment Branch

- 2.5D U-Net or multi-plane fusion model.
- Distillation from the best 3D ensemble.
- StrokeSeg-style decoupled pipeline: Anima/BIDS preprocessing → ONNX Float16 inference → post-processing.
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
- HD-BET (Isensee et al. 2019, Human Brain Mapping): https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.24750 (arXiv: https://arxiv.org/abs/1901.11341)
- deepbet (Fisch et al. 2024, Comput. Biol. Med.): https://www.sciencedirect.com/science/article/abs/pii/S0010482524009302 (arXiv: https://arxiv.org/abs/2308.07003)
- Neuro-TransUNet (Nouman et al. 2024): https://arxiv.org/abs/2406.06017
- BrainSegFounder (Cox/Liu et al. 2024, Med. Image Anal.): https://www.sciencedirect.com/science/article/abs/pii/S1361841524002263 (arXiv: https://arxiv.org/abs/2406.10395)
- qATLAS / qSynth (Chalcroft et al., MICCAI 2025): https://arxiv.org/abs/2412.03318 (code: https://github.com/liamchalcroft/qsynth)
- StrokeSeg deployment framework: https://arxiv.org/abs/2510.24378
- SAMRI (Wang, Dai, Dao, Bollmann, Sun, Engstrom, Chandra 2025): https://arxiv.org/abs/2510.26635
- SAM brain MRI tumor paper: https://arxiv.org/abs/2304.07875
- Brain-SAM preprint: https://www.medrxiv.org/content/10.64898/2026.01.30.26345164v1.full-text
- 2024 deep learning stroke segmentation survey: https://doi.org/10.1016/j.compbiomed.2024.108509
- Stroke lesion segmentation deep learning review: https://www.mdpi.com/2306-5354/11/1/86
- MMStrokeNet code: https://github.com/LounesMD/MMStrokeNet

Related fibers: [[research/challenges/atlas-r2]],
[[research/challenges/isles22]], [[models/status]], [[lessons/problems]].
