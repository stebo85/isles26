---
name: ISLES'22 multimodal challenge context
tags: [research, challenge, isles22]
description: Adjacent DWI/ADC/FLAIR evidence about generalization and ensemble construction.
---

# ISLES'22 Multimodal Challenge Context

This file covers the ISLES'22 DWI/ADC/FLAIR task only as adjacent context. ISLES'26 is closer to ATLAS R2.0 because it uses T1w MRI, but the ISLES'22 task gives useful lessons about challenge design, generalization, and ensemble construction.

## Task

ISLES'22 Task 1 asked participants to segment acute and sub-acute ischemic stroke lesions from multimodal MRI:

- DWI,
- ADC,
- FLAIR.

The dataset paper describes 400 multi-vendor MRI cases with high variability in lesion size, number, and location, split into 250 training cases and 150 hidden test cases. The dataset was intentionally heterogeneous across centers and scanners.

## Why It Is Still Useful

ISLES'22 is less relevant architecturally because DWI/ADC/FLAIR lesion contrast is different from T1w lesion contrast. It is still useful because:

- it used similar evaluation logic,
- it tested hidden center generalization,
- it exposed the same small-lesion and scattered-infarct problems,
- the post-challenge DeepISLES work shows how to combine top challenge submissions into a more robust model.

## Key Findings

The ISLES'22 dataset paper says the challenge was designed to evaluate both large-lesion boundary segmentation and detection of small punctiform infarcts. It argues that Dice alone is insufficient because small separated embolic infarcts barely move Dice when a large lesion is present.

DeepISLES reports:

- 476 registered participants,
- 325 dataset downloads,
- 20 teams validated Docker algorithms in the sanity-check phase,
- 15 final deep-learning submissions,
- 12 submissions met the participation criteria.

Top methods differed substantially:

- SEALS ranked first and led detection-oriented metrics such as lesion-wise F1 and absolute lesion count difference.
- NVAUTO ranked second and led segmentation-derived metrics such as Dice and absolute volume difference.
- SWAN was favored over PAT in post-challenge ranking analysis for third place.

DeepISLES then ensembled the top-ranked approaches and outperformed individual challenge submissions across downstream analyses.

## Generalization Lessons

DeepISLES analyzed performance across:

- seen and unseen centers,
- lesion sizes,
- acute versus sub-acute phase,
- stroke pattern,
- vascular territory.

Important findings:

- Unseen-center performance can be comparable to seen-center performance when the model is robust.
- Smaller lesions depress Dice and make metric interpretation difficult.
- Acute scans can have lower Dice than sub-acute scans even when lesion-wise F1 is similar.
- Ensembles of diverse methods can improve clinically relevant derived tasks such as stroke-pattern and vascular-territory classification.

## Practical Lessons For ISLES'26

- Subgroup evaluation is mandatory: center, chronicity, lesion size, lesion count, and vascular territory if available.
- Detection metrics must sit next to Dice in all reports.
- Keep diverse high-performing models around long enough to test ensemble complementarity.
- A single model may lead one metric family while another leads a different family; rank aggregation rewards balance.
- Once a strong ensemble exists, compress or distill it into a smaller deployment model.

## Sources

- ISLES'22 Grand Challenge page: https://isles22.grand-challenge.org/
- ISLES'22 dataset paper: https://www.nature.com/articles/s41597-022-01875-5
- DeepISLES paper: https://www.nature.com/articles/s41467-025-62373-x
- DeepISLES code: https://github.com/ezequieldlrosa/DeepIsles

Related fibers: [[research/methods]], [[research/challenges/atlas-r2]],
[[challenge/overview]].
