# Model leaderboard

generated 2026-06-25 04:35:29 UTC by compare_models.py

## eval set: soop_bench

| Method | Modalities | n | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (GT as prediction) | — | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | upper-bound sanity check: ground truth copied as prediction (Dice should be ~1.0) |
| DeepISLES ensemble (SEALS+NVAUTO+SWAN) | DWI+ADC+FLAIR | 12 | 0.539 | 0.704 | 0.582 | 0.584 | 0.763 | 18.929 | 18.894 | pretrained apptainer ensemble; consumes the DWI/ADC/FLAIR modalities the soop_bench GT is defined on |
| OOD blend: R3.0 250ep nnU-Net + synth-plus T1w | T1w | 12 | 0.289 | 0.202 | — | — | — | — | — | exploratory T1w probability blend; weight/threshold selected on this same 12-case soop_bench grid |
| OOD blend: R3.0 1000ep nnU-Net + synth-plus T1w | T1w | 12 | 0.288 | 0.199 | — | — | — | — | — | exploratory T1w probability blend; weight/threshold selected on this same 12-case soop_bench grid |
| nnU-Net R3.0 default 250ep, 5-fold+TTA on soop_bench | T1w | 12 | 0.255 | 0.036 | 0.239 | 0.189 | 0.302 | 45.776 | 20.470 | T1w-only OOD proxy; R3.0 includes SOOP-source subjects, so this can be optimistic |
| nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250ep, T1w) | T1w | 12 | 0.188 | 0.000 | 0.201 | 0.154 | 0.200 | 50.531 | 25.750 | single fold, reduced 250-epoch schedule, no TTA, no ensemble; T1w prediction warped to TRACE space via ANTs rigid registration |
| Empty (all-background prediction) | — | 12 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 41.573 | lower-bound sanity check: null prediction (Dice 0); confirms metrics handle empty masks |

## eval set: atlas_r21_fold0_val

| Method | Modalities | n | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250ep, T1w) | T1w | 194 | 0.640 | 0.725 | 0.654 | 0.750 | 0.756 | 18.761 | 5.035 | in-distribution held-out: the 194 fold-0 validation cases never seen during training; no skull-strip/registration needed (ATLAS T1w already brain-extracted, GT is T1w-native) |

## eval set: atlas_r30_5fold_cv

| Method | Modalities | n | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nnU-Net R3.0 Dice+TopK10 1000ep, 5-fold CV | T1w | 1450 | 0.655 | 0.763 | — | — | — | — | — | current best in-distribution model; MAPPING-style Dice+TopK10 hard-example loss |
| nnU-Net R3.0 default 1000ep, 5-fold CV | T1w | 1450 | 0.653 | 0.763 | — | — | — | — | — | full default schedule; +1.55 Dice points over 250ep on the same 1450-case CV |
| nnU-Net R3.0 default 250ep, 5-fold CV | T1w | 1450 | 0.637 | 0.744 | — | — | — | — | — | baseline 250-epoch 5-fold out-of-fold CV over 1450 usable converted R3.0 cases |

Note: lower is better for HD95/AVD; higher is better for the remaining metrics.
