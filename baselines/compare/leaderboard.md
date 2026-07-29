# Model leaderboard

generated 2026-07-29 00:21:21 UTC by compare_models.py

## eval set: soop_bench

| Method | Modalities | n | Evaluator (n scored) | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (GT as prediction) | — | 12 | unspecified (n=12) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | upper-bound sanity check: ground truth copied as prediction (Dice should be ~1.0) |
| DeepISLES ensemble (SEALS+NVAUTO+SWAN) | DWI+ADC+FLAIR | 12 | unspecified (n=12) | 0.539 | 0.704 | 0.582 | 0.584 | 0.763 | 18.929 | 18.894 | pretrained apptainer ensemble; consumes the DWI/ADC/FLAIR modalities the soop_bench GT is defined on |
| OOD blend: R3.0 TopK10 nnU-Net + synth-plus T1w | T1w | 12 | unspecified (n=12) | 0.301 | 0.257 | — | — | — | — | — | best T1w-only OOD blend so far; weight/threshold selected on this same 12-case soop_bench grid |
| OOD blend: R3.0 250ep nnU-Net + synth-plus T1w | T1w | 12 | unspecified (n=12) | 0.289 | 0.202 | — | — | — | — | — | exploratory T1w probability blend; weight/threshold selected on this same 12-case soop_bench grid |
| OOD blend: R3.0 1000ep nnU-Net + synth-plus T1w | T1w | 12 | unspecified (n=12) | 0.288 | 0.199 | — | — | — | — | — | exploratory T1w probability blend; weight/threshold selected on this same 12-case soop_bench grid |
| nnU-Net R3.0 default 250ep, 5-fold+TTA on soop_bench | T1w | 12 | unspecified (n=12) | 0.255 | 0.036 | 0.239 | 0.189 | 0.302 | 45.776 | 20.470 | T1w-only OOD proxy; R3.0 includes SOOP-source subjects, so this can be optimistic |
| nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250ep, T1w) | T1w | 12 | unspecified (n=12) | 0.188 | 0.000 | 0.201 | 0.154 | 0.200 | 50.531 | 25.750 | single fold, reduced 250-epoch schedule, no TTA, no ensemble; T1w prediction warped to TRACE space via ANTs rigid registration |
| Empty (all-background prediction) | — | 12 | unspecified (n=12) | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 41.573 | lower-bound sanity check: null prediction (Dice 0); confirms metrics handle empty masks |

## eval set: atlas_r21_fold0_val

| Method | Modalities | n | Evaluator (n scored) | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250ep, T1w) | T1w | 194 | unspecified (n=194) | 0.640 | 0.725 | 0.654 | 0.750 | 0.756 | 18.761 | 5.035 | in-distribution held-out: the 194 fold-0 validation cases never seen during training; no skull-strip/registration needed (ATLAS T1w already brain-extracted, GT is T1w-native) |

## eval set: atlas_r30_5fold_cv

| Method | Modalities | n | Evaluator (n scored) | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nnU-Net R3.0 Dice+TopK10 1000ep, 5-fold CV | T1w | 1450 | unspecified (n=1450) | 0.656 | 0.764 | 0.674 | 0.707 | 0.768 | 18.099 | 4.943 | current best single nnU-Net in-distribution model; rich OOF eval via repo harness |
| MAPPING input single TopK10 OOF argmax | T1w | 1450 | unspecified (n=1450) | 0.656 | 0.763 | 0.674 | 0.707 | 0.768 | 18.099 | 4.943 | Single-family argmax from step-35 validation softmax; exact TopK10 input source for 3-family ensemble comparison |
| nnU-Net R3.0 Dice+TopK10 + synthetic lesion insertion, 5-fold CV | T1w | 1450 | unspecified (n=1450) | 0.656 | 0.764 | 0.665 | 0.713 | 0.765 | 18.187 | 4.969 | Dice+TopK10 baseline with size-rebalanced synthetic lesion insertion augmentation targeting tiny/small-lesion recall |
| MAPPING 3-family softmax ensemble OOF CV | T1w | 1450 | unspecified (n=1450) | 0.655 | 0.770 | 0.673 | 0.700 | 0.768 | 17.602 | 5.160 | MAPPING-style 3-family per-fold validation softmax average; threshold 0.5; no Dataset504/self-training member |
| MAPPING 3-family softmax ensemble OOF CV, threshold 0.54 | T1w | 1450 | unspecified (n=1450) | 0.654 | 0.768 | 0.676 | 0.697 | 0.767 | 17.633 | 5.159 | Threshold selected by sweeping saved OOF probability NIfTIs only; thresholds=0.35:0.55:0.01; primary_score=lesion_f1_mean; no re-inference |
| MAPPING input single default 1000ep OOF argmax | T1w | 1450 | unspecified (n=1450) | 0.653 | 0.764 | 0.665 | 0.704 | 0.759 | 18.700 | 5.296 | Single-family argmax from step-35 validation softmax; exact default 1000ep input source for 3-family ensemble comparison |
| nnU-Net R3.0 default 1000ep, 5-fold CV | T1w | 1450 | unspecified (n=1448) | 0.653 | 0.763 | — | — | — | — | — | full default schedule; +1.55 Dice points over 250ep on the same 1450-case CV |
| MAPPING input single ResEnc-L OOF argmax | T1w | 1450 | unspecified (n=1450) | 0.639 | 0.754 | 0.638 | 0.693 | 0.755 | 18.170 | 5.390 | Single-family argmax from step-35 validation softmax; exact ResEnc-L input source for 3-family ensemble comparison |
| nnU-Net R3.0 default 250ep, 5-fold CV | T1w | 1450 | unspecified (n=1448) | 0.637 | 0.744 | — | — | — | — | — | baseline 250-epoch 5-fold out-of-fold CV over 1450 usable converted R3.0 cases |
| TopK10 out-of-fold, grouped by center / chronicity / lesion size | T1w | 1450 | nnunet_summary_grouped (analysis_nnunet_29_report_generalization_cv.sh) (n=1450) | — | — | — | — | — | — | — | TRIAGE ONLY, not a model row: this is the SAME TopK10 out-of-fold prediction set as nnunet_topk10_crossval, re-aggregated by center / chronicity / lesion-size group to find which groups are weak. It reports Dice only (n_valid 1447 of 1450), and its folds are stratified-RANDOM, so the per-center numbers still contain center leakage - 52 of 55 centers appear on both sides of every fold. For an honest per-center estimate use the center-grouped Dataset507 reports instead. Its chronicity strata are also affected by the CHRONICITY-parsing bug: 270 cases binned 'unknown' are in fact chronic. |

## eval set: atlas_r30_center_grouped_cv

| Method | Modalities | n | Evaluator (n scored) | Dice mean | Dice median | Lesion F1 | Lesion recall | Surface Dice 3mm | HD95 mm | AVD mL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nnU-Net corrected R3.0 Dice+TopK10, held-CENTER 5-fold CV (raw) | T1w | 1452 | repo_harness (n=1452) | 0.635 | 0.740 | 0.650 | 0.690 | 0.746 | 20.475 | 6.278 | HONEST GENERALIZATION ESTIMATE: the five folds hold out whole CENTERS (each of the 55 ATLAS sites appears in exactly one validation fold), so no scanner/site is shared between train and validation. Dice 0.6347 / lesion-F1 0.6498 / HD95 20.47 mm / AVD 6.28 mL over n=1452. Not comparable with the atlas_r30_5fold_cv rows, which leak centers across folds and are ~2 Dice points optimistic; the case set also differs (1452 corrected Dataset507 cases vs 1450 in Dataset502). Identical to nnunet_center_grouped_topk10_crossval (postprocessed) because nnU-Net selected no postprocessing (postprocessing_fns == []). Lesion F1 here is the repo harness's lenient 1-voxel many-to-many criterion, not the official panoptica RQ. |
| ISLES'26 official metrics -- TopK10 out-of-fold, Dataset507 center-grouped folds (HONEST) | T1w | 1452 | isles26_official (work/isles26_upstream/utils/eval_utils.py) (n=1452) | — | — | — | — | — | — | — | Official five-metric scoring on folds that hold out whole CENTERS. This is the honest generalization surface and the one the shipped operating point is selected on, because the hidden ISLES'26 test set spans 60+ centers. |

Note: lower is better for HD95/AVD; higher is better for the remaining metrics.

Provenance: the `Evaluator (n scored)` column names the code that produced the row and how many cases entered the Dice mean. Rows from different evaluators use different empty-case conventions (`repo_harness` scores empty-on-both as 1.0, `nnunet_summary` drops those cases) and must not be compared at the third decimal.
