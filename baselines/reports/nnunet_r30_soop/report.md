# nnU-Net v2 3D fullres (ATLAS R3.0, 5-fold ensemble + TTA, T1w-only) on soop_bench

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2546 | 0.0364 | 0.3026 | 0.0000 | 0.7249 |
| iou | 12 | 0.1854 | 0.0189 | 0.2290 | 0.0000 | 0.5685 |
| sensitivity | 12 | 0.2262 | 0.0190 | 0.2845 | 0.0000 | 0.7217 |
| precision | 11 | 0.4081 | 0.6579 | 0.3763 | 0.0000 | 0.8544 |
| specificity | 12 | 0.9989 | 0.9999 | 0.0020 | 0.9933 | 1.0000 |
| hd95_mm | 11 | 45.7765 | 37.0326 | 24.2159 | 10.2796 | 92.3525 |
| assd_mm | 11 | 22.2566 | 20.7206 | 17.6759 | 2.1774 | 63.9326 |
| surface_dice_3mm | 11 | 0.3017 | 0.1421 | 0.3201 | 0.0000 | 0.7838 |
| abs_volume_diff_ml | 12 | 20.4696 | 2.4283 | 42.5224 | 0.0807 | 152.8736 |
| rel_abs_volume_diff | 12 | 0.5651 | 0.5471 | 0.3911 | 0.0088 | 1.0000 |
| lesion_f1 | 12 | 0.2394 | 0.1000 | 0.2633 | 0.0000 | 0.6667 |
| lesion_precision | 12 | 0.4583 | 0.4167 | 0.4363 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1888 | 0.0714 | 0.2145 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 2.3333 | 1.5000 | 2.2485 | 0.0000 | 7.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 2 | 0.222 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 4 | 0.364 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.3633 | 0.3633 | 0.5714 | 0.4500 | 57.452 | 77.143 |
| acute | 9 | 0.1979 | 0.0000 | 0.1182 | 0.0961 | 45.032 | 9.969 |
| chronic | 1 | 0.5471 | 0.5471 | 0.6667 | 0.5000 | 28.382 | 1.629 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.3269 | 0.2857 | 43.453 | 0.924 |
| Philips|Ingenia Ambition X | 6 | 0.1501 | 0.1717 | 54.018 | 13.424 |
| Philips|Ingenia Elition X | 2 | 0.7139 | 0.3500 | 23.354 | 4.777 |
| Philips|Achieva | 2 | 0.0364 | 0.2857 | 49.918 | 76.846 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 74.025 | 0.139 | 12.186 | 1 | 1 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 49.873 | 0.647 | 1.081 | 3 | 6 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 35.303 | 0.367 | 0.448 | 1 | 1 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 21.963 | 0.010 | 0.828 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 40.029 | 0.163 | 8.530 | 12 | 5 |
| sub-3 | mixed | Philips|Achieva | 0.0728 | 0.5714 | 0.4000 | 77.872 | 7.119 | 159.993 | 5 | 2 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.3537 | 0.3636 | 0.2222 | 92.353 | 24.548 | 79.740 | 9 | 3 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.5471 | 0.6667 | 0.5000 | 28.382 | 3.543 | 5.172 | 2 | 1 |
| sub-1 | mixed | Philips|Intera | 0.6537 | 0.5714 | 0.5000 | 37.033 | 109.028 | 110.442 | 4 | 3 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.7028 | 0.5000 | 0.5000 | 10.280 | 25.507 | 34.332 | 2 | 4 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.7249 | 0.2000 | 0.1429 | 36.429 | 82.170 | 82.898 | 7 | 3 |

## Method Notes

- nnU-Net v2 3D full-resolution, T1w-only, trained on ATLAS R3.0 (Dataset502, 1450 cases) for 250 epochs.
- Inference uses the full 5-fold ensemble with test-time augmentation ENABLED.
- soop_bench T1w inputs are forced to RAS canonical orientation and skull-stripped with FreeSurfer SynthStrip before inference, then rigidly warped to TRACE/DWI space.
- CAVEAT: R3.0 training included the 169 sub-soop* subjects; soop_bench (sub-1..14) may overlap, so this OOD number can be optimistically biased.

