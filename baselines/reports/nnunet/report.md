# nnU-Net v2 3D fullres (ATLAS R2.1, fold 0, 250 ep, T1w-only) on soop_bench

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1876 | 0.0000 | 0.2682 | 0.0000 | 0.6589 |
| iou | 12 | 0.1320 | 0.0000 | 0.1928 | 0.0000 | 0.4913 |
| sensitivity | 12 | 0.1623 | 0.0000 | 0.2421 | 0.0000 | 0.6489 |
| precision | 12 | 0.2856 | 0.0000 | 0.3411 | 0.0000 | 0.7702 |
| specificity | 12 | 0.9991 | 0.9999 | 0.0021 | 0.9924 | 1.0000 |
| hd95_mm | 12 | 50.5308 | 40.4242 | 29.9003 | 12.5259 | 113.0934 |
| assd_mm | 12 | 32.4814 | 20.3369 | 28.4727 | 4.1276 | 86.0765 |
| surface_dice_3mm | 12 | 0.1997 | 0.0000 | 0.2756 | 0.0000 | 0.6491 |
| abs_volume_diff_ml | 12 | 25.7497 | 5.7756 | 44.9087 | 0.0147 | 157.2518 |
| rel_abs_volume_diff | 12 | 0.6367 | 0.8063 | 0.3822 | 0.0295 | 0.9942 |
| lesion_f1 | 12 | 0.2006 | 0.0000 | 0.2622 | 0.0000 | 0.6667 |
| lesion_precision | 12 | 0.3542 | 0.0000 | 0.4385 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1536 | 0.0000 | 0.2096 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 2.3333 | 1.0000 | 2.8087 | 0.0000 | 8.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 1 | 0.111 |
| huge_>=10k | 4 | 3 | 0.750 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 2 | 0.500 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.3297 | 0.3297 | 0.4167 | 0.3500 | 63.433 | 80.253 |
| acute | 9 | 0.1185 | 0.0000 | 0.1007 | 0.0714 | 50.133 | 16.299 |
| chronic | 1 | 0.5253 | 0.5253 | 0.6667 | 0.5000 | 28.305 | 1.797 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.3197 | 0.2500 | 44.587 | 1.853 |
| Philips|Ingenia Ambition X | 6 | 0.0876 | 0.1111 | 49.120 | 17.433 |
| Philips|Ingenia Elition X | 2 | 0.5334 | 0.4533 | 25.741 | 21.310 |
| Philips|Achieva | 2 | 0.0100 | 0.1667 | 85.496 | 79.038 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 68.582 | 0.169 | 12.186 | 1 | 2 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 50.983 | 0.629 | 1.081 | 3 | 5 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 17.606 | 0.462 | 0.448 | 1 | 1 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 25.242 | 0.022 | 3.228 | 1 | 2 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 82.317 | 0.005 | 0.828 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 41.893 | 0.232 | 8.530 | 12 | 4 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 113.093 | 0.477 | 79.740 | 9 | 1 |
| sub-3 | mixed | Philips|Achieva | 0.0200 | 0.3333 | 0.2000 | 88.675 | 2.741 | 159.993 | 5 | 1 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.4079 | 0.6667 | 0.5000 | 12.526 | 12.366 | 34.332 | 2 | 2 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.5253 | 0.6667 | 0.5000 | 28.305 | 3.375 | 5.172 | 2 | 1 |
| sub-1 | mixed | Philips|Intera | 0.6394 | 0.5000 | 0.5000 | 38.191 | 113.695 | 110.442 | 4 | 4 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6589 | 0.2400 | 0.1429 | 38.955 | 62.244 | 82.898 | 7 | 4 |

## Method Notes

- Training uses nnU-Net v2 3D full-resolution, T1w-only, fold 0 only, for 250 epochs.
- This is a time-bounded reduced-schedule baseline, not the full 5-fold 1000-epoch nnU-Net ensemble.
- soop_bench T1w inputs are forced to RAS canonical orientation and skull-stripped with FreeSurfer SynthStrip before inference.
- Inference uses fold 0 only with test-time augmentation disabled.

