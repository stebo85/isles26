# Our SynthStroke (isles26_synth_k_dwi) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0468 | 0.0033 | 0.1034 | 0.0000 | 0.3739 |
| iou | 12 | 0.0273 | 0.0016 | 0.0633 | 0.0000 | 0.2300 |
| sensitivity | 12 | 0.0442 | 0.0027 | 0.0760 | 0.0000 | 0.2516 |
| precision | 12 | 0.0812 | 0.0048 | 0.1985 | 0.0000 | 0.7279 |
| specificity | 12 | 0.9905 | 0.9930 | 0.0076 | 0.9712 | 0.9984 |
| hd95_mm | 12 | 86.6683 | 87.2262 | 22.2625 | 52.6762 | 119.7032 |
| assd_mm | 12 | 48.3452 | 54.3852 | 21.7794 | 17.1603 | 81.3377 |
| surface_dice_3mm | 12 | 0.0385 | 0.0115 | 0.0770 | 0.0000 | 0.2861 |
| abs_volume_diff_ml | 12 | 64.5383 | 71.2938 | 48.6347 | 15.0482 | 192.7918 |
| rel_abs_volume_diff | 12 | 30.5458 | 4.3368 | 63.5259 | 0.2514 | 233.6885 |
| lesion_f1 | 12 | 0.0833 | 0.0678 | 0.1002 | 0.0000 | 0.3529 |
| lesion_precision | 12 | 0.0585 | 0.0484 | 0.0759 | 0.0000 | 0.2727 |
| lesion_recall | 12 | 0.1957 | 0.1131 | 0.2096 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 19.0833 | 17.5000 | 9.7593 | 6.0000 | 37.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 3 | 0.333 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2054 | 0.2054 | 0.2610 | 0.4500 | 58.707 | 78.959 |
| acute | 9 | 0.0148 | 0.0000 | 0.0425 | 0.1054 | 90.828 | 60.391 |
| chronic | 1 | 0.0171 | 0.0171 | 0.0952 | 0.5000 | 105.152 | 73.026 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.1870 | 0.1765 | 84.534 | 45.790 |
| Philips|Ingenia Ambition X | 6 | 0.0039 | 0.0526 | 93.992 | 89.454 |
| Philips|Ingenia Elition X | 2 | 0.0633 | 0.0810 | 65.134 | 17.946 |
| Philips|Achieva | 2 | 0.0184 | 0.0845 | 88.365 | 55.132 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 111.193 | 83.965 | 12.186 | 1 | 16 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 108.681 | 20.393 | 1.081 | 3 | 24 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 91.011 | 105.027 | 0.448 | 1 | 24 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 83.442 | 196.020 | 3.228 | 1 | 19 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 119.703 | 25.441 | 0.828 | 1 | 18 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0009 | 0.0667 | 0.0833 | 103.783 | 32.268 | 8.530 | 12 | 18 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0057 | 0.1538 | 0.2222 | 69.373 | 8.931 | 79.740 | 9 | 17 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0138 | 0.0930 | 0.5000 | 77.592 | 19.284 | 34.332 | 2 | 39 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0171 | 0.0952 | 0.5000 | 105.152 | 78.198 | 5.172 | 2 | 19 |
| sub-3 | mixed | Philips|Achieva | 0.0368 | 0.1690 | 0.4000 | 57.028 | 74.342 | 159.993 | 5 | 28 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.1129 | 0.0690 | 0.1429 | 52.676 | 62.054 | 82.898 | 7 | 44 |
| sub-1 | mixed | Philips|Intera | 0.3739 | 0.3529 | 0.5000 | 60.387 | 38.174 | 110.442 | 4 | 11 |