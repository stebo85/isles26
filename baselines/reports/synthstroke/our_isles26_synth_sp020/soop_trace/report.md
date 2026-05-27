# Our SynthStroke (isles26_synth_sp020) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| iou | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sensitivity | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| precision | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| specificity | 12 | 0.9943 | 0.9969 | 0.0050 | 0.9837 | 0.9993 |
| hd95_mm | 12 | 89.1617 | 87.3648 | 18.0876 | 61.7599 | 127.7394 |
| assd_mm | 12 | 63.9337 | 61.0258 | 17.9515 | 36.6390 | 97.9625 |
| surface_dice_3mm | 12 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0003 |
| abs_volume_diff_ml | 12 | 45.1486 | 53.7131 | 29.1043 | 3.4432 | 86.8377 |
| rel_abs_volume_diff | 12 | 7.9438 | 3.3460 | 9.7721 | 0.3965 | 31.6393 |
| lesion_f1 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lesion_precision | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lesion_recall | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| abs_lesion_count_diff | 12 | 2.9167 | 2.0000 | 2.2158 | 0.0000 | 7.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 0 | 0.000 |
| large_1k-10k | 9 | 0 | 0.000 |
| huge_>=10k | 4 | 0 | 0.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 0 | 0.000 |
| large_10-50mL | 4 | 0 | 0.000 |
| vlarge_>=50mL | 4 | 0 | 0.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 79.463 | 75.139 |
| acute | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 92.169 | 38.156 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 81.494 | 48.104 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.0000 | 0.0000 | 90.025 | 54.276 |
| Philips|Ingenia Ambition X | 6 | 0.0000 | 0.0000 | 92.424 | 46.117 |
| Philips|Ingenia Elition X | 2 | 0.0000 | 0.0000 | 81.243 | 44.165 |
| Philips|Achieva | 2 | 0.0000 | 0.0000 | 86.431 | 34.098 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-1 | mixed | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 69.699 | 23.604 | 110.442 | 4 | 6 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 127.739 | 88.161 | 12.186 | 1 | 8 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 81.494 | 53.277 | 5.172 | 2 | 4 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 110.351 | 22.797 | 1.081 | 3 | 5 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 92.036 | 14.607 | 0.448 | 1 | 3 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 94.258 | 15.106 | 82.898 | 7 | 7 |
| sub-3 | mixed | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 89.227 | 96.553 | 159.993 | 5 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 85.503 | 62.550 | 3.228 | 1 | 4 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 83.635 | 5.584 | 0.828 | 1 | 2 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 61.760 | 11.973 | 8.530 | 12 | 7 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 106.011 | 4.039 | 79.740 | 9 | 4 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 68.228 | 13.795 | 34.332 | 2 | 8 |