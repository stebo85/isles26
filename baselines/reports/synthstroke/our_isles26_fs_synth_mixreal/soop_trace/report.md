# Our SynthStroke (isles26_fs_synth_mixreal) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0041 | 0.0000 | 0.0137 | 0.0000 | 0.0497 |
| iou | 12 | 0.0021 | 0.0000 | 0.0070 | 0.0000 | 0.0255 |
| sensitivity | 12 | 0.0023 | 0.0000 | 0.0076 | 0.0000 | 0.0276 |
| precision | 12 | 0.0204 | 0.0000 | 0.0676 | 0.0000 | 0.2445 |
| specificity | 12 | 0.9917 | 0.9916 | 0.0048 | 0.9821 | 0.9983 |
| hd95_mm | 12 | 93.8501 | 93.0792 | 19.5922 | 61.9627 | 125.2791 |
| assd_mm | 12 | 59.8073 | 55.9317 | 22.9034 | 24.4940 | 98.8218 |
| surface_dice_3mm | 12 | 0.0060 | 0.0000 | 0.0197 | 0.0000 | 0.0713 |
| abs_volume_diff_ml | 12 | 57.8092 | 60.0884 | 30.7430 | 21.1861 | 118.3206 |
| rel_abs_volume_diff | 12 | 21.6934 | 3.4638 | 39.8402 | 0.5183 | 147.5410 |
| lesion_f1 | 12 | 0.0203 | 0.0000 | 0.0674 | 0.0000 | 0.2439 |
| lesion_precision | 12 | 0.0198 | 0.0000 | 0.0658 | 0.0000 | 0.2381 |
| lesion_recall | 12 | 0.0208 | 0.0000 | 0.0691 | 0.0000 | 0.2500 |
| abs_lesion_count_diff | 12 | 27.4167 | 27.5000 | 11.1165 | 5.0000 | 45.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 0 | 0.000 |
| large_1k-10k | 9 | 0 | 0.000 |
| huge_>=10k | 4 | 1 | 0.250 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 0 | 0.000 |
| large_10-50mL | 4 | 0 | 0.000 |
| vlarge_>=50mL | 4 | 1 | 0.250 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.0248 | 0.0248 | 0.1220 | 0.1250 | 69.547 | 90.437 |
| acute | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 99.763 | 49.142 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 89.243 | 70.561 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.0248 | 0.1220 | 90.623 | 61.972 |
| Philips|Ingenia Ambition X | 6 | 0.0000 | 0.0000 | 99.503 | 66.470 |
| Philips|Ingenia Elition X | 2 | 0.0000 | 0.0000 | 76.967 | 32.373 |
| Philips|Achieva | 2 | 0.0000 | 0.0000 | 97.002 | 53.100 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 123.325 | 66.335 | 12.186 | 1 | 30 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 89.243 | 75.733 | 5.172 | 2 | 46 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 110.877 | 27.071 | 1.081 | 3 | 19 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 94.188 | 66.475 | 0.448 | 1 | 38 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 91.971 | 39.606 | 82.898 | 7 | 52 |
| sub-3 | mixed | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 68.725 | 77.074 | 159.993 | 5 | 33 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 86.311 | 121.549 | 3.228 | 1 | 34 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 125.279 | 24.109 | 0.828 | 1 | 28 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 106.802 | 29.716 | 8.530 | 12 | 38 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 97.149 | 11.164 | 79.740 | 9 | 14 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 61.963 | 55.786 | 34.332 | 2 | 24 |
| sub-1 | mixed | Philips|Intera | 0.0497 | 0.2439 | 0.2500 | 70.368 | 12.487 | 110.442 | 4 | 21 |