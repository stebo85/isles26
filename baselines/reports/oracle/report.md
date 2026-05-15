# Sanity: oracle (GT-as-prediction)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| iou | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| sensitivity | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| precision | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| specificity | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| hd95_mm | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| assd_mm | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| surface_dice_3mm | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| abs_volume_diff_ml | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rel_abs_volume_diff | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lesion_f1 | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lesion_precision | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lesion_recall | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 5 | 1.000 |
| small_10-100 | 19 | 19 | 1.000 |
| medium_100-1k | 11 | 11 | 1.000 |
| large_1k-10k | 9 | 9 | 1.000 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 11 | 1.000 |
| small_0.1-1mL | 18 | 18 | 1.000 |
| med_1-10mL | 11 | 11 | 1.000 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.000 |
| acute | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.000 |
| chronic | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.000 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 1.0000 | 1.0000 | 0.000 | 0.000 |
| Philips|Ingenia Ambition X | 6 | 1.0000 | 1.0000 | 0.000 | 0.000 |
| Philips|Ingenia Elition X | 2 | 1.0000 | 1.0000 | 0.000 | 0.000 |
| Philips|Achieva | 2 | 1.0000 | 1.0000 | 0.000 | 0.000 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-1 | mixed | Philips|Intera | 1.0000 | 1.0000 | 1.0000 | 0.000 | 110.442 | 110.442 | 4 | 4 |
| sub-10 | acute | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 12.186 | 12.186 | 1 | 1 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 5.172 | 5.172 | 2 | 2 |
| sub-13 | acute | Philips|Intera | 1.0000 | 1.0000 | 1.0000 | 0.000 | 1.081 | 1.081 | 3 | 3 |
| sub-14 | acute | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.448 | 0.448 | 1 | 1 |
| sub-2 | acute | Philips|Ingenia Elition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 82.898 | 82.898 | 7 | 7 |
| sub-3 | mixed | Philips|Achieva | 1.0000 | 1.0000 | 1.0000 | 0.000 | 159.993 | 159.993 | 5 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 3.228 | 3.228 | 1 | 1 |
| sub-5 | acute | Philips|Achieva | 1.0000 | 1.0000 | 1.0000 | 0.000 | 0.828 | 0.828 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 8.530 | 8.530 | 12 | 12 |
| sub-8 | acute | Philips|Ingenia Ambition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 79.740 | 79.740 | 9 | 9 |
| sub-9 | acute | Philips|Ingenia Elition X | 1.0000 | 1.0000 | 1.0000 | 0.000 | 34.332 | 34.332 | 2 | 2 |