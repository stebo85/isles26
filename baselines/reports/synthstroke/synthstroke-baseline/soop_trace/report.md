# SynthStroke synthstroke-baseline on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| iou | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sensitivity | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| precision | 0 | nan | nan | nan | nan | nan |
| specificity | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| hd95_mm | 0 | nan | nan | nan | nan | nan |
| assd_mm | 0 | nan | nan | nan | nan | nan |
| surface_dice_3mm | 0 | nan | nan | nan | nan | nan |
| abs_volume_diff_ml | 12 | 41.5731 | 10.3577 | 51.4173 | 0.4475 | 159.9930 |
| rel_abs_volume_diff | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lesion_f1 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lesion_precision | 12 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| lesion_recall | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| abs_lesion_count_diff | 12 | 4.0000 | 2.5000 | 3.4641 | 1.0000 | 12.0000 |

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
| mixed | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | 135.217 |
| acute | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | 24.808 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | 5.172 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.0000 | 0.0000 | nan | 55.762 |
| Philips|Ingenia Ambition X | 6 | 0.0000 | 0.0000 | nan | 18.217 |
| Philips|Ingenia Elition X | 2 | 0.0000 | 0.0000 | nan | 58.615 |
| Philips|Achieva | 2 | 0.0000 | 0.0000 | nan | 80.411 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-1 | mixed | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 110.442 | 4 | 0 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 12.186 | 1 | 0 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 5.172 | 2 | 0 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 1.081 | 3 | 0 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.448 | 1 | 0 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 82.898 | 7 | 0 |
| sub-3 | mixed | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 159.993 | 5 | 0 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 8.530 | 12 | 0 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 79.740 | 9 | 0 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 34.332 | 2 | 0 |