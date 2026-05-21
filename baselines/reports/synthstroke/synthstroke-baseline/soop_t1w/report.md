# SynthStroke synthstroke-baseline on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0988 | 0.0000 | 0.1699 | 0.0000 | 0.4648 |
| iou | 12 | 0.0618 | 0.0000 | 0.1099 | 0.0000 | 0.3027 |
| sensitivity | 12 | 0.0679 | 0.0000 | 0.1226 | 0.0000 | 0.3376 |
| precision | 8 | 0.4155 | 0.3980 | 0.3984 | 0.0000 | 0.9462 |
| specificity | 12 | 0.9998 | 0.9999 | 0.0005 | 0.9981 | 1.0000 |
| hd95_mm | 8 | 53.3604 | 49.5680 | 22.1939 | 24.6575 | 90.8186 |
| assd_mm | 8 | 29.9881 | 20.5096 | 21.5305 | 6.9524 | 64.4081 |
| surface_dice_3mm | 8 | 0.1565 | 0.0465 | 0.1925 | 0.0000 | 0.5412 |
| abs_volume_diff_ml | 12 | 36.6372 | 9.8625 | 47.4803 | 0.5869 | 159.5717 |
| rel_abs_volume_diff | 12 | 0.9289 | 0.9970 | 0.1983 | 0.5206 | 1.3115 |
| lesion_f1 | 12 | 0.1336 | 0.0000 | 0.1704 | 0.0000 | 0.5000 |
| lesion_precision | 12 | 0.5466 | 0.5833 | 0.4023 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1327 | 0.0000 | 0.1852 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 3.5000 | 2.5000 | 3.2532 | 0.0000 | 12.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 0 | 0.000 |
| huge_>=10k | 4 | 3 | 0.750 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 2 | 0.182 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 2 | 0.500 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2326 | 0.2326 | 0.3205 | 0.2250 | 71.923 | 111.516 |
| acute | 9 | 0.0294 | 0.0000 | 0.0514 | 0.0714 | 49.236 | 23.769 |
| chronic | 1 | 0.4564 | 0.4564 | 0.5000 | 0.5000 | 36.857 | 2.692 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2324 | 0.1667 | 53.028 | 32.271 |
| Philips|Ingenia Ambition X | 6 | 0.0761 | 0.0833 | 53.068 | 17.618 |
| Philips|Ingenia Elition X | 2 | 0.1321 | 0.2311 | 35.383 | 54.498 |
| Philips|Achieva | 2 | 0.0002 | 0.1538 | 90.819 | 80.200 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 59.020 | 0.990 | 12.186 | 1 | 3 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 1.081 | 3 | 0 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 32.607 | 1.034 | 0.448 | 1 | 4 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 8.530 | 12 | 0 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 83.787 | 0.263 | 79.740 | 9 | 1 |
| sub-3 | mixed | Philips|Achieva | 0.0005 | 0.3077 | 0.2000 | 90.819 | 0.421 | 159.993 | 5 | 3 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0862 | 0.2400 | 0.1429 | 46.108 | 3.956 | 82.898 | 7 | 4 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.1781 | 0.2222 | 0.5000 | 24.658 | 4.279 | 34.332 | 2 | 7 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4564 | 0.5000 | 0.5000 | 36.857 | 2.480 | 5.172 | 2 | 2 |
| sub-1 | mixed | Philips|Intera | 0.4648 | 0.3333 | 0.2500 | 53.028 | 46.981 | 110.442 | 4 | 2 |