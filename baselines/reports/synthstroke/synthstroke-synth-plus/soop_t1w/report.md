# SynthStroke synthstroke-synth-plus on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2173 | 0.0428 | 0.2774 | 0.0000 | 0.6701 |
| iou | 12 | 0.1536 | 0.0221 | 0.2039 | 0.0000 | 0.5039 |
| sensitivity | 12 | 0.1932 | 0.0226 | 0.2616 | 0.0000 | 0.6880 |
| precision | 10 | 0.4957 | 0.6219 | 0.3441 | 0.0000 | 1.0000 |
| specificity | 12 | 0.9989 | 0.9997 | 0.0020 | 0.9926 | 1.0000 |
| hd95_mm | 10 | 50.4600 | 46.3328 | 25.6739 | 13.5177 | 83.9685 |
| assd_mm | 10 | 27.5914 | 22.3159 | 22.6836 | 3.9242 | 63.9222 |
| surface_dice_3mm | 10 | 0.2715 | 0.1503 | 0.2821 | 0.0000 | 0.6848 |
| abs_volume_diff_ml | 12 | 24.8251 | 5.2003 | 45.0911 | 0.8282 | 158.9517 |
| rel_abs_volume_diff | 12 | 1.0495 | 0.9185 | 1.2124 | 0.0598 | 4.8689 |
| lesion_f1 | 12 | 0.2308 | 0.2253 | 0.2310 | 0.0000 | 0.6667 |
| lesion_precision | 12 | 0.5324 | 0.5833 | 0.4049 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.2076 | 0.1528 | 0.2103 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 2.8333 | 2.0000 | 2.7938 | 0.0000 | 10.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 4 | 0.444 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 6 | 0.545 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2936 | 0.2936 | 0.4396 | 0.4500 | 60.768 | 83.463 |
| acute | 9 | 0.1694 | 0.0000 | 0.1359 | 0.1213 | 49.498 | 14.333 |
| chronic | 1 | 0.4965 | 0.4965 | 0.6667 | 0.5000 | 36.580 | 1.981 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2872 | 0.1538 | 60.360 | 4.650 |
| Philips|Ingenia Ambition X | 6 | 0.1138 | 0.1905 | 52.960 | 16.930 |
| Philips|Ingenia Elition X | 2 | 0.6691 | 0.3736 | 18.560 | 13.619 |
| Philips|Achieva | 2 | 0.0065 | 0.2857 | 81.962 | 79.890 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 12.186 | 1 | 0 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 81.145 | 2.409 | 1.081 | 3 | 3 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 53.091 | 2.626 | 0.448 | 1 | 3 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 68.850 | 0.492 | 3.228 | 1 | 1 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-3 | mixed | Philips|Achieva | 0.0129 | 0.5714 | 0.4000 | 81.962 | 1.041 | 159.993 | 5 | 2 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0728 | 0.3333 | 0.2222 | 83.969 | 4.905 | 79.740 | 9 | 3 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1135 | 0.1429 | 0.0833 | 22.309 | 0.866 | 8.530 | 12 | 2 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4965 | 0.6667 | 0.5000 | 36.580 | 3.191 | 5.172 | 2 | 1 |
| sub-1 | mixed | Philips|Intera | 0.5743 | 0.3077 | 0.5000 | 39.575 | 102.468 | 110.442 | 4 | 9 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6680 | 0.3333 | 0.5000 | 13.518 | 36.386 | 34.332 | 2 | 4 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6701 | 0.4138 | 0.2857 | 23.603 | 57.713 | 82.898 | 7 | 4 |