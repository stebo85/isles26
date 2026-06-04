# Our SynthStroke (isles26_synth_h_tversky) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2189 | 0.0509 | 0.2605 | 0.0000 | 0.6620 |
| iou | 12 | 0.1503 | 0.0262 | 0.1880 | 0.0000 | 0.4948 |
| sensitivity | 12 | 0.1956 | 0.0271 | 0.2436 | 0.0000 | 0.5985 |
| precision | 12 | 0.3236 | 0.3940 | 0.2932 | 0.0000 | 0.7859 |
| specificity | 12 | 0.9981 | 0.9995 | 0.0035 | 0.9872 | 1.0000 |
| hd95_mm | 12 | 45.3797 | 47.4770 | 17.4076 | 10.9193 | 71.5062 |
| assd_mm | 12 | 22.7783 | 18.1123 | 17.3865 | 2.8072 | 52.8677 |
| surface_dice_3mm | 12 | 0.2540 | 0.1089 | 0.2752 | 0.0000 | 0.7525 |
| abs_volume_diff_ml | 12 | 27.6380 | 8.2654 | 43.2024 | 0.0514 | 154.4234 |
| rel_abs_volume_diff | 12 | 5.1651 | 0.5888 | 15.4240 | 0.0159 | 56.3115 |
| lesion_f1 | 12 | 0.2370 | 0.2540 | 0.2103 | 0.0000 | 0.6486 |
| lesion_precision | 12 | 0.2087 | 0.1714 | 0.1916 | 0.0000 | 0.5714 |
| lesion_recall | 12 | 0.3303 | 0.3429 | 0.3162 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 6.0833 | 4.0000 | 6.1841 | 1.0000 | 24.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 7 | 0.778 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2824 | 0.2824 | 0.5465 | 0.5750 | 40.438 | 90.680 |
| acute | 9 | 0.1768 | 0.0140 | 0.1628 | 0.2571 | 47.234 | 16.559 |
| chronic | 1 | 0.4705 | 0.4705 | 0.2857 | 0.5000 | 38.576 | 1.269 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2667 | 0.3243 | 51.157 | 13.912 |
| Philips|Ingenia Ambition X | 6 | 0.1291 | 0.2055 | 52.791 | 18.805 |
| Philips|Ingenia Elition X | 2 | 0.6436 | 0.2593 | 13.796 | 18.002 |
| Philips|Achieva | 2 | 0.0157 | 0.2222 | 48.954 | 77.500 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 58.050 | 0.195 | 1.081 | 3 | 1 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 71.506 | 25.648 | 0.448 | 1 | 25 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 50.690 | 3.279 | 3.228 | 1 | 6 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 61.296 | 0.252 | 0.828 | 1 | 2 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0140 | 0.4444 | 1.0000 | 56.110 | 3.529 | 12.186 | 1 | 7 |
| sub-3 | mixed | Philips|Achieva | 0.0314 | 0.4444 | 0.4000 | 36.612 | 5.570 | 159.993 | 5 | 18 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0705 | 0.4000 | 0.4444 | 39.465 | 6.197 | 79.740 | 9 | 11 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.2199 | 0.1026 | 0.0833 | 60.396 | 4.422 | 8.530 | 12 | 15 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4705 | 0.2857 | 0.5000 | 38.576 | 3.903 | 5.172 | 2 | 5 |
| sub-1 | mixed | Philips|Intera | 0.5334 | 0.6486 | 0.7500 | 44.264 | 137.378 | 110.442 | 4 | 7 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6253 | 0.2963 | 0.2857 | 16.673 | 54.769 | 82.898 | 7 | 13 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6620 | 0.2222 | 0.5000 | 10.919 | 26.459 | 34.332 | 2 | 7 |