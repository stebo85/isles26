# Our SynthStroke (isles26_synth_g_fade) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1144 | 0.0207 | 0.1511 | 0.0000 | 0.4610 |
| iou | 12 | 0.0682 | 0.0105 | 0.0948 | 0.0000 | 0.2996 |
| sensitivity | 12 | 0.1707 | 0.1486 | 0.1692 | 0.0000 | 0.4831 |
| precision | 12 | 0.1009 | 0.0111 | 0.1412 | 0.0000 | 0.4409 |
| specificity | 12 | 0.9778 | 0.9823 | 0.0149 | 0.9369 | 0.9908 |
| hd95_mm | 12 | 85.6591 | 87.3243 | 21.4485 | 58.7469 | 118.6092 |
| assd_mm | 12 | 49.6195 | 56.5330 | 25.0373 | 18.2113 | 88.5582 |
| surface_dice_3mm | 12 | 0.1158 | 0.0361 | 0.1314 | 0.0000 | 0.3446 |
| abs_volume_diff_ml | 12 | 107.2005 | 78.2740 | 112.0228 | 3.1923 | 426.0729 |
| rel_abs_volume_diff | 12 | 49.1003 | 10.5791 | 78.7988 | 0.0400 | 278.7869 |
| lesion_f1 | 12 | 0.1015 | 0.0553 | 0.1439 | 0.0000 | 0.5000 |
| lesion_precision | 12 | 0.0760 | 0.0340 | 0.1207 | 0.0000 | 0.4286 |
| lesion_recall | 12 | 0.2220 | 0.1667 | 0.2268 | 0.0000 | 0.6000 |
| abs_lesion_count_diff | 12 | 28.0000 | 26.5000 | 12.3626 | 9.0000 | 55.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 5 | 0.556 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 2 | 0.500 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.3553 | 0.3553 | 0.2857 | 0.4250 | 62.130 | 43.034 |
| acute | 9 | 0.0723 | 0.0000 | 0.0651 | 0.1459 | 89.492 | 109.749 |
| chronic | 1 | 0.0107 | 0.0107 | 0.0606 | 0.5000 | 98.223 | 212.600 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2305 | 0.0357 | 81.375 | 36.763 |
| Philips|Ingenia Ambition X | 6 | 0.0274 | 0.0614 | 91.771 | 164.951 |
| Philips|Ingenia Elition X | 2 | 0.2485 | 0.1388 | 65.922 | 51.211 |
| Philips|Achieva | 2 | 0.1248 | 0.2500 | 91.344 | 60.375 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 117.907 | 154.203 | 12.186 | 1 | 25 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 102.568 | 64.024 | 1.081 | 3 | 39 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 91.533 | 125.210 | 0.448 | 1 | 34 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 83.116 | 429.301 | 3.228 | 1 | 15 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 118.609 | 46.092 | 0.828 | 1 | 36 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0107 | 0.0606 | 0.5000 | 98.223 | 217.772 | 5.172 | 2 | 31 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0307 | 0.0500 | 0.0833 | 101.101 | 89.593 | 8.530 | 12 | 28 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.1232 | 0.2581 | 0.4444 | 58.747 | 76.547 | 79.740 | 9 | 33 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.1579 | 0.0851 | 0.5000 | 71.096 | 119.998 | 34.332 | 2 | 43 |
| sub-3 | mixed | Philips|Achieva | 0.2497 | 0.5000 | 0.6000 | 64.078 | 235.478 | 159.993 | 5 | 14 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.3392 | 0.1925 | 0.2857 | 60.749 | 99.656 | 82.898 | 7 | 62 |
| sub-1 | mixed | Philips|Intera | 0.4610 | 0.0714 | 0.2500 | 60.183 | 121.025 | 110.442 | 4 | 24 |