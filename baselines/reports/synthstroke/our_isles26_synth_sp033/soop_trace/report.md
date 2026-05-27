# Our SynthStroke (isles26_synth_sp033) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0387 | 0.0000 | 0.0744 | 0.0000 | 0.2502 |
| iou | 12 | 0.0213 | 0.0000 | 0.0420 | 0.0000 | 0.1430 |
| sensitivity | 12 | 0.0243 | 0.0000 | 0.0463 | 0.0000 | 0.1507 |
| precision | 12 | 0.1874 | 0.0000 | 0.3145 | 0.0000 | 0.9811 |
| specificity | 12 | 0.9988 | 0.9993 | 0.0012 | 0.9965 | 1.0000 |
| hd95_mm | 12 | 85.0488 | 79.0289 | 25.7278 | 27.8950 | 135.1792 |
| assd_mm | 12 | 52.4853 | 51.8794 | 23.7011 | 11.0910 | 97.3055 |
| surface_dice_3mm | 12 | 0.0309 | 0.0000 | 0.0605 | 0.0000 | 0.2051 |
| abs_volume_diff_ml | 12 | 34.1092 | 7.3497 | 43.9553 | 0.2274 | 127.1614 |
| rel_abs_volume_diff | 12 | 0.8923 | 0.7782 | 0.6309 | 0.0440 | 2.6349 |
| lesion_f1 | 12 | 0.0793 | 0.0000 | 0.1021 | 0.0000 | 0.2759 |
| lesion_precision | 12 | 0.1497 | 0.0000 | 0.2854 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1003 | 0.0000 | 0.1485 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 6.5000 | 7.0000 | 4.9917 | 0.0000 | 19.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 1 | 0.091 |
| large_1k-10k | 9 | 0 | 0.000 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 1 | 0.091 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.1353 | 0.1353 | 0.2432 | 0.2250 | 51.801 | 115.403 |
| acute | 9 | 0.0216 | 0.0000 | 0.0517 | 0.0838 | 94.137 | 19.809 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 69.746 | 0.227 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.0102 | 0.1053 | 92.309 | 53.247 |
| Philips|Ingenia Ambition X | 6 | 0.0014 | 0.0333 | 96.418 | 15.938 |
| Philips|Ingenia Elition X | 2 | 0.0931 | 0.1326 | 73.608 | 39.633 |
| Philips|Achieva | 2 | 0.1251 | 0.1379 | 55.122 | 63.963 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 111.255 | 21.466 | 12.186 | 1 | 20 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 69.746 | 4.945 | 5.172 | 2 | 9 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 108.910 | 3.931 | 1.081 | 3 | 4 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 93.700 | 1.100 | 0.448 | 1 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 73.508 | 3.866 | 3.228 | 1 | 11 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 82.350 | 0.063 | 0.828 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 95.121 | 3.111 | 8.530 | 12 | 11 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0081 | 0.2000 | 0.1111 | 135.179 | 0.332 | 79.740 | 9 | 1 |
| sub-1 | mixed | Philips|Intera | 0.0204 | 0.2105 | 0.2500 | 75.708 | 6.797 | 110.442 | 4 | 11 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0483 | 0.0833 | 0.1429 | 71.697 | 25.074 | 82.898 | 7 | 17 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.1378 | 0.1818 | 0.5000 | 75.518 | 12.891 | 34.332 | 2 | 9 |
| sub-3 | mixed | Philips|Achieva | 0.2502 | 0.2759 | 0.2000 | 27.895 | 32.832 | 159.993 | 5 | 9 |