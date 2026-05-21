# Our SynthStroke (isles26_fs_synth_mixreal) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1619 | 0.0868 | 0.1955 | 0.0000 | 0.5680 |
| iou | 12 | 0.1023 | 0.0454 | 0.1346 | 0.0000 | 0.3967 |
| sensitivity | 12 | 0.1858 | 0.1297 | 0.1931 | 0.0000 | 0.5340 |
| precision | 12 | 0.1890 | 0.0844 | 0.2187 | 0.0000 | 0.6067 |
| specificity | 12 | 0.9959 | 0.9966 | 0.0020 | 0.9908 | 0.9979 |
| hd95_mm | 12 | 63.8345 | 60.0506 | 36.1064 | 24.8987 | 166.7192 |
| assd_mm | 12 | 28.2790 | 26.7583 | 15.7228 | 7.9137 | 50.0943 |
| surface_dice_3mm | 12 | 0.1726 | 0.1505 | 0.1791 | 0.0000 | 0.6296 |
| abs_volume_diff_ml | 12 | 31.5730 | 20.3203 | 33.6544 | 2.0566 | 130.7453 |
| rel_abs_volume_diff | 12 | 8.8744 | 2.3824 | 16.4543 | 0.0186 | 60.5246 |
| lesion_f1 | 12 | 0.0801 | 0.0437 | 0.0930 | 0.0000 | 0.3230 |
| lesion_precision | 12 | 0.0545 | 0.0228 | 0.0747 | 0.0000 | 0.2708 |
| lesion_recall | 12 | 0.3187 | 0.3095 | 0.2894 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 58.1667 | 62.0000 | 18.2475 | 15.0000 | 83.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 4 | 0.211 |
| medium_100-1k | 11 | 5 | 0.455 |
| large_1k-10k | 9 | 4 | 0.444 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 5 | 0.278 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2928 | 0.2928 | 0.2050 | 0.3250 | 27.900 | 66.401 |
| acute | 9 | 0.1400 | 0.0765 | 0.0552 | 0.2972 | 72.248 | 25.988 |
| chronic | 1 | 0.0970 | 0.0970 | 0.0541 | 0.5000 | 59.985 | 12.186 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2614 | 0.0435 | 42.507 | 7.242 |
| Philips|Ingenia Ambition X | 6 | 0.0730 | 0.0416 | 81.443 | 29.099 |
| Philips|Ingenia Elition X | 2 | 0.4593 | 0.1506 | 44.708 | 21.906 |
| Philips|Achieva | 2 | 0.0314 | 0.1615 | 51.462 | 72.993 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 60.116 | 13.510 | 1.081 | 3 | 64 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 88.534 | 27.533 | 0.448 | 1 | 84 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 73.714 | 23.322 | 3.228 | 1 | 78 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 72.022 | 16.070 | 0.828 | 1 | 36 |
| sub-3 | mixed | Philips|Achieva | 0.0629 | 0.3230 | 0.4000 | 30.901 | 29.248 | 159.993 | 5 | 48 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0765 | 0.1367 | 0.5556 | 33.600 | 20.465 | 79.740 | 9 | 77 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0970 | 0.0541 | 0.5000 | 59.985 | 17.358 | 5.172 | 2 | 70 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.1244 | 0.0333 | 1.0000 | 166.719 | 47.591 | 12.186 | 1 | 59 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1401 | 0.0256 | 0.3333 | 66.107 | 29.076 | 8.530 | 12 | 75 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.3505 | 0.1600 | 0.2857 | 43.306 | 43.194 | 82.898 | 7 | 63 |
| sub-1 | mixed | Philips|Intera | 0.5227 | 0.0870 | 0.2500 | 24.899 | 108.385 | 110.442 | 4 | 19 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.5680 | 0.1412 | 0.5000 | 46.110 | 30.223 | 34.332 | 2 | 73 |