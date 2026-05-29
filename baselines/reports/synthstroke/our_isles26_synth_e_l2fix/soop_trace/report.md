# Our SynthStroke (isles26_synth_e_l2fix) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1308 | 0.0262 | 0.1733 | 0.0000 | 0.4762 |
| iou | 12 | 0.0804 | 0.0133 | 0.1126 | 0.0000 | 0.3125 |
| sensitivity | 12 | 0.3078 | 0.2881 | 0.2515 | 0.0000 | 0.6579 |
| precision | 12 | 0.0951 | 0.0136 | 0.1343 | 0.0000 | 0.3733 |
| specificity | 12 | 0.9632 | 0.9679 | 0.0113 | 0.9337 | 0.9781 |
| hd95_mm | 12 | 88.3323 | 87.5924 | 21.3969 | 51.5690 | 120.8045 |
| assd_mm | 12 | 50.8984 | 60.5778 | 24.5149 | 13.0885 | 86.8506 |
| surface_dice_3mm | 12 | 0.1235 | 0.0419 | 0.1375 | 0.0000 | 0.3752 |
| abs_volume_diff_ml | 12 | 200.2789 | 172.9976 | 94.7917 | 84.1113 | 447.7958 |
| rel_abs_volume_diff | 12 | 102.6954 | 19.9424 | 165.5424 | 0.7616 | 592.2787 |
| lesion_f1 | 12 | 0.1367 | 0.1149 | 0.1213 | 0.0000 | 0.4148 |
| lesion_precision | 12 | 0.0951 | 0.0694 | 0.1035 | 0.0000 | 0.3889 |
| lesion_recall | 12 | 0.4047 | 0.4365 | 0.3333 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 17.7500 | 13.5000 | 12.1595 | 6.0000 | 53.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 5 | 0.455 |
| large_1k-10k | 9 | 7 | 0.778 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 8 | 0.727 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.4744 | 0.4744 | 0.1868 | 0.4500 | 56.357 | 104.796 |
| acute | 9 | 0.0674 | 0.0065 | 0.1249 | 0.3840 | 94.220 | 212.795 |
| chronic | 1 | 0.0135 | 0.0135 | 0.1429 | 0.5000 | 99.296 | 278.598 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2381 | 0.1154 | 84.733 | 131.036 |
| Philips|Ingenia Ambition X | 6 | 0.0299 | 0.1517 | 94.464 | 251.220 |
| Philips|Ingenia Elition X | 2 | 0.2197 | 0.1352 | 75.682 | 157.509 |
| Philips|Achieva | 2 | 0.2371 | 0.1149 | 86.187 | 159.468 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 108.323 | 179.043 | 1.081 | 3 | 31 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 93.202 | 265.503 | 0.448 | 1 | 12 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 81.983 | 451.024 | 3.228 | 1 | 12 |
| sub-5 | acute | Philips|Achieva | 0.0017 | 0.0870 | 1.0000 | 120.804 | 194.283 | 0.828 | 1 | 22 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0065 | 0.2857 | 1.0000 | 115.832 | 258.152 | 12.186 | 1 | 12 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0135 | 0.1429 | 0.5000 | 99.296 | 283.770 | 5.172 | 2 | 12 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0389 | 0.0667 | 0.0833 | 106.786 | 176.563 | 8.530 | 12 | 18 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.1204 | 0.4148 | 0.4444 | 69.685 | 181.612 | 79.740 | 9 | 18 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.1826 | 0.0870 | 0.5000 | 75.272 | 199.826 | 34.332 | 2 | 21 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.2568 | 0.1834 | 0.4286 | 76.092 | 232.422 | 82.898 | 7 | 60 |
| sub-3 | mixed | Philips|Achieva | 0.4726 | 0.1429 | 0.4000 | 51.569 | 285.474 | 159.993 | 5 | 23 |
| sub-1 | mixed | Philips|Intera | 0.4762 | 0.2308 | 0.5000 | 61.144 | 194.553 | 110.442 | 4 | 20 |