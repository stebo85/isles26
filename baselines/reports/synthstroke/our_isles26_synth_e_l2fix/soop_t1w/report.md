# Our SynthStroke (isles26_synth_e_l2fix) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2154 | 0.1319 | 0.2349 | 0.0000 | 0.6336 |
| iou | 12 | 0.1429 | 0.0708 | 0.1690 | 0.0000 | 0.4637 |
| sensitivity | 12 | 0.2287 | 0.1337 | 0.2467 | 0.0000 | 0.6673 |
| precision | 12 | 0.2646 | 0.2191 | 0.2586 | 0.0000 | 0.7257 |
| specificity | 12 | 0.9962 | 0.9983 | 0.0052 | 0.9810 | 0.9995 |
| hd95_mm | 12 | 53.8003 | 55.5168 | 21.8232 | 22.4059 | 89.8443 |
| assd_mm | 12 | 28.1433 | 18.6617 | 21.5735 | 4.1424 | 67.1445 |
| surface_dice_3mm | 12 | 0.2472 | 0.2257 | 0.2345 | 0.0000 | 0.6662 |
| abs_volume_diff_ml | 12 | 33.3539 | 11.8843 | 42.0114 | 0.8877 | 145.1343 |
| rel_abs_volume_diff | 12 | 12.4725 | 0.8632 | 36.9771 | 0.0728 | 135.0328 |
| lesion_f1 | 12 | 0.1860 | 0.1548 | 0.1621 | 0.0000 | 0.4356 |
| lesion_precision | 12 | 0.1561 | 0.1125 | 0.1567 | 0.0000 | 0.4783 |
| lesion_recall | 12 | 0.3164 | 0.3429 | 0.2909 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 11.3333 | 11.0000 | 4.2492 | 7.0000 | 21.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 1 | 0.053 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 6 | 0.667 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2824 | 0.2824 | 0.3917 | 0.4500 | 35.971 | 107.166 |
| acute | 9 | 0.1864 | 0.1066 | 0.1461 | 0.2663 | 57.147 | 20.285 |
| chronic | 1 | 0.3428 | 0.3428 | 0.1333 | 0.5000 | 59.334 | 3.353 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2541 | 0.1739 | 61.048 | 37.343 |
| Philips|Ingenia Ambition X | 6 | 0.1292 | 0.1728 | 58.454 | 25.629 |
| Philips|Ingenia Elition X | 2 | 0.6225 | 0.2058 | 26.532 | 12.558 |
| Philips|Achieva | 2 | 0.0283 | 0.2178 | 59.858 | 73.337 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 80.026 | 6.569 | 1.081 | 3 | 14 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 75.008 | 60.877 | 0.448 | 1 | 22 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 51.700 | 10.652 | 3.228 | 1 | 9 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 89.844 | 2.368 | 0.828 | 1 | 8 |
| sub-3 | mixed | Philips|Achieva | 0.0567 | 0.4356 | 0.4000 | 29.872 | 14.859 | 159.993 | 5 | 23 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1066 | 0.1429 | 0.1667 | 74.749 | 24.874 | 8.530 | 12 | 24 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.1572 | 0.4275 | 0.4444 | 30.526 | 14.406 | 79.740 | 9 | 17 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.1685 | 0.3333 | 1.0000 | 59.410 | 13.073 | 12.186 | 1 | 15 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3428 | 0.1333 | 0.5000 | 59.334 | 8.525 | 5.172 | 2 | 13 |
| sub-1 | mixed | Philips|Intera | 0.5081 | 0.3478 | 0.5000 | 42.070 | 179.640 | 110.442 | 4 | 15 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6115 | 0.1667 | 0.5000 | 30.659 | 27.886 | 34.332 | 2 | 10 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6336 | 0.2449 | 0.2857 | 22.406 | 64.230 | 82.898 | 7 | 14 |