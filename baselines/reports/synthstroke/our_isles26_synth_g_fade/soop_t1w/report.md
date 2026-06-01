# Our SynthStroke (isles26_synth_g_fade) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2255 | 0.1623 | 0.2389 | 0.0000 | 0.6446 |
| iou | 12 | 0.1501 | 0.0885 | 0.1722 | 0.0000 | 0.4755 |
| sensitivity | 12 | 0.2265 | 0.1705 | 0.2335 | 0.0000 | 0.6128 |
| precision | 12 | 0.3005 | 0.2578 | 0.2925 | 0.0000 | 0.7862 |
| specificity | 12 | 0.9973 | 0.9987 | 0.0037 | 0.9868 | 1.0000 |
| hd95_mm | 12 | 50.1809 | 49.9117 | 22.4299 | 13.4117 | 95.3014 |
| assd_mm | 12 | 23.4153 | 19.5694 | 16.1201 | 3.8278 | 57.8864 |
| surface_dice_3mm | 12 | 0.2522 | 0.2471 | 0.2446 | 0.0000 | 0.7045 |
| abs_volume_diff_ml | 12 | 29.8748 | 10.4225 | 42.3728 | 0.7119 | 153.2853 |
| rel_abs_volume_diff | 12 | 9.0639 | 0.8539 | 27.5463 | 0.0656 | 100.4098 |
| lesion_f1 | 12 | 0.1498 | 0.1193 | 0.1422 | 0.0000 | 0.4444 |
| lesion_precision | 12 | 0.1283 | 0.0718 | 0.1507 | 0.0000 | 0.5000 |
| lesion_recall | 12 | 0.3210 | 0.3095 | 0.3143 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 13.0833 | 15.0000 | 6.6012 | 1.0000 | 25.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 7 | 0.778 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 6 | 0.545 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2808 | 0.2808 | 0.3422 | 0.5750 | 40.542 | 92.049 |
| acute | 9 | 0.2019 | 0.1340 | 0.1119 | 0.2447 | 51.208 | 19.056 |
| chronic | 1 | 0.3269 | 0.3269 | 0.1053 | 0.5000 | 60.214 | 2.898 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2689 | 0.1200 | 72.171 | 15.907 |
| Philips|Ingenia Ambition X | 6 | 0.1484 | 0.1188 | 58.124 | 22.680 |
| Philips|Ingenia Elition X | 2 | 0.6270 | 0.2000 | 15.706 | 18.303 |
| Philips|Achieva | 2 | 0.0119 | 0.2222 | 38.836 | 76.999 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 95.301 | 2.083 | 1.081 | 3 | 11 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 68.572 | 45.383 | 0.448 | 1 | 26 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 50.783 | 8.371 | 3.228 | 1 | 19 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 45.630 | 0.116 | 0.828 | 1 | 2 |
| sub-3 | mixed | Philips|Achieva | 0.0238 | 0.4444 | 0.4000 | 32.043 | 6.708 | 159.993 | 5 | 20 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1340 | 0.0741 | 0.0833 | 71.508 | 23.206 | 8.530 | 12 | 30 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.1905 | 0.3333 | 0.3333 | 35.324 | 12.111 | 79.740 | 9 | 12 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.2390 | 0.2000 | 1.0000 | 62.343 | 11.386 | 12.186 | 1 | 18 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3269 | 0.1053 | 0.5000 | 60.214 | 8.070 | 5.172 | 2 | 17 |
| sub-1 | mixed | Philips|Intera | 0.5377 | 0.2400 | 0.7500 | 49.041 | 141.254 | 110.442 | 4 | 21 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6094 | 0.2667 | 0.2857 | 18.000 | 52.461 | 82.898 | 7 | 16 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6446 | 0.1333 | 0.5000 | 13.412 | 28.163 | 34.332 | 2 | 13 |