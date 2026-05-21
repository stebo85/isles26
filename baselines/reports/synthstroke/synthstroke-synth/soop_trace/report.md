# SynthStroke synthstroke-synth on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2613 | 0.1499 | 0.2766 | 0.0000 | 0.8220 |
| iou | 12 | 0.1849 | 0.0811 | 0.2201 | 0.0000 | 0.6979 |
| sensitivity | 12 | 0.3454 | 0.3176 | 0.3264 | 0.0000 | 0.8402 |
| precision | 12 | 0.2490 | 0.0883 | 0.2770 | 0.0000 | 0.8047 |
| specificity | 12 | 0.9958 | 0.9952 | 0.0025 | 0.9919 | 0.9995 |
| hd95_mm | 12 | 70.9027 | 70.0834 | 22.2403 | 37.9009 | 105.5529 |
| assd_mm | 12 | 34.9637 | 35.2101 | 21.5825 | 4.9680 | 72.8169 |
| surface_dice_3mm | 12 | 0.2609 | 0.2065 | 0.2500 | 0.0000 | 0.7664 |
| abs_volume_diff_ml | 12 | 15.3157 | 8.9590 | 11.8322 | 0.8877 | 34.9276 |
| rel_abs_volume_diff | 12 | 7.6594 | 0.7086 | 15.3242 | 0.0442 | 53.6066 |
| lesion_f1 | 12 | 0.1276 | 0.1484 | 0.1060 | 0.0000 | 0.3544 |
| lesion_precision | 12 | 0.1363 | 0.1056 | 0.1376 | 0.0000 | 0.4667 |
| lesion_recall | 12 | 0.2442 | 0.1556 | 0.2869 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 6.0000 | 6.5000 | 2.9439 | 1.0000 | 10.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 3 | 0.333 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.7483 | 0.7483 | 0.1944 | 0.2250 | 42.219 | 6.694 |
| acute | 9 | 0.1450 | 0.0612 | 0.1111 | 0.2200 | 78.282 | 18.835 |
| chronic | 1 | 0.3336 | 0.3336 | 0.1429 | 0.5000 | 61.853 | 0.888 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.3373 | 0.0833 | 67.921 | 16.125 |
| Philips|Ingenia Ambition X | 6 | 0.0883 | 0.0703 | 74.726 | 15.580 |
| Philips|Ingenia Elition X | 2 | 0.4721 | 0.2605 | 61.591 | 22.235 |
| Philips|Achieva | 2 | 0.4933 | 0.2111 | 71.727 | 6.795 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 102.072 | 7.520 | 12.186 | 1 | 11 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 89.306 | 27.008 | 1.081 | 3 | 8 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 81.987 | 24.437 | 0.448 | 1 | 7 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 54.183 | 7.644 | 3.228 | 1 | 8 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0612 | 0.1538 | 0.1111 | 58.252 | 44.812 | 79.740 | 9 | 8 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1352 | 0.1250 | 0.0833 | 90.006 | 33.121 | 8.530 | 12 | 8 |
| sub-5 | acute | Philips|Achieva | 0.1646 | 0.2000 | 1.0000 | 105.553 | 7.352 | 0.828 | 1 | 9 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3336 | 0.1429 | 0.5000 | 61.853 | 4.284 | 5.172 | 2 | 12 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.4567 | 0.3544 | 0.2857 | 44.869 | 72.046 | 82.898 | 7 | 15 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.4876 | 0.1667 | 0.5000 | 78.314 | 67.950 | 34.332 | 2 | 10 |
| sub-1 | mixed | Philips|Intera | 0.6745 | 0.1667 | 0.2500 | 46.536 | 104.119 | 110.442 | 4 | 8 |
| sub-3 | mixed | Philips|Achieva | 0.8220 | 0.2222 | 0.2000 | 37.901 | 167.059 | 159.993 | 5 | 4 |