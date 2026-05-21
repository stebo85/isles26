# SynthStroke synthstroke-synth on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1452 | 0.0000 | 0.2103 | 0.0000 | 0.5840 |
| iou | 12 | 0.0941 | 0.0000 | 0.1400 | 0.0000 | 0.4124 |
| sensitivity | 12 | 0.1047 | 0.0000 | 0.1618 | 0.0000 | 0.5049 |
| precision | 8 | 0.6367 | 0.7852 | 0.3786 | 0.0000 | 1.0000 |
| specificity | 12 | 0.9996 | 1.0000 | 0.0012 | 0.9955 | 1.0000 |
| hd95_mm | 8 | 62.6307 | 66.1122 | 27.7918 | 13.4117 | 93.8341 |
| assd_mm | 8 | 36.3865 | 30.6797 | 29.4361 | 5.0638 | 84.9770 |
| surface_dice_3mm | 8 | 0.2300 | 0.2085 | 0.2279 | 0.0000 | 0.5535 |
| abs_volume_diff_ml | 12 | 32.2328 | 10.2696 | 45.7052 | 0.4475 | 159.9882 |
| rel_abs_volume_diff | 12 | 0.8624 | 0.9886 | 0.2199 | 0.2708 | 1.0000 |
| lesion_f1 | 12 | 0.2217 | 0.0909 | 0.2626 | 0.0000 | 0.6667 |
| lesion_precision | 12 | 0.7431 | 1.0000 | 0.3687 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1628 | 0.0556 | 0.2051 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 2.8333 | 1.0000 | 3.2872 | 1.0000 | 12.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 1 | 0.111 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2920 | 0.2920 | 0.4524 | 0.3500 | 69.354 | 94.950 |
| acute | 9 | 0.0796 | 0.0000 | 0.1209 | 0.0838 | 65.031 | 21.533 |
| chronic | 1 | 0.4413 | 0.4413 | 0.6667 | 0.5000 | 37.183 | 3.096 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2920 | 0.2857 | 69.737 | 15.485 |
| Philips|Ingenia Ambition X | 6 | 0.0757 | 0.1414 | 67.493 | 17.732 |
| Philips|Ingenia Elition X | 2 | 0.3519 | 0.4533 | 33.012 | 44.308 |
| Philips|Achieva | 2 | 0.0000 | 0.1667 | 93.069 | 80.408 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 79.612 | 0.176 | 12.186 | 1 | 2 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 93.834 | 0.023 | 1.081 | 3 | 2 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.448 | 1 | 0 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 8.530 | 12 | 0 |
| sub-3 | mixed | Philips|Achieva | 0.0001 | 0.3333 | 0.2000 | 93.069 | 0.005 | 159.993 | 5 | 1 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0131 | 0.1818 | 0.1111 | 85.684 | 0.659 | 79.740 | 9 | 2 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.3422 | 0.6667 | 0.5000 | 13.412 | 7.702 | 34.332 | 2 | 3 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.3616 | 0.2400 | 0.1429 | 52.612 | 20.913 | 82.898 | 7 | 4 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4413 | 0.6667 | 0.5000 | 37.183 | 2.076 | 5.172 | 2 | 1 |
| sub-1 | mixed | Philips|Intera | 0.5840 | 0.5714 | 0.5000 | 45.640 | 80.531 | 110.442 | 4 | 3 |