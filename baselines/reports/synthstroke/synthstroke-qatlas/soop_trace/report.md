# SynthStroke synthstroke-qatlas on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0523 | 0.0011 | 0.1519 | 0.0000 | 0.5547 |
| iou | 12 | 0.0351 | 0.0005 | 0.1053 | 0.0000 | 0.3838 |
| sensitivity | 12 | 0.0784 | 0.0011 | 0.1637 | 0.0000 | 0.5744 |
| precision | 12 | 0.0525 | 0.0010 | 0.1466 | 0.0000 | 0.5363 |
| specificity | 12 | 0.9907 | 0.9901 | 0.0045 | 0.9825 | 0.9979 |
| hd95_mm | 12 | 63.0592 | 61.3973 | 14.7829 | 35.0117 | 84.1690 |
| assd_mm | 12 | 36.3582 | 38.0893 | 10.5854 | 9.7977 | 48.3782 |
| surface_dice_3mm | 12 | 0.0426 | 0.0061 | 0.0826 | 0.0000 | 0.3031 |
| abs_volume_diff_ml | 12 | 48.0380 | 35.3940 | 40.6666 | 4.3933 | 147.0861 |
| rel_abs_volume_diff | 12 | 22.3337 | 5.0611 | 39.4190 | 0.0710 | 137.1639 |
| lesion_f1 | 12 | 0.1871 | 0.0769 | 0.2144 | 0.0000 | 0.5714 |
| lesion_precision | 12 | 0.1785 | 0.0833 | 0.2013 | 0.0000 | 0.5000 |
| lesion_recall | 12 | 0.2188 | 0.0714 | 0.2929 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 3.2500 | 3.5000 | 2.3139 | 0.0000 | 7.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 2 | 0.222 |
| huge_>=10k | 4 | 3 | 0.750 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 2 | 0.500 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2812 | 0.2812 | 0.3333 | 0.3250 | 50.649 | 77.464 |
| acute | 9 | 0.0050 | 0.0000 | 0.1198 | 0.1640 | 64.631 | 35.781 |
| chronic | 1 | 0.0200 | 0.0200 | 0.5000 | 0.5000 | 73.737 | 99.496 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2774 | 0.1111 | 48.473 | 14.212 |
| Philips|Ingenia Ambition X | 6 | 0.0105 | 0.2374 | 66.031 | 54.472 |
| Philips|Ingenia Elition X | 2 | 0.0011 | 0.0769 | 57.296 | 8.026 |
| Philips|Achieva | 2 | 0.0039 | 0.2222 | 74.493 | 102.575 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 61.934 | 21.664 | 1.081 | 3 | 9 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 60.861 | 61.831 | 0.448 | 1 | 7 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 50.977 | 38.054 | 3.228 | 1 | 5 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 82.699 | 58.893 | 0.828 | 1 | 4 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 81.086 | 44.492 | 8.530 | 12 | 5 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 60.695 | 29.939 | 34.332 | 2 | 7 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0021 | 0.1538 | 0.1429 | 53.897 | 94.557 | 82.898 | 7 | 6 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0071 | 0.3529 | 0.3333 | 45.356 | 56.547 | 79.740 | 9 | 8 |
| sub-3 | mixed | Philips|Achieva | 0.0077 | 0.4444 | 0.4000 | 66.286 | 12.907 | 159.993 | 5 | 6 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0200 | 0.5000 | 0.5000 | 73.737 | 104.668 | 5.172 | 2 | 2 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0358 | 0.5714 | 1.0000 | 84.169 | 84.155 | 12.186 | 1 | 5 |
| sub-1 | mixed | Philips|Intera | 0.5547 | 0.2222 | 0.2500 | 35.012 | 118.283 | 110.442 | 4 | 5 |