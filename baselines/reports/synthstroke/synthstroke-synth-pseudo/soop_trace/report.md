# SynthStroke synthstroke-synth-pseudo on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2616 | 0.1584 | 0.2665 | 0.0000 | 0.7906 |
| iou | 12 | 0.1815 | 0.0863 | 0.2047 | 0.0000 | 0.6537 |
| sensitivity | 12 | 0.4135 | 0.4768 | 0.3561 | 0.0000 | 0.8955 |
| precision | 12 | 0.2280 | 0.1285 | 0.2381 | 0.0000 | 0.7077 |
| specificity | 12 | 0.9930 | 0.9929 | 0.0042 | 0.9836 | 0.9996 |
| hd95_mm | 12 | 64.9341 | 66.2243 | 23.4271 | 16.2699 | 96.5476 |
| assd_mm | 12 | 33.7207 | 29.1345 | 21.9483 | 3.4857 | 70.2112 |
| surface_dice_3mm | 12 | 0.2452 | 0.1633 | 0.2329 | 0.0000 | 0.6847 |
| abs_volume_diff_ml | 12 | 28.7746 | 25.2861 | 17.5207 | 1.3499 | 58.7355 |
| rel_abs_volume_diff | 12 | 11.2965 | 0.9166 | 19.2139 | 0.2313 | 65.3770 |
| lesion_f1 | 12 | 0.1666 | 0.1822 | 0.1430 | 0.0000 | 0.4286 |
| lesion_precision | 12 | 0.1376 | 0.1080 | 0.1321 | 0.0000 | 0.4286 |
| lesion_recall | 12 | 0.3126 | 0.2500 | 0.2952 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 6.0833 | 7.0000 | 3.9678 | 0.0000 | 12.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 4 | 0.211 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 3 | 0.333 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 5 | 0.278 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.6745 | 0.6745 | 0.3185 | 0.4250 | 33.207 | 50.600 |
| acute | 9 | 0.1592 | 0.0688 | 0.1343 | 0.2668 | 73.432 | 26.972 |
| chronic | 1 | 0.3573 | 0.3573 | 0.1538 | 0.5000 | 51.907 | 1.350 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2792 | 0.1250 | 68.162 | 48.150 |
| Philips|Ingenia Ambition X | 6 | 0.1124 | 0.0978 | 68.476 | 20.648 |
| Philips|Ingenia Elition X | 2 | 0.5234 | 0.2768 | 59.605 | 32.997 |
| Philips|Achieva | 2 | 0.4297 | 0.3047 | 56.409 | 29.555 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 86.143 | 15.869 | 12.186 | 1 | 10 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 86.179 | 38.647 | 1.081 | 3 | 10 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 80.086 | 29.705 | 0.448 | 1 | 12 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 54.801 | 23.242 | 3.228 | 1 | 9 |
| sub-5 | acute | Philips|Achieva | 0.0688 | 0.2222 | 1.0000 | 96.548 | 17.474 | 0.828 | 1 | 8 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.1313 | 0.2105 | 0.2222 | 48.271 | 61.294 | 79.740 | 9 | 15 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1856 | 0.2222 | 0.2500 | 89.650 | 59.670 | 8.530 | 12 | 10 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3573 | 0.1538 | 0.5000 | 51.907 | 3.822 | 5.172 | 2 | 11 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.4769 | 0.1250 | 0.5000 | 77.648 | 79.012 | 34.332 | 2 | 14 |
| sub-1 | mixed | Philips|Intera | 0.5584 | 0.2500 | 0.2500 | 50.144 | 169.177 | 110.442 | 4 | 4 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.5698 | 0.4286 | 0.4286 | 41.561 | 104.213 | 82.898 | 7 | 7 |
| sub-3 | mixed | Philips|Achieva | 0.7906 | 0.3871 | 0.6000 | 16.270 | 202.458 | 159.993 | 5 | 7 |