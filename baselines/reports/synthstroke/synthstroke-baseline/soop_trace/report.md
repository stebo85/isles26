# SynthStroke synthstroke-baseline on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0438 | 0.0000 | 0.1448 | 0.0000 | 0.5239 |
| iou | 12 | 0.0296 | 0.0000 | 0.0981 | 0.0000 | 0.3549 |
| sensitivity | 12 | 0.0307 | 0.0000 | 0.1015 | 0.0000 | 0.3672 |
| precision | 12 | 0.0763 | 0.0000 | 0.2525 | 0.0000 | 0.9137 |
| specificity | 12 | 0.9971 | 0.9983 | 0.0026 | 0.9929 | 0.9999 |
| hd95_mm | 12 | 60.1290 | 56.9123 | 12.9224 | 43.1562 | 82.2517 |
| assd_mm | 12 | 35.6913 | 36.4872 | 13.1770 | 12.9046 | 52.0645 |
| surface_dice_3mm | 12 | 0.0387 | 0.0000 | 0.1040 | 0.0000 | 0.3783 |
| abs_volume_diff_ml | 12 | 40.2067 | 30.4974 | 42.4534 | 1.8854 | 157.7167 |
| rel_abs_volume_diff | 12 | 11.9529 | 0.9277 | 23.0251 | 0.0883 | 78.1803 |
| lesion_f1 | 12 | 0.0208 | 0.0000 | 0.0496 | 0.0000 | 0.1667 |
| lesion_precision | 12 | 0.0160 | 0.0000 | 0.0376 | 0.0000 | 0.1250 |
| lesion_recall | 12 | 0.0301 | 0.0000 | 0.0730 | 0.0000 | 0.2500 |
| abs_lesion_count_diff | 12 | 6.4167 | 5.5000 | 3.5227 | 1.0000 | 12.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 0 | 0.000 |
| large_1k-10k | 9 | 1 | 0.111 |
| huge_>=10k | 4 | 1 | 0.250 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 0 | 0.000 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 1 | 0.250 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2620 | 0.2620 | 0.0833 | 0.1250 | 63.427 | 111.886 |
| acute | 9 | 0.0001 | 0.0000 | 0.0093 | 0.0123 | 57.472 | 25.849 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 77.451 | 26.066 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2620 | 0.0833 | 58.372 | 39.771 |
| Philips|Ingenia Ambition X | 6 | 0.0002 | 0.0139 | 58.012 | 21.862 |
| Philips|Ingenia Elition X | 2 | 0.0000 | 0.0000 | 50.097 | 39.561 |
| Philips|Achieva | 2 | 0.0000 | 0.0000 | 78.270 | 96.323 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 43.156 | 2.098 | 12.186 | 1 | 2 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 77.451 | 31.238 | 5.172 | 2 | 7 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 64.178 | 14.568 | 1.081 | 3 | 12 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 71.359 | 35.435 | 0.448 | 1 | 12 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 54.479 | 6.807 | 82.898 | 7 | 10 |
| sub-3 | mixed | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 74.288 | 2.276 | 159.993 | 5 | 7 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 46.171 | 1.343 | 3.228 | 1 | 10 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 82.252 | 35.757 | 0.828 | 1 | 13 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 59.345 | 0.533 | 8.530 | 12 | 2 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.0000 | 0.0000 | 0.0000 | 45.715 | 31.302 | 34.332 | 2 | 7 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0011 | 0.0833 | 0.1111 | 50.589 | 29.590 | 79.740 | 9 | 15 |
| sub-1 | mixed | Philips|Intera | 0.5239 | 0.1667 | 0.2500 | 52.566 | 44.387 | 110.442 | 4 | 8 |