# SynthStroke synthstroke-qsynth on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1645 | 0.0115 | 0.2259 | 0.0000 | 0.6126 |
| iou | 12 | 0.1086 | 0.0058 | 0.1551 | 0.0000 | 0.4415 |
| sensitivity | 12 | 0.1272 | 0.0058 | 0.1847 | 0.0000 | 0.5066 |
| precision | 8 | 0.5451 | 0.6607 | 0.3475 | 0.0000 | 0.9872 |
| specificity | 12 | 0.9992 | 0.9999 | 0.0019 | 0.9929 | 1.0000 |
| hd95_mm | 8 | 52.5157 | 52.3790 | 28.3573 | 13.1771 | 101.8519 |
| assd_mm | 8 | 27.7635 | 13.1153 | 26.3572 | 3.4435 | 82.2578 |
| surface_dice_3mm | 8 | 0.2650 | 0.2132 | 0.2496 | 0.0000 | 0.6350 |
| abs_volume_diff_ml | 12 | 30.7242 | 9.8404 | 45.6502 | 0.8282 | 158.1042 |
| rel_abs_volume_diff | 12 | 1.6082 | 0.9933 | 2.7019 | 0.2051 | 10.5246 |
| lesion_f1 | 12 | 0.2865 | 0.1220 | 0.3119 | 0.0000 | 0.7500 |
| lesion_precision | 12 | 0.7444 | 1.0000 | 0.3713 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.2536 | 0.0714 | 0.3108 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 3.1667 | 1.5000 | 3.2102 | 1.0000 | 12.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 2 | 0.222 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 2 | 0.500 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2538 | 0.2538 | 0.5556 | 0.4500 | 62.003 | 90.379 |
| acute | 9 | 0.1093 | 0.0000 | 0.1845 | 0.1825 | 51.952 | 20.567 |
| chronic | 1 | 0.4824 | 0.4824 | 0.6667 | 0.5000 | 36.358 | 2.832 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2423 | 0.3333 | 43.792 | 11.868 |
| Philips|Ingenia Ambition X | 6 | 0.0928 | 0.2361 | 54.538 | 18.343 |
| Philips|Ingenia Elition X | 2 | 0.4547 | 0.4553 | 38.985 | 37.982 |
| Philips|Achieva | 2 | 0.0115 | 0.2222 | 80.214 | 79.466 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 1.081 | 3 | 0 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 60.966 | 5.157 | 0.448 | 1 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 8.530 | 12 | 0 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 101.852 | 0.132 | 79.740 | 9 | 2 |
| sub-3 | mixed | Philips|Achieva | 0.0230 | 0.4444 | 0.4000 | 80.214 | 1.889 | 159.993 | 5 | 4 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0744 | 0.7500 | 1.0000 | 18.974 | 1.034 | 12.186 | 1 | 5 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.2969 | 0.2439 | 0.1429 | 64.793 | 18.815 | 82.898 | 7 | 6 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4824 | 0.6667 | 0.5000 | 36.358 | 2.340 | 5.172 | 2 | 1 |
| sub-1 | mixed | Philips|Intera | 0.4846 | 0.6667 | 0.5000 | 43.792 | 87.787 | 110.442 | 4 | 2 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6126 | 0.6667 | 0.5000 | 13.177 | 22.453 | 34.332 | 2 | 1 |