# Our SynthStroke (isles26_synth_f_sp100) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1297 | 0.0808 | 0.1627 | 0.0000 | 0.5152 |
| iou | 12 | 0.0787 | 0.0422 | 0.1080 | 0.0000 | 0.3470 |
| sensitivity | 12 | 0.1866 | 0.1501 | 0.1988 | 0.0000 | 0.6973 |
| precision | 12 | 0.1359 | 0.0532 | 0.1620 | 0.0000 | 0.4643 |
| specificity | 12 | 0.9942 | 0.9946 | 0.0047 | 0.9800 | 0.9979 |
| hd95_mm | 12 | 54.2137 | 50.0325 | 14.2361 | 37.2367 | 77.7476 |
| assd_mm | 12 | 25.5882 | 26.8662 | 11.4958 | 8.4960 | 43.4756 |
| surface_dice_3mm | 12 | 0.1502 | 0.1327 | 0.1449 | 0.0000 | 0.4793 |
| abs_volume_diff_ml | 12 | 39.0546 | 30.9939 | 31.5572 | 5.4501 | 116.3128 |
| rel_abs_volume_diff | 12 | 11.9277 | 2.1787 | 25.4176 | 0.1587 | 94.5574 |
| lesion_f1 | 12 | 0.3355 | 0.3431 | 0.2408 | 0.0000 | 0.7059 |
| lesion_precision | 12 | 0.3597 | 0.2500 | 0.3220 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.4325 | 0.4365 | 0.3400 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 2.9167 | 3.0000 | 1.4977 | 0.0000 | 5.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 1 | 0.200 |
| small_10-100 | 19 | 1 | 0.053 |
| medium_100-1k | 11 | 6 | 0.545 |
| large_1k-10k | 9 | 6 | 0.667 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 1 | 0.091 |
| small_0.1-1mL | 18 | 3 | 0.167 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.3222 | 0.3222 | 0.6387 | 0.5750 | 41.025 | 97.201 |
| acute | 9 | 0.0908 | 0.0324 | 0.2498 | 0.3933 | 58.866 | 29.425 |
| chronic | 1 | 0.0950 | 0.0950 | 0.5000 | 0.5000 | 38.724 | 9.427 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2576 | 0.3529 | 59.978 | 45.015 |
| Philips|Ingenia Ambition X | 6 | 0.0460 | 0.2527 | 54.307 | 35.879 |
| Philips|Ingenia Elition X | 2 | 0.3021 | 0.4492 | 45.314 | 16.883 |
| Philips|Achieva | 2 | 0.0808 | 0.4524 | 57.070 | 64.791 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 77.748 | 13.023 | 1.081 | 3 | 8 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 58.867 | 42.764 | 0.448 | 1 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 46.674 | 37.870 | 3.228 | 1 | 3 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0123 | 0.5581 | 0.4444 | 44.418 | 12.838 | 79.740 | 9 | 4 |
| sub-5 | acute | Philips|Achieva | 0.0324 | 0.3333 | 1.0000 | 74.297 | 14.098 | 0.828 | 1 | 5 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0666 | 0.3333 | 1.0000 | 66.567 | 43.072 | 12.186 | 1 | 5 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0950 | 0.5000 | 0.5000 | 38.724 | 14.599 | 5.172 | 2 | 2 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1018 | 0.1250 | 0.1667 | 70.592 | 39.631 | 8.530 | 12 | 10 |
| sub-3 | mixed | Philips|Achieva | 0.1292 | 0.5714 | 0.4000 | 39.842 | 43.680 | 159.993 | 5 | 3 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.1799 | 0.3529 | 0.4286 | 53.391 | 54.581 | 82.898 | 7 | 10 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.4243 | 0.5455 | 0.5000 | 37.237 | 28.882 | 34.332 | 2 | 5 |
| sub-1 | mixed | Philips|Intera | 0.5152 | 0.7059 | 0.7500 | 42.209 | 188.531 | 110.442 | 4 | 3 |