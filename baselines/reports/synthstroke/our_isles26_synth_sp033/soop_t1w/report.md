# Our SynthStroke (isles26_synth_sp033) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1379 | 0.0206 | 0.1927 | 0.0000 | 0.5218 |
| iou | 12 | 0.0872 | 0.0104 | 0.1269 | 0.0000 | 0.3530 |
| sensitivity | 12 | 0.1209 | 0.0160 | 0.1543 | 0.0000 | 0.3952 |
| precision | 12 | 0.2243 | 0.0402 | 0.3030 | 0.0000 | 0.7746 |
| specificity | 12 | 0.9980 | 0.9984 | 0.0016 | 0.9935 | 0.9998 |
| hd95_mm | 12 | 72.5085 | 77.2410 | 24.1146 | 40.6012 | 119.6638 |
| assd_mm | 12 | 35.8664 | 34.4702 | 19.5574 | 8.6532 | 68.1421 |
| surface_dice_3mm | 12 | 0.1568 | 0.0402 | 0.2033 | 0.0000 | 0.5605 |
| abs_volume_diff_ml | 12 | 34.2934 | 16.0409 | 41.7875 | 0.0436 | 152.6315 |
| rel_abs_volume_diff | 12 | 8.3521 | 0.8932 | 24.5358 | 0.0526 | 89.7049 |
| lesion_f1 | 12 | 0.1423 | 0.1357 | 0.1250 | 0.0000 | 0.3478 |
| lesion_precision | 12 | 0.1203 | 0.0851 | 0.1101 | 0.0000 | 0.3077 |
| lesion_recall | 12 | 0.2790 | 0.1825 | 0.2961 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 10.1667 | 10.5000 | 4.5795 | 1.0000 | 20.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 4 | 0.444 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2731 | 0.2731 | 0.3406 | 0.4500 | 44.375 | 103.115 |
| acute | 9 | 0.1024 | 0.0146 | 0.1010 | 0.2165 | 79.659 | 21.881 |
| chronic | 1 | 0.1867 | 0.1867 | 0.1176 | 0.5000 | 64.423 | 8.363 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2609 | 0.1667 | 74.236 | 27.918 |
| Philips|Ingenia Ambition X | 6 | 0.0434 | 0.0986 | 78.763 | 22.656 |
| Philips|Ingenia Elition X | 2 | 0.4242 | 0.2174 | 57.640 | 33.536 |
| Philips|Achieva | 2 | 0.0122 | 0.1739 | 66.885 | 76.338 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 104.478 | 3.319 | 1.081 | 3 | 13 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 80.413 | 40.592 | 0.448 | 1 | 21 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 80.457 | 8.525 | 3.228 | 1 | 10 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 89.016 | 0.872 | 0.828 | 1 | 6 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0146 | 0.2353 | 0.2222 | 47.818 | 13.365 | 79.740 | 9 | 8 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0168 | 0.1538 | 1.0000 | 119.664 | 15.832 | 12.186 | 1 | 12 |
| sub-3 | mixed | Philips|Achieva | 0.0245 | 0.3478 | 0.4000 | 44.754 | 7.362 | 159.993 | 5 | 13 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0421 | 0.0851 | 0.0833 | 79.802 | 20.640 | 8.530 | 12 | 23 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.1867 | 0.1176 | 0.5000 | 64.423 | 13.536 | 5.172 | 2 | 15 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.3812 | 0.2609 | 0.5000 | 74.680 | 14.362 | 34.332 | 2 | 17 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.4672 | 0.1739 | 0.1429 | 40.601 | 35.796 | 82.898 | 7 | 18 |
| sub-1 | mixed | Philips|Intera | 0.5218 | 0.3333 | 0.5000 | 43.995 | 56.842 | 110.442 | 4 | 12 |