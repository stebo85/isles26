# Our SynthStroke (isles26_synth_h_tversky) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1457 | 0.0870 | 0.1549 | 0.0000 | 0.4118 |
| iou | 12 | 0.0866 | 0.0460 | 0.0956 | 0.0000 | 0.2593 |
| sensitivity | 12 | 0.2013 | 0.1856 | 0.1937 | 0.0000 | 0.4939 |
| precision | 12 | 0.1489 | 0.0498 | 0.1714 | 0.0000 | 0.4780 |
| specificity | 12 | 0.9891 | 0.9905 | 0.0060 | 0.9704 | 0.9934 |
| hd95_mm | 12 | 82.3129 | 81.9798 | 20.8719 | 54.9457 | 118.9521 |
| assd_mm | 12 | 42.7783 | 41.9516 | 23.1698 | 10.6955 | 76.6823 |
| surface_dice_3mm | 12 | 0.1430 | 0.1016 | 0.1408 | 0.0000 | 0.3686 |
| abs_volume_diff_ml | 12 | 58.1038 | 45.7999 | 44.5548 | 23.9315 | 198.5875 |
| rel_abs_volume_diff | 12 | 28.6477 | 3.9447 | 44.1735 | 0.2702 | 151.4918 |
| lesion_f1 | 12 | 0.1229 | 0.0755 | 0.1286 | 0.0000 | 0.3967 |
| lesion_precision | 12 | 0.0820 | 0.0542 | 0.0904 | 0.0000 | 0.2963 |
| lesion_recall | 12 | 0.3470 | 0.3651 | 0.3246 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 25.7500 | 26.5000 | 10.4970 | 3.0000 | 46.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 8 | 0.889 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 8 | 0.727 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.3317 | 0.3317 | 0.3614 | 0.6750 | 59.680 | 47.331 |
| acute | 9 | 0.1157 | 0.0070 | 0.0766 | 0.2571 | 87.731 | 61.724 |
| chronic | 1 | 0.0445 | 0.0445 | 0.0625 | 0.5000 | 78.819 | 47.070 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.1301 | 0.1630 | 84.243 | 61.859 |
| Philips|Ingenia Ambition X | 6 | 0.0650 | 0.0898 | 87.359 | 69.830 |
| Philips|Ingenia Elition X | 2 | 0.3477 | 0.1063 | 64.627 | 37.413 |
| Philips|Achieva | 2 | 0.2017 | 0.1983 | 82.930 | 39.861 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 110.436 | 73.368 | 1.081 | 3 | 31 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 85.140 | 68.243 | 0.448 | 1 | 32 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 85.799 | 201.816 | 3.228 | 1 | 26 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 104.551 | 37.321 | 0.828 | 1 | 15 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0070 | 0.1765 | 1.0000 | 118.952 | 44.708 | 12.186 | 1 | 31 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0445 | 0.0625 | 0.5000 | 78.819 | 52.242 | 5.172 | 2 | 30 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1295 | 0.0741 | 0.0833 | 96.192 | 53.059 | 8.530 | 12 | 15 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.2090 | 0.2260 | 0.4444 | 59.251 | 51.266 | 79.740 | 9 | 33 |
| sub-1 | mixed | Philips|Intera | 0.2601 | 0.3261 | 0.7500 | 58.050 | 59.010 | 110.442 | 4 | 24 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.2837 | 0.0769 | 0.5000 | 74.308 | 85.226 | 34.332 | 2 | 48 |
| sub-3 | mixed | Philips|Achieva | 0.4033 | 0.3967 | 0.6000 | 61.309 | 116.763 | 159.993 | 5 | 27 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.4118 | 0.1356 | 0.2857 | 54.946 | 106.830 | 82.898 | 7 | 45 |