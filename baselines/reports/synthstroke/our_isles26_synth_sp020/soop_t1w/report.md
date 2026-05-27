# Our SynthStroke (isles26_synth_sp020) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1363 | 0.0556 | 0.1794 | 0.0000 | 0.5853 |
| iou | 12 | 0.0848 | 0.0289 | 0.1220 | 0.0000 | 0.4137 |
| sensitivity | 12 | 0.1443 | 0.0375 | 0.1750 | 0.0000 | 0.5292 |
| precision | 12 | 0.2143 | 0.0905 | 0.2873 | 0.0000 | 0.8628 |
| specificity | 12 | 0.9968 | 0.9965 | 0.0020 | 0.9933 | 0.9995 |
| hd95_mm | 12 | 80.6011 | 79.4350 | 20.4685 | 44.1170 | 117.4985 |
| assd_mm | 12 | 41.8173 | 40.6820 | 23.8499 | 9.5115 | 84.1177 |
| surface_dice_3mm | 12 | 0.1591 | 0.0891 | 0.1910 | 0.0000 | 0.6231 |
| abs_volume_diff_ml | 12 | 39.8259 | 28.2730 | 40.4228 | 3.0948 | 147.2556 |
| rel_abs_volume_diff | 12 | 10.1873 | 1.6437 | 24.6845 | 0.1919 | 91.2787 |
| lesion_f1 | 12 | 0.1468 | 0.1356 | 0.1282 | 0.0000 | 0.3333 |
| lesion_precision | 12 | 0.1331 | 0.1111 | 0.1435 | 0.0000 | 0.5000 |
| lesion_recall | 12 | 0.2767 | 0.1964 | 0.2928 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 6.2500 | 6.5000 | 3.0311 | 1.0000 | 11.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 4 | 0.444 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.1414 | 0.1414 | 0.3288 | 0.3250 | 62.754 | 118.916 |
| acute | 9 | 0.1373 | 0.0091 | 0.1025 | 0.2412 | 84.948 | 24.632 |
| chronic | 1 | 0.1169 | 0.1169 | 0.1818 | 0.5000 | 77.178 | 18.392 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.1315 | 0.1667 | 77.062 | 47.531 |
| Philips|Ingenia Ambition X | 6 | 0.0639 | 0.1388 | 84.913 | 31.293 |
| Philips|Ingenia Elition X | 2 | 0.4844 | 0.1356 | 67.074 | 22.371 |
| Philips|Achieva | 2 | 0.0099 | 0.1622 | 84.731 | 75.175 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 95.501 | 5.568 | 1.081 | 3 | 4 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 86.127 | 41.296 | 0.448 | 1 | 9 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 81.692 | 45.757 | 3.228 | 1 | 12 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 102.577 | 3.923 | 0.828 | 1 | 6 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0091 | 0.2222 | 1.0000 | 117.498 | 26.484 | 12.186 | 1 | 8 |
| sub-3 | mixed | Philips|Achieva | 0.0198 | 0.3243 | 0.4000 | 66.884 | 12.737 | 159.993 | 5 | 11 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0914 | 0.3333 | 0.4444 | 44.117 | 26.084 | 79.740 | 9 | 15 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.1169 | 0.1818 | 0.5000 | 77.178 | 23.564 | 5.172 | 2 | 9 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1662 | 0.0952 | 0.0833 | 102.867 | 26.561 | 8.530 | 12 | 9 |
| sub-1 | mixed | Philips|Intera | 0.2631 | 0.3333 | 0.2500 | 58.624 | 19.865 | 110.442 | 4 | 2 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.3834 | 0.1379 | 0.1429 | 72.256 | 44.744 | 82.898 | 7 | 15 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.5853 | 0.1333 | 0.5000 | 61.893 | 27.744 | 34.332 | 2 | 13 |