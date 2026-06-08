# Our SynthStroke (isles26_synth_i_tversky_a08) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0902 | 0.0153 | 0.1221 | 0.0000 | 0.3530 |
| iou | 12 | 0.0519 | 0.0077 | 0.0726 | 0.0000 | 0.2143 |
| sensitivity | 12 | 0.0926 | 0.0324 | 0.1185 | 0.0000 | 0.3333 |
| precision | 12 | 0.1151 | 0.0209 | 0.1516 | 0.0000 | 0.3828 |
| specificity | 12 | 0.9938 | 0.9946 | 0.0032 | 0.9841 | 0.9974 |
| hd95_mm | 12 | 88.6505 | 86.4898 | 27.4934 | 36.7105 | 132.0816 |
| assd_mm | 12 | 49.7169 | 51.1614 | 26.2591 | 13.5816 | 97.4856 |
| surface_dice_3mm | 12 | 0.0882 | 0.0291 | 0.1081 | 0.0000 | 0.2961 |
| abs_volume_diff_ml | 12 | 43.7488 | 32.1396 | 33.7974 | 4.2318 | 105.3206 |
| rel_abs_volume_diff | 12 | 17.0486 | 2.1755 | 28.7148 | 0.1115 | 102.1148 |
| lesion_f1 | 12 | 0.1192 | 0.1000 | 0.1136 | 0.0000 | 0.3636 |
| lesion_precision | 12 | 0.0875 | 0.0764 | 0.0873 | 0.0000 | 0.2857 |
| lesion_recall | 12 | 0.2909 | 0.2540 | 0.2933 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 27.0833 | 28.5000 | 11.6007 | 5.0000 | 52.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 2 | 0.182 |
| large_1k-10k | 9 | 6 | 0.667 |
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
| mixed | 2 | 0.1639 | 0.1639 | 0.3152 | 0.4500 | 52.429 | 88.324 |
| acute | 9 | 0.0814 | 0.0011 | 0.0762 | 0.2324 | 93.659 | 36.714 |
| chronic | 1 | 0.0218 | 0.0218 | 0.1143 | 0.5000 | 116.018 | 17.915 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.0627 | 0.1818 | 91.045 | 56.746 |
| Philips|Ingenia Ambition X | 6 | 0.0170 | 0.0731 | 97.062 | 46.038 |
| Philips|Ingenia Elition X | 2 | 0.3263 | 0.1807 | 65.276 | 6.738 |
| Philips|Achieva | 2 | 0.1012 | 0.1333 | 84.396 | 60.895 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 113.942 | 40.449 | 1.081 | 3 | 30 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 92.621 | 46.146 | 0.448 | 1 | 31 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 80.359 | 108.549 | 3.228 | 1 | 53 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 132.082 | 20.094 | 0.828 | 1 | 15 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0011 | 0.0645 | 1.0000 | 113.909 | 29.617 | 12.186 | 1 | 30 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0089 | 0.1739 | 0.2222 | 73.587 | 14.789 | 79.740 | 9 | 14 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0218 | 0.1143 | 0.5000 | 116.018 | 23.087 | 5.172 | 2 | 31 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0702 | 0.0857 | 0.0833 | 105.880 | 33.441 | 8.530 | 12 | 34 |
| sub-1 | mixed | Philips|Intera | 0.1255 | 0.3636 | 0.5000 | 68.148 | 36.318 | 110.442 | 4 | 21 |
| sub-3 | mixed | Philips|Achieva | 0.2023 | 0.2667 | 0.4000 | 36.710 | 57.469 | 159.993 | 5 | 45 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.2995 | 0.2105 | 0.5000 | 78.289 | 38.564 | 34.332 | 2 | 30 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.3530 | 0.1509 | 0.2857 | 52.263 | 73.654 | 82.898 | 7 | 39 |