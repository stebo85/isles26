# Our SynthStroke (isles26_synth_k_dwi) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2409 | 0.0730 | 0.2722 | 0.0000 | 0.6796 |
| iou | 12 | 0.1676 | 0.0380 | 0.1990 | 0.0000 | 0.5147 |
| sensitivity | 12 | 0.2149 | 0.0398 | 0.2509 | 0.0000 | 0.5951 |
| precision | 11 | 0.3745 | 0.4740 | 0.3013 | 0.0000 | 0.7921 |
| specificity | 12 | 0.9983 | 0.9994 | 0.0027 | 0.9900 | 1.0000 |
| hd95_mm | 11 | 44.9881 | 41.9652 | 21.1848 | 12.1774 | 84.8644 |
| assd_mm | 11 | 21.5780 | 15.7099 | 17.7076 | 3.1095 | 61.3470 |
| surface_dice_3mm | 11 | 0.2901 | 0.1641 | 0.2868 | 0.0000 | 0.7324 |
| abs_volume_diff_ml | 12 | 24.8901 | 7.6798 | 42.8615 | 0.1945 | 152.4862 |
| rel_abs_volume_diff | 12 | 4.7380 | 0.2847 | 14.1653 | 0.0617 | 51.7049 |
| lesion_f1 | 12 | 0.1959 | 0.1511 | 0.2069 | 0.0000 | 0.5714 |
| lesion_precision | 12 | 0.2554 | 0.1548 | 0.2884 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.2562 | 0.1845 | 0.2633 | 0.0000 | 0.7500 |
| abs_lesion_count_diff | 12 | 5.9167 | 4.0000 | 6.4738 | 1.0000 | 26.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 5 | 0.455 |
| large_1k-10k | 9 | 6 | 0.667 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2948 | 0.2948 | 0.4898 | 0.5750 | 49.958 | 79.652 |
| acute | 9 | 0.2028 | 0.0000 | 0.1246 | 0.1583 | 46.815 | 15.406 |
| chronic | 1 | 0.4760 | 0.4760 | 0.2500 | 0.5000 | 20.436 | 0.726 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2696 | 0.2857 | 63.415 | 3.507 |
| Philips|Ingenia Ambition X | 6 | 0.1583 | 0.1427 | 46.399 | 18.200 |
| Philips|Ingenia Elition X | 2 | 0.6758 | 0.2575 | 15.848 | 14.578 |
| Philips|Achieva | 2 | 0.0253 | 0.2041 | 57.951 | 76.657 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 51.490 | 1.020 | 12.186 | 1 | 5 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 84.864 | 1.276 | 1.081 | 3 | 5 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 67.748 | 23.586 | 0.448 | 1 | 27 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 41.704 | 3.852 | 3.228 | 1 | 7 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-3 | mixed | Philips|Achieva | 0.0505 | 0.4082 | 0.4000 | 57.951 | 7.507 | 159.993 | 5 | 12 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0955 | 0.5263 | 0.5556 | 37.427 | 8.931 | 79.740 | 9 | 12 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.3783 | 0.0800 | 0.0833 | 59.588 | 5.795 | 8.530 | 12 | 13 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4760 | 0.2500 | 0.5000 | 20.436 | 4.446 | 5.172 | 2 | 6 |
| sub-1 | mixed | Philips|Intera | 0.5392 | 0.5714 | 0.7500 | 41.965 | 117.260 | 110.442 | 4 | 13 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.6721 | 0.2222 | 0.5000 | 12.177 | 25.791 | 34.332 | 2 | 7 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6796 | 0.2927 | 0.2857 | 19.518 | 62.283 | 82.898 | 7 | 10 |