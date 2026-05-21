# SynthStroke synthstroke-synth-pseudo on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1798 | 0.0062 | 0.2193 | 0.0000 | 0.5428 |
| iou | 12 | 0.1163 | 0.0031 | 0.1453 | 0.0000 | 0.3725 |
| sensitivity | 12 | 0.1366 | 0.0031 | 0.1778 | 0.0000 | 0.5199 |
| precision | 11 | 0.3841 | 0.5000 | 0.3752 | 0.0000 | 1.0000 |
| specificity | 12 | 0.9992 | 0.9999 | 0.0021 | 0.9921 | 1.0000 |
| hd95_mm | 11 | 48.9703 | 54.2438 | 21.6453 | 6.2985 | 79.9784 |
| assd_mm | 11 | 29.7879 | 27.4961 | 22.8236 | 2.2405 | 62.2625 |
| surface_dice_3mm | 11 | 0.2340 | 0.0320 | 0.2631 | 0.0000 | 0.7238 |
| abs_volume_diff_ml | 12 | 28.5784 | 8.5872 | 45.1722 | 0.4108 | 157.9977 |
| rel_abs_volume_diff | 12 | 0.7644 | 0.8450 | 0.2725 | 0.0841 | 1.0000 |
| lesion_f1 | 12 | 0.2845 | 0.1071 | 0.3302 | 0.0000 | 1.0000 |
| lesion_precision | 12 | 0.4464 | 0.4286 | 0.4307 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.2536 | 0.0714 | 0.3108 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 2.0833 | 0.5000 | 3.2264 | 0.0000 | 11.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 1 | 0.111 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2775 | 0.2775 | 0.5357 | 0.4500 | 47.929 | 83.642 |
| acute | 9 | 0.1339 | 0.0000 | 0.1862 | 0.1825 | 51.018 | 19.266 |
| chronic | 1 | 0.3978 | 0.3978 | 0.6667 | 0.5000 | 34.668 | 2.267 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2714 | 0.2500 | 48.850 | 5.170 |
| Philips|Ingenia Ambition X | 6 | 0.0663 | 0.1111 | 62.536 | 17.101 |
| Philips|Ingenia Elition X | 2 | 0.4168 | 0.3379 | 33.876 | 35.679 |
| Philips|Achieva | 2 | 0.1919 | 0.7857 | 30.271 | 79.318 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 68.788 | 3.397 | 12.186 | 1 | 1 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 56.087 | 0.029 | 1.081 | 3 | 1 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 60.673 | 0.037 | 0.448 | 1 | 1 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 68.571 | 0.144 | 8.530 | 12 | 1 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 79.978 | 0.213 | 79.740 | 9 | 4 |
| sub-3 | mixed | Philips|Achieva | 0.0123 | 0.5714 | 0.4000 | 54.244 | 1.995 | 159.993 | 5 | 5 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.3012 | 0.4615 | 0.5000 | 15.646 | 8.711 | 34.332 | 2 | 7 |
| sub-5 | acute | Philips|Achieva | 0.3714 | 1.0000 | 1.0000 | 6.299 | 0.189 | 0.828 | 1 | 1 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3978 | 0.6667 | 0.5000 | 34.668 | 2.905 | 5.172 | 2 | 1 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.5324 | 0.2143 | 0.1429 | 52.105 | 37.161 | 82.898 | 7 | 7 |
| sub-1 | mixed | Philips|Intera | 0.5428 | 0.5000 | 0.5000 | 41.614 | 101.155 | 110.442 | 4 | 4 |