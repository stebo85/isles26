# SynthStroke synthstroke-qsynth on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.2457 | 0.1286 | 0.2667 | 0.0022 | 0.8148 |
| iou | 12 | 0.1716 | 0.0700 | 0.2102 | 0.0011 | 0.6875 |
| sensitivity | 12 | 0.6152 | 0.7077 | 0.2847 | 0.0036 | 0.9836 |
| precision | 12 | 0.1933 | 0.0730 | 0.2339 | 0.0016 | 0.7393 |
| specificity | 12 | 0.9871 | 0.9894 | 0.0052 | 0.9790 | 0.9956 |
| hd95_mm | 12 | 71.4491 | 77.4613 | 24.8069 | 11.6797 | 106.2682 |
| assd_mm | 12 | 36.1512 | 36.3613 | 21.9678 | 2.4704 | 76.7667 |
| surface_dice_3mm | 12 | 0.2320 | 0.1255 | 0.2417 | 0.0128 | 0.7675 |
| abs_volume_diff_ml | 12 | 64.2215 | 62.7588 | 28.7802 | 15.1642 | 126.7090 |
| rel_abs_volume_diff | 12 | 31.9314 | 4.0725 | 49.5513 | 0.2275 | 152.1639 |
| lesion_f1 | 12 | 0.2161 | 0.2053 | 0.1266 | 0.0741 | 0.4286 |
| lesion_precision | 12 | 0.1445 | 0.1213 | 0.0979 | 0.0417 | 0.3333 |
| lesion_recall | 12 | 0.6578 | 0.5857 | 0.2862 | 0.0833 | 1.0000 |
| abs_lesion_count_diff | 12 | 11.8333 | 13.5000 | 6.4528 | 1.0000 | 21.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 1 | 0.053 |
| medium_100-1k | 11 | 9 | 0.818 |
| large_1k-10k | 9 | 9 | 1.000 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 4 | 0.222 |
| med_1-10mL | 11 | 11 | 1.000 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.6654 | 0.6654 | 0.4286 | 0.6750 | 38.387 | 54.930 |
| acute | 9 | 0.1725 | 0.0284 | 0.1827 | 0.6715 | 78.293 | 67.042 |
| chronic | 1 | 0.0650 | 0.0650 | 0.0909 | 0.5000 | 75.975 | 57.422 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2614 | 0.2513 | 77.194 | 100.081 |
| Philips|Ingenia Ambition X | 6 | 0.1010 | 0.1648 | 76.852 | 59.251 |
| Philips|Ingenia Elition X | 2 | 0.4879 | 0.2739 | 61.970 | 63.850 |
| Philips|Achieva | 2 | 0.4216 | 0.2768 | 58.974 | 43.644 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0022 | 0.2222 | 1.0000 | 98.136 | 27.350 | 12.186 | 1 | 8 |
| sub-13 | acute | Philips|Intera | 0.0067 | 0.0741 | 0.3333 | 89.294 | 127.790 | 1.081 | 3 | 24 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0128 | 0.0909 | 1.0000 | 81.126 | 68.544 | 0.448 | 1 | 21 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0187 | 0.2105 | 1.0000 | 58.924 | 98.953 | 3.228 | 1 | 17 |
| sub-5 | acute | Philips|Achieva | 0.0284 | 0.1250 | 1.0000 | 106.268 | 51.710 | 0.828 | 1 | 15 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0650 | 0.0909 | 0.5000 | 75.975 | 62.594 | 5.172 | 2 | 20 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.1923 | 0.0800 | 0.0833 | 89.184 | 56.045 | 8.530 | 12 | 13 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.3152 | 0.2941 | 0.5556 | 57.769 | 151.326 | 79.740 | 9 | 25 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.3667 | 0.2000 | 0.5000 | 78.948 | 122.719 | 34.332 | 2 | 8 |
| sub-1 | mixed | Philips|Intera | 0.5160 | 0.4286 | 0.7500 | 65.094 | 183.895 | 110.442 | 4 | 10 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6091 | 0.3478 | 0.5714 | 44.993 | 122.211 | 82.898 | 7 | 20 |
| sub-3 | mixed | Philips|Achieva | 0.8148 | 0.4286 | 0.6000 | 11.680 | 196.399 | 159.993 | 5 | 9 |