# SynthStroke synthstroke-qatlas on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1148 | 0.0001 | 0.1750 | 0.0000 | 0.5533 |
| iou | 12 | 0.0715 | 0.0000 | 0.1151 | 0.0000 | 0.3825 |
| sensitivity | 12 | 0.1132 | 0.0000 | 0.1911 | 0.0000 | 0.5404 |
| precision | 9 | 0.4217 | 0.5253 | 0.3580 | 0.0000 | 1.0000 |
| specificity | 12 | 0.9990 | 0.9996 | 0.0017 | 0.9939 | 1.0000 |
| hd95_mm | 9 | 59.7033 | 51.0799 | 28.5439 | 19.5956 | 130.7718 |
| assd_mm | 9 | 29.7167 | 28.6202 | 17.4066 | 6.8573 | 66.7521 |
| surface_dice_3mm | 9 | 0.1531 | 0.0467 | 0.1665 | 0.0000 | 0.3792 |
| abs_volume_diff_ml | 12 | 31.7121 | 8.5248 | 45.5832 | 0.8282 | 156.3607 |
| rel_abs_volume_diff | 12 | 1.6000 | 0.9886 | 2.5417 | 0.1898 | 9.9508 |
| lesion_f1 | 12 | 0.1780 | 0.0909 | 0.2233 | 0.0000 | 0.6667 |
| lesion_precision | 12 | 0.5870 | 0.8000 | 0.4424 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.1587 | 0.0556 | 0.1943 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 3.3333 | 2.5000 | 2.3570 | 1.0000 | 8.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 0 | 0.000 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 0 | 0.000 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 1 | 0.250 |
| vlarge_>=50mL | 4 | 3 | 0.750 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2883 | 0.2883 | 0.4286 | 0.3250 | 62.309 | 88.661 |
| acute | 9 | 0.0551 | 0.0000 | 0.1219 | 0.0838 | 60.458 | 21.689 |
| chronic | 1 | 0.3044 | 0.3044 | 0.1818 | 0.5000 | 49.965 | 8.019 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2767 | 0.1429 | 51.080 | 11.022 |
| Philips|Ingenia Ambition X | 6 | 0.0508 | 0.0636 | 68.443 | 17.741 |
| Philips|Ingenia Elition X | 2 | 0.2480 | 0.4487 | 35.249 | 47.434 |
| Philips|Achieva | 2 | 0.0117 | 0.2857 | 73.539 | 78.594 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 46.678 | 3.155 | 12.186 | 1 | 6 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 1.081 | 3 | 0 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 52.150 | 4.901 | 0.448 | 1 | 5 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 3.228 | 1 | 0 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 62.649 | 10.512 | 8.530 | 12 | 7 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0002 | 0.2000 | 0.1111 | 130.772 | 0.006 | 79.740 | 9 | 1 |
| sub-3 | mixed | Philips|Achieva | 0.0233 | 0.5714 | 0.4000 | 73.539 | 3.632 | 159.993 | 5 | 3 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.2078 | 0.2308 | 0.1429 | 50.902 | 12.285 | 82.898 | 7 | 5 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.2882 | 0.6667 | 0.5000 | 19.596 | 10.078 | 34.332 | 2 | 1 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.3044 | 0.1818 | 0.5000 | 49.965 | 13.191 | 5.172 | 2 | 9 |
| sub-1 | mixed | Philips|Intera | 0.5533 | 0.2857 | 0.2500 | 51.080 | 89.480 | 110.442 | 4 | 3 |