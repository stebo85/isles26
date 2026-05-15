# DeepISLES (SEALS+NVAUTO+SWAN) on soop_bench

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.5395 | 0.7041 | 0.3270 | 0.0000 | 0.9062 |
| iou | 12 | 0.4341 | 0.5442 | 0.2922 | 0.0000 | 0.8286 |
| sensitivity | 12 | 0.4638 | 0.5674 | 0.3112 | 0.0000 | 0.8920 |
| precision | 10 | 0.8688 | 0.9210 | 0.1743 | 0.3895 | 1.0000 |
| specificity | 12 | 0.9997 | 0.9999 | 0.0006 | 0.9979 | 1.0000 |
| hd95_mm | 10 | 18.9285 | 6.8587 | 22.3669 | 0.8984 | 77.1054 |
| assd_mm | 10 | 6.6084 | 1.7601 | 11.1916 | 0.2318 | 37.8577 |
| surface_dice_3mm | 10 | 0.7627 | 0.8350 | 0.2290 | 0.3058 | 0.9863 |
| abs_volume_diff_ml | 12 | 18.8940 | 3.0783 | 32.4697 | 0.2201 | 98.4656 |
| rel_abs_volume_diff | 12 | 0.4701 | 0.4170 | 0.3355 | 0.0315 | 1.0000 |
| lesion_f1 | 12 | 0.5817 | 0.6077 | 0.3288 | 0.0000 | 1.0000 |
| lesion_precision | 12 | 0.8028 | 1.0000 | 0.2562 | 0.3333 | 1.0000 |
| lesion_recall | 12 | 0.5837 | 0.5357 | 0.3487 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 2.2500 | 1.0000 | 2.5207 | 0.0000 | 9.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 11 | 0.579 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 4 | 0.444 |
| huge_>=10k | 4 | 3 | 0.750 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 2 | 0.182 |
| small_0.1-1mL | 18 | 11 | 0.611 |
| med_1-10mL | 11 | 3 | 0.273 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 2 | 0.500 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.5358 | 0.5358 | 0.6077 | 0.5500 | 52.754 | 51.754 |
| acute | 9 | 0.6002 | 0.7344 | 0.6406 | 0.6561 | 10.472 | 13.116 |
| chronic | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | 5.172 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2130 | 0.5077 | 56.542 | 49.502 |
| Philips|Ingenia Ambition X | 6 | 0.4455 | 0.5397 | 6.312 | 15.238 |
| Philips|Ingenia Elition X | 2 | 0.8200 | 0.5636 | 10.827 | 15.507 |
| Philips|Achieva | 2 | 0.8675 | 0.8000 | 14.650 | 2.642 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 5.172 | 2 | 0 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 79.740 | 9 | 0 |
| sub-1 | mixed | Philips|Intera | 0.1654 | 0.6154 | 0.5000 | 77.105 | 11.976 | 110.442 | 4 | 5 |
| sub-13 | acute | Philips|Intera | 0.2606 | 0.4000 | 0.3333 | 35.979 | 0.544 | 1.081 | 3 | 2 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.5176 | 1.0000 | 1.0000 | 7.616 | 1.137 | 3.228 | 1 | 2 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.6739 | 1.0000 | 1.0000 | 6.101 | 0.227 | 0.448 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.7344 | 0.6667 | 0.5000 | 6.000 | 7.833 | 8.530 | 12 | 8 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.7468 | 0.5714 | 1.0000 | 5.529 | 8.679 | 12.186 | 1 | 5 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.7560 | 0.7273 | 0.5714 | 19.258 | 54.533 | 82.898 | 7 | 8 |
| sub-5 | acute | Philips|Achieva | 0.8288 | 1.0000 | 1.0000 | 0.898 | 0.586 | 0.828 | 1 | 1 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.8840 | 0.4000 | 0.5000 | 2.396 | 31.683 | 34.332 | 2 | 6 |
| sub-3 | mixed | Philips|Achieva | 0.9062 | 0.6000 | 0.6000 | 28.403 | 154.951 | 159.993 | 5 | 5 |