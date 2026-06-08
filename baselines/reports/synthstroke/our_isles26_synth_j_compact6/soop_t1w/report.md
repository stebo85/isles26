# Our SynthStroke (isles26_synth_j_compact6) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1947 | 0.0521 | 0.2412 | 0.0000 | 0.6097 |
| iou | 12 | 0.1303 | 0.0268 | 0.1683 | 0.0000 | 0.4385 |
| sensitivity | 12 | 0.1672 | 0.0276 | 0.2182 | 0.0000 | 0.5854 |
| precision | 12 | 0.3215 | 0.3040 | 0.3075 | 0.0000 | 0.8264 |
| specificity | 12 | 0.9983 | 0.9995 | 0.0037 | 0.9861 | 1.0000 |
| hd95_mm | 12 | 52.1699 | 46.2885 | 25.3876 | 14.4558 | 113.3531 |
| assd_mm | 12 | 27.4929 | 17.7389 | 28.1691 | 4.2655 | 106.7805 |
| surface_dice_3mm | 12 | 0.2222 | 0.1035 | 0.2494 | 0.0000 | 0.6446 |
| abs_volume_diff_ml | 12 | 28.1576 | 9.5593 | 42.9914 | 0.7168 | 153.5371 |
| rel_abs_volume_diff | 12 | 2.2623 | 0.6927 | 5.4185 | 0.2780 | 20.2131 |
| lesion_f1 | 12 | 0.1512 | 0.1241 | 0.1452 | 0.0000 | 0.4688 |
| lesion_precision | 12 | 0.1196 | 0.0833 | 0.1262 | 0.0000 | 0.3846 |
| lesion_recall | 12 | 0.3377 | 0.3095 | 0.3232 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 11.4167 | 10.5000 | 7.6535 | 0.0000 | 29.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 8 | 0.889 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 6 | 0.545 |
| large_10-50mL | 4 | 4 | 1.000 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2764 | 0.2764 | 0.3648 | 0.6750 | 39.643 | 92.636 |
| acute | 9 | 0.1493 | 0.0195 | 0.1046 | 0.2447 | 55.674 | 16.798 |
| chronic | 1 | 0.4399 | 0.4399 | 0.1429 | 0.5000 | 45.683 | 1.438 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2559 | 0.1304 | 58.326 | 16.394 |
| Philips|Ingenia Ambition X | 6 | 0.1028 | 0.1155 | 53.809 | 16.787 |
| Philips|Ingenia Elition X | 2 | 0.5836 | 0.1955 | 18.201 | 25.063 |
| Philips|Achieva | 2 | 0.0205 | 0.2344 | 75.065 | 77.127 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 74.142 | 0.029 | 1.081 | 3 | 1 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 65.795 | 9.493 | 0.448 | 1 | 30 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 46.894 | 2.062 | 3.228 | 1 | 7 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 113.353 | 0.111 | 0.828 | 1 | 1 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0195 | 0.1538 | 1.0000 | 61.192 | 2.113 | 12.186 | 1 | 12 |
| sub-3 | mixed | Philips|Achieva | 0.0410 | 0.4688 | 0.6000 | 36.776 | 6.456 | 159.993 | 5 | 26 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0633 | 0.3077 | 0.3333 | 35.761 | 5.507 | 79.740 | 9 | 21 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0939 | 0.0889 | 0.0833 | 67.530 | 3.763 | 8.530 | 12 | 21 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4399 | 0.1429 | 0.5000 | 45.683 | 3.734 | 5.172 | 2 | 12 |
| sub-1 | mixed | Philips|Intera | 0.5118 | 0.2609 | 0.7500 | 42.510 | 142.177 | 110.442 | 4 | 19 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.5575 | 0.1053 | 0.5000 | 14.456 | 17.474 | 34.332 | 2 | 17 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.6097 | 0.2857 | 0.2857 | 21.947 | 49.630 | 82.898 | 7 | 14 |