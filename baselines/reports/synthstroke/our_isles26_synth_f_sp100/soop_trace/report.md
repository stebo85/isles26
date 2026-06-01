# Our SynthStroke (isles26_synth_f_sp100) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1554 | 0.0637 | 0.2036 | 0.0000 | 0.6104 |
| iou | 12 | 0.0998 | 0.0329 | 0.1417 | 0.0000 | 0.4393 |
| sensitivity | 12 | 0.3691 | 0.3002 | 0.3209 | 0.0000 | 0.8120 |
| precision | 12 | 0.1133 | 0.0338 | 0.1617 | 0.0000 | 0.4982 |
| specificity | 12 | 0.9684 | 0.9738 | 0.0124 | 0.9342 | 0.9792 |
| hd95_mm | 12 | 87.0150 | 86.4290 | 26.6848 | 37.9032 | 132.0336 |
| assd_mm | 12 | 47.9742 | 55.4603 | 24.2521 | 8.7209 | 92.4388 |
| surface_dice_3mm | 12 | 0.1239 | 0.0821 | 0.1398 | 0.0000 | 0.4459 |
| abs_volume_diff_ml | 12 | 172.4768 | 155.6309 | 91.4510 | 92.9543 | 444.9053 |
| rel_abs_volume_diff | 12 | 70.8159 | 15.7307 | 98.7362 | 0.5810 | 337.1475 |
| lesion_f1 | 12 | 0.1871 | 0.1538 | 0.1675 | 0.0000 | 0.6061 |
| lesion_precision | 12 | 0.1468 | 0.1010 | 0.1729 | 0.0000 | 0.6667 |
| lesion_recall | 12 | 0.4235 | 0.4167 | 0.3524 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 6.6667 | 5.0000 | 4.6428 | 0.0000 | 17.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 4 | 0.211 |
| medium_100-1k | 11 | 6 | 0.545 |
| large_1k-10k | 9 | 5 | 0.556 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 5 | 0.278 |
| med_1-10mL | 11 | 7 | 0.636 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.5590 | 0.5590 | 0.2674 | 0.5250 | 44.125 | 94.832 |
| acute | 9 | 0.0826 | 0.0583 | 0.1654 | 0.3924 | 94.726 | 182.147 |
| chronic | 1 | 0.0027 | 0.0027 | 0.2222 | 0.5000 | 103.398 | 240.735 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2538 | 0.0769 | 78.055 | 135.662 |
| Philips|Ingenia Ambition X | 6 | 0.0361 | 0.2079 | 94.693 | 218.049 |
| Philips|Ingenia Elition X | 2 | 0.2632 | 0.1484 | 74.988 | 146.682 |
| Philips|Achieva | 2 | 0.3068 | 0.2738 | 84.968 | 98.369 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 105.764 | 175.695 | 1.081 | 3 | 15 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 91.763 | 151.327 | 0.448 | 1 | 6 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 79.543 | 448.133 | 3.228 | 1 | 2 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0027 | 0.2222 | 0.5000 | 103.398 | 245.907 | 5.172 | 2 | 7 |
| sub-5 | acute | Philips|Achieva | 0.0031 | 0.1667 | 1.0000 | 132.034 | 104.612 | 0.828 | 1 | 11 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0583 | 0.2857 | 1.0000 | 117.868 | 155.641 | 12.186 | 1 | 6 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0692 | 0.1333 | 0.3333 | 107.111 | 176.469 | 8.530 | 12 | 12 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0867 | 0.6061 | 0.5556 | 68.475 | 240.122 | 79.740 | 9 | 6 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.2266 | 0.1429 | 0.5000 | 68.880 | 211.746 | 34.332 | 2 | 12 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.2998 | 0.1538 | 0.1429 | 81.095 | 198.848 | 82.898 | 7 | 24 |
| sub-1 | mixed | Philips|Intera | 0.5077 | 0.1538 | 0.2500 | 50.346 | 207.151 | 110.442 | 4 | 9 |
| sub-3 | mixed | Philips|Achieva | 0.6104 | 0.3810 | 0.8000 | 37.903 | 252.947 | 159.993 | 5 | 12 |