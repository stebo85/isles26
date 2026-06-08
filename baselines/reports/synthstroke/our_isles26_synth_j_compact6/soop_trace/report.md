# Our SynthStroke (isles26_synth_j_compact6) on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.0781 | 0.0039 | 0.1111 | 0.0000 | 0.3246 |
| iou | 12 | 0.0444 | 0.0019 | 0.0648 | 0.0000 | 0.1938 |
| sensitivity | 12 | 0.1073 | 0.0362 | 0.1271 | 0.0000 | 0.3461 |
| precision | 12 | 0.0673 | 0.0021 | 0.1016 | 0.0000 | 0.3057 |
| specificity | 12 | 0.9736 | 0.9783 | 0.0146 | 0.9372 | 0.9897 |
| hd95_mm | 12 | 91.0875 | 90.3895 | 21.4896 | 65.8967 | 124.0910 |
| assd_mm | 12 | 52.3790 | 62.2291 | 23.8275 | 19.5349 | 88.8282 |
| surface_dice_3mm | 12 | 0.0808 | 0.0159 | 0.0996 | 0.0000 | 0.2678 |
| abs_volume_diff_ml | 12 | 125.7816 | 90.9681 | 110.7476 | 14.6232 | 424.2462 |
| rel_abs_volume_diff | 12 | 66.9819 | 9.3094 | 103.1639 | 0.1324 | 346.9508 |
| lesion_f1 | 12 | 0.0893 | 0.0575 | 0.0953 | 0.0000 | 0.2767 |
| lesion_precision | 12 | 0.0560 | 0.0339 | 0.0630 | 0.0000 | 0.1842 |
| lesion_recall | 12 | 0.3542 | 0.3810 | 0.2926 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 42.5000 | 42.0000 | 14.3904 | 24.0000 | 66.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 2 | 0.105 |
| medium_100-1k | 11 | 4 | 0.364 |
| large_1k-10k | 9 | 7 | 0.778 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 2 | 0.111 |
| med_1-10mL | 11 | 8 | 0.727 |
| large_10-50mL | 4 | 3 | 0.750 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2539 | 0.2539 | 0.1548 | 0.4250 | 66.039 | 53.079 |
| acute | 9 | 0.0476 | 0.0012 | 0.0768 | 0.3223 | 95.015 | 127.801 |
| chronic | 1 | 0.0009 | 0.0009 | 0.0714 | 0.5000 | 105.839 | 253.016 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.1629 | 0.0360 | 89.254 | 39.398 |
| Philips|Ingenia Ambition X | 6 | 0.0094 | 0.0813 | 97.183 | 178.168 |
| Philips|Ingenia Elition X | 2 | 0.1855 | 0.1223 | 70.585 | 54.502 |
| Philips|Achieva | 2 | 0.0916 | 0.1339 | 95.136 | 126.285 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 94.434 | 155.714 | 0.448 | 1 | 45 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 86.345 | 427.474 | 3.228 | 1 | 25 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | 124.091 | 161.863 | 0.828 | 1 | 41 |
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0005 | 0.0800 | 1.0000 | 119.340 | 132.443 | 12.186 | 1 | 48 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.0009 | 0.0714 | 0.5000 | 105.839 | 258.188 | 5.172 | 2 | 26 |
| sub-13 | acute | Philips|Intera | 0.0012 | 0.0303 | 0.3333 | 112.611 | 65.254 | 1.081 | 3 | 63 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0066 | 0.0594 | 0.0833 | 108.907 | 83.164 | 8.530 | 12 | 65 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0487 | 0.2767 | 0.5556 | 68.234 | 121.328 | 79.740 | 9 | 38 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.1114 | 0.0556 | 0.5000 | 69.127 | 124.733 | 34.332 | 2 | 68 |
| sub-3 | mixed | Philips|Achieva | 0.1831 | 0.2679 | 0.6000 | 66.181 | 251.528 | 159.993 | 5 | 29 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.2596 | 0.1890 | 0.4286 | 72.043 | 101.502 | 82.898 | 7 | 66 |
| sub-1 | mixed | Philips|Intera | 0.3246 | 0.0417 | 0.2500 | 65.897 | 125.065 | 110.442 | 4 | 44 |