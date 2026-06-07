# Our SynthStroke (isles26_synth_i_tversky_a08) on soop_bench (T1w in TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.1663 | 0.0159 | 0.2189 | 0.0000 | 0.5122 |
| iou | 12 | 0.1080 | 0.0080 | 0.1452 | 0.0000 | 0.3442 |
| sensitivity | 12 | 0.1331 | 0.0081 | 0.1844 | 0.0000 | 0.5155 |
| precision | 10 | 0.3609 | 0.3476 | 0.3097 | 0.0000 | 0.8369 |
| specificity | 12 | 0.9988 | 0.9996 | 0.0026 | 0.9901 | 1.0000 |
| hd95_mm | 10 | 44.5792 | 44.6674 | 18.0498 | 14.4421 | 73.7238 |
| assd_mm | 10 | 21.7222 | 19.9695 | 16.6031 | 5.1943 | 61.3063 |
| surface_dice_3mm | 10 | 0.2316 | 0.0942 | 0.2456 | 0.0000 | 0.5968 |
| abs_volume_diff_ml | 12 | 27.8008 | 6.5846 | 44.7143 | 0.7483 | 156.0459 |
| rel_abs_volume_diff | 12 | 2.0615 | 0.8144 | 4.6238 | 0.0131 | 17.3607 |
| lesion_f1 | 12 | 0.1854 | 0.1916 | 0.1766 | 0.0000 | 0.4444 |
| lesion_precision | 12 | 0.3368 | 0.3015 | 0.3291 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.2169 | 0.1845 | 0.2132 | 0.0000 | 0.5000 |
| abs_lesion_count_diff | 12 | 7.0000 | 4.5000 | 5.4924 | 1.0000 | 21.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 0 | 0.000 |
| medium_100-1k | 11 | 3 | 0.273 |
| large_1k-10k | 9 | 5 | 0.556 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 1 | 0.056 |
| med_1-10mL | 11 | 5 | 0.455 |
| large_10-50mL | 4 | 2 | 0.500 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.2641 | 0.2641 | 0.4097 | 0.4500 | 44.667 | 78.748 |
| acute | 9 | 0.1106 | 0.0000 | 0.1244 | 0.1336 | 47.141 | 19.355 |
| chronic | 1 | 0.4721 | 0.4721 | 0.2857 | 0.5000 | 26.470 | 1.922 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.2561 | 0.2222 | 45.640 | 1.266 |
| Philips|Ingenia Ambition X | 6 | 0.0939 | 0.1115 | 52.837 | 17.538 |
| Philips|Ingenia Elition X | 2 | 0.4523 | 0.3683 | 19.719 | 34.488 |
| Philips|Achieva | 2 | 0.0080 | 0.1875 | 43.695 | 78.437 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 73.724 | 0.763 | 12.186 | 1 | 5 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 1.081 | 3 | 0 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 62.537 | 8.217 | 0.448 | 1 | 22 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 51.205 | 3.976 | 3.228 | 1 | 11 |
| sub-5 | acute | Philips|Achieva | 0.0000 | 0.0000 | 0.0000 | nan | 0.000 | 0.828 | 1 | 0 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.0158 | 0.2857 | 0.3333 | 38.822 | 1.775 | 79.740 | 9 | 12 |
| sub-3 | mixed | Philips|Achieva | 0.0160 | 0.3750 | 0.4000 | 43.695 | 3.947 | 159.993 | 5 | 17 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.0753 | 0.0976 | 0.0833 | 64.262 | 3.130 | 8.530 | 12 | 17 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.3947 | 0.4167 | 0.5000 | 14.442 | 10.595 | 34.332 | 2 | 14 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.4721 | 0.2857 | 0.5000 | 26.470 | 3.250 | 5.172 | 2 | 5 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.5099 | 0.3200 | 0.2857 | 24.995 | 37.659 | 82.898 | 7 | 11 |
| sub-1 | mixed | Philips|Intera | 0.5122 | 0.4444 | 0.5000 | 45.640 | 111.892 | 110.442 | 4 | 10 |