# SynthStroke synthstroke-synth-plus on soop_bench (TRACE)

Subjects evaluated: **12**

## Overall metrics

| metric | n_valid | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| dice | 12 | 0.4470 | 0.4909 | 0.3136 | 0.0000 | 0.8695 |
| iou | 12 | 0.3419 | 0.3277 | 0.2704 | 0.0000 | 0.7691 |
| sensitivity | 12 | 0.4528 | 0.4792 | 0.3452 | 0.0000 | 0.9079 |
| precision | 12 | 0.5271 | 0.6495 | 0.3437 | 0.0000 | 0.8816 |
| specificity | 12 | 0.9990 | 0.9993 | 0.0010 | 0.9969 | 1.0000 |
| hd95_mm | 12 | 35.2963 | 28.2370 | 27.0261 | 5.5221 | 88.2591 |
| assd_mm | 12 | 16.6587 | 4.9061 | 24.5051 | 1.2107 | 77.1204 |
| surface_dice_3mm | 12 | 0.4973 | 0.5627 | 0.2969 | 0.0000 | 0.8851 |
| abs_volume_diff_ml | 12 | 10.9031 | 5.4614 | 14.4227 | 0.1907 | 54.3075 |
| rel_abs_volume_diff | 12 | 0.9680 | 0.5536 | 1.5721 | 0.0272 | 6.0741 |
| lesion_f1 | 12 | 0.4781 | 0.4722 | 0.3035 | 0.0000 | 1.0000 |
| lesion_precision | 12 | 0.5639 | 0.5000 | 0.3555 | 0.0000 | 1.0000 |
| lesion_recall | 12 | 0.4923 | 0.5000 | 0.3439 | 0.0000 | 1.0000 |
| abs_lesion_count_diff | 12 | 2.0833 | 1.0000 | 2.9569 | 0.0000 | 11.0000 |

## Size-binned lesion recall (pooled across cases, voxels)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| tiny_<10 | 5 | 0 | 0.000 |
| small_10-100 | 19 | 2 | 0.105 |
| medium_100-1k | 11 | 7 | 0.636 |
| large_1k-10k | 9 | 5 | 0.556 |
| huge_>=10k | 4 | 4 | 1.000 |

## Size-binned lesion recall (pooled across cases, mL)

| bin | n_GT lesions | n_detected | recall |
|---|---:|---:|---:|
| vsmall_<0.1mL | 11 | 0 | 0.000 |
| small_0.1-1mL | 18 | 4 | 0.222 |
| med_1-10mL | 11 | 8 | 0.727 |
| large_10-50mL | 4 | 2 | 0.500 |
| vlarge_>=50mL | 4 | 4 | 1.000 |

## By chronicity

| group | n | Dice mean | Dice median | Lesion F1 mean | Lesion recall mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 2 | 0.8108 | 0.8108 | 0.5556 | 0.4500 | 36.043 | 13.185 |
| acute | 9 | 0.3833 | 0.4264 | 0.4585 | 0.5009 | 35.310 | 11.155 |
| chronic | 1 | 0.2924 | 0.2924 | 0.5000 | 0.5000 | 33.680 | 4.072 |

## By center (Manufacturer | Model)

| center | n | Dice mean | Lesion F1 mean | HD95 mm mean | AVD mL mean |
|---|---:|---:|---:|---:|---:|
| Philips|Intera | 2 | 0.3761 | 0.3333 | 65.881 | 14.293 |
| Philips|Ingenia Ambition X | 6 | 0.2684 | 0.4771 | 36.287 | 12.604 |
| Philips|Ingenia Elition X | 2 | 0.7883 | 0.3818 | 17.654 | 10.719 |
| Philips|Achieva | 2 | 0.7124 | 0.7222 | 19.383 | 2.596 |

## Per-case table (sorted by Dice, worst first)

| subject | chronicity | center | Dice | Lesion F1 | Lesion recall | HD95 mm | Pred mL | GT mL | GT comps | Pred comps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-10 | acute | Philips|Ingenia Ambition X | 0.0000 | 0.0000 | 0.0000 | 88.259 | 0.095 | 12.186 | 1 | 1 |
| sub-13 | acute | Philips|Intera | 0.0000 | 0.0000 | 0.0000 | 86.364 | 7.650 | 1.081 | 3 | 7 |
| sub-4 | acute | Philips|Ingenia Ambition X | 0.0747 | 0.5000 | 1.0000 | 53.905 | 0.895 | 3.228 | 1 | 3 |
| sub-14 | acute | Philips|Ingenia Ambition X | 0.1892 | 1.0000 | 1.0000 | 13.242 | 0.638 | 0.448 | 1 | 1 |
| sub-11 | chronic | Philips|Ingenia Ambition X | 0.2924 | 0.5000 | 0.5000 | 33.680 | 1.100 | 5.172 | 2 | 2 |
| sub-8 | acute | Philips|Ingenia Ambition X | 0.4264 | 0.5769 | 0.5556 | 15.773 | 25.432 | 79.740 | 9 | 10 |
| sub-5 | acute | Philips|Achieva | 0.5553 | 1.0000 | 1.0000 | 12.077 | 1.666 | 0.828 | 1 | 1 |
| sub-7 | acute | Philips|Ingenia Ambition X | 0.6276 | 0.2857 | 0.1667 | 12.861 | 11.158 | 8.530 | 12 | 1 |
| sub-1 | mixed | Philips|Intera | 0.7522 | 0.6667 | 0.5000 | 45.398 | 88.425 | 110.442 | 4 | 2 |
| sub-2 | acute | Philips|Ingenia Elition X | 0.7580 | 0.3636 | 0.2857 | 29.786 | 68.946 | 82.898 | 7 | 4 |
| sub-9 | acute | Philips|Ingenia Elition X | 0.8186 | 0.4000 | 0.5000 | 5.522 | 41.819 | 34.332 | 2 | 3 |
| sub-3 | mixed | Philips|Achieva | 0.8695 | 0.4444 | 0.4000 | 26.688 | 155.639 | 159.993 | 5 | 4 |