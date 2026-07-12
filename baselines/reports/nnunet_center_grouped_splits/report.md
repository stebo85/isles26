# Center-grouped 5-fold split plan

- Manifest: `/scratch/users/sciget/isles26challenge/work/nnunet/nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv`
- Manifest SHA-256: `fe6b935782d15dce3ce03b62c3ad1cefa87e08d53936e67863252a8138265aea`
- Assignment-input SHA-256: `168683caeabdb8bb7000b431972e4493f5b030e28ed223c4d839bb5d0b390492`
- Assignment-input fields: `nnunet_id, site, chronicity_bin, size_bin, lesion_volume_ml`
- Algorithm: `center_grouped_v1`; seed `42`; restarts `256`
- Cases: 1452; centers: 55
- Optimizer objective: 0.49072415

## Validation

All invariants passed: **true**. Each case appears in validation exactly once, every training set is its validation complement, and every center belongs to exactly one validation fold.

## Fold balance

| fold | val centers | val cases | case delta | chronicity | lesion size | volume known | volume sum (mL) | volume delta (mL) |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 0 | 13 | 281 | -9.4 | ge180d=145, lt180d=90, unknown=46 | large_10-50mL=63, med_1-10mL=100, small_0.1-1mL=56, vlarge_>=50mL=53, vsmall_<0.1mL=9 | 281 | 8797.649 | +670.848 |
| 1 | 10 | 284 | -6.4 | ge180d=145, lt180d=89, unknown=50 | large_10-50mL=67, med_1-10mL=101, small_0.1-1mL=58, vlarge_>=50mL=49, vsmall_<0.1mL=9 | 284 | 7331.723 | -795.077 |
| 2 | 10 | 287 | -3.4 | ge180d=140, lt180d=91, unknown=56 | large_10-50mL=64, med_1-10mL=105, small_0.1-1mL=52, vlarge_>=50mL=54, vsmall_<0.1mL=12 | 287 | 7721.375 | -405.426 |
| 3 | 12 | 283 | -7.4 | ge180d=138, lt180d=95, unknown=50 | large_10-50mL=65, med_1-10mL=106, small_0.1-1mL=53, vlarge_>=50mL=50, vsmall_<0.1mL=9 | 283 | 8226.416 | +99.615 |
| 4 | 10 | 317 | +26.6 | ge180d=81, lt180d=68, unknown=168 | large_10-50mL=83, med_1-10mL=112, small_0.1-1mL=55, vlarge_>=50mL=58, vsmall_<0.1mL=9 | 317 | 8556.840 | +430.039 |

## Center assignments

| fold | center | cases | chronicity | lesion size | volume sum (mL) |
|---:|---|---:|---|---|---:|
| 0 | R010 | 26 | unknown=26 | large_10-50mL=1, med_1-10mL=10, small_0.1-1mL=14, vsmall_<0.1mL=1 | 45.297 |
| 0 | R014 | 9 | unknown=9 | small_0.1-1mL=5, vsmall_<0.1mL=4 | 1.398 |
| 0 | R016 | 1 | unknown=1 | small_0.1-1mL=1 | 0.904 |
| 0 | R018 | 11 | ge180d=11 | large_10-50mL=3, med_1-10mL=2, small_0.1-1mL=1, vlarge_>=50mL=5 | 467.249 |
| 0 | R029 | 10 | ge180d=9, lt180d=1 | large_10-50mL=3, med_1-10mL=5, small_0.1-1mL=1, vlarge_>=50mL=1 | 226.994 |
| 0 | R033 | 2 | lt180d=2 | large_10-50mL=2 | 31.930 |
| 0 | R035 | 15 | ge180d=15 | large_10-50mL=4, med_1-10mL=6, small_0.1-1mL=2, vlarge_>=50mL=2, vsmall_<0.1mL=1 | 306.771 |
| 0 | R040 | 86 | ge180d=18, lt180d=67, unknown=1 | large_10-50mL=20, med_1-10mL=29, small_0.1-1mL=10, vlarge_>=50mL=26, vsmall_<0.1mL=1 | 4576.012 |
| 0 | R047 | 48 | ge180d=46, lt180d=2 | large_10-50mL=16, med_1-10mL=18, small_0.1-1mL=5, vlarge_>=50mL=9 | 1544.904 |
| 0 | R052 | 32 | ge180d=32 | large_10-50mL=7, med_1-10mL=11, small_0.1-1mL=7, vlarge_>=50mL=7 | 998.163 |
| 0 | R056 | 15 | ge180d=14, unknown=1 | large_10-50mL=3, med_1-10mL=7, small_0.1-1mL=3, vlarge_>=50mL=2 | 417.812 |
| 0 | R065 | 18 | lt180d=12, unknown=6 | large_10-50mL=1, med_1-10mL=8, small_0.1-1mL=6, vlarge_>=50mL=1, vsmall_<0.1mL=2 | 98.105 |
| 0 | R070 | 8 | lt180d=6, unknown=2 | large_10-50mL=3, med_1-10mL=4, small_0.1-1mL=1 | 82.110 |
| 1 | R011 | 29 | unknown=29 | med_1-10mL=13, small_0.1-1mL=11, vsmall_<0.1mL=5 | 39.675 |
| 1 | R020 | 1 | lt180d=1 | med_1-10mL=1 | 5.074 |
| 1 | R024 | 21 | ge180d=21 | large_10-50mL=4, med_1-10mL=6, small_0.1-1mL=6, vlarge_>=50mL=4, vsmall_<0.1mL=1 | 439.612 |
| 1 | R027 | 32 | ge180d=29, unknown=3 | large_10-50mL=6, med_1-10mL=11, small_0.1-1mL=5, vlarge_>=50mL=10 | 1335.559 |
| 1 | R030 | 15 | ge180d=14, lt180d=1 | large_10-50mL=3, med_1-10mL=2, small_0.1-1mL=3, vlarge_>=50mL=7 | 787.246 |
| 1 | R038 | 94 | ge180d=18, lt180d=76 | large_10-50mL=27, med_1-10mL=36, small_0.1-1mL=16, vlarge_>=50mL=15 | 2276.133 |
| 1 | R042 | 35 | ge180d=34, lt180d=1 | large_10-50mL=11, med_1-10mL=11, small_0.1-1mL=4, vlarge_>=50mL=8, vsmall_<0.1mL=1 | 1303.869 |
| 1 | R049 | 24 | lt180d=6, unknown=18 | large_10-50mL=3, med_1-10mL=7, small_0.1-1mL=12, vsmall_<0.1mL=2 | 119.312 |
| 1 | R064 | 6 | ge180d=6 | large_10-50mL=1, med_1-10mL=3, vlarge_>=50mL=2 | 212.708 |
| 1 | R067 | 27 | ge180d=23, lt180d=4 | large_10-50mL=12, med_1-10mL=11, small_0.1-1mL=1, vlarge_>=50mL=3 | 812.536 |
| 2 | R002 | 12 | ge180d=12 | large_10-50mL=6, med_1-10mL=5, vlarge_>=50mL=1 | 326.613 |
| 2 | R003 | 15 | ge180d=15 | large_10-50mL=4, med_1-10mL=5, small_0.1-1mL=2, vlarge_>=50mL=4 | 627.864 |
| 2 | R004 | 37 | ge180d=20, unknown=17 | large_10-50mL=17, med_1-10mL=7, vlarge_>=50mL=13 | 1903.011 |
| 2 | R009 | 111 | ge180d=61, lt180d=50 | large_10-50mL=13, med_1-10mL=46, small_0.1-1mL=35, vlarge_>=50mL=5, vsmall_<0.1mL=12 | 771.781 |
| 2 | R015 | 23 | unknown=23 | large_10-50mL=7, med_1-10mL=5, small_0.1-1mL=4, vlarge_>=50mL=7 | 817.998 |
| 2 | R019 | 14 | lt180d=13, unknown=1 | large_10-50mL=3, med_1-10mL=6, small_0.1-1mL=4, vlarge_>=50mL=1 | 118.441 |
| 2 | R023 | 15 | unknown=15 | large_10-50mL=2, med_1-10mL=6, vlarge_>=50mL=7 | 913.529 |
| 2 | R032 | 37 | ge180d=9, lt180d=28 | large_10-50mL=4, med_1-10mL=19, small_0.1-1mL=1, vlarge_>=50mL=13 | 1804.238 |
| 2 | R061 | 15 | ge180d=15 | large_10-50mL=6, med_1-10mL=5, small_0.1-1mL=3, vlarge_>=50mL=1 | 237.986 |
| 2 | R062 | 8 | ge180d=8 | large_10-50mL=2, med_1-10mL=1, small_0.1-1mL=3, vlarge_>=50mL=2 | 199.915 |
| 3 | R005 | 37 | ge180d=35, lt180d=1, unknown=1 | large_10-50mL=5, med_1-10mL=10, small_0.1-1mL=13, vlarge_>=50mL=8, vsmall_<0.1mL=1 | 1231.270 |
| 3 | R008 | 7 | lt180d=6, unknown=1 | med_1-10mL=3, small_0.1-1mL=3, vsmall_<0.1mL=1 | 9.207 |
| 3 | R017 | 16 | ge180d=15, unknown=1 | large_10-50mL=5, med_1-10mL=4, small_0.1-1mL=3, vlarge_>=50mL=4 | 471.518 |
| 3 | R028 | 26 | ge180d=25, lt180d=1 | large_10-50mL=8, med_1-10mL=11, small_0.1-1mL=3, vlarge_>=50mL=4 | 635.038 |
| 3 | R031 | 37 | ge180d=1, lt180d=36 | large_10-50mL=10, med_1-10mL=18, small_0.1-1mL=3, vlarge_>=50mL=6 | 1072.484 |
| 3 | R044 | 4 | ge180d=4 | large_10-50mL=1, med_1-10mL=2, vlarge_>=50mL=1 | 157.807 |
| 3 | R048 | 44 | ge180d=44 | large_10-50mL=10, med_1-10mL=17, small_0.1-1mL=8, vlarge_>=50mL=9 | 1669.287 |
| 3 | R050 | 15 | lt180d=6, unknown=9 | large_10-50mL=1, med_1-10mL=1, small_0.1-1mL=10, vsmall_<0.1mL=3 | 26.725 |
| 3 | R053 | 5 | lt180d=5 | med_1-10mL=2, small_0.1-1mL=1, vsmall_<0.1mL=2 | 4.229 |
| 3 | R057 | 14 | ge180d=14 | large_10-50mL=5, med_1-10mL=3, small_0.1-1mL=1, vlarge_>=50mL=4, vsmall_<0.1mL=1 | 574.147 |
| 3 | R059 | 38 | unknown=38 | large_10-50mL=7, med_1-10mL=16, small_0.1-1mL=6, vlarge_>=50mL=8, vsmall_<0.1mL=1 | 1482.150 |
| 3 | R069 | 40 | lt180d=40 | large_10-50mL=13, med_1-10mL=19, small_0.1-1mL=2, vlarge_>=50mL=6 | 892.554 |
| 4 | R001 | 39 | ge180d=39 | large_10-50mL=8, med_1-10mL=18, small_0.1-1mL=6, vlarge_>=50mL=5, vsmall_<0.1mL=2 | 623.286 |
| 4 | R034 | 24 | ge180d=24 | large_10-50mL=5, med_1-10mL=9, small_0.1-1mL=7, vlarge_>=50mL=2, vsmall_<0.1mL=1 | 336.463 |
| 4 | R039 | 4 | lt180d=4 | med_1-10mL=1, small_0.1-1mL=2, vlarge_>=50mL=1 | 187.118 |
| 4 | R041 | 3 | lt180d=3 | small_0.1-1mL=2, vsmall_<0.1mL=1 | 0.692 |
| 4 | R045 | 4 | ge180d=4 | med_1-10mL=3, small_0.1-1mL=1 | 22.446 |
| 4 | R046 | 13 | ge180d=13 | large_10-50mL=1, med_1-10mL=8, small_0.1-1mL=2, vlarge_>=50mL=2 | 263.089 |
| 4 | R058 | 35 | lt180d=35 | large_10-50mL=10, med_1-10mL=11, small_0.1-1mL=11, vlarge_>=50mL=2, vsmall_<0.1mL=1 | 394.257 |
| 4 | R063 | 1 | ge180d=1 | med_1-10mL=1 | 4.036 |
| 4 | R071 | 26 | lt180d=26 | large_10-50mL=3, med_1-10mL=17, small_0.1-1mL=6 | 148.905 |
| 4 | SOOP | 168 | unknown=168 | large_10-50mL=56, med_1-10mL=44, small_0.1-1mL=18, vlarge_>=50mL=46, vsmall_<0.1mL=4 | 6576.547 |
