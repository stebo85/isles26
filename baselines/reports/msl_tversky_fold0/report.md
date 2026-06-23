# MSL vs binary baseline (fold-0 cross-validation, merged to binary)

Cases: binary n=292, MSL n=292

| model | mean Dice | median Dice |
|---|---:|---:|
| binary baseline | 0.6374 | 0.7243 |
| MSL (merged) | 0.6186 | 0.7172 |

## Size-binned lesion recall (pooled, voxels)

| bin | binary recall | MSL recall |
|---|---:|---:|
| tiny_<10 | 0.183 | 0.140 |
| small_10-100 | 0.339 | 0.454 |
| med_100-1k | 0.589 | 0.628 |
| large_1k-10k | 0.829 | 0.810 |
| huge_>=10k | 0.952 | 0.945 |
