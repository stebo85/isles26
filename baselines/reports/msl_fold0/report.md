# MSL vs binary baseline (fold-0 cross-validation, merged to binary)

Cases: binary n=292, MSL n=292

| model | mean Dice | median Dice |
|---|---:|---:|
| binary baseline | 0.6374 | 0.7243 |
| MSL (merged) | 0.6298 | 0.7077 |

## Size-binned lesion recall (pooled, voxels)

| bin | binary recall | MSL recall |
|---|---:|---:|
| tiny_<10 | 0.183 | 0.177 |
| small_10-100 | 0.339 | 0.486 |
| med_100-1k | 0.589 | 0.641 |
| large_1k-10k | 0.829 | 0.848 |
| huge_>=10k | 0.952 | 0.959 |
