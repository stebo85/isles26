# DBL vs binary baseline (fold-0 cross-validation, merged to binary)

Cases: binary n=292, DBL n=292

| model | mean Dice | median Dice |
|---|---:|---:|
| binary baseline | 0.6374 | 0.7243 |
| DBL (merged) | 0.6317 | 0.7225 |

## Size-binned lesion recall (pooled, voxels)

| bin | binary recall | DBL recall |
|---|---:|---:|
| tiny_<10 | 0.183 | 0.204 |
| small_10-100 | 0.339 | 0.353 |
| med_100-1k | 0.589 | 0.586 |
| large_1k-10k | 0.829 | 0.804 |
| huge_>=10k | 0.952 | 0.945 |
