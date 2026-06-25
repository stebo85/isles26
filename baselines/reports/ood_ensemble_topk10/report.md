# OOD ensemble: nnU-Net(T1w) + synth-plus(T1w) on soop_bench

Cases: 12. Both models run on the skull-stripped T1w input, lesion probabilities warped to TRACE, blended, thresholded; Dice via the eval framework.

| model | weight_nn | best threshold | mean Dice | median Dice |
|---|---:|---:|---:|---:|
| nnU-Net only | 1.0 | 0.30 | 0.2866 | 0.1538 |
| synth-plus only | 0.0 | 0.30 | 0.2731 | 0.2090 |
| blend | 0.6 | 0.30 | 0.3006 | 0.2566 |

**Best overall:** blend w_nn=0.6 (w_nn=0.6, thr=0.30) -> mean Dice 0.3006 (median 0.2566).

Reference: nnU-Net 5-fold+TTA soop baseline (mask@0.5) = 0.2546 mean; synth-plus on the DWI/TRACE image directly = 0.447 (needs DWI, not available at T1w test time).

## Full grid (mean Dice)

| weight_nn \ thr | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.287 | 0.281 | 0.276 | 0.270 | 0.265 | 0.261 | 0.256 |
| 0.7 | 0.289 | 0.283 | 0.277 | 0.270 | 0.264 | 0.256 | 0.246 |
| 0.6 | 0.301 | 0.294 | 0.277 | 0.270 | 0.260 | 0.250 | 0.235 |
| 0.5 | 0.300 | 0.293 | 0.285 | 0.276 | 0.250 | 0.243 | 0.235 |
| 0.4 | 0.294 | 0.285 | 0.272 | 0.266 | 0.261 | 0.256 | 0.240 |
| 0.3 | 0.277 | 0.272 | 0.270 | 0.266 | 0.262 | 0.258 | 0.254 |
| 0.0 | 0.273 | 0.270 | 0.267 | 0.265 | 0.263 | 0.261 | 0.258 |
