# OOD ensemble: nnU-Net(T1w) + synth-plus(T1w) on soop_bench

Cases: 12. Both models run on the skull-stripped T1w input, lesion probabilities warped to TRACE, blended, thresholded; Dice via the eval framework.

| model | weight_nn | best threshold | mean Dice | median Dice |
|---|---:|---:|---:|---:|
| nnU-Net only | 1.0 | 0.30 | 0.2555 | 0.0516 |
| synth-plus only | 0.0 | 0.30 | 0.2731 | 0.2090 |
| blend | 0.5 | 0.30 | 0.2884 | 0.1993 |

**Best overall:** blend w_nn=0.5 (w_nn=0.5, thr=0.30) -> mean Dice 0.2884 (median 0.1993).

Reference: nnU-Net 5-fold+TTA soop baseline (mask@0.5) = 0.2546 mean; synth-plus on the DWI/TRACE image directly = 0.447 (needs DWI, not available at T1w test time).

## Full grid (mean Dice)

| weight_nn \ thr | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.255 | 0.254 | 0.252 | 0.250 | 0.248 | 0.246 | 0.245 |
| 0.7 | 0.265 | 0.259 | 0.256 | 0.252 | 0.249 | 0.246 | 0.243 |
| 0.6 | 0.286 | 0.281 | 0.261 | 0.255 | 0.250 | 0.245 | 0.233 |
| 0.5 | 0.288 | 0.285 | 0.279 | 0.274 | 0.247 | 0.235 | 0.229 |
| 0.4 | 0.287 | 0.285 | 0.275 | 0.266 | 0.261 | 0.255 | 0.234 |
| 0.3 | 0.281 | 0.273 | 0.270 | 0.267 | 0.263 | 0.258 | 0.254 |
| 0.0 | 0.273 | 0.270 | 0.267 | 0.265 | 0.263 | 0.261 | 0.258 |
