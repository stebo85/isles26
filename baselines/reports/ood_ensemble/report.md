# OOD ensemble: nnU-Net(T1w) + synth-plus(T1w) on soop_bench

Cases: 12. Both models run on the skull-stripped T1w input, lesion probabilities warped to TRACE, blended, thresholded; Dice via the eval framework.

| model | weight_nn | best threshold | mean Dice | median Dice |
|---|---:|---:|---:|---:|
| nnU-Net only | 1.0 | 0.30 | 0.2576 | 0.0396 |
| synth-plus only | 0.0 | 0.30 | 0.2732 | 0.2091 |
| blend | 0.4 | 0.30 | 0.2886 | 0.2021 |

**Best overall:** blend w_nn=0.4 (w_nn=0.4, thr=0.30) -> mean Dice 0.2886 (median 0.2021).

Reference: nnU-Net 5-fold+TTA soop baseline (mask@0.5) = 0.2546 mean; synth-plus on the DWI/TRACE image directly = 0.447 (needs DWI, not available at T1w test time).

## Full grid (mean Dice)

| weight_nn \ thr | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.258 | 0.257 | 0.257 | 0.256 | 0.255 | 0.255 | 0.254 |
| 0.7 | 0.265 | 0.261 | 0.260 | 0.259 | 0.258 | 0.256 | 0.254 |
| 0.6 | 0.285 | 0.281 | 0.264 | 0.260 | 0.259 | 0.256 | 0.245 |
| 0.5 | 0.288 | 0.286 | 0.282 | 0.279 | 0.254 | 0.242 | 0.237 |
| 0.4 | 0.289 | 0.287 | 0.278 | 0.268 | 0.263 | 0.257 | 0.240 |
| 0.3 | 0.283 | 0.274 | 0.271 | 0.268 | 0.264 | 0.260 | 0.255 |
| 0.0 | 0.273 | 0.270 | 0.267 | 0.265 | 0.263 | 0.261 | 0.258 |
