# Head-to-head: shipped model vs stock nnU-Net baseline vs DA5

n = 19 cases, all out-of-fold for all three models. Operating point for
every model: p >= 0.35, components < 20 voxels dropped (26-connectivity).

`ours` is evaluated on center-held-out folds and `baseline` on center-leaking
folds, so those two columns are NOT a like-for-like model comparison.
`ours_same_split` is: it is our loss trained on the baseline's own split.

| case | site | GT mL | ours Dice | baseline Dice | ours-same-split Dice | DA5 Dice | GT mL | ours mL | baseline mL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ATLAS_0809 | R041 | 0.00 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0.02 | 0.00 |
| ATLAS_0310 | R011 | 0.45 | 0.804 | 0.791 | 0.782 | 0.820 | 0.45 | 0.43 | 0.45 |
| ATLAS_0712 | R038 | 1.17 | 0.561 | 0.628 | 0.613 | 0.487 | 1.17 | 0.81 | 0.78 |
| ATLAS_0800 | R040 | 2.42 | 0.806 | 0.800 | 0.816 | 0.808 | 2.42 | 2.82 | 2.33 |
| ATLAS_1247 | R069 | 5.38 | 0.409 | 0.619 | 0.595 | 0.621 | 5.38 | 3.77 | 7.21 |
| ATLAS_0635 | R038 | 14.81 | 0.884 | 0.915 | 0.914 | 0.885 | 14.81 | 12.49 | 14.23 |
| ATLAS_1138 | R061 | 34.78 | 0.857 | 0.823 | 0.832 | 0.842 | 34.78 | 33.01 | 28.63 |
| ATLAS_1372 | SOOP | 71.63 | 0.616 | 0.838 | 0.838 | 0.366 | 71.63 | 42.88 | 87.14 |
| ATLAS_0104 | R005 | 387.63 | 0.865 | 0.862 | 0.866 | 0.796 | 387.63 | 393.93 | 377.02 |
| ATLAS_0390 | R023 | 49.82 | 0.704 | 0.713 | 0.645 | 0.736 | 49.82 | 46.16 | 46.81 |
| ATLAS_0628 | R038 | 25.84 | 0.749 | 0.767 | 0.771 | 0.665 | 25.84 | 22.60 | 23.28 |
| ATLAS_1024 | R052 | 3.98 | 0.772 | 0.859 | 0.565 | 0.865 | 3.98 | 5.87 | 4.40 |
| ATLAS_0193 | R009 | 2.15 | 0.463 | 0.580 | 0.466 | 0.474 | 2.15 | 0.72 | 0.96 |
| ATLAS_0774 | R040 | 0.68 | 0.815 | 0.804 | 0.831 | 0.813 | 0.68 | 0.49 | 0.49 |
| ATLAS_0980 | R049 | 0.55 | 0.693 | 0.704 | 0.754 | 0.740 | 0.55 | 0.30 | 0.30 |
| ATLAS_0904 | R047 | 146.72 | 0.917 | 0.919 | 0.919 | 0.919 | 146.72 | 140.42 | 143.11 |
| ATLAS_0785 | R040 | 152.15 | 0.883 | 0.895 | 0.933 | 0.865 | 152.15 | 127.08 | 133.62 |
| ATLAS_0217 | R009 | 0.06 | 0.126 | 0.223 | 0.225 | 0.150 | 0.06 | 0.95 | 0.51 |
| ATLAS_0163 | R009 | 0.07 | 0.000 | 0.000 | 0.000 | 0.069 | 0.07 | 0.00 | 0.15 |

| model | mean Dice | median Dice | cases better than baseline |
|---|---:|---:|---:|
| ours | 0.6276 | 0.7493 | 5/19 |
| baseline | 0.7230 | 0.8000 | 0/19 |
| ours_same_split | 0.7033 | 0.7816 | 8/19 |
| da5 | 0.6800 | 0.7961 | 9/19 |

This n is far too small to rank models -- the corpus-level numbers in
`work/official_eval/` do that. It exists so the masks can be looked at.

## Viewing

```
fsleyes work/head_to_head/ATLAS_1138_t1w.nii.gz \
  work/head_to_head/ATLAS_1138_gt.nii.gz -cm green -a 40 \
  work/head_to_head/ATLAS_1138_ours_pred.nii.gz -cm red -a 40 \
  work/head_to_head/ATLAS_1138_baseline_pred.nii.gz -cm blue -a 40
```
