# Mirroring TTA ablation -- what the shipped container actually gives up

Fold 0 of Dataset507 (center-grouped), n = 281 cases, same weights,
same grid. TTA is the only variable. Positive delta = no-TTA minus TTA.

## At the shipped operating point (p >= 0.35, minCC 20)

| metric | TTA | no TTA | delta | paired t | better |
|---|---:|---:|---:|---:|---|
| dice | 0.6605 | 0.6464 | -0.0141 | -2.73 | TTA |
| abs_volume_diff_ml | 6.5153 | 7.3713 | +0.8560 | +3.46 | TTA |
| abs_lesion_count_diff | 1.1566 | 1.2740 | +0.1174 | +2.02 | TTA |
| lesion_f1 | 0.6601 | 0.6396 | -0.0205 | -2.56 | TTA |
| pr_auc_flatten_if_empty | 0.7768 | 0.7577 | -0.0191 | -4.01 | TTA |

## Operating point re-selected on no-TTA probabilities (top 8 of 54)

| config | Dice | AVD mL | count diff | lesion-F1 | PR-AUC | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| thr0.25_k50 | 0.6494 | 7.1362 | 1.1815 | 0.6600 | 0.7577 | 12.700 |
| thr0.20_k50 | 0.6497 | 7.0395 | 1.2313 | 0.6529 | 0.7577 | 13.800 |
| thr0.30_k50 | 0.6486 | 7.2373 | 1.1673 | 0.6623 | 0.7577 | 14.200 |
| thr0.20_k35 | 0.6496 | 7.0405 | 1.3096 | 0.6461 | 0.7577 | 17.300 |
| thr0.35_k50 | 0.6460 | 7.3732 | 1.1744 | 0.6584 | 0.7577 | 17.500 |
| thr0.45_k50 | 0.6423 | 7.6834 | 1.1637 | 0.6571 | 0.7597 | 17.500 |
| thr0.25_k35 | 0.6494 | 7.1362 | 1.2491 | 0.6508 | 0.7577 | 17.600 |
| thr0.30_k35 | 0.6487 | 7.2362 | 1.2242 | 0.6536 | 0.7577 | 17.800 |

Shipped `thr0.35_k20` ranks 16 of 54 on no-TTA maps; best is `thr0.25_k50` (Dice 0.6494 vs 0.6464).

Change the shipped operating point only if the gap is larger than the
single-fold noise this table is measured with.
