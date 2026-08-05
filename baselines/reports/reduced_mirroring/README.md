# Reduced mirroring TTA -- can a cheaper subset fit the CPU budget?

Fold 0 of Dataset507 (center-grouped), n = 281, same weights, same grid,
scored at the shipped operating point (`thr0.35_k20`). The only variable is which
spatial axes mirroring may flip. Axis 0 is left-right, 1 posterior-anterior,
2 inferior-superior (volumes are RAS-canonical, transpose_forward [0,1,2]).

Full TTA is worth **+0.0141 Dice** over no-TTA here. Cost at 5
folds scales with the pass count: ~2.0 min/case at 1 pass, ~12.9 min at 8,
against a ~10 minute budget.

| mirror axes | passes | est. min/case @5 folds | Dice | AVD mL | count diff | lesion-F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (shipped) | 1 | 2.0 | 0.6464 | 7.3713 | 1.2740 | 0.6396 | 0.7577 |
| (0,) L-R | 2 | 4.0 | 0.6505 | 7.3879 | 1.2278 | 0.6467 | 0.7657 |
| (1,) P-A | 2 | 4.0 | 0.6510 | 7.3912 | 1.2278 | 0.6467 | 0.7623 |
| (2,) I-S | 2 | 4.0 | 0.6487 | 7.4578 | 1.2527 | 0.6479 | 0.7635 |
| (0,1) | 4 | 8.0 | 0.6577 | 7.4038 | 1.2064 | 0.6525 | 0.7748 |
| (0,1,2) full | 8 | 16.0 | 0.6605 | 6.5153 | 1.1566 | 0.6601 | 0.7768 |

## Paired deltas vs no-TTA (same 281 cases, at `thr0.35_k20`)

| mirror axes | dDice | t | dAVD | t | dCount | t | dRQ | t | dPR-AUC | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (0,) L-R | +0.0041 | +2.25 | +0.0166 | +0.38 | -0.0463 | -1.07 | +0.0071 | +1.26 | +0.0080 | +3.28 |
| (1,) P-A | +0.0046 | +1.40 | +0.0199 | +0.53 | -0.0463 | -1.18 | +0.0071 | +1.22 | +0.0045 | +2.80 |
| (2,) I-S | +0.0023 | +1.51 | +0.0864 | +1.08 | -0.0214 | -0.43 | +0.0084 | +1.55 | +0.0058 | +2.36 |
| (0,1) | +0.0113 | +2.30 | +0.0324 | +0.56 | -0.0676 | -1.37 | +0.0129 | +1.71 | +0.0171 | +3.58 |
| (0,1,2) full | +0.0141 | +2.73 | -0.8560 | -3.46 | -0.1174 | -2.02 | +0.0205 | +2.56 | +0.0191 | +4.01 |

## Decision (rule fixed before the numbers were read)

Adopt the cheapest subset that recovers >= half of full TTA's Dice gain
(>= +0.0071), loses nothing else beyond noise (paired |t| < 2
against no-TTA on the other four metrics), and leaves >= 2x CPU headroom
at 5 folds (<= 5.0 min/case). Otherwise keep no-TTA.

- **(0,) L-R** (4.0 min/case): reject -- Dice gain +0.0041 < +0.0071
- **(1,) P-A** (4.0 min/case): reject -- Dice gain +0.0046 < +0.0071
- **(2,) I-S** (4.0 min/case): reject -- Dice gain +0.0023 < +0.0071
- **(0,1)** (8.0 min/case): reject -- 8.0 min/case exceeds the 5.0 min ceiling

**Result: no subset qualifies. The container ships unchanged, 5 folds
with mirroring off.** Reduced mirroring is now measured and closed.
