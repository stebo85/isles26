# R1+R2 (instance-balanced "blob" loss + soft volume term): FAILS THE GATE

Dataset507 (center-grouped) fold 0, 1000 epochs, n=281 validation cases, scored
with the official five metrics at the shipped operating point (thr 0.35, k=20).
Control is the existing fold-0 `nnUNetTrainerDiceTopK10Loss` at the same 1000
epochs -- exactly schedule-matched, which is why 1000 epochs was chosen over a
cheap 250.

| metric | TopK10 (control) | blob+volume | delta | paired t |
|---|---:|---:|---:|---:|
| Dice | 0.6605 | 0.6664 | **+0.0059** | +1.15 |
| absolute volume difference (mL) | 6.5153 | 6.5318 | +0.0165 | +0.09 |
| absolute lesion count difference | 1.1566 | 1.3416 | +0.1851 | — |
| **lesion-F1 (RQ)** | 0.6601 | 0.6278 | **-0.0323** | **-2.75** |
| PR-AUC | 0.7673 | 0.7624 | -0.0049 | — |

**Gate (fixed before the run): promote iff RQ >= +0.020 or AVD <= -0.30 mL, with
Dice not dropping more than 0.005. Result: DO NOT PROMOTE.**

## It moved the target metric the wrong way, significantly

The blob term exists to improve Recognition Quality by weighting every ground-truth
instance equally. It made RQ **worse** by 0.032 (t = -2.75), while nudging Dice up
by 0.006 (not significant). Absolute lesion count difference also worsened
(+0.185).

The most likely mechanism, consistent with all three numbers: weighting small
instances equally pushes the model to put probability mass around every small
lesion, which produces **more predicted components**. Under the official
one-to-one IoU >= 0.25 matching, a predicted component that does not reach IoU
0.25 with any reference instance is penalised twice -- it is a false positive AND
it leaves a reference instance unmatched -- so extra marginal detections cost RQ
even when they raise voxel overlap. The count metric rising alongside is the same
effect seen directly.

This is the fourth time in this campaign that a recall-oriented intervention
(MSL, DBL, synthetic-lesion insertion, and now the blob loss) has traded
precision for recall and lost under the real metric. The pattern is now
well-evidenced enough to treat as settled: **on this task, under one-to-one
IoU >= 0.25 matching, interventions that add marginal detections lose.**

## The one genuinely useful signal

The volume term did nothing (AVD +0.017 mL, t = +0.09). Combined with Problems
21-22 -- where post-hoc volume calibration also failed and the threshold was shown
to have less leverage than the volume error -- absolute volume difference now looks
resistant to both post-processing *and* a direct loss term. It should not absorb
more effort before the deadline.

Note this arm's own rank-optimal configuration is `thr0.20_k35` (Dice 0.6697,
AVD 6.194, RQ 0.6397), i.e. even tuned in its own favour it does not reach the
control's RQ of 0.6601.

## Status of the sibling arm

R3+R4 (`nnUNetTrainerDA5TopK10`, heavy domain/resolution augmentation) is still
training and is unaffected by this result -- it targets the center-shift and
spacing losses, not instance balance.
