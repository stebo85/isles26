# R3+R4 (DA5 heavy domain/resolution augmentation): PASSES THE GATE — promoted

Dataset507 (center-grouped) fold 0, 1000 epochs, n=281 validation cases, scored
with the official five metrics at the shipped operating point (thr 0.35, k=20).
Control is the existing fold-0 `nnUNetTrainerDiceTopK10Loss` at the same schedule.

| metric | TopK10 (control) | DA5+TopK10 | delta | paired t | R1+R2 (failed) |
|---|---:|---:|---:|---:|---:|
| **Dice** | 0.6605 | 0.6799 | **+0.0194** | **+2.61** | 0.6664 |
| **absolute volume difference (mL)** | 6.5153 | 6.1181 | **-0.3972** | -1.64 | 6.5318 |
| absolute lesion count difference | 1.1566 | 1.3238 | +0.1673 | — | 1.3416 |
| lesion-F1 (RQ) | 0.6601 | 0.6727 | +0.0126 | +1.11 | 0.6278 |
| PR-AUC | 0.7673 | 0.7791 | +0.0118 | — | 0.7624 |

**Gate (fixed before the run): promote iff RQ >= +0.020 or AVD <= -0.30 mL, with
Dice not dropping more than 0.005. Result: PROMOTE**, on the AVD criterion.

## Read this honestly

Four of five ranked metrics improve; absolute lesion count difference worsens
(+0.167).

The criterion that actually fired -- AVD -0.397 mL -- is **not individually
significant** (t = -1.64). The gate was pre-registered on effect size, not on
significance, and it is met as written. What genuinely carries the decision is
the **Dice gain of +0.0194 at t = +2.61**, which is significant, together with
the consistent positive direction across four metrics. Do not quote the AVD
improvement as an established effect until the full 5-fold run is in.

## Why this is the one lever that worked

It is the only intervention that targeted a loss which was both large and
previously unaddressed:

- center shift: Dice -0.021, AVD +27% when folds hold out whole centers
- spacing: Dice 0.670 at 1 mm isotropic vs 0.374 at (1.5, 1.5, 1.5)

and the hidden test set spans 60+ centers of native-space clinical data. Every
other lever tried in this campaign either chased marginal detections (MSL, DBL,
synthetic-lesion insertion, the R1+R2 blob loss -- all four lost under one-to-one
IoU >= 0.25 matching) or tried to recover from the soft map information that is
not in it (Problems 21-22).

Note also that AVD moved here after resisting both post-hoc calibration and a
direct volume loss term. That is consistent with the diagnosis in Problem 22:
volume error is a property of the model's domain robustness, not something
recoverable at the output.

## Promotion

Folds 1-4 submitted (jobs 37118736/7/42/8), same 1000-epoch schedule, same
center-grouped split. Auto-resume is in place, so preemption costs at most the
current epoch rather than the whole run.

Full 5-fold out-of-fold scoring must repeat this comparison before the container's
weights are swapped. **Do not ship the fold-0 result.**
