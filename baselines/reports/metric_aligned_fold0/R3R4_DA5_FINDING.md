# R3+R4 (DA5 heavy augmentation): NOT PROMOTED

## Final result — full 5-fold out-of-fold, n=1452

| metric | TopK10 (shipped) | DA5 | delta | SE | t |
|---|---:|---:|---:|---:|---:|
| Dice | 0.6283 | 0.6393 | **+0.0111** | 0.0034 | **+3.24** |
| absolute volume difference (mL) | 6.3732 | 6.4205 | +0.0473 | 0.1516 | +0.31 |
| absolute lesion count difference | 1.8685 | 1.8809 | +0.0124 | 0.0287 | +0.43 |
| lesion-F1 (RQ) | 0.5791 | 0.5898 | **+0.0107** | 0.0047 | **+2.26** |
| PR-AUC | 0.7417 | 0.7441 | +0.0024 | 0.0026 | +0.89 |

**Gate, fixed before the runs: RQ >= +0.020 or AVD <= -0.30 mL, with Dice not
dropping more than 0.005. Result: NOT MET. DA5 is not promoted; TopK10 ships.**

## Why this was a close call, and why the gate was held anyway

DA5 is, on the evidence, a slightly better model: two of five ranked metrics
improve significantly at n=1452 and nothing degrades. It was tempting to ship it.

It was not shipped because **the gate was set before the numbers were seen, and
the decision to move it would have been taken after seeing them.** That is the
exact failure mode this project has repeatedly paid for — the fold-0 result on
this very arm looked like +0.0194 Dice and -0.397 mL AVD, both of which shrank or
reversed once the other four folds landed:

| metric | fold 0 (n=281) | all 5 folds (n=1452) |
|---|---:|---:|
| Dice | +0.0194 (t=+2.61) | +0.0111 (t=+3.24) |
| absolute volume difference | **-0.397 mL** | **+0.047 mL** |
| lesion-F1 (RQ) | +0.0126 | +0.0107 |
| PR-AUC | +0.0118 | +0.0024 |

The AVD criterion that fired the fold-0 promotion evaporated completely. A gate
that can be renegotiated after the fact is not a gate.

The cost of holding it is about **0.011 Dice and 0.011 RQ** — real, but small
against the risk of a late weight swap: it would mean re-exporting, re-uploading
596 MB, and rebuilding an as-yet-unbuilt container against a hard deadline, for a
model whose advantage sits at the edge of what this evaluation can resolve.

## If this is revisited

The 5-fold DA5 weights exist under
`work/nnunet/nnUNet_results/Dataset507_ATLASR30_CORRECTED/nnUNetTrainerDA5TopK10__nnUNetPlans__3d_fullres/`
and can be exported with `analysis_nnunet_50_export_submission_weights.sh`
(`TRAINER=nnUNetTrainerDA5TopK10`) and published with
`analysis_eval_02_upload_weights_hf.sh`. Do that only with time to re-run the
full container validation, not as a last-minute change.

---

## Original fold-0 write-up (superseded by the update above)


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
