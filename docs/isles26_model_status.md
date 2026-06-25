# ISLES'26 Model Status — Living Leaderboard

Goal: a model that is **extremely good on the training data (ATLAS T1w)** and
**works OK on the SOOP benchmark** (DWI-defined OOD proxy, `data/soop_bench`,
n=12). Challenge task: native T1w -> binary infarct mask. Docker submission
closes 2026-08-15 23:59 CET.

Metrics:
- **In-dist** = ATLAS R3.0 5-fold cross-validation Dice over all 1450 cases
  (the canonical nnU-Net number; consolidated via find_best_configuration).
- **OOD proxy** = mean Dice on `data/soop_bench` (predicted, warped to TRACE).
  CAVEAT: R3.0 training INCLUDED the 169 `sub-soop*` subjects, so soop numbers
  may be optimistically biased.

## Current standings

| Model | In-dist CV Dice | soop_bench Dice (mean / median) | Verdict |
|---|---:|---:|---|
| nnU-Net default, R2.1, fold-0 | 0.640 (n=194) | 0.188 / — | prior baseline |
| nnU-Net default, R3.0, 250ep, 5-fold+TTA | 0.6372 | 0.255 / 0.036 | superseded |
| nnU-Net default, R3.0, 1000ep, 5-fold | 0.6528 | — | +1.56% over 250ep |
| **nnU-Net Dice+TopK10, R3.0, 1000ep, 5-fold** | **0.6551** | _~0.26 (T1w)_ | **BEST in-dist (MAPPING recipe)** |
| nnU-Net ResEnc-L, R3.0, 250ep | 0.6389 | — | tie — no gain |
| **OOD blend: nnU-Net(0.4)+synth-plus(0.6) on T1w** | — | **0.2886 / 0.202** | **best OOD-realistic** |
| OOD blend: 1000ep nnU-Net(0.5)+synth-plus(0.5) on T1w | — | 0.2884 / 0.199 | same as 250ep |
| synth-plus on DWI/TRACE directly | 0.458 (R2.1) | 0.447 | needs DWI (not at T1w test time) |
| DeepISLES (DWI specialist) | — | 0.540 / 0.70 | reference |

## What worked / didn't (this campaign)

- **Longer schedule (250→1000 epochs): WIN.** Consolidated CV Dice 0.6372 →
  **0.6528**; all 5 folds improved (0.645/0.655/0.645/0.635/0.685).
- **Dice+TopK10 loss @ 1000ep (MAPPING recipe): WIN, new best.** Consolidated CV
  **0.6551** (per-fold 0.647/0.661/0.651/0.638/0.681), edging plain 1000ep
  (0.6528). TopK10 hard-example mining gives a small additional in-dist gain.
  Postprocessing again chose none.
- **OOD T1w ensemble: WIN.** Blending nnU-Net + synth-plus probabilities, both
  run on the **T1w input only** (challenge-realistic) and warped to TRACE,
  lifts soop mean Dice 0.2546 → **0.2886** and median 0.040 → **0.202** at
  w_nn=0.4, thr=0.30. synth-plus carries the acute-DWI cases nnU-Net misses.
- **ResEnc-L: no gain** (0.6389 vs 0.6376 — within noise).
- **Connected-component postprocessing: none chosen** (no CV gain; ATLAS
  lesions are multifocal so pruning hurts).
- **MSL (size-stratified labels): qualified result** (fold-0). Substantially
  raises small/medium-lesion recall — small (10-100 vox) **0.339 → 0.486**
  (+14.7%), medium (100-1k) 0.589 → 0.641 — but costs ~0.8% Dice (0.6374 →
  0.6298, a precision tradeoff from extra small-lesion false positives). Tiny
  (<10 vox) unchanged (~0.18). Adopt only if the challenge's lesion-wise
  detection metric outweighs the small Dice cost.
- **Precision-aware MSL + Tversky(0.7/0.3): FAILED** (fold-0). Intended to
  recover MSL's Dice cost via an FP penalty, but it made both axes worse:
  Dice 0.6298 → **0.6186**, small recall 0.486 → 0.454, tiny 0.177 → 0.140. The
  FP penalty suppressed true positives without a Dice gain. Plain MSL remains
  the better recall lever; Tversky-on-MSL is not worth pursuing.
- **1000ep OOD blend: same as 250ep** — blending 1000ep nnU-Net + synth-plus on
  T1w gives soop 0.2884 (w_nn=0.5), i.e. the longer schedule helps in-dist but
  not OOD; the blend value comes from synth-plus, not the nnU-Net schedule.
- **Leaderboard is reconciled.** `baselines/compare/leaderboard.*` now includes
  R3.0 250ep/1000ep/TopK10 CV rows, the R3.0 soop row, and the OOD blend rows.
- **Hidden-generalization scaffolding exists.** Leave-center-out and
  leave-chronicity-out split plans live in
  `baselines/reports/hidden_generalization_splits/`; TopK10 out-of-fold grouped
  triage lives in `baselines/reports/nnunet_topk10_hidden_generalization/`.

## Production recommendation (current)

- **Primary submission (T1w in-dist):** nnU-Net R3.0, **Dice+TopK10 loss, 1000
  epochs**, 5-fold ensemble + TTA. **~0.655 CV** (best). (Plain 1000ep 0.653 is
  a near-equivalent fallback.)
- **For OOD robustness:** add synth-plus and blend probabilities in native T1w
  space (w_nn≈0.4, thr≈0.30) before warping. Improves soop without DWI.

## Next levers (priority)

Everything cheap/obvious has now been tried (see "What worked / didn't" above
and Problem 16 in `problems_and_lessons.md`). Remaining untested ideas, in
rough priority:

1. **TopK10 OOD blend:** rerun the T1w probability blend using the actual
   Dice+TopK10 model, not default 250ep/1000ep nnU-Net. Submitted 2026-06-25:
   probability array `31143608`, dependent eval `31143609`, writing
   `work/predictions_ood_ens_topk10/` and
   `baselines/reports/ood_ensemble_topk10/`.
2. **Self-training / pseudo-labels** on opt-in unlabeled T1w. Scripts
   `analysis_nnunet_26_selftrain_pseudolabels.sh` and
   `analysis_nnunet_27_prepare_selftrain_dataset.sh` build Dataset504 with
   TopK10 pseudo-labels and train-only pseudo cases; `analysis_nnunet_33_train_selftrain.sh`
   trains the self-training model once legitimate unlabeled inputs are staged.
3. **True hidden-generalization retraining:** use the generated split plan to
   run leave-center-out and leave-chronicity-out jobs for the worst/high-risk
   groups surfaced by the TopK10 grouped report (not every tiny center first).
4. **DBL / MSL+DBL:** scripts `analysis_nnunet_30_prepare_dbl.sh` through
   `analysis_nnunet_32_eval_dbl.sh` add the next small-lesion relabeling branch.
   Run DBL fold 0 first; try `DBL_SCHEME=msl_dbl` if DBL improves recall without
   the MSL Dice cost. Submitted 2026-06-25: DBL prepare `31144153`, fold-0 train
   `31144154`, eval `31144155`.
5. **Package the final pipeline:** TopK10 5-fold ensemble + any proven
   OOD/self-training/DBL additions into the submission container.

## Git Note

`main` was synced with `origin/main` before this experiment. The current work is
on branch `self-training-hidden-generalization`; push still requires human
credentials outside the cluster.
