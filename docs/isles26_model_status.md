# ISLES'26 Model Status — Living Leaderboard

Goal: a model that is **extremely good on the training data (ATLAS T1w)** and
**works OK on the SOOP benchmark** (DWI-defined OOD proxy, `data/soop_bench`,
n=12). Challenge task: native T1w -> binary infarct mask. Docker submission
closes 2026-08-15 23:59 CET.

Metrics:
- **In-dist** = ATLAS R3.0 5-fold cross-validation over all 1450 cases.
  nnU-Net's own consolidated summaries are retained as `crossval_summary.json`;
  leaderboard/report rows use the repo evaluator when available so Dice,
  lesion-F1, HD95, surface Dice, and volume error are head-to-head comparable.
- **OOD proxy** = mean Dice on `data/soop_bench` (predicted, warped to TRACE).
  CAVEAT: R3.0 training INCLUDED the 169 `sub-soop*` subjects, so soop numbers
  may be optimistically biased.

## Current standings

| Model | In-dist CV Dice | soop_bench Dice (mean / median) | Verdict |
|---|---:|---:|---|
| nnU-Net default, R2.1, fold-0 | 0.640 (n=194) | 0.188 / — | prior baseline |
| nnU-Net default, R3.0, 250ep, 5-fold+TTA | 0.6372 | 0.255 / 0.036 | superseded |
| nnU-Net default, R3.0, 1000ep, 5-fold | 0.6528 | — | +1.56% over 250ep |
| **nnU-Net Dice+TopK10, R3.0, 1000ep, 5-fold** | **0.6558** | 0.287 / 0.154 (nn-only) | **BEST single model (MAPPING recipe)** |
| nnU-Net Dice+TopK10 **+ synth-lesion insertion**, R3.0, 1000ep, 5-fold | 0.6558 | — | recall↑ / F1↓ — no net gain |
| nnU-Net ResEnc-L, R3.0, 250ep | 0.6389 | — | tie — no gain |
| **OOD blend: TopK10 nnU-Net(0.6)+synth-plus(0.4) on T1w** | — | **0.3006 / 0.257** | **best T1w-only OOD blend** |
| OOD blend: 250ep nnU-Net(0.4)+synth-plus(0.6) on T1w | — | 0.2886 / 0.202 | superseded |
| OOD blend: 1000ep nnU-Net(0.5)+synth-plus(0.5) on T1w | — | 0.2884 / 0.199 | same as 250ep |
| synth-plus on DWI/TRACE directly | 0.458 (R2.1) | 0.447 | needs DWI (not at T1w test time) |
| DeepISLES (DWI specialist) | — | 0.540 / 0.70 | reference |

## What worked / didn't (this campaign)

- **Longer schedule (250→1000 epochs): WIN.** Consolidated CV Dice 0.6372 →
  **0.6528**; all 5 folds improved (0.645/0.655/0.645/0.635/0.685).
- **Dice+TopK10 loss @ 1000ep (MAPPING recipe): WIN, new best.** nnU-Net
  consolidated CV is **0.6551** (per-fold 0.647/0.661/0.651/0.638/0.681),
  edging plain 1000ep (0.6528). The rich repo-harness OOF report is now
  **Dice 0.6558 / lesion-F1 0.6735 / surface Dice 0.7679 / HD95 18.10 mm**,
  so it can be compared directly to the MAPPING-style ensemble. Postprocessing
  again chose none.
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
- **TopK10 OOD blend: best T1w-only OOD so far.** TopK10 nn-only on soop is
  0.2866 / 0.1538; blending TopK10 + synth-plus reaches **0.3006 / 0.2566** at
  w_nn=0.6, thr=0.30. This is still selected on the same 12-case soop grid, so
  treat it as a robustness signal rather than a blind validation score.
- **DBL fold-0: modest small-lesion recall gain, small Dice cost.** DBL merged
  to binary gives Dice 0.6317 vs binary 0.6374. Tiny recall improves 0.183 →
  0.204 and small recall 0.339 → 0.353, but medium/large/huge recall dip
  slightly. This is less attractive than MSL's small-recall jump unless the
  hidden metric strongly rewards tiny lesions.
- **Synthetic-lesion insertion augmentation (full 5-fold CV): neutral, not a
  win.** Size-rebalanced synthetic lesion insertion on top of Dice+TopK10, run
  as a full 1450-case 5-fold CV (not just fold-0). It does what it was built to
  do at the margin: lesion recall rises 0.7065 → **0.7131** and every size bin
  ticks up (tiny <10 vox 0.156 → 0.162, small 10-100 0.288 → 0.293, medium
  0.607 → 0.625). But it is a precision trade — lesion-F1 **drops 0.6735 →
  0.6647**, surface Dice 0.7679 → 0.7647, HD95 18.10 → 18.19 mm, AVD 4.94 →
  4.97 mL, and Dice is flat (0.65584 → 0.65580). Same pattern as MSL/DBL:
  recall-positive, precision-negative, no aggregate gain. Keep plain TopK10 as
  the primary; revisit only if the hidden metric is lesion-detection-weighted.
  Report: `baselines/reports/nnunet_synthlesion_topk10_crossval/`; scripts
  `analysis_nnunet_40_train_synthlesion_topk10.sh` /
  `analysis_nnunet_41_postprocess_eval_synthlesion_topk10.sh`. Postprocessing
  again chose none.
- **Leaderboard is reconciled.** `baselines/compare/leaderboard.*` now includes
  R3.0 250ep/1000ep/TopK10 CV rows, the R3.0 soop row, and the OOD blend rows.
- **Hidden-generalization scaffolding exists.** Leave-center-out and
  leave-chronicity-out split plans live in
  `baselines/reports/hidden_generalization_splits/`; TopK10 out-of-fold grouped
  triage lives in `baselines/reports/nnunet_topk10_hidden_generalization/`.

## Production recommendation (current)

- **Primary submission (T1w in-dist):** nnU-Net R3.0, **Dice+TopK10 loss, 1000
  epochs**, 5-fold ensemble + TTA. **~0.656 CV** (best). (Plain 1000ep 0.653 is
  a near-equivalent fallback.)
- **For OOD robustness:** add synth-plus and blend probabilities in native T1w
  space (w_nn≈0.4, thr≈0.30) before warping. Improves soop without DWI.

## Next levers (priority)

Everything cheap/obvious has now been tried (see "What worked / didn't" above
and Problem 16 in `problems_and_lessons.md`). The recurring blocker is now
clear: **every small-lesion recall lever we have tried (MSL, DBL, synth-lesion
insertion) trades precision for recall and leaves Dice flat-to-worse.** So the
next levers split into (a) ensembling/packaging proven members, and (b) one
last attempt to convert recall into Dice rather than chasing more recall.

Remaining untested ideas, in rough priority:

0. **Decide whether recall even matters for ranking.** Before spending more
   compute on recall levers, confirm the hidden challenge metric. If it is
   Dice/overlap-weighted, stop tuning recall and ship TopK10 as-is; if it is
   lesion-wise detection (F1/recall), then MSL — not synth-lesion — is the
   strongest recall lever measured so far and should be promoted to a full CV.
1. **Tune synth-lesion to keep recall without the precision bleed (cheap):**
   the augmentation works mechanically — the loss is false positives. Re-run a
   single fold with a lower insertion rate / smaller synthetic lesions, or add a
   small-component FP penalty at inference (min-size pruning on *added* detections
   only), and check whether F1 recovers above 0.6735 while holding recall. Only
   promote to full CV if fold-0 beats plain TopK10 on F1 *and* Dice.
2. **Package the TopK10 OOD blend:** the actual TopK10 blend is complete and is
   now the best T1w-only OOD result: `baselines/reports/ood_ensemble_topk10/`.
3. **Self-training / pseudo-labels** on opt-in unlabeled T1w. Scripts
   `analysis_nnunet_26_selftrain_pseudolabels.sh` and
   `analysis_nnunet_27_prepare_selftrain_dataset.sh` build Dataset504 with
   TopK10 pseudo-labels and train-only pseudo cases; `analysis_nnunet_33_train_selftrain.sh`
   trains the self-training model once legitimate unlabeled inputs are staged.
4. **MAPPING-style softmax ensemble package:** `analysis_nnunet_34_mapping_ensemble_predict.sh`
   now runs TopK10 + default 1000ep + ResEnc-L + Dataset504 self-training members
   on nnU-Net-format T1w inputs, saves each member's softmax, and averages them via
   `nnunet_helpers/average_softmax_ensemble.py`. It is blocked only on the Dataset504
   checkpoints if self-training has not been run yet.
5. **True hidden-generalization retraining:** use the generated split plan to
   run leave-center-out and leave-chronicity-out jobs for the worst/high-risk
   groups surfaced by the TopK10 grouped report (not every tiny center first).
6. **MSL+DBL only if we want another small-lesion gamble:** DBL fold-0 is
   complete and only modestly helps tiny/small recall while costing Dice. Try
   `DBL_SCHEME=msl_dbl` only if we decide the small-lesion objective is worth
   another fold-0 experiment.
7. **Package the final container:** TopK10 5-fold ensemble + any proven
   OOD/self-training/DBL additions into the submission container.

## Git Note

`main` was synced with `origin/main` before this experiment. The current work is
on branch `self-training-hidden-generalization`; push still requires human
credentials outside the cluster.
