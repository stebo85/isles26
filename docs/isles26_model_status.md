# ISLES'26 Model Status — Living Leaderboard

Goal: a model that is **extremely good on the training data (ATLAS T1w)** and
**works OK on the SOOP benchmark** (DWI-defined OOD proxy, `data/soop_bench`,
n=12). Challenge task: native T1w → binary infarct mask. Closes 2026-08-01.

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
| **nnU-Net default, R3.0, 1000ep, 5-fold** | **0.6528** | _eval pending_ | **best in-dist (+1.56%)** |
| nnU-Net ResEnc-L, R3.0, 250ep | 0.6389 | — | tie — no gain |
| **OOD blend: nnU-Net(0.4)+synth-plus(0.6) on T1w** | — | **0.2886 / 0.202** | **best OOD-realistic** |
| synth-plus on DWI/TRACE directly | 0.458 (R2.1) | 0.447 | needs DWI (not at T1w test time) |
| DeepISLES (DWI specialist) | — | 0.540 / 0.70 | reference |

## What worked / didn't (this campaign)

- **Longer schedule (250→1000 epochs): WIN.** Consolidated CV Dice 0.6372 →
  **0.6528**; all 5 folds improved (0.645/0.655/0.645/0.635/0.685). The single
  best in-distribution lever found so far.
- **OOD T1w ensemble: WIN.** Blending nnU-Net + synth-plus probabilities, both
  run on the **T1w input only** (challenge-realistic) and warped to TRACE,
  lifts soop mean Dice 0.2546 → **0.2886** and median 0.040 → **0.202** at
  w_nn=0.4, thr=0.30. synth-plus carries the acute-DWI cases nnU-Net misses.
- **ResEnc-L: no gain** (0.6389 vs 0.6376 — within noise).
- **Connected-component postprocessing: none chosen** (no CV gain; ATLAS
  lesions are multifocal so pruning hurts).
- **MSL (size-stratified labels): in progress** (fold-0 hypothesis test).

## Production recommendation (current)

- **Primary submission (T1w in-dist):** nnU-Net R3.0 default plans, **1000
  epochs**, 5-fold ensemble + TTA. ~0.653 CV.
- **For OOD robustness:** add synth-plus and blend probabilities in native T1w
  space (w_nn≈0.4, thr≈0.30) before warping. Improves soop without DWI.

## In flight (jobs)

- 1000ep find_best postprocessing pick (job 30596908) — consolidated 0.6528.
- 1000ep OOD ensemble refresh (30596963→30596964) — rebuild blend on the
  stronger 1000ep nnU-Net probs.
- MSL fold-0 (30582935 → eval 30582936) — small-lesion recall test.

## Next levers (priority)

1. Finish MSL fold-0 eval; scale to 5-fold only if small-lesion recall rises
   without Dice loss.
2. 1000ep soop binary eval + 1000ep-based OOD blend (refresh underway).
3. nnU-Net ResEnc-L at 1000 epochs (only if 1000ep default keeps scaling).

## Push note

Cluster has no GitHub credentials (no `gh`, token, SSH key, or helper), so
`git push` fails here. Commits land locally on `main`; a human must push
(`origin` = https://github.com/stebo85/isles26). Local `main` is several
commits ahead of `origin/main`.
