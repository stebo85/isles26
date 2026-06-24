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
| nnU-Net default, R3.0, 1000ep, 5-fold | 0.6528 | — | +1.56% over 250ep |
| **nnU-Net Dice+TopK10, R3.0, 1000ep, 5-fold** | **0.6551** | _~0.26 (T1w)_ | **BEST in-dist (MAPPING recipe)** |
| nnU-Net ResEnc-L, R3.0, 250ep | 0.6389 | — | tie — no gain |
| **OOD blend: nnU-Net(0.4)+synth-plus(0.6) on T1w** | — | **0.2886 / 0.202** | **best OOD-realistic** |
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

## Production recommendation (current)

- **Primary submission (T1w in-dist):** nnU-Net R3.0, **Dice+TopK10 loss, 1000
  epochs**, 5-fold ensemble + TTA. **~0.655 CV** (best). (Plain 1000ep 0.653 is
  a near-equivalent fallback.)
- **For OOD robustness:** add synth-plus and blend probabilities in native T1w
  space (w_nn≈0.4, thr≈0.30) before warping. Improves soop without DWI.

## Next levers (priority)

1. MSL is a Dice-vs-recall tradeoff at 250ep/fold-0; before scaling to 5-fold,
   consider a precision-aware loss (Tversky) or higher size thresholds to keep
   the recall gain while recovering Dice.
2. nnU-Net ResEnc-L at 1000 epochs (only if 1000ep default keeps scaling — but
   ResEnc was a tie at 250ep, so low priority).
3. Try 1000ep + region-based / TopK loss (MAPPING used TopK10).

## Push note

Cluster has no GitHub credentials (no `gh`, token, SSH key, or helper), so
`git push` fails here. Commits land locally on `main`; a human must push
(`origin` = https://github.com/stebo85/isles26). Local `main` is several
commits ahead of `origin/main`.
