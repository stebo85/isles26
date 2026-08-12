---
name: ISLES'26 model status
status: active
tags: [isles26, models, decisions]
description: Canonical current standings, retractions, production choice, and next actions.
---

# ISLES'26 Model Status — Living Leaderboard

Challenge task: native-space T1w -> binary infarct mask. **Sanity/submission
phase opens 2026-07-30**; Docker submission closes 2026-08-15 23:59 CET.

> **READ [the official metrics reference](../evaluation/official-metrics.md) FIRST.**
> The challenge ranks on five metrics — Dice, absolute volume difference,
> absolute lesion count difference, lesion-wise F1 (panoptica RQ at **one-to-one
> IoU >= 0.25**), and **PR-AUC over a soft map**. Every `lesion_f1` number
> written in this repo before 2026-07-28 used a *lenient* 1-voxel any-overlap
> many-to-many criterion and is NOT the challenge metric. PR-AUC had never been
> measured at all. See the Retractions section below before trusting any
> historical row.

Evaluation surfaces, in decreasing order of how much you should believe them:

- **Center-held-out CV** (`Dataset507`, n=1452) — folds hold out whole CENTERS.
  This is the honest estimate, because the hidden test set spans 60+ centers.
  **Dice 0.6347.**
- **Random-fold CV** (`Dataset502`, n=1450) — stratified random folds in which
  52 of 55 centers appear on both sides of every fold. **Dice 0.6558, and this
  is center-leaking.** It is the number this project quoted as its headline for
  weeks; the honest figure is 2.1 points lower, and absolute volume difference
  is 27% worse (4.94 -> 6.28 mL).
- **soop_bench** (n=12, DWI-defined) — **retired as a decision surface.** R3.0
  training included the SOOP subjects, the labels are defined on a modality we
  do not see at test time, and every "win" measured on it was an in-sample
  argmax over a 49-point grid on twelve cases.

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

- **Ship:** nnU-Net 3D fullres, Dice+TopK10 loss, 1000 epochs, **5-fold
  ensemble + mirroring TTA**, from **Dataset507** (center-grouped folds,
  corrected 1452-case conversion). Weights staged by
  `analysis_nnunet_50_export_submission_weights.sh` (596 MB, tensor-identity
  verified).
- **Emit BOTH** a binary mask and a float32 probability map. PR-AUC is one of
  the five ranked metrics and a mask-only submission forfeits it.
- **Operating point** (threshold, min-component-size) is selected on the
  center-held-out surface against all five official metrics, not on Dice —
  see `baselines/reports/official_metrics_d507_topk10/`.
- **Do NOT ship the 3-family MAPPING ensemble.** Measured on the same 1450 cases
  it is worse than TopK10 alone on Dice (-0.0005), lenient lesion-F1 (-0.0005)
  and absolute volume difference (+0.217 mL) at 3x the inference cost and 5x the
  weight bytes, because its weakest member (ResEnc-L, Dice 0.639) is averaged in
  at equal weight.
- **Do NOT put a second network in the container for OOD.** The synth-plus blend
  "win" is an in-sample grid maximum: 0.3006 in-sample vs **0.2827** under honest
  leave-one-out selection, paired difference +0.0140 with SE 0.0179 (t=0.78),
  and dropping the single best-responding subject leaves -0.0015.

## Retractions (2026-07-28)

Each of these was believed, is wrong or unsupported, and is corrected here.

| Claim | Status | Why |
|---|---|---|
| "lesion-F1 0.6735" (and every `lesion_f1` in `baselines/reports/` and the leaderboard) | **RETRACTED — relabelled** | Computed with 1-voxel any-overlap many-to-many matching. The challenge uses one-to-one IoU >= 0.25. A bound reconstructed from the committed per-case artifacts puts the official value at <= 0.6015; the bias is model-dependent (-0.070 to -0.084), i.e. 6-14x larger than the 0.0025-0.003 deltas model selection was being made on. |
| "Connected-component postprocessing: none chosen (pruning hurts)" | **RETRACTED** | `nnUNetv2_find_best_configuration` only ever tests `remove_all_but_largest_component`, and only accepts it on a Dice gain. **Min-size pruning was never a candidate.** Two ranked metrics depend on it directly. Being re-tested by `analysis_nnunet_47_official_metric_sweep.sh`. |
| Threshold 0.54 | **RETRACTED** | Selected in-sample on the lenient F1 by +0.0025 (z=1.69, not significant) while losing Dice by -0.0011 (z=-5.62, significant). The 0.35-0.55 grid was also truncated at its own Dice optimum — Dice was still rising at the 0.35 edge. Grid widened to 0.15-0.75 and selection is now nested. |
| "CV Dice 0.6558 estimates test performance" | **RETRACTED** | Center-held-out is **0.6347**, and AVD is 27% worse. Quote 0.635. |
| "MSL / DBL are the small-lesion recall levers to promote" | **ABANDONED** | Their recall gains were measured with the 1-voxel criterion; 52% of the lesions counted as "detected" under that rule have overlap/|gt| < 0.25 and cannot match officially. `msl_fold0` and `dbl_fold0` have no `per_case.json`, so they cannot even be re-scored. Do not spend GPU-days here. |
| "Synthetic-lesion insertion: neutral" | **WORSE THAN NEUTRAL, and confounded** | Under the official criterion it loses RQ -0.0158 and |Δcount| +0.0993. Separately, its `contrast_std` parameter was **inert** — a `torch.minimum` clamp against a global low quantile dominated the local contrast term, so every synthetic lesion was painted near the brain's low quantile and the augmentation could not produce the subtle lesions it existed to produce. Fixed; the committed CV result tested a degenerate corner and is void. |
| "soop_bench is our OOD compass" | **RETIRED** | Contaminated (SOOP subjects are in R3.0 training), n=12, DWI-defined labels. The center-held-out Dataset507 out-of-fold surface (n=1452) is the correct compass and existed all along. |
| "1450 usable cases / 3 conversion skips" | **CORRECTED** | Dataset507 has 1452 usable, 1 skip (ATLAS_1424). |
| Chronicity distribution {ge180d 649, lt180d 433, unknown 371} | **CORRECTED** | The `CHRONICITY` column the challenge supplies was never parsed. 270 of the 369 cases with missing/non-positive `DAYS_POST_STROKE` carry CHRONICITY=1. True distribution is approximately {919, 433, 101}. NOTE: CHRONICITY=1 means **chronic (>=180d)** — verified on the 26 cases where both fields exist — so `docs/challenge.md` is the document that is wrong. |
| "5-fold ensemble + TTA, ~0.656 CV" | **MISATTRIBUTED** | 0.656 is a single-model out-of-fold number; nnU-Net predicts each case with the one fold that held it out. The 5-fold test-time ensemble has never been measured. |
| "Dice+TopK10 is a WIN over 1000ep" | **RIGHT CALL, WRONG EVIDENCE** | The Dice delta is +0.0026 at t=1.16 (a tie). TopK10 is still the right primary because it wins significantly on AVD (-0.352 mL, t=-3.10) and lesion precision (t=+2.47). Cite those. |

## Retraining experiments (fold-0, center-grouped, 1000 epochs)

Schedule-matched against the existing fold-0 TopK10 model, scored with the
official five metrics, gate fixed before the runs (RQ >= +0.020 or AVD <= -0.30 mL,
Dice drop <= 0.005).

| arm (5-fold OOF, n=1452) | Dice | AVD mL | count diff | RQ | PR-AUC | verdict |
|---|---:|---:|---:|---:|---:|---|
| **TopK10 (SHIPPED)** | 0.6283 | 6.373 | 1.869 | 0.5791 | 0.7417 | **in the container** |
| DA5 heavy augmentation | 0.6393 | 6.421 | 1.881 | 0.5898 | 0.7441 | **NOT promoted** |
| R1+R2 blob + volume loss (fold 0) | 0.6664 | 6.532 | 1.342 | 0.6278 | 0.7624 | NOT promoted |

**DA5 improves Dice +0.0111 (t=+3.24) and RQ +0.0107 (t=+2.26) with nothing
degraded — and was still not promoted**, because the gate fixed before the runs
required RQ >= +0.020 or AVD <= -0.30 mL. Holding it cost ~0.011 Dice. The reason
to hold it: on fold 0 alone this arm looked like +0.0194 Dice and **-0.397 mL
AVD**, and the AVD claim reversed to +0.047 once the other four folds landed. A
gate renegotiated after seeing the result is not a gate. Full analysis:
`baselines/reports/metric_aligned_fold0/R3R4_DA5_FINDING.md`.

R1+R2 (instance-balanced blob loss + volume term) moved its own target metric
backwards: RQ -0.0323 at fold 0 (t=-2.75). Under one-to-one IoU>=0.25 matching an
unmatched predicted component is penalised twice, so interventions that add
marginal detections lose. That is now the fourth such result (MSL, DBL,
synthetic-lesion insertion, blob loss) and should be treated as settled.

## Shipped configuration (locked 2026-08-04, amended 2026-08-11)

nnU-Net v2 3d_fullres, Dice+TopK10, 1000 epochs, Dataset507 center-grouped,
**5-fold ensemble, mirroring TTA ON when a GPU is present and OFF on CPU**,
threshold 0.35, min connected component 20 voxels (26-connectivity), constant
soft map when the mask is empty.

### 2026-08-11: first real submission failed, and what it changed

Submission `0b843f9e` failed the Grand Challenge health check. Two findings, both
now fixed in `submission/isles26_algorithm/`.

**1. The weights were baked to `/opt/ml/model`, which Grand Challenge overmounts.**
That path is GC's mount point for a separately uploaded model tarball, and GC
bind-mounts it read-only **even when no tarball was uploaded** — an empty
directory that shadowed all 596 MB. `initialize_from_trained_model_folder` raised
`FileNotFoundError: /opt/ml/model/dataset.json` inside the FastAPI lifespan, which
runs *before* uvicorn binds port 4743, so the server never listened and the only
symptom GC surfaced was "health check not passed within 300 seconds". Weights now
live at `/opt/app/model`; `resolve_model_dir()` uses the GC mount only when it
actually contains a model.

**CI could not have caught this**, and that is the more important lesson: the
organizers' `do_test_run.sh` mounts the host `model/` directory over
`/opt/ml/model`, and CI had just filled that directory with the real weights. The
mount shadowed the image layer with *identical content*. A test that provisions
the resource under test cannot detect that the resource is missing in
deployment. CI now empties `model/` before the end-to-end run.

**2. Grand Challenge gave us No GPU and 2 CPU threads.** Every runtime figure in
this repo is an 8-core measurement, so the CPU path is roughly 4x worse than the
tables below: 5-fold no-TTA is ~8 min/case, not 2.0, against a ~10 minute budget.

**TTA is back on for the GPU path.** The no-TTA decision was correct but
conditional — it rested on the stated assumption that "Grand Challenge does not
guarantee a GPU, so the CPU figure is the one that must fit". A T4 (16 GB) can be
requested, and on GPU 5 folds *with* mirroring runs ~30 s/case. That recovers
Dice +0.0141, lesion-F1 +0.0205, PR-AUC +0.0191 and AVD −0.86 mL — four of the
five ranked metrics. It is wired as `device.type == "cuda"` rather than a hard
`on`, so a job scheduled without a GPU degrades to no-TTA instead of timing out.
**On the GPU path the quoted 0.6283 is now the schedule-matched forecast rather
than an optimistic bound**, since every out-of-fold number here was measured with
TTA.

Requesting the T4 is a Grand Challenge algorithm *setting*, not a container
property. If it is not set, the container silently runs the CPU fallback.

Both open decisions are now closed, and neither moved the config.

**TTA vs folds — trigger fired, choice held.** The container carried a
pre-registered trigger: revisit in favour of 1 fold + TTA if TTA proved worth
more than ~0.005 Dice. `analysis_nnunet_61` measured **+0.0141 Dice** (fold 0,
n=281, TTA the only variable), so the revisit happened. It failed on a fact the
trigger did not have: **1 fold + TTA is slower than 5 folds without it** —
183.9 s vs 114.2 s on ATLAS_0001, measured through the container's own
`inference.py` on CPU (`analysis_nnunet_66`). The alternative costs ~60% more
wall time for a single-model prediction, and would need the 5-fold ensemble to be
worth less than +0.0141 Dice to win. That ensemble gain is unmeasurable here
(every case was seen by 4 of 5 members), so this is a judgement under one
unmeasured quantity, recorded as such.

**Operating point — held.** `thr0.35_k20` is rank 1 of 54 by mean rank across the
five official metrics on the center-held-out surface (n=1452), and rank 16 of 54
on the deployment-matched no-TTA surface (n=281) where the argmax is
`thr0.25_k50`. Not moved: that argmax is in-sample over a 54-point grid on 281
single-fold cases and worth 0.0030 Dice — the same construction as the retracted
threshold-0.54 decision above. PR-AUC is threshold-invariant (0.7417 everywhere),
so one of the five metrics cannot respond to the change at all.

**Cost of shipping no-TTA**, to be carried into any leaderboard comparison:
Dice −0.0141, lesion-F1 −0.0205, PR-AUC −0.0191, AVD +0.86 mL. **The quoted
0.6283 Dice was measured WITH TTA — it is an optimistic bound, not a forecast.**

**Reduced mirroring — measured, rejected, closed.** A subset of
`allowed_mirroring_axes` costs 2^k passes instead of 8, so one axis fits the CPU
budget at 5 folds. It does not buy enough: single axes give +0.0041 (L-R),
+0.0046 (P-A) and +0.0023 (I-S) Dice against full TTA's +0.0141, i.e. the gain is
strongly super-additive and no single axis recovers half of it. The two-axis pair
reaches +0.0113 but costs ~7-8 min/case, inside the ~10 min budget only if Grand
Challenge's CPUs are no slower than ours. Absolute volume difference — a ranked
metric — improves only under full mirroring (-0.856 mL, t=-3.46); every subset is
marginally worse than no-TTA on it. The rule fixed before the run rejects all
four. Report: `baselines/reports/reduced_mirroring/`.

Weights published at `https://huggingface.co/sbollmann/isles26-nnunet-d507-topk10`.

## Next levers (priority)

The model is essentially done; nothing available in the remaining time beats it
by more than noise. **All remaining value is in packaging, geometry correctness,
and choosing the operating point against the right five metrics on the right
surface.**

1. **Ship a valid container on day one of the sanity phase (2026-07-30).** This
   is the only existential risk. `submission/isles26_algorithm/` now contains a
   Dockerfile, a Grand Challenge entrypoint, exact-inverse geometry handling and
   a passing 48-orientation round-trip test. The remaining blocker is that
   **no container can be built on Sherlock** — no docker/podman/buildah/skopeo,
   and apptainer cannot emit a `docker save` tarball. A build host is needed.
2. **Answer the open interface questions** in
   `submission/isles26_algorithm/FORUM_QUESTIONS.md` — output socket slugs, and
   whether a soft map is a required second output.
3. **Select threshold and min-component-size** on the center-held-out surface
   using all five official metrics (`analysis_nnunet_47`/`49`).
4. **Nested holdout:** predict the Dataset502 fold-0 validation cases with folds
   1-4 only. This is the first honest measurement of the 5-fold test-time
   ensembling gain, and it recalibrates the threshold for an *ensembled* softmax
   (averaging pulls probability mass off 0 and 1; every operating point we own
   was fitted on single-model softmax).
5. **10-model ensemble at zero training cost:** average Dataset502 and
   Dataset507 TopK10 folds. Free split diversity; validate on (4) and adopt only
   if non-losing.
6. **Use the sanity phase as a measurement instrument** — one variable per
   submission, V1 as control. It is the only out-of-distribution signal we will
   ever get, and it costs no compute.

Explicitly NOT doing: metadata conditioning (measured post-hoc gain 0.0000;
CHRONICITY is fully redundant with DAYS_POST_STROKE where both exist), further
MSL/DBL/synth-lesion work, self-training/Dataset504, the 3-family MAPPING
ensemble.

## Git Note

`main` was synced with `origin/main` before this experiment. The current work is
on branch `self-training-hidden-generalization`; push still requires human
credentials outside the cluster.

Related fibers: [[challenge/overview]], [[evaluation/official-metrics]],
[[models/synthstroke-lessons]], [[lessons/problems]].
