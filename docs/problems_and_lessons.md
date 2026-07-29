# Problems, Failure Modes, And Lessons

This file turns the ATLAS and ISLES'22 literature into concrete risks for ISLES'26.

## Problem 1: Dice Can Hide Missed Lesions

Dice is dominated by large lesions. If a scan contains one large infarct plus several punctate lesions, a model can get a good Dice score while missing clinically meaningful small lesions.

Both ATLAS and ISLES'22 therefore used lesion-wise metrics. The ISLES'22 dataset paper states that the evaluation should capture both large-lesion borders and detection of small punctiform infarcts. The DeepISLES paper repeats the same point: lesion-wise F1 reflects localization and detection of distinct lesions, while Dice is voxel-overlap biased toward larger lesions.

Action:

- Always report lesion-wise F1, precision, and recall by lesion-size bin.
- Keep Dice, but do not select checkpoints by Dice alone.
- Add connected-component visual reviews for cases with low F1 but acceptable Dice.

## Problem 2: Small Lesions Are Underrepresented By Voxel Count

ATLAS v2.0 has many small connected components but few small-lesion voxels. Shang et al. report that 517 training lesions had volume under 100 voxels, yet their total voxel count was smaller than a single very large lesion can be.

Action:

- Oversample patches containing tiny lesions.
- Stratify folds by lesion volume and lesion count.
- Test MSL/DBL or a similar auxiliary labeling strategy.
- Consider lesion-wise sampling weights instead of purely voxel-wise loss.

## Problem 3: T1w Lesion Appearance Is Subtle And Phase-Dependent

The ISLES'22 dataset paper contrasts the multimodal acute task with ATLAS: chronic T1w lesions are reduced to glial scars, smaller infarcts are difficult to visualize, and etiology can be ambiguous. ISLES'26 expands this T1w problem across acute, sub-acute, and chronic stages, so appearance will vary more than in ATLAS v2.0.

Action:

- Use `DAYS_POST_STROKE` and `CHRONICITY` for stratified validation and subgroup reporting.
- Do not assume a single intensity signature for lesions.
- Build intensity augmentation that can mimic phase and scanner variation.

**Empirically confirmed (2026-05-20).** Our first nnU-Net v2 baseline trained
on ATLAS R2.1 (chronic T1w) scored 0.640 Dice / 0.654 lesion-F1 on the ATLAS
fold-0 held-out set (in-distribution, ~MAPPING-level) but only **0.188 Dice /
0.201 lesion-F1 on soop_bench** (out-of-distribution). The soop_bench
per-chronicity breakdown is monotonic — **chronic 0.53 > mixed 0.33 >
acute 0.12** — and the model massively under-segments acute cases (e.g. 0.48 mL
predicted vs 79.7 mL GT). Mechanism: ATLAS lesions are chronic encephalomalacia
on T1w; soop_bench GT is DWI/TRACE-defined acute infarcts that are bright on DWI
but subtle/invisible on T1w. **A chronic-T1w-trained model is the wrong tool for
an acute DWI-defined benchmark, and the gap is modality+phase, not model
quality.** Implication for ISLES'26: a T1w-only model must be validated on a
T1w-defined test set across all phases; do not read a low score on a
DWI-defined benchmark as a model-quality verdict. Full numbers + caveats in
`baselines/reports/nnunet/PIPELINE_STATUS.md`.

## Problem 4: Center And Scanner Shift Matter

ATLAS v2.0 was designed with a hidden generalizability set from separate cohorts. ISLES'22 multimodal data included a test center unseen during training, and DeepISLES analyzed performance by center. ISLES'26 has 60+ centers, so center shift is not a side issue.

Action:

- Create leave-one-center-out or grouped-center validation if enough center labels exist.
- Track per-center metrics and worst-center performance, not only mean performance.
- Avoid preprocessing choices that depend on hidden-test distributions.

## Problem 5: Public Leaderboards Encourage Overfitting

The ATLAS ranking weighted the hidden Docker test result 4x more heavily than public-test ranking. That is a direct warning: public test submissions can be gamed, even unintentionally.

Action:

- Keep an internal validation set untouched until final model selection.
- Limit public leaderboard-driven tuning.
- Treat a public leaderboard improvement without internal confirmation as suspect.

## Problem 6: Post-Processing Has Metric Tradeoffs

MAPPING's post-processing improved public Dice from 0.664 to 0.667 and lesion-wise F1 from 0.555 to 0.564, but post-processing rules are brittle if tuned to one dataset. Component removal helps false positives but can delete true tiny lesions.

Action:

- Tune thresholds on internal validation only.
- Report precision/recall tradeoffs before and after post-processing.
- Keep a no-postprocessing baseline for sanity.

## Problem 7: Human Ground Truth Is Imperfect

The ISLES'22 dataset paper reports that prior annotations often underestimated infarct borders and missed small embolic infarcts. The authors used a hybrid human-algorithm workflow plus expert review because manual delineation is expensive and variable.

Action:

- Expect label noise, especially at lesion boundaries and tiny lesions.
- Use robust losses and evaluate boundary tolerance where possible.
- Review apparent false positives manually; some may be plausible lesions absent from the label.

## Problem 8: Published "Strong T1w" Weights Are Effectively Unreachable

The strongest published T1w stroke segmenters all gate their weights behind
interactive auth that a compute cluster cannot script through (verified
2026-05-20):

- **MAPPING / CTRL** (ATLAS R2 winner): weights on a KCL SharePoint folder —
  returns HTTP 403 `Forms_Based_Auth_Required` from the cluster.
- **Halai et al. 2026** (`AjayHalai/Lesion-Segmentation-nnUNet`): weights on a
  Cambridge OneDrive share.
- **BrainSegFounder** (`lab-smile/BrainSegFounder`): weights on Google Drive.
- **qSynth** (`liamchalcroft/qsynth`): weights "coming soon", not yet released.

Action:

- Budget for **training our own** baseline rather than assuming a drop-in
  pretrained model; this is why the canonical move is "train nnU-Net on the
  data we have" (we have ATLAS R2.1 decrypted locally).
- If a published model is essential, ask a human to fetch the weights via
  browser and stage them on `/scratch`; do not burn cycles scripting around
  SharePoint/OneDrive auth.

## Problem 9: Adapted Loss Code Can Silently Stay In Warmup Forever

The vendored SynthStroke `DiceCEL2Loss` switches from an L2 warmup to Dice+CE
based on a stateful `self.epoch` attribute that must be advanced by the caller
each epoch. Our adaptation `our_train.py` constructed the loss with
`l2_epochs=50` but **never updated `crit.epoch`**, so every step of every run
before 2026-05-27 trained on the L2 regression objective only — never on the
segmentation objective. Symptoms: low train losses on the wrong scale, val
lesion dice barely above zero until very late, and apparent plateau ~0.10–0.20.
We initially misread this as "synth from-scratch can't compete with synth-plus."

After the fix (`crit.epoch = epoch` each epoch) val lesion dice climbs roughly
5× faster (0.28 by ~ep110 vs the broken trainer's 0.18 at ep400).

Action:

- Whenever you vendor or adapt a loss/transform that carries epoch- or
  schedule-dependent state, write a one-line unit check that the state is
  advanced during training.
- Treat suspiciously-flat learning curves on a documented-good recipe as a
  trainer bug first, model bug second.
- Re-judge any "X doesn't work" conclusion that was made against a broken
  trainer. Specifically: Run A / C / D in this repo (and their reports) are
  confounded by this bug; do not cite their absolute numbers as evidence
  about synth-prob, compact6, or fade.

Full lineage: [docs/synthstroke_training_lessons.md](synthstroke_training_lessons.md).

## Problem 10: Skull-Stripped Inputs Need A Brain Mask At Inference

Diagnosis on the first SynthStroke runs showed the model **hallucinated lesion
voxels out in the image background / air** on ATLAS T1w (which is curator
skull-stripped, so background is exactly 0). On one probe case predicted
lesion volume was 8× ground truth, much of it scattered FPs outside the head.
Adding a simple post-hoc brain mask — `binary_fill_holes(input > 0)` reduced
to the largest 26-connected component — lifts ATLAS Dice **0.280 → 0.375**
and lesion-F1 **0.109 → 0.202** on the full 194 fold-0 val (single-checkpoint,
no retrain).

Action:

- Apply a brain mask at inference whenever the training input is
  skull-stripped and the model is allowed to emit a background-class
  prediction in tissue space.
- Keep the saved probability map *unmasked* so post-processing tuning still
  sees the raw network output.
- For non-stripped test inputs (raw DWI, soop_bench TRACE), `(input>0)` only
  removes pure-air FPs; a real skull-strip (SynthStrip) is needed to do
  better. Acceptable for now since the ATLAS gain dominates.

Implementation: `--brain-mask` in
[our_predict.py](../baselines/models/synthstroke/synthstroke_helpers/our_predict.py),
default-on in
[analysis_synth_07_predict_eval.sh](../baselines/models/synthstroke/analysis_synth_07_predict_eval.sh)
via the `BRAIN_MASK` env.

## Problem 11: Patch-Cropping Before Normalization Mismatches Inference Stats

Our `real_transform` originally cropped to 128³ patches and *then* applied
`HistogramNormalizeD` + `NormalizeIntensityD`, while `our_predict.py`
normalizes the **full volume** before sliding-window inference. The training
intensity statistics (per-patch percentiles, per-patch z-score) drifted from
the test statistics (full-volume), creating a hidden distribution shift on
the real-image path.

Action:

- Apply per-image / per-volume normalizers **before** spatial cropping in the
  training pipeline so train and test see the same intensity distribution.
- The synth path can keep post-synthesis normalization (synthesis is
  inherently patch-based); only the real path matters here.
- More generally: any test-time preprocessing that depends on global stats
  must also be applied at training BEFORE the train-only crop / mosaic /
  patch sampling.

## Problem 12: SLURM Scripts Can Mask Failure Without `-e` And Wait Checks

`analysis_synth_06_train.sh` used `set -uo pipefail` (no `-e`) and then
`wait "$PY_PID"; echo "[done] ..."` — a non-zero Python exit was silently
followed by a `[done]` line, making failed runs look successful in
`logs/`/`squeue`. `analysis_synth_08_tune_postproc.sh` had the same shape.

Action:

- Use `set -euo pipefail` in batch scripts (Lmod `module load` exit codes are
  fine in practice for these scripts), or at minimum capture `wait`'s exit
  code and `exit "$RC"` on failure before any "done" line.
- Wrap the terminal Python step in `if ! ...; then exit 1; fi` so the script
  fails the SLURM job (and any `afterok` dependents) instead of printing
  success.
- For long-running trainers with `--requeue` and a USR1 trap, set
  `PY_PID=""` before launch and guard the trap with `if [[ -n "$PY_PID" ]]`
  to avoid trap-induced errors before the process starts.



- Selecting models by Dice only.
- Random k-fold validation that leaks center/chronicity structure.
- Building a browser-small model before having a strong teacher.
- Trusting a novel architecture without beating nnU-Net under identical folds.
- Removing tiny components blindly in post-processing.
- Ignoring metadata that the challenge explicitly provides.

## Problem 13: A Custom nnU-Net Trainer With `*args` __init__ Crashes At Instantiation

nnU-Net's base `nnUNetTrainer.__init__` records its keyword arguments by
iterating the **subclass** `__init__` signature and reading `locals()[name]`
for each parameter. A trainer subclass written as
`def __init__(self, *args, **kwargs)` therefore makes it look up
`locals()['args']` in the base frame, raising `KeyError: 'args'` before training
starts (cost: a few minutes, not GPU-days — it fails at construction). **Always
give a custom trainer the explicit signature**
`(plans, configuration, fold, dataset_json, device=...)` and call
`super().__init__(plans, configuration, fold, dataset_json, device)`, exactly
like `nnUNetTrainer_250epochs`. (Hit when adding the Tversky+CE trainer.)

## Problem 14: Warped Predictions And GT Can Share A Shape But Differ In Orientation

soop_bench GT masks are stored `LAS`; predictions warped into TRACE space via
`as_closest_canonical(TRACE)` come out `RAS`. They have the **same array shape**
but a left↔right flip baked into the affine, so a naive `np.asarray(pred) vs
np.asarray(gt)` voxel comparison scores a mirror-flipped mask (~0.016 Dice
instead of ~0.255). `eval.run_eval` is affine-aware and handles this; ad-hoc
in-memory comparisons are not. **When comparing two NIfTIs directly, reconcile
orientation first** — here, `nib.as_closest_canonical(gt)` made both `RAS` and
aligned them voxel-for-voxel. Symptom to watch for: a model scoring near-zero in
a custom evaluator but normally under the project's eval harness. (Hit in the
OOD ensemble eval; nn-only then reproduced the 0.2576 baseline.)

## Problem 15: nibabel Infers File Type From The Suffix — `.nii.gz.tmp` Fails

Writing to a temp path like `<name>.nii.gz.tmp` and renaming is a common
atomic-write pattern, but `nib.save` raises `ImageFileError: Cannot work out
file type` because it dispatches on the filename suffix. Use a temp name that
**ends in the real extension**, e.g. `<name>.tmp.nii.gz`, then `os.replace` to
the final path. (Hit building the MSL relabeled dataset.)

## Problem 16: Per-Lever Results (What Actually Moved The Number)

Full table and operating points live in `docs/isles26_model_status.md`. Summary
of the R3.0 campaign so the conclusions are not lost if that file is trimmed:

- **WIN — longer schedule:** 250→1000 epochs lifted consolidated 5-fold CV Dice
  0.6372 → 0.6528 (all folds up).
- **WIN — Dice+TopK10 (MAPPING recipe) @ 1000ep:** nnU-Net consolidated CV
  **0.6551**; rich repo-harness OOF report **Dice 0.6558 / lesion-F1 0.6735 /
  surface Dice 0.7679 / HD95 18.10 mm**; postprocessing chose none.
- **WIN — OOD T1w ensemble:** blending nnU-Net + synth-plus probabilities, both
  on the T1w input only and warped to TRACE, lifted soop mean Dice 0.255 →
  0.289 (median 0.04 → 0.20). synth-plus carries the acute-DWI cases.
- **NEGATIVE — ResEnc-L:** 0.639 vs 0.638, a tie at 250ep (no gain on this task).
- **NEGATIVE — connected-component postprocessing:** `find_best_configuration`
  chose "no postprocessing" every time (ATLAS lesions are multifocal; pruning
  components hurts CV Dice).
- **TRADEOFF — MSL (size-stratified labels):** small-lesion recall 0.339 →
  0.486, medium 0.589 → 0.641, but Dice 0.6374 → 0.6298 (precision cost).
- **NEGATIVE — precision-aware MSL+Tversky(0.7/0.3):** worsened both Dice
  (0.6298 → 0.6186) and small recall (0.486 → 0.454); the FP penalty suppressed
  true positives. Do not pursue Tversky-on-MSL.

## Problem 17 — We optimised a metric the challenge does not use (2026-07-28)

**Symptom.** Nine months of model selection, threshold tuning and go/no-go calls
on "lesion-F1", against a criterion the leaderboard does not use.

**Cause.** `eval/metrics.py` matches lesions at **1-voxel any-overlap,
many-to-many** — the ATLAS convention, and also what the organizers' code did in
its first published version. The organizers **rewrote `eval_utils.py` on
2026-07-24** to use panoptica Recognition Quality under **one-to-one IoU >= 0.25**
matching, and to score **PR-AUC** (which we had never computed) instead of
ROC-AUC. We were reasoning from a nine-day-old snapshot.

**Magnitude.** 52% of the GT lesions our harness counted as "detected" have
overlap/|gt| < 0.25 and cannot match officially; 77 were credited on a single
voxel. The reconstructed bound puts official lesion-F1 at <= 0.6015 against a
reported 0.6735, and the bias is **model-dependent** (-0.070 to -0.084), i.e.
6-14x larger than the 0.0025-0.003 deltas we were selecting models on.

**Lessons.**
1. **Pin the evaluator, not your reading of it.** Clone the organizers' code and
   `import` it. `eval/official_metrics.py` never reimplements a formula.
2. **Re-check the upstream metric code before every decision milestone.**
   Challenge evaluation code is not frozen. A `git log` on it is five seconds.
3. **Measure the semantics you depend on.** We *assumed* 26-connectivity and an
   IoU threshold. `eval/probe_panoptica_semantics.py` now proves both with
   phantoms that have known answers. Had connectivity been 6, every lesion count
   in the repo would have been wrong.
4. **Keep the diagnostic harness, demote it.** `eval/metrics.py` still provides
   size-binned recall, boundary metrics and per-center strata that the official
   scorer does not. The official one *decides*; the rich one *explains*.

## Problem 18 — A decision made on a search space that never contained the option

**Symptom.** "Connected-component postprocessing: none chosen — pruning hurts"
was recorded as a settled negative result and generalised in the docs.

**Cause.** `nnUNetv2_find_best_configuration` tests exactly one operation,
`remove_all_but_largest_component`, and accepts it only on a foreground-Dice
gain. **Min-size pruning was never a candidate.** Meanwhile two of the five
ranked metrics — absolute lesion count difference, and RQ under one-to-one
matching — are directly sensitive to spurious small components, and neither was
being measured.

**Lesson.** When a tool reports "no configuration selected", write down *what it
searched*. A negative result is only as broad as its search space, and
generalising it is how a lever stays untried for months.

## Problem 19 — Geometry is the highest-risk part of a segmentation submission

**Symptom.** Every number this project ever produced was computed on the
RAS-canonicalised copy in `work/nnunet/nnUNet_raw/`. There was no end-to-end
raw-file to prediction path anywhere in the repo, 19 days before the deadline.

**Why it matters.** The models were trained on `nib.as_closest_canonical` output
and nnU-Net does **not** reorient at inference (`transpose_forward [0,1,2]`),
while the release stores volumes as RAS 869 / LAS 575 / PSR 8 / PSL 1. A
permutation error produces a well-formed mask of plausible size — it fails
*silently*. This project already paid for it once (commit `f8d88c1`: 0.016 Dice
where 0.2576 was correct).

**What we built.** `submission/isles26_algorithm/geometry.py` derives the exact
inverse of the canonicalisation rather than guessing it, and asserts the
round-trip per case. `test_geometry.py` proves exactness over all 48
axis-aligned orientations plus real ATLAS cases in every orientation present in
the release.

**The lesson about the TEST, not the code.** The first version of
`analysis_nnunet_51` compared container-on-raw Dice against the recorded
out-of-fold Dice and reported MISMATCH on 4 of 7 cases (-0.011 to -0.042). That
looked like a geometry bug and was not. Adding a **control** — feed the same case
from the already-canonical copy, which exercises the identical code path with an
identity orientation transform — showed geometry deltas of **exactly 0.000000**
on every case including PSR/PSL, and isolated the residual as a different effect
(Problem 20). *A test that cannot separate two candidate causes will indict the
wrong one.*

## Problem 20 — nnU-Net's training-time validation is not the deployment path

**Symptom.** Predicting a case through `predict_from_files` scores a mean
**-0.017 Dice** (range -0.042 to 0.000, n=7) versus the number nnU-Net recorded
for that same case, same fold, during training-time validation.

**Cause (measured, not assumed).** Not geometry — the control above pins the
geometry delta at exactly zero. Training-time validation predicts from the cached
preprocessed arrays in `nnUNet_preprocessed`, while `predict_from_files`
re-preprocesses from the NIfTI. These are different code paths and they do not
agree exactly.

**Consequence.** Every cross-validation number in this repo is a
*training-time-validation* number, and the container will score systematically
lower than it. Quote CV Dice as an upper bound on deployed performance, not an
estimate of it.

**Lesson.** Validate the artifact you ship, through the path you ship it on. An
internal validation score is a property of the training harness, not of the
model.

## Sources

- ISLES'22 dataset paper: https://www.nature.com/articles/s41597-022-01875-5
- ATLAS v2.0 dataset paper: https://www.nature.com/articles/s41597-022-01401-7
- MAPPING winner report: https://arxiv.org/abs/2211.15486
- DeepISLES paper: https://www.nature.com/articles/s41467-025-62373-x
- MSL/DBL small-lesion paper: https://arxiv.org/abs/2408.02929
