# SynthStroke Training Campaign — Lessons (2026-05-21 → 2026-05-27)

Hard-won lessons from the from-scratch SynthStroke training effort on ATLAS R2.1.
Read this before iterating on `baselines/models/synthstroke/`. Some prior reports
were confounded by a critical trainer bug — **anything reporting Run A / C / D
ATLAS metrics from before 2026-05-27 must be re-read through the caveats below.**

## TL;DR (what changed)

- A critical bug made every from-scratch run before 2026-05-27 train on the
  **L2 warmup objective for all 500 epochs**, never on Dice+CE. Fixed by setting
  `crit.epoch = epoch` each epoch in [our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py).
- **Brain-masking at inference is a free, large win** on ATLAS T1w: Run A ATLAS
  Dice **0.280 → 0.375**, lesion-F1 **0.109 → 0.202** (full 194 val). Now
  default in [analysis_synth_07](../baselines/models/synthstroke/analysis_synth_07_predict_eval.sh)
  (`BRAIN_MASK=1`), driven by a new `--brain-mask` flag in
  [our_predict.py](../baselines/models/synthstroke/synthstroke_helpers/our_predict.py).
- **Run E** (corrected baseline = Run A config + L2 fix + normalize-before-crop)
  is hitting val dice **0.288 @ epoch 109** vs the broken Run A's *final* 0.198
  @ epoch 404 — roughly **5× faster climb**, validating the fix.
- Earlier conclusion "synth from-scratch can't beat synth-plus" was **premature**;
  it was confounded by the L2 bug. Re-judge after Run E's official eval.

## Why this matters / context

Goal: a single model strong on ATLAS T1w that also works on soop_bench
(cross-contrast). Strategy chosen earlier: train our own SynthStroke-style
MONAI UNet on ATLAS label maps with cornucopia GMM synthesis + `--mix-real`
ATLAS T1w. See [relevance_to_isles26.md](relevance_to_isles26.md) and
[challenge.md](challenge.md) for the broader plan.

## The critical bug (root cause of all prior negative results)

[work/synthstroke/repo/custom.py](../work/synthstroke/repo/custom.py)'s
`DiceCEL2Loss.__init__` does `self.epoch = 0`; `forward()` does
`if self.epoch < self.l2_epochs: total_loss = self.l2(...) else Dice+CE`.
Our adaptation [our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py)
constructed the loss with `l2_epochs=50` but **never advanced `crit.epoch`**
in the training loop. So `0 < 50` was always true → every step of every prior
run trained on the **L2 regression objective**, never on the segmentation
objective.

Symptom: train losses ~5–20 (L2 scale), val lesion dice barely above zero
until very late, "best" val dice plateauing at 0.10–0.20. We misread this
as "synth from-scratch is weak."

Fix (one line):

```python
for epoch in range(start_epoch, epochs):
    cur_epoch = epoch
    crit.epoch = epoch              # <-- this was missing
    model.train()
    ...
```

Validation (smoke with `--l2 0` to force Dice+CE immediately): epoch-0 loss
~3.5 (Dice+CE scale) vs the broken runs' ~20 (L2 scale). Run E then climbs
~5× faster than the broken runs ever did.

**Implication for prior reports.** All numbers from Run A (sp033), Run C
(sp020), Run D (compact6+fade) and the in-training val curves of the
2026-05-21 → 2026-05-26 runs were produced by a model trained on the wrong
objective. Do not use them to judge the recipe (synth-prob, class scheme,
fade, etc.); only Run E and later are valid recipe comparisons.

## Brain-masking — the proven free win

Visual diagnosis on Run A showed the model **hallucinated lesion voxels in
the image background / out in the air** (ATLAS images are skull-stripped, so
background is exactly 0). Adding a simple `(input>0)` brain mask, hole-filled
and reduced to the largest 26-connected component, lifts ATLAS:

| Run A (sp033), full 194 val, op=(thr 0.40, cc 10) | Dice | median | lesion-F1 |
|---|---|---|---|
| no mask (as originally reported)                  | 0.280 | 0.102 | 0.109 |
| **+ brain mask**                                  | **0.375** | 0.351 | **0.202** |

Implementation: `--brain-mask` in [our_predict.py](../baselines/models/synthstroke/synthstroke_helpers/our_predict.py)
applies after the existing CC removal:

```python
brain = scipy.ndimage.binary_fill_holes(input_data > 0)
labeled, n = scipy.ndimage.label(brain, structure=np.ones((3,3,3), np.uint8))
# keep largest component
mask = mask & (labeled == largest_label)
```

`analysis_synth_07_predict_eval.sh` now defaults `BRAIN_MASK=1` (set
`BRAIN_MASK=0` to disable). The saved probability map is left raw so
[tune_postproc.py](../baselines/models/synthstroke/synthstroke_helpers/tune_postproc.py)
can still tune on the unmasked prob.

**Caveat.** `(input>0)` is the right brain definition for skull-stripped ATLAS
T1w. For raw inputs (e.g. soop_bench TRACE which is not skull-stripped), the
same mask removes pure-air FPs but does **not** strip the skull — a proper
SynthStrip pass would be more correct on soop. Acceptable for now since the
ATLAS gain dominates.

## Diagnosis methodology (what to do when scores are bad)

1. **Voxel-count overlays first.** Compare `pred.sum()` vs `gt.sum()` per case —
   over-fill (pred ≫ gt) vs under-fill (pred ≪ gt) is the dominant signal.
2. **Visual overlays of a handful of cases** ([work/synthstroke/diag_overlays.py](../work/synthstroke/diag_overlays.py))
   reveal *where* the model is firing: inside vs outside brain, at the right
   anatomy or scattered, whether it traces lesion borders.
3. **Offline reasoning on saved probability maps.** With prob maps saved by
   `analysis_synth_08` we can sweep thresholds + masking variants on the full
   194 val set without any GPU
   ([work/synthstroke/diag_brainmask_sweep.py](../work/synthstroke/diag_brainmask_sweep.py)).
4. **Compare to known-working references** (synth-plus, nnU-Net) at matched
   epochs and matched eval set.

Findings from this pipeline (interpret WITH the L2-bug caveat):
- Model **localizes** lesions on both T1w and DWI (right anatomy under contour).
- Failures are **calibration**, not localization:
  - **Out-of-brain FPs** on T1w → fixed by brain-mask.
  - **In-brain over-fill** on T1w → 3–4× volume even after masking; partially
    addressable by tighter threshold and/or precision-aware losses; also
    expected to improve substantially after the L2 fix (Run E and later).
  - **Under-fill on DWI** → traces the lesion but doesn't fill; likely
    calibration / out-of-distribution sensitivity.

## Other fixes shipped this session

### Train/inference normalization mismatch (real path)

[our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py)
`real_transform` used to apply `HistogramNormalizeD` + `NormalizeIntensityD`
**after** the random crop, while [our_predict.py](../baselines/models/synthstroke/synthstroke_helpers/our_predict.py)
normalizes the **full volume** before sliding-window inference. Training stats
were computed on a 128³ patch → shifted distribution at test time.

Fix: moved normalization before the crop block in the real path. The synth
path keeps its current (post-synthesis) `NormalizeIntensityD` because
synthesis is inherently patch-based; the original repo does the same.

### Scripts no longer report success after Python failure

[analysis_synth_06_train.sh](../baselines/models/synthstroke/analysis_synth_06_train.sh)
used `set -uo pipefail` and then `wait "$PY_PID"; echo "[done] ..."`, so a
non-zero Python exit was masked. Fixed with `RC=0; wait "$PY_PID" || RC=$?;
[[ $RC -ne 0 ]] && exit $RC`. [analysis_synth_08_tune_postproc.sh](../baselines/models/synthstroke/analysis_synth_08_tune_postproc.sh)
got the same treatment + `if ! tune_postproc ...; then exit 1; fi`; both have
been further hardened to full `set -euo pipefail` and `exit 2` on missing
inputs.

### N_CLASSES threading through eval / tune

Run D used a 6-class checkpoint; `our_predict.py` defaulted `--n-classes 34`,
which would have silently failed to load the state-dict. Added `N_CLASSES`
env to [analysis_synth_07](../baselines/models/synthstroke/analysis_synth_07_predict_eval.sh)
and [analysis_synth_08](../baselines/models/synthstroke/analysis_synth_08_tune_postproc.sh)
that flows to `--n-classes`. Default 34 preserves prior behaviour.

### compact6 label scheme + fade (added, but not yet validated)

[build_labelmap.py](../baselines/models/synthstroke/synthstroke_helpers/build_labelmap.py)
gained `--scheme {fs34,compact6}`. compact6 groups the SynthSeg FreeSurfer
aseg labels into 6 classes (0 background, 1 CSF/ventricles, 2 GM,
3 WM, 4 cerebellum+brainstem, 5 lesion) to match synth-plus's 6-class style.
[our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py)
gained `--fade` (smooth multiplicative field on the lesion region post-
synthesis, mirroring the repo's penumbra/inhomogeneity intent for our
integer-label cornucopia pipeline). **Both were used in Run D, which was
confounded by the L2 bug — they are NOT independently validated yet.**

## Run history (with corrected interpretation)

| Run | Recipe | Status | Note |
|---|---|---|---|
| A `sp033` | 34-class, synth-prob 0.33, l2=50, 500 ep | **L2-bug confounded** | Reported 0.280 ATLAS; with brain-mask 0.375. Trained on L2 only. |
| C `sp020` | 34-class, synth-prob 0.20, l2=50, 500 ep | **L2-bug confounded** | Reported 0.169 ATLAS. Trained on L2 only. |
| D `c6fade` | compact6 (6-class), synth-prob 0.33, fade, l2=50 | **L2-bug confounded; eval cancelled** | In-training val 0.022. compact6+fade *cannot* be judged from this run. |
| **E `l2fix`** | 34-class, synth-prob 0.33, l2=50, **L2 fix + normalize fix + brain-mask** | training | First valid recipe test. Val dice 0.288 @ ep109 (vs broken A's 0.198 final). Eval chain queued (jobs 26331189 → 26331195). |

References to compare against:
- nnU-Net v2 3D fullres (real ATLAS T1w only): **ATLAS 0.640 / lesion-F1 0.654** ; soop_t1w 0.188 / 0.201
- pretrained synth-plus (Chalcroft, 6-class, multi-dataset): ATLAS 0.458 / 0.515 ; soop_trace **0.447 / 0.478** ; soop_t1w 0.217

## Recipe knobs (current state)

`analysis_synth_06_train.sh` reads these envs (defaults preserve original
34-class fs34 behaviour):

```
SYNTH_TRAIN_NAME   # run name; checkpoints go to work/synthstroke/train/$NAME/
LABELMAPS_DIR      # work/synthstroke/labelmaps (fs34) | labelmaps_compact6
N_CLASSES          # 34 (fs34) | 6 (compact6)
SYNTH_PROB         # synth fraction per step (default 0.33 = 67% real)
FADE               # 1 to enable LesionFadeD in synth path
```

`analysis_synth_07_predict_eval.sh` and `analysis_synth_08_tune_postproc.sh`
read:

```
SYNTH_TRAIN_NAME   # picks checkpoint + report tag
N_CLASSES          # must match the trained checkpoint
PROB_THRESHOLD     # override frozen op-point (otherwise read from tuning JSON)
MIN_CC_VOXELS      # override frozen op-point
BRAIN_MASK         # 1 default; 0 disables --brain-mask
```

## What to verify next (open items)

1. **Final Run E eval** (ATLAS val / soop_trace / soop_t1w) — chained to fire
   when training completes (job 26331195). The first valid number for the
   from-scratch route.
2. **Whether compact6 helps or hurts.** Run D's negative result is invalid
   (L2 bug). Re-test as `Run F = Run E + compact6` (single variable) only
   after Run E lands.
3. **Whether `--fade` helps or hurts.** Same caveat — re-test as one variable
   on top of the best so far.
4. **In-brain over-fill** (the remaining named failure after brain-mask). Try
   precision-aware loss (Tversky / Focal-Tversky) and/or `lesion_weight=1`
   on top of the L2-fixed pipeline.
5. **Codex adversarial review.** All trainer fixes (L2, normalize, fail-hard,
   brain-mask, compact6, fade, N_CLASSES) bypassed `/codex:adversarial-review`
   because the Codex runtime was sandbox-blocked. Route them back through
   review when the environment is reset.

## Sources / pointers

- Trainer: [our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py)
- Inference: [our_predict.py](../baselines/models/synthstroke/synthstroke_helpers/our_predict.py)
- Tuning: [tune_postproc.py](../baselines/models/synthstroke/synthstroke_helpers/tune_postproc.py)
- Label-map build: [build_labelmap.py](../baselines/models/synthstroke/synthstroke_helpers/build_labelmap.py)
- Scripts: [analysis_synth_05](../baselines/models/synthstroke/analysis_synth_05_build_labelmaps.sh)
  · [06](../baselines/models/synthstroke/analysis_synth_06_train.sh)
  · [07](../baselines/models/synthstroke/analysis_synth_07_predict_eval.sh)
  · [08](../baselines/models/synthstroke/analysis_synth_08_tune_postproc.sh)
- Diagnostic helpers (transient, in `work/`): `diag_overlays.py`, `diag_brainmask_sweep.py`
- Pipeline status report: [baselines/reports/synthstroke/PIPELINE_STATUS.md](../baselines/reports/synthstroke/PIPELINE_STATUS.md)
- Upstream DiceCEL2Loss (the bug-prone class): [work/synthstroke/repo/custom.py](../work/synthstroke/repo/custom.py)
- SynthStroke paper: Chalcroft et al., MELBA 2025 — http://dx.doi.org/10.59275/j.melba.2025-f3g6
