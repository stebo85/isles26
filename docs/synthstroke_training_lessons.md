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
- **Run E** (corrected baseline = Run A config + L2 fix + normalize-before-crop
  + brain-mask) is the first VALID from-scratch result: **ATLAS Dice 0.480 —
  beats pretrained synth-plus (0.458)**, confirming the fix and the from-scratch
  route. (The broken Run A reached val 0.198 at epoch 404; Run E hit 0.288 by
  epoch 109 — ~5× faster climb.)
- **But the cross-contrast goal is NOT met by the mix-real recipe**: Run E
  scores only **soop_trace Dice 0.131 vs synth-plus 0.447**. Mixing 67% real
  T1w anchors the model to T1w and destroys the contrast-agnostic property.
  Run F (synth-only) is the pivotal test of whether dropping real mixing
  recovers it.
- Earlier conclusion "synth from-scratch can't beat synth-plus" was **premature**
  (confounded by the L2 bug) — Run E disproves it on ATLAS. The "works on soop"
  half remains open.

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
| **E `l2fix`** | 34-class, synth-prob 0.33, l2=50, **L2 fix + normalize fix + brain-mask** | **DONE — VALID** | First valid result. See full numbers below. ATLAS Dice **0.480** (> synth-plus 0.458); soop_trace 0.131 (≪ synth-plus 0.447). |
| F `sp100` | 34-class, **synth-prob 1.0 (synth-only)**, l2=50, fixes | **DONE** | Cross-contrast NOT recovered: soop_trace 0.155 (≪ synth-plus 0.447); ATLAS collapsed to 0.139. The ratio knob is not the cross-contrast lever. |
| **G `fade`** | 34-class, synth-prob 0.33, **`--fade`**, l2=50, fixes | **DONE** | `--fade` helps in-domain: **ATLAS 0.504** (> E 0.480 > synth-plus 0.458). Keep fade. soop_trace 0.114 (no cross-contrast gain). |
| **H `tversky`** | Run G + **lesion-targeted Tversky** (`--loss tverskycel2` α0.7/β0.3) | **DONE — best ATLAS** | Precision-aware loss works: **ATLAS Dice 0.566 / lesion-F1 0.450** (vs G 0.504/0.268), **lesion-precision 0.18→0.41** at held recall 0.76. Best synth model in-domain. soop_trace 0.146 (cross-contrast still unsolved). See "Run H" section. |
| **I `tversky_a08`** | Run H + **α0.8/β0.2** (more FP penalty) | **DONE — best F1** | α-sweep result: precision 0.41→**0.45**, lesion-F1 0.450→**0.479** (best yet), recall held 0.75, small Dice cost 0.566→0.553. Cross-contrast slightly worse (soop_trace 0.090). Monotonic precision lever; sweet spot ~α0.7–0.8, stop here. |
| J `compact6` | Run H + **compact6 (6-class)** label scheme | **DONE — negative** | Single-variable swap to the 6-class scheme that matches synth-plus's structure. **compact6 did NOT help cross-contrast:** soop_trace **0.078** (worst of all valid runs, < H's 0.146), ATLAS Dice 0.559 (≈ H/I) but lesion-F1 **0.392** (< H 0.450, I 0.479). Matching synth-plus's class count is not the DWI lever. See "Run J" section. |
| K `dwi` | Run H + **DWI-aware synth aug** (`--dwi-prob`, lesion forced hyperintense in a fraction of synth samples) | **CODE READY — to run** | First attack on the synthesis itself rather than loss/ratio/scheme. New `DWIContrastD` transform overrides the random GMM lesion intensity with a restricted-diffusion *bright* blob (tissue-p95 × U(1.5,3) + smooth texture) for a `--dwi-prob` fraction of synthetic patches. Hypothesis: teaching the bright-lesion DWI cue lifts soop_trace without hurting ATLAS (default-off, single-variable on top of best in-domain model H). |

References to compare against:
- nnU-Net v2 3D fullres (real ATLAS T1w only): **ATLAS 0.640 / lesion-F1 0.654** ; soop_t1w 0.188 / 0.201
- pretrained synth-plus (Chalcroft, 6-class, multi-dataset): ATLAS 0.458 / 0.515 ; soop_trace **0.447 / 0.478** ; soop_t1w 0.217

## Run E — first valid from-scratch result (2026-05-27)

Corrected baseline (= Run A config + the three fixes). Brain-masked, tuned
operating point thr=0.3 / min_cc=10 (tuned on ATLAS val only). 500 epochs,
fold 0, synth-prob 0.33, 34-class, no requeues.

| Eval set | n | Dice | lesion-F1 | lesion-recall | lesion-prec | surf-Dice 3mm | HD95 | AVD mL | |abs lesion-count diff| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **ATLAS val** (T1w, in-dist) | 194 | **0.480** | 0.255 | 0.759 | 0.181 | 0.561 | 40.2 | 10.8 | 12.0 |
| soop_trace (DWI, OOD)        | 12  | 0.131 | 0.137 | 0.405 | 0.095 | 0.124 | 88.3 | 200.3 | 17.8 |
| soop_t1w (T1w-in-TRACE)      | 12  | 0.215 | 0.186 | 0.316 | 0.156 | 0.247 | 53.8 | 33.4 | 11.3 |

**Reading it:**

1. **ATLAS Dice 0.480 beats pretrained synth-plus (0.458)** and the broken
   Run A (0.375 brain-masked) — the from-scratch route works once the trainer
   is correct. Still below nnU-Net's 0.640 (a real-data specialist).
2. **lesion-F1 0.255 ≪ synth-plus's 0.515.** High recall (0.76) but low
   precision (0.18) → the model finds lesions but emits **too many spurious
   components**. Over-segmentation/fragmentation is now the dominant weakness,
   not localization. (`|abs lesion-count diff|` 12 is far better than broken
   Run A's ~29, but precision is still the bottleneck.)
3. **Cross-contrast NOT recovered: soop_trace 0.131 vs synth-plus 0.447.** AVD
   200 mL on soop_trace shows gross volume miscalibration on DWI. Training with
   67% real T1w anchored the model to T1w appearance; the GMM-synthetic fraction
   (33%) was not enough to keep it contrast-agnostic. soop_t1w (0.215) does
   match synth-plus (0.217) and beats nnU-Net (0.188), so the T1w-input story is
   fine — it's specifically DWI generalization that's lost.

**Net:** a strong *in-domain T1w* model (ATLAS Dice > synth-plus), but the
"works on soop_bench (acute/DWI)" half of the goal is unmet with mix-real.
Two levers to test: (a) **less real / synth-only** (Run F) to recover
contrast-agnosticism; (b) **precision-aware loss / lesion_weight=1** to fix
the F1 over-segmentation on top of the L2-fixed pipeline.

## Runs F & G — synth-only and fade (2026-05-28)

Both are single-variable changes on top of Run E, fully evaluated (brain-masked,
each with its own tuned operating point). Headline comparison:

| Run (op-point) | ATLAS Dice / F1 | soop_trace Dice / F1 | soop_t1w Dice / F1 |
|---|---|---|---|
| E baseline sp0.33 (thr0.3/cc10) | 0.480 / 0.255 | 0.131 / 0.137 | 0.215 / 0.186 |
| **G fade sp0.33 (thr0.4/cc5)** | **0.504 / 0.268** | 0.114 / 0.101 | 0.225 / 0.150 |
| F synth-only sp1.0 (thr0.3/cc50) | 0.139 / 0.242 | **0.155 / 0.187** | 0.130 / 0.335 |
| *synth-plus (pretrained)* | *0.458 / 0.515* | ***0.447 / 0.478*** | *0.217 / 0.231* |
| *nnU-Net (real T1w)* | ***0.640 / 0.654*** | *—* | *0.188 / 0.201* |

**Two clear findings:**

1. **`--fade` helps in-domain — Run G is our best ATLAS model (Dice 0.504),
   beating Run E (0.480) and pretrained synth-plus (0.458).** Small but
   consistent lift on Dice, F1, recall, precision, and AVD (8.3 vs 10.8 mL).
   Fade's penumbra/inhomogeneity simulation makes the lesion class easier to
   learn. This vindicates re-testing it on a correct trainer (Run D's negative
   fade verdict was L2-confounded and wrong). **Fade should be kept.**

2. **Synth-only (Run F) did NOT recover cross-contrast — the key negative
   result.** The whole point of F was to drop the real-T1w anchor and regain
   synth-plus-like DWI generalization. soop_trace went 0.131 → only 0.155 (vs
   synth-plus's 0.447) while ATLAS collapsed to 0.139. So **the synth:real ratio
   is NOT the lever for cross-contrast** — every from-scratch variant we have
   sits at soop_trace ≈ 0.11–0.16, an order of magnitude short of synth-plus on
   DWI, with gross over-segmentation (AVD 107–200 mL). (One curiosity: F's
   soop_t1w lesion-F1 0.335 / precision 0.360 is the highest of any run —
   synth-only emits far fewer false-positive components on T1w, just at low
   Dice.)

**Why the cross-contrast gap persists (hypotheses, unproven):** our cornucopia
`SynthFromLabelTransform` is driven by **ATLAS-only label maps** and our default
synthesis parameters; pretrained synth-plus is "synth **+ real multi-dataset**"
and a tuned 6-class recipe. The contrast-agnostic property that lets synth-plus
hit 0.447 on DWI likely comes from (a) training label/intensity diversity we
don't reproduce, and/or (b) more aggressive synthesis. Our GMM synthesis is
evidently not covering DWI-like acute-lesion appearance well enough to transfer.

**Strategic readout.** The "**very strong on ATLAS**" goal is essentially met:
Run G (0.504) is our best single model and beats the pretrained weights
in-domain. The "**works on soop_bench (acute DWI)**" goal is **not** met by any
from-scratch synth variant — synth-plus (0.447) remains the only thing that
works cross-contrast, and the ratio knob doesn't close it. Realistic paths
forward: (i) **ship Run G for ATLAS/T1w** and use **synth-plus for DWI** (two
specialists, or ensemble→distill); (ii) attack cross-contrast at its likely
root — **multi-dataset synthesis and/or more aggressive cornucopia params**,
not the mix ratio; (iii) fix the persistent **lesion-F1/over-segmentation**
(precision ≈ 0.18–0.19) with a precision-aware loss. The mix-ratio lever is
spent.

## Recipe knobs (current state)

`analysis_synth_06_train.sh` reads these envs (defaults preserve original
34-class fs34 behaviour):

```
SYNTH_TRAIN_NAME   # run name; checkpoints go to work/synthstroke/train/$NAME/
LABELMAPS_DIR      # work/synthstroke/labelmaps (fs34) | labelmaps_compact6
N_CLASSES          # 34 (fs34) | 6 (compact6)
SYNTH_PROB         # synth fraction per step (default 0.33 = 67% real)
FADE               # 1 to enable LesionFadeD in synth path
DWI_PROB           # 0..1: fraction of synth samples rendered DWI-style
                   #   (lesion forced hyperintense, restricted-diffusion cue).
                   #   Unset/0 = off, byte-identical to prior behaviour.
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

1. ~~Final Run E eval~~ **DONE** — see the Run E table above. ATLAS Dice
   **0.480** (> synth-plus 0.458), soop_trace 0.131 (≪ synth-plus 0.447).
   From-scratch beats the pretrained weights in-domain but not cross-contrast.
2. **Run F (synth-only, synth-prob 1.0)** — IN PROGRESS (job 26630839). Pivotal
   test of whether dropping real-T1w mixing recovers synth-plus-like
   cross-contrast on soop_trace (~0.447). ATLAS in-domain will drop (real
   anchor removed) — the question is the soop trade.
3. **Run G (Run E + `--fade`)** — IN PROGRESS (job 26630843). Isolates fade on a
   correct trainer (Run D's fade verdict was L2-confounded, invalid).
4. **lesion-F1 / over-segmentation** — was the dominant weakness after the L2 fix
   (ATLAS precision 0.18 vs recall 0.76). **SOLVED by Run H** (lesion-targeted
   Tversky `--loss tverskycel2` α0.7/β0.3): precision 0.18 → 0.41 at held recall
   0.76, lesion-F1 0.268 → 0.450, Dice 0.504 → 0.566. See the "Run H" section.
   **Run I (α0.8/β0.2, DONE)** confirmed the lever is monotonic: precision
   0.41→0.45, lesion-F1 0.450→**0.479** (best), recall held 0.75, Dice 0.566→0.553.
   Sweet spot ≈ α0.7–0.8; further α gives diminishing returns and starts costing
   Dice/recall — **lever considered tapped**. Ship choice: Run H for Dice (0.566),
   Run I for lesion-F1 (0.479).
5. **real-image intensity augmentation** (`--real-aug`, added 2026-05-27: bias
   field + Gaussian noise + gamma + intensity shift on the real path) —
   implemented, gated default-off, **not yet run**. Candidate to harden the
   real path's cross-contrast robustness.
6. **compact6 scheme** — ~~Re-test on a correct trainer~~ **DONE (Run J, negative)**.
   `Run H + compact6` made cross-contrast worse (soop_trace 0.078) and regressed
   in-domain lesion-F1 (0.392). Class count is not the DWI lever. See "Run J".
8. **DWI-aware synthesis / contrast augmentation** — **CODE READY (Run K, to run)**.
   Every cheap lever (loss, synth:real ratio, label scheme) has failed to close the
   modality gap. Attack the synthesis: `DWIContrastD` (`--dwi-prob p`,
   [our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py))
   forces the lesion hyperintense (tissue-p95 × U(1.5,3) × smooth texture, overriding
   the random GMM draw) on a fraction `p` of synthetic patches, so the model learns
   the restricted-diffusion bright-blob prior that DWI inference relies on. Synth-path
   only; default `p=0` is byte-identical (transform not appended to the Compose).
   Implemented via Codex + `/codex:adversarial-review`; review fixes applied
   (NaN/non-positive tissue reference is filtered and bails out no-op rather than
   corrupting the lesion; the texture field is drawn from the transform's `self.R`
   so seeded replay is deterministic) and validated by a CPU functional test
   (hyperintensity, NaN-safety, no-lesion no-op, seeded determinism, p=0 no-op).
   **Caveat (same as `--fade`/`--synth-prob`):** `dwi_prob` is *not* in the resume
   `objective_config` guard — toggling `DWI_PROB` on a requeue under the same
   `--name` silently mixes policies; always use a fresh run name (the SLURM recipe
   does). First run: `Run H recipe + DWI_PROB≈0.5`, sweep `p` after.
7. **Codex adversarial review — NOW UNBLOCKED (2026-06-01).** The earlier
   "sandbox-blocked" state was not transient namespace exhaustion: this is a
   RHEL7 / kernel-3.10 login node (no Landlock), and the cluster sets
   `max_net_namespaces=0` / `max_uts_namespaces=0`, so Codex's bwrap read-only
   sandbox can never `--unshare-net`/`--unshare-uts` (the misleading ENOSPC).
   Fix (user-authorized): an env-gated `CODEX_FORCE_SANDBOX=danger-full-access`
   override patched into the codex plugin's `runAppServerTurn` (both marketplace
   and cache copies). With it, `/codex:adversarial-review` runs here. The
   previously-bypassed trainer fixes (L2/normalize/fade/brain-mask/real-aug) were
   reviewed 2026-06-01 against base `6190485` — **no correctness faults in the
   fixes themselves**; the review instead surfaced three eval-pipeline issues
   (items 8–10 below).

## Eval-pipeline review findings (2026-06-01, `/codex:adversarial-review`)

These came out of reviewing the committed trainer/eval scripts. They affect the
**absolute** numbers and **re-run hygiene**, not the relative Run-G-vs-Run-H
comparison (kept apples-to-apples by identical methodology). Fix before any
final/absolute claim.

8. **[F1] Prediction cache is config-blind**
   ([analysis_synth_07_predict_eval.sh](../baselines/models/synthstroke/analysis_synth_07_predict_eval.sh)).
   `predict_one` skips any existing `<case>.nii.gz`, but the cache path is keyed
   only by `TAG`/case — not by checkpoint SHA, `N_CLASSES`, threshold, `min_cc`,
   or brain-mask. Re-running the same `SYNTH_TRAIN_NAME` after changing any of
   those silently reuses stale masks. Mitigation in use: every run uses a fresh
   `SYNTH_TRAIN_NAME`. Proper fix: checkpoint-SHA + config sidecars like the
   nnU-Net pipeline already has.
9. **[F2] Tuning optimizes a different pipeline than deployment**
   ([tune_postproc.py](../baselines/models/synthstroke/synthstroke_helpers/tune_postproc.py)).
   `analysis_synth_08` sweeps (thr, cc) on the **raw, unmasked** saved prob maps,
   but `analysis_synth_07` deploys with **brain-masking on** by default. The
   selected operating point can therefore be slightly off the masked pipeline it
   is applied to. Consistent across all runs, so relative comparisons hold; fix:
   apply the same brain mask during tuning and persist `brain_mask` in the
   operating-point JSON.
10. **[F3] Requeue resume is not bit-deterministic**
    ([our_train.py](../baselines/models/synthstroke/synthstroke_helpers/our_train.py)).
    The checkpoint restores global Python/NumPy/Torch/CUDA RNG + the synth/real
    mixing RNG, but **not** MONAI `Randomizable` transform states or the
    DataLoader sampler position. A preempted+requeued run resumes with different
    crops/flips/case-order than an uninterrupted one. This is best-effort
    continuation, not reproducible resume — it adds noise to an interrupted run
    but does not change the training objective (now guarded by `objective_config`
    on resume). Downgrade expectations or persist transform RNG + sampler offset.

## Run H — lesion-targeted precision-aware loss (DONE — best ATLAS, 2026-06-03)

**RESULT (tuned op-point thr 0.3 / cc 10, brain-masked):**

| Eval set | n | Dice | lesion-F1 | lesion-prec | lesion-recall | surf-Dice 3mm | AVD mL | |Δlesion-count| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **ATLAS val** (T1w, in-dist) | 194 | **0.566** | **0.450** | **0.407** | 0.763 | 0.691 | 6.1 | 4.6 |
| soop_trace (DWI, OOD) | 12 | 0.146 | 0.123 | 0.082 | 0.347 | 0.143 | 58.1 | 25.8 |
| soop_t1w (T1w-in-TRACE) | 12 | 0.219 | 0.237 | 0.209 | 0.330 | 0.254 | 27.6 | 6.1 |

**The precision-aware loss worked, exactly as hypothesized.** vs Run G
(0.504 / 0.268, precision ~0.18): lesion **precision 0.18 → 0.407 (2.3×) at held
recall 0.76** — the intended recall-for-precision trade — lifting **lesion-F1
0.268 → 0.450 (+68%)** and Dice 0.504 → 0.566. The false-positive-component
problem is largely fixed (|Δlesion-count| ~4.6 vs Run E's ~12; AVD 6.1 mL). Run H
is now **our strongest from-scratch synth model in-domain** — beats pretrained
synth-plus on Dice (0.566 > 0.458) and closes most of the F1 gap (synth-plus
0.515). Still below the real-data nnU-Net (0.640 / 0.654).

**Cross-contrast unchanged (expected):** soop_trace 0.114 → 0.146 (still ≪
synth-plus 0.447) — a loss tweak can't fix the modality gap; that remains the
separate, unsolved problem. soop_t1w F1 improved (0.150 → 0.237) via the same
precision gain.

**Recipe** — single-variable change from Run G: swap the region term
of the loss from class-averaged Dice to a **lesion-channel soft Tversky**
(`--loss tverskycel2`, `alpha=0.7` penalizes false positives, `beta=0.3`),
keeping the L2 warmup, the full multi-class CE (`ce_weight[lesion]=2`), `--fade`,
`synth-prob 0.33`, and `--mix-real`. Rationale: Run G's dominant weakness is
lesion over-segmentation (precision ~0.18 / recall ~0.76); an FP-weighted Tversky
on the lesion channel should trade some recall for precision and lift lesion-F1.

Implementation notes (all via Codex + `/codex:adversarial-review`, 2026-06-01):
- The Tversky is computed **only on the lesion channel** (`softmax(logits)[:,lesion]`
  vs the lesion one-hot), not MONAI's class-averaged `TverskyLoss` — otherwise the
  FP penalty would be diluted ~1/33 across foreground classes and barely move
  lesion precision (review finding).
- Resume is guarded by an `objective_config` (loss/alpha/beta/lesion_weight/l2) in
  the checkpoint: a requeue that finds a mismatching objective aborts with a clear
  message instead of silently continuing a different objective (review finding).
- Per-epoch logs now print `region=` / `ce=` components so CE-vs-Tversky dominance
  is visible.
- Validated end-to-end with a GPU `--smoke` run (`--l2 0` to exercise the Tversky
  branch): `region=0.98 ce=2.71`, no NaN, checkpoint written.
- Default `--loss dicecel2` path is byte-identical to Run G.
- Knobs: `LOSS=tverskycel2 TVERSKY_ALPHA=0.7 TVERSKY_BETA=0.3` in
  [analysis_synth_06_train.sh](../baselines/models/synthstroke/analysis_synth_06_train.sh).

## Run J — compact6 label scheme (DONE — negative on cross-contrast, 2026-06-08)

**Hypothesis:** synth-plus generalizes to DWI (soop_trace 0.447) while our
from-scratch synth does not (~0.08–0.15). One structural difference is the label
scheme — synth-plus uses a ~6-class grouping, we use the 34-class FreeSurfer
scheme. Run J is the single-variable test: **Run H recipe + compact6 (6-class)
label maps** (`--scheme compact6`, `N_CLASSES=6`, lesion idx 5). If the coarse
class structure is what lets the GMM pipeline stay contrast-agnostic, compact6
should lift soop_trace.

**RESULT (tuned op-point thr 0.5 / cc 5, brain-masked):**

| Eval set | n | Dice | lesion-F1 | lesion-prec | lesion-recall | surf-Dice 3mm | AVD mL | |Δlesion-count| |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **ATLAS val** (T1w, in-dist) | 194 | 0.559 | 0.392 | 0.325 | 0.767 | 0.671 | 6.0 | 6.4 |
| soop_trace (DWI, OOD) | 12 | **0.078** | 0.089 | 0.056 | 0.354 | 0.081 | 125.8 | 42.5 |
| soop_t1w (T1w-in-TRACE) | 12 | 0.195 | 0.151 | 0.120 | 0.338 | 0.222 | 28.2 | 11.4 |

**Reading it — the hypothesis failed:**

1. **Cross-contrast got WORSE, not better.** soop_trace Dice **0.078** is the
   lowest of any valid run (E 0.131, G 0.114, H 0.146, I 0.090), and AVD blew up
   to 126 mL. Collapsing 34 → 6 classes did not make the model contrast-agnostic;
   the class count is **not** the DWI lever. Synth-plus's cross-contrast strength
   must come from something else (its multi-dataset, multi-contrast training corpus
   — which we cannot replicate without that data).
2. **In-domain roughly held but lesion-F1 regressed.** ATLAS Dice 0.559 ≈ H/I,
   but lesion-F1 0.392 < H 0.450 and I 0.479, and precision dropped to 0.325 (vs
   H 0.41) — the op-point landed at thr 0.5 / cc 5 (less aggressive FP pruning
   than H's thr 0.3 / cc 10), and the coarse scheme gives the lesion channel a
   less distinctive context. **No reason to prefer compact6.**

**Verdict:** compact6 is a dead end for the cross-contrast goal and a mild
regression in-domain. **Keep the 34-class scheme; ship H (Dice) / I (F1).** The
modality gap is now confirmed to survive every cheap from-scratch lever we have
tried — loss (H/I), synth:real ratio (F), and label scheme (J). The remaining
untried lever is the **synthesis itself**: the GMM contrast model never sees a
DWI-like appearance (restricted-diffusion *bright* lesion, distinct CSF/tissue
DWI contrast), so the model has no reason to recognize one. **Next run attacks
this directly — see "Run K (DWI-aware synthesis)" / the contrast-augmentation
work below.**

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
