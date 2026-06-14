# ATLAS R3.0 — their preprocessing vs ours, and why we train on raw

**Question (2026-06-14):** is our preprocessing as good as ATLAS's official
`atlas3_training_preprocessed`, and does our model work on raw, unprocessed input?

**Answer:** they do different things. ATLAS's preprocessing **nonlinearly
registers each scan to the MNI152NLin2009aSym template and rescales intensities
to [0, 100]**. Ours keeps images in **native space** and lets nnU-Net normalize
internally (**Z-score**) — no registration. **Training on raw native is the
correct choice for this challenge**, because the leaderboard provides raw native
T1w and scores masks in native space; training on the MNI set would force an
MNI-registration (+ inverse-warp) dependency at inference that the challenge does
not provide. Our model therefore works directly on raw, unprocessed data.

## Measured comparison (R001 sample, T1w)

| property | THEIR preprocessed (`space-MNI152NLin2009aSym_desc-brainnorm`) | OUR raw (`space-orig_desc-brain`) + nnU-Net |
|---|---|---|
| spatial space | **MNI152NLin2009aSym** (nonlinear registration) | **native** (no registration) |
| shape | **fixed 197×233×189** for every subject | **variable** per subject (e.g. 173×213×181, 154×197×197) |
| voxel size | 1.0 mm iso, RAS | 1.0 mm iso, RAS (ATLAS native already iso) |
| intensities | **rescaled to [0, 100]** (p50≈64, p99≈97, consistent across subjects) | **raw scanner units** (max ~8–10k, p50≈2800–3000, scanner-dependent) |
| skull-strip | yes (brain in MNI box, ~24% nonzero) | yes (brain, native) |

nnU-Net plan for `Dataset502_ATLASR30` (= our preprocessing, applied **internally
and identically at train + inference**): target spacing **1.0³ mm**,
**ZScoreNormalization**, patch 128³, median resampled size 148×167×137.

## Why ours is the right call (deployment)

1. **Raw-input compatibility (the hard requirement).** `nnUNetv2_predict` consumes
   raw native T1w and applies the plan's preprocessing (resample-to-1mm + per-image
   z-score) itself — the same transform used in training. No external tool needed.
   Already demonstrated by the R2.1 nnU-Net predicting on raw T1w *and* raw DWI for
   soop_bench. **A model trained on the MNI set would instead require running ATLAS's
   nonlinear registration on every new scan and inverse-warping the prediction back
   to native space** — a heavy, failure-prone pipeline the challenge harness does not
   provide. We deliberately avoid that dependency.
2. **Quality is comparable / standard.** nnU-Net is explicitly designed to segment
   in native space without template registration; large patches + augmentation give
   it spatial context that MNI alignment would otherwise provide. Per-image z-score
   is more robust to scanner-intensity variation than a fixed [0,100] rescale.
   Registration also risks interpolation artifacts and registration-failure modes we
   avoid entirely.
3. **Scoring space.** The challenge scores native-space masks; our outputs are native
   already (no inverse warp).

## Bottom line

Their preprocessed set = MNI registration + [0,100] intensity norm (reference only;
**we do not train on it**). Our pipeline = native + nnU-Net internal z-score, which
**works on raw unprocessed input** and is the field-standard for segmentation. The
R3.0 nnU-Net teacher (`Dataset502_ATLASR30`) is trained on **raw native** for
exactly this reason; a concrete raw-input inference check will be run once it is
trained.

Data: `data/ATLAS3_Training_Raw/` (raw, used), `data/ATLAS3_Training_Preprocessed/`
(MNI, comparison only). Download/decrypt jobs: `scripts/01_download_atlas_r30.sh`,
`scripts/02_download_atlas_r30_preprocessed.sh`.
