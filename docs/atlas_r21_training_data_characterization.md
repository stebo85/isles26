# ATLAS R2.1 Raw Training Data — Characterization

Source tarball: `data/atlas21_training_raw.tar.gz` (openssl AES-256-CBC, base64-armored; see `scripts/download_data.sh`).
Decrypted + extracted to `data/ATLAS_R2.1_raw/Training_Raw/` (gitignored).

Pipeline: [scripts/analysis/characterize_training_data.py](../scripts/analysis/characterize_training_data.py) (SLURM-submitted via [analysis_01_characterize_training_data.sh](../scripts/analysis/analysis_01_characterize_training_data.sh)) → per-session CSV; aggregated by [aggregate_training_data.py](../scripts/analysis/aggregate_training_data.py); QC slices by [render_qc_examples.py](../scripts/analysis/render_qc_examples.py). All outputs live under [baselines/reports/training_data_characterization/](../baselines/reports/training_data_characterization/).

Job 25415567: COMPLETED in 17m32s, MaxRSS 3.9 GB.

## Top-line numbers

| Property | Value |
|---|---|
| Sessions read OK | 955 / 955 |
| Sites (centers, `R001..R052`) | 33 |
| Sub-folders per session present | T1w `nii.gz`, lesion `nii.gz`, metadata `csv` (no `.DS_Store` ignored) |
| Read errors / missing files / shape mismatches / affine mismatches | 0 / 0 / 0 / 0 |
| One session per subject | yes (`ses-1` for every subject; no longitudinal pairs) |
| Voxel-size isotropy ≈1 mm (0.9–1.1 mm³) | 82.1 % |
| Orientations seen | LAS 503, RAS 452 (no other codes) |
| T1w dtype distribution | float64 910, float32 45 (no integer storage) |
| Lesion-mask dtype distribution | uint16 437, float32 218, int32 143, uint8 141, float64 16 |

## Why this matters for the ISLES'26 pipeline (per `agents.md`, push lessons up)

1. **Voxel geometry is heterogeneous but mostly 1 mm iso.** Median voxel volume = 1.0 mm³; 82 % fall inside the 0.9–1.1 mm³ band. Outliers: R049 (median 3.375 mm³, ~1.5 mm iso), R015 (median 0.25 mm³, ~0.5×0.5×1 mm), R027 (mixed 1 mm/non-iso). nnU-Net's auto-resampling handles this; for a browser-deployable smaller model, fix a target spacing (1 mm iso) and verify the hi-res outlier sites do not lose tiny lesions in resampling.
2. **Orientation is split LAS/RAS.** Always apply `nib.as_closest_canonical` (or set `orientations.io_orientation`) before training, otherwise left/right priors flip across half the dataset.
3. **Raw intensity scale varies by ~5 orders of magnitude.** T1w p99 ranges from 26 to 2.5 × 10⁶ (mean 32 k, median 427). No site labels imply a common scaling. Per-image normalization (z-score within brain, or 1–99 percentile clip) is mandatory before feeding any model; site-conditional normalization will not help — the chaos is also intra-site (e.g. mixed float64 storage with un-rescaled DICOM values).
4. **T1w stored as float64.** 95 % of T1 volumes are float64 (likely from upstream resampling). For a small/in-browser model the on-disk type does not matter, but be aware that simple `astype(np.uint8)` after percentile clipping is fine and saves I/O.
5. **Two-class binary masks, but R039 stores them as float64 with value 0.9999999776482582 (4 sessions: `sub-r039s001/2/3/5`).** `> 0` thresholding handles it; any downstream code using `== 1` (e.g. some nnU-Net plans-authoring helpers) would silently discard the lesion on those sessions. The characterization CSV now rounds before stringifying (`int(round(u))`), so all 955 sessions show `lesion_unique_labels = "0,1"` — but the underlying float-encoded values are unchanged on disk, so the **re-binarize-on-load step in the preprocessing block is still mandatory**.
6. **Lesion volumes are log-distributed and skewed small.** Median 4.3 mL, p95 136 mL, max 388 mL; p5 = 0.14 mL (≈140 voxels at 1 mm iso); 14 sessions have a tiny lesion (<0.05 mL = ≤50 voxels). These dominate lesion-wise F1 and surface Dice@3mm — match the eval framework's existing size-binned recall.
7. **Strong lesion fragmentation.** Connected-component counts use 26-connectivity (matches [eval/metrics.py](../eval/metrics.py)). Median CC = 2, p75 = 3, p95 = 8, max = 38, mean ≈ 2.58. Lesion-wise F1 and Simple Lesion Count metrics will swing heavily on small CCs; post-processing that removes tiny CCs (a common trick) will help Dice but hurt F1.
8. **Chronicity coverage is uneven and metadata is partially missing.** Distribution after the DPS-sanitization rule (`unknown = NaN ∨ DPS ≤ 0`): **chronic ≥180 d 540, acute/subacute (0, 180) d 264, unknown 151** (126 NaN + 24 DPS=0 from R049/R050 + 1 DPS=-4 in `sub-r049s018`). Six sites supply no DPS at all (R010, R011, R015 most prominently); R049 reports DPS=0 for 16/24 sessions (treated as `unknown` here, since R049 is the same site with the negative-DPS error). Center-level chronicity bias is extreme: R031 ≈ 3 % chronic vs. R047/R048/R001/R005/R024/R034/R052 = 100 % chronic. Random CV will be confounded — stratify val splits by site **and** by chronicity (matches the agents.md lesson). If the challenge organizers confirm `DPS=0` legitimately means "hyperacute" rather than "missing", re-bin those 24 sessions into `lt180d` and the unknown count drops back to 127.
9. **Center imbalance is severe.** Top 3 sites (R009, R038, R040) supply 30 % of all sessions (291/955); 6 sites have ≤9 sessions (R014, R029, R039, R044, R045, R050). Any per-site fold will under-sample these centers — consider leave-one-site-out only for the larger sites and a multi-site held-out fold for the smaller ones.
10. **No longitudinal data.** Each subject has exactly one session — no subject leakage risk across splits from same-patient repeats. Subjects can be split = sessions.

## Data-quality flags (raised by the per-session characterization)

| Flag | Count | Action |
|---|---:|---|
| `non_binary_mask` after `int(round(...))` fix | 0 | (R039 float-encoded `0.9999…` masks are now classified `0,1`; still must re-binarize on load — keep test fixture against `sub-r039s002`/`s005`.) |
| `tiny_lesion_lt0p05ml` (≤50 voxels @1 mm³) | 14 | Keep in train; do not auto-drop in CC post-processing; report recall on size bin 0–1 mL separately. |
| `huge_lesion_gt250ml` | 4 | Sanity-check segmentation visually (could be hemispheric or include CSF cavity). |
| `missing_days_post_stroke` | 126 | Treat as a third chronicity stratum (`unknown`); do not impute 0. |
| `DPS == 0` (R049 × 16, R050 × 8) | 24 | Likely coded "missing" — confirm with challenge organizers; otherwise treat as `unknown`. |
| `DPS == -4` (`sub-r049s018`) | 1 | Definitely metadata error; treat as `unknown`. |

## Site-level snapshot (top 12 by size)

| Site | n | median lesion (mL) | p95 lesion (mL) | 1 mm iso frac | median CC | median DPS | chronic frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| R009 | 111 | 1.11 | 47 | 1.00 | 3 | 426 | 0.55 |
| R038 |  94 | 4.61 | 116 | 1.00 | 2 | 100 | 0.19 |
| R040 |  86 | 16.10 | 226 | 1.00 | 1 | 74 | 0.21 |
| R047 |  48 | 10.58 | 155 | 0.00 (non-iso) | 2 | 1024 | 0.96 |
| R048 |  44 | 8.47 | 165 | 1.00 | 2 | 1404 | 1.00 |
| R001 |  38 | 4.47 | 81 | 1.00 | 2 | 698 | 1.00 |
| R031 |  37 | 6.73 | 118 | 1.00 | 3 | 138 | 0.03 |
| R004 |  37 | 40.01 | 141 | 0.95 | 2 | 1701 | 0.54 |
| R042 |  35 | 11.29 | 154 | 1.00 | 3 | 728 | 0.97 |
| R005 |  34 | 1.85 | 161 | 1.00 | 1 | 985 | 0.97 |
| R052 |  32 | 6.02 | 124 | 1.00 | 2.5 | 1181 | 1.00 |
| R027 |  32 | 11.16 | 176 | 0.59 (mixed) | 2 | 610 | 0.91 |

Full table: [per_site_summary.csv](../baselines/reports/training_data_characterization/per_site_summary.csv).

## Concrete recommendations for the modelling effort

- **Preprocessing block (universal), strict order:**
  1. **Re-binarize the mask first** (`mask = (raw > 0.5).astype(np.uint8)`) — R039 stores masks as `0.9999999776482582`; nnU-Net `dataset_fingerprint` helpers that use `== 1` would silently empty those 2 cases.
  2. `as_closest_canonical` for both T1 and mask (folds LAS/RAS into RAS).
  3. Resample to 1.0 mm iso (linear T1, nearest mask).
  4. `1–99 percentile clip` on T1 (use brain-only voxels via `T1>0` or HD-BET hull).
  5. `z-score within brain mask`.
  6. Cast T1 to float16, mask to uint8.

  Run as a one-off SLURM array writing to `work/atlas_r21_pre/` (gitignored). After resampling, re-check lesion voxel counts: drop any session whose post-resample lesion is `< 30 voxels` (~0.03 mL) to a manual-QC list rather than silently into training (R035 has the tightest case at 0.06 mL pre-resample).
- **Validation splits:** 5-fold stratified by `(site, chronicity_bin)` where `chronicity_bin ∈ {acute_<180d, chronic_≥180d, unknown}`; this matches the agents.md "stratify validation by center/chronicity/lesion size" rule and the ATLAS-MICCAI winner setup. Hold R044/R045/R039/R014 (smallest sites with `n ≤ 9`) entirely out of training as an external generalization probe.
- **Reporting metrics:** keep the eval framework's lesion-wise F1, HD95/ASSD, surface Dice @3 mm, size-binned recall (use [0, 0.5), [0.5, 5), [5, 50), [50, ∞) mL bins). Add per-site Dice/F1 tables to every model report so site bias is visible.
- **Data sanity gate in training data loader:**
  - Re-binarize mask on load (`> 0.5`).
  - Assert affine matches T1 (already true for all 955 cases — keep the check).
  - Drop / log anything outside `LAS|RAS` (none today, but defensive).
  - If `DAYS_POST_STROKE ∈ {NaN, 0, <0}` on R049 or R050 → tag as chronicity `unknown`.
- **Inference-time:** keep native voxel grid for the **output** mask; only resample T1 → 1 mm for the model and project predictions back with nearest-neighbour. Otherwise small lesions in the 0.45 mm in-plane R015 / R027 cases get aliased.

## QC images

All under [baselines/reports/training_data_characterization/qc/](../baselines/reports/training_data_characterization/qc/):

| File | What it shows |
|---|---|
| `lesion_volume_hist.png` | log10(volume mL); reference lines at 0.1/1/10/100 mL |
| `voxel_xz_scatter.png` | in-plane × slice-thickness cloud — surfaces the non-iso outliers (R049, R015, R027) |
| `days_post_stroke_hist.png` | DPS with 180-day cutoff line |
| `sessions_per_site.png` | per-site n |
| `cc_vs_volume.png` | fragmentation versus burden — shows that tiny lesions are not the fragmented ones |
| `example_smallest_nonzero__sub-r011s011_ses-1.png` | 0.01 mL lesion — the kind that breaks Dice but matters for F1 |
| `example_median__sub-r023s015_ses-1.png` | representative mid-size case |
| `example_largest__sub-r005s015_ses-1.png` | 388 mL "hemispheric" infarct |
| `example_most_fragmented__sub-r017s119_ses-1.png` | 60-CC multifocal case |
| `example_site_R{009,038,040,047,048}_random__*.png` | top-5 site exemplars |

## Reproducing

```bash
# 1. decrypt + extract (one-time)
cd data
export ATLAS_PW='<password>'
openssl aes-256-cbc -md sha256 -d -a -in atlas21_training_raw.tar.gz \
    -pass env:ATLAS_PW | tar xz -C ATLAS_R2.1_raw
unset ATLAS_PW

# 2. characterize (~18 min on owners partition)
cd ..
sbatch scripts/analysis/analysis_01_characterize_training_data.sh

# 3. aggregate + plots (~10 s on login node)
module load python/3.12.1 && source .venv-eval/bin/activate
python scripts/analysis/aggregate_training_data.py \
    --csv baselines/reports/training_data_characterization/per_session.csv \
    --out-dir baselines/reports/training_data_characterization
python scripts/analysis/render_qc_examples.py \
    --csv baselines/reports/training_data_characterization/per_session.csv \
    --out-dir baselines/reports/training_data_characterization/qc
```

Per-session CSV (`per_session.csv`) is committed alongside `summary.json`, `per_site_summary.csv`, `per_chronicity_summary.csv`, and the QC PNGs so the report is reviewable without re-running. Adversarial-review trail (defects 1–7 with file/line refs and the fixes that were applied): [baselines/reports/training_data_characterization/adversarial_review.md](../baselines/reports/training_data_characterization/adversarial_review.md).
