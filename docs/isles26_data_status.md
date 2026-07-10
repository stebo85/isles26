# ISLES'26 Data Status, Timeline, Access (as of 2026-06-25)

Authoritative sources: <https://isles-26.grand-challenge.org/> and its
`/dataset/` page. Data questions go to the challenge forum or the listed
organizers.

## TL;DR

- **Task:** native-space **T1w** MRI plus per-case metadata JSON
  (`DAYS_POST_STROKE`, `CHRONICITY`, `CENTER`) -> **binary infarct mask**.
- **Public training data is now released.** The whole Batch 1 + Batch 2
  training release is **N=1453** and corresponds to ATLAS v3.0 / R3.0.
- **Local state:** the release is staged at
  [data/ATLAS3_Training_Raw/](../data), characterized in
  [baselines/reports/training_data_characterization_r30/](../baselines/reports/training_data_characterization_r30/),
  and converted to nnU-Net `Dataset502_ATLASR30`.
- **Usable nnU-Net cases:** 1450/1453. Three sessions are intentionally skipped
  by the converter because of shape/affine problems; see
  `work/nnunet/nnUNet_raw/Dataset502_ATLASR30/conversion_skips.csv`.
- The old "Batch 2 format unknown / human download still required" blocker is
  resolved for this workspace.

## Key Dates

| Milestone | Date |
|---|---|
| Batch 1 release | 2026-04-24 |
| Batch 2 release | 2026-06-12 |
| Submission system opens, sanity phase | 2026-07-15 |
| Challenge closes, Docker submission | 2026-08-15 23:59 CET |

## Public Training Release

The dataset page describes the whole training release as **Batches 1 and 2,
N=1453**. It includes:

- ATLAS v2.0 train/validation/test masks: 955 cases.
- SOOP cohort: 169 cases.
- Newly added cases: 329 cases.

The challenge overview still describes the broader challenge corpus as roughly
2,000 T1w scans across 60+ centers; for model development in this repo, the
public labeled training release is the 1453-session ATLAS R3.0 set.

## Metadata Variables

Each case ships a per-case metadata JSON. The stroke-timing variables interact
as follows:

- **`DAYS_POST_STROKE`** — the primary, most informative timing variable
  (continuous days between stroke onset and scan). Preferred whenever available,
  but for some cases it could not be retrieved and is missing.
- **`CHRONICITY`** — a secondary, coarse fallback. It is provided as a partial
  substitute *because `DAYS_POST_STROKE` was not retrievable for some cases*;
  when known it gives a rough sense of lesion maturation. It is effectively a
  single-valued flag: **`1` = chronic (180+ days post-stroke)** or **`NaN` = not
  available**. There is no explicit "acute/subacute" code — that state is only
  inferrable from `DAYS_POST_STROKE`, not from `CHRONICITY`. So `CHRONICITY`
  alone can confirm "this case is 180+ days out" but cannot, on its own, place a
  case in the acute/subacute window.

Practical consequence for stratification: a case's chronicity bin is derived by
combining both fields — `DAYS_POST_STROKE` when present, else `CHRONICITY==1`
implies `chronic_≥180d`, and everything remaining stays `unknown`. See
[atlas_r21_training_data_characterization.md](atlas_r21_training_data_characterization.md)
for the exact DPS-sanitization rule and the observed bin counts.

## Local Layout

Raw release:

```text
data/ATLAS3_Training_Raw/<SITE>/sub-*/ses-1/anat/
  *_space-orig_desc-brain_T1w.nii.gz
  *_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
```

Converted nnU-Net release:

```text
work/nnunet/nnUNet_raw/Dataset502_ATLASR30/
  imagesTr/ATLAS_####_0000.nii.gz
  labelsTr/ATLAS_####.nii.gz
  conversion_manifest.csv
  conversion_skips.csv
work/nnunet/nnUNet_preprocessed/Dataset502_ATLASR30/
  splits_final.json
  splits_summary.json
```

The converter forces canonical orientation, checks shape/affine agreement,
re-binarizes masks on load, and writes stratified folds by
`site x chronicity_bin x size_bin`.

## Current Data Caveats

- `Dataset502_ATLASR30` has 1450 converted cases, not 1453. The skipped cases
  are documented, reproducible converter decisions, not silent data loss.
- R3.0 includes SOOP-source subjects, so `soop_bench` is useful as a T1w/DWI
  stress test but is not a clean external validation set for R3.0-trained
  models.
- The strongest current validation need is no longer ingestion; it is hidden
  generalization: center-held-out and chronicity-held-out splits/reports.

## Open Work

1. Use [baselines/reports/hidden_generalization_splits/](../baselines/reports/hidden_generalization_splits/)
   to run true leave-center-out and leave-chronicity-out retraining where needed.
2. Package the TopK10 5-fold inference path and the T1w OOD blend into the final
   submission container.
3. Treat any additional unlabeled T1w for self-training as opt-in input only;
   never pseudo-label hidden challenge/evaluation data used for reporting.
