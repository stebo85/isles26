# ISLES'26 Data — Status, Timeline, Access (as of 2026-06-01)

Authoritative source: <https://isles-26.grand-challenge.org/> and its `/dataset/`
page. Data questions go to the **Forum** or **ezequiel.delarosa@uzh.ch**.

## TL;DR

- **Task:** native-space **T1w** MRI → **binary infarct mask**, across acute /
  sub-acute / chronic. Metadata JSON per case: `DAYS_POST_STROKE`, `CHRONICITY`
  (1 if < 180 days), `CENTER`. Final corpus ≈ **2,000 scans / 60+ centers**.
- **Batch 1 = ATLAS v2.0 (955 scans).** The dataset page states Batch 1 is
  downloaded from the *NITRC ATLAS v2.0 Download Portal* ("Raw Data" folder),
  native/skull-stripped, lesion masks for all subjects. **This is exactly the
  data we already have** decrypted at [data/ATLAS_R2.1_raw/Training_Raw/](../data)
  (955 sessions / 33 sites), already characterized
  ([baselines/reports/training_data_characterization/](../baselines/reports/training_data_characterization/))
  and already converted to nnU-Net `Dataset501_ATLASR21`. **So our work to date
  is on the REAL ISLES'26 Batch 1 — not a proxy.**
- **Batch 2 = the new, un-ingested expansion** (toward ~2,000 / 60+ centers).
  Scheduled release **2026-05-29**. Its **format/layout is not published** on the
  dataset page yet ("still to come"); whether it follows ATLAS's BIDS-ish layout
  or a different one is **unknown** until we see it. Both batches are for
  **training**.

## Key dates

| Milestone | Date |
|---|---|
| Batch 1 release (ATLAS) | 2026-04-24 |
| Batch 2 release | 2026-05-29 |
| Submission system opens | 2026-06-15 |
| Challenge closes | 2026-08-01 (23:59 CET) |

## Access (auth-gated — human action required)

- A **verified Grand Challenge account** is mandatory; must **join the challenge**
  before downloading. This cannot be scripted from the cluster (interactive
  auth). A human must download Batch 2 via browser and stage it on `/scratch`.
- Batch 1 (ATLAS) is already staged locally (see above).

## Batch-1 (ATLAS) on-disk layout (known-good, for reference)

```
data/ATLAS_R2.1_raw/Training_Raw/<SITE=Rxxx>/sub-rXXXsYYY/ses-1/anat/
  sub-rXXXsYYY_ses-1_space-orig_desc-brain_T1w.nii.gz              # T1w (skull-stripped)
  sub-rXXXsYYY_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz  # binary lesion mask
```
Per-session metadata (DAYS_POST_STROKE etc.) comes from the ATLAS metadata CSV,
joined in [characterize_training_data.py](../scripts/analysis/characterize_training_data.py)
and [convert_atlas_r21_to_nnunet.py](../baselines/models/nnunet_atlas_t1w/nnunet_helpers/convert_atlas_r21_to_nnunet.py).
ATLAS gotchas (re-binarize float masks, force canonical orientation, per-image
percentile + z-score normalization, 126 sessions missing DAYS_POST_STROKE) are in
[atlas_r21_training_data_characterization.md](atlas_r21_training_data_characterization.md).

## What "prep real data" means here

Batch 1 is done. The open work is to make **Batch 2 drop-in**:
1. Human downloads Batch 2 (verified GC account) → stage under `data/isles26_batch2/`.
2. Run the **format-discover** tool to report Batch 2's actual structure
   (per-subject T1w / mask / metadata candidates + anomalies).
3. Point the (pattern/config-driven) characterizer + nnU-Net converter at it;
   build a combined **Batch1+Batch2** nnU-Net dataset with stratified splits
   (center × chronicity × lesion-size, the existing discipline).
4. Retrain the strong nnU-Net teacher (multi-fold + ResEnc + ensemble) on the
   real combined corpus — the campaign we deliberately deferred off the
   ATLAS-only proxy.

The characterizer/converter generalization is best **finalized once Batch 2's
real layout is known** (via discover), to avoid guessing the format.
