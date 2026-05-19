# nnU-Net v2 baseline — pipeline status

Submitted 2026-05-19. Final report lands here (`report.md`, `report.json`,
`per_case.json`) once the chain completes.

## Why nnU-Net v2

Canonical strong T1w stroke baseline from the literature (MAPPING / ATLAS R2
winner is built on nnU-Net). All "stronger" published methods (MAPPING, Halai
2026, BrainSegFounder, Neuro-TransUNet) gate pretrained weights behind
SharePoint/OneDrive auth that the cluster cannot reach, so the most reliable
"strong baseline" path is to train nnU-Net v2 ourselves on the
already-decrypted ATLAS R2.1 training data.

## Configuration

- **Dataset**: `Dataset501_ATLASR21` — ATLAS R2.1 T1w (curator brain-extracted),
  stratified 5-fold split by site × chronicity × lesion-size bin (RNG seed 42).
- **Architecture**: nnU-Net v2 default 3D fullres (auto-configured by nnU-Net's
  planner on this dataset).
- **Training**: fold 0 only, 250 epochs (reduced from default 1000 for a
  time-bounded baseline; documented honestly in the final report).
- **Test-time augmentation**: **disabled** (`--disable_tta`) for speed.

## Pipeline files

| Order | Script | Purpose |
|------:|---|---|
| 01 | `analysis_nnunet_01_prepare_dataset.sh` | Build `.venv-nnunet/`; convert ATLAS R2.1 → nnU-Net raw format with RAS canonical + mask re-binarization + source SHA-256 manifest; nnU-Net planning/preprocessing; write stratified `splits_final.json` |
| 02 | `analysis_nnunet_02_train.sh` | Train fold 0 / 3D fullres / 250 epochs. Preempt-safe via `--requeue` + USR1 trap → `scontrol requeue`; checkpoint every 10 epochs |
| 03 | `analysis_nnunet_03_predict_soop_bench.sh` | 12-task array. SynthStrip skull-strip → nnU-Net predict in T1w-brain space → ANTs rigid T1w→TRACE → `antsApplyTransforms -n GenericLabel` warp to TRACE; checkpoint-SHA sidecars |
| 04 | `analysis_nnunet_04_evaluate.sh` | Run `eval.run_eval` on soop_bench predictions; fail if any subject missing |
| 05a | `analysis_nnunet_05a_compute_atlas_val_array.sh` | Compute fold-0 val list + print `--array=0-N`; run after prepare finishes |
| 05 | `analysis_nnunet_05_predict_atlas_val.sh` | SLURM array — one task per held-out ATLAS val case |
| 06 | `analysis_nnunet_06_evaluate_atlas_val.sh` | Run `run_atlas_eval.py` (parallel to soop's `eval.run_eval`) and write the second report |

## Reports

- **`baselines/reports/nnunet/`** — soop_bench (12 subjects, OOD: Philips
  multimodal scanners; T1w + FLAIR + DWI/TRACE/ADC available, only T1w used
  by this model).
- **`baselines/reports/nnunet_atlas_val/`** — ATLAS R2.1 fold-0 held-out val
  (~190 cases, in-distribution but never seen during training). Stratifies by
  site (ATLAS R001…R050), chronicity (lt180d / ge180d / unknown), and lesion
  size bin (mL).

## Known limitations (already documented in report Method Notes)

- Reduced-schedule single-fold (not the full MAPPING-style 5-fold 1000-epoch
  ensemble).
- TTA off.
- soop_bench T1w are SynthStrip-stripped at inference, ATLAS T1w are
  curator-stripped at training → mask-boundary distribution mismatch may hurt
  cortical-edge lesion Dice on soop_bench.
- ANTs rigid T1w→TRACE registration is per-subject and unsupervised; QC PNGs
  in `work/qc_t1_to_trace/` should be inspected before trusting soop_bench
  numbers.

## Adversarial-review log

Two rounds of codex adversarial review were applied before submission:

- Round 1 (15 defects): blocker on preempt-without-requeue + blocker on
  T1w-space-vs-TRACE-space evaluation. Fixed all blockers + majors.
- Round 2 (1 blocker + 2 majors): TRACE was 4-D singleton `(X,Y,Z,1)` while
  ANTs called `-d 3` (squeeze required); registration tail level too coarse
  (added `x1` final level); single batched ATLAS val job was not
  preempt-safe (converted to SLURM array of one-case-per-task).

All Round-1 and Round-2 findings applied. See `agents.md` "nnU-Net v2 baseline"
bullet for the durable summary.
