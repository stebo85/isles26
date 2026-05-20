# nnU-Net v2 baseline — pipeline status

**COMPLETED 2026-05-20.** Reports: `report.md` (soop_bench OOD) here and
`baselines/reports/nnunet_atlas_val/report.md` (ATLAS fold-0 held-out).

## Headline results

| Eval set | n | Dice mean / median | Lesion F1 | Surface Dice 3mm | HD95 mm |
|---|---:|---:|---:|---:|---:|
| **ATLAS R2.1 fold-0 held-out** (in-distribution) | 194 | **0.640 / 0.725** | 0.654 | 0.756 | 18.8 |
| **soop_bench** (OOD, T1w-only) | 12 | **0.188 / 0.000** | 0.201 | 0.200 | 50.5 |
| soop_bench — DeepISLES reference (DWI/ADC/FLAIR) | 12 | 0.540 / 0.704 | 0.582 | 0.763 | 18.9 |

Training reached **Mean Validation Dice 0.6436** (matches the eval-framework's
0.640), in line with the MAPPING ATLAS winner (~0.667) despite only fold 0 /
250 epochs / no ensemble / no TTA.

## Interpretation — the key finding

The model is **strong in-distribution and collapses out-of-distribution**, and
the failure is structural, not a bug:

- **ATLAS held-out (0.64 Dice / 0.65 lesion-F1)** confirms the training is sound
  and the stratified split is honest. Performance is roughly flat across
  chronicity (ge180d 0.66, lt180d 0.59).
- **soop_bench (0.19 Dice)** is far below DeepISLES (0.54). The per-chronicity
  breakdown explains why: Dice is **chronic 0.53 > mixed 0.33 > acute 0.12**,
  monotonic with chronicity. ATLAS R2.1 is a **chronic T1w** lesion dataset, so
  the model learns chronic encephalomalacia/cavity appearance on T1w. soop_bench
  GT is **DWI/TRACE-defined**, dominated by **acute** infarcts that are bright on
  DWI but subtle/invisible on T1w — so the model massively under-segments them
  (e.g. sub-8: 0.48 mL predicted vs 79.7 mL GT). DeepISLES wins on soop_bench
  precisely because it consumes the DWI/ADC/FLAIR modalities the acute lesions
  are defined on.

**Takeaway for ISLES'26:** a T1w-only chronic-trained nnU-Net is a legitimate
strong baseline for *chronic T1w* lesion segmentation (ATLAS-like), but is the
wrong tool for acute DWI-defined benchmarks. The soop_bench number is a
domain-and-modality-mismatch result, not a model-quality result. For a fair
T1w-vs-T1w comparison we would need a T1w-defined acute/subacute test set, or to
add the DWI modality to match soop_bench's target.

## Caveats baked into these numbers

- Reduced schedule: single fold (fold 0), 250 epochs (vs MAPPING's 5×1000 +
  ensemble), TTA disabled.
- soop_bench predictions are warped T1w→TRACE via per-subject ANTs rigid
  registration; QC overlays in `work/qc_t1_to_trace/*.png` should be eyeballed
  before over-interpreting individual soop cases.
- soop_bench T1w skull-stripped with SynthStrip at inference vs ATLAS curator
  brain-extraction at training — a secondary domain shift on top of the
  chronicity/modality mismatch.

---

## (original pre-run plan below)

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
