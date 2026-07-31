# nnU-Net v2 3D fullres — ATLAS R2.1 T1w baseline

Our own canonical strong T1w baseline. nnU-Net v2 (default 3D full-resolution,
auto-configured) trained from scratch on ATLAS R2.1.

| | |
|---|---|
| **Family** | nnU-Net v2 |
| **Input** | T1w only (1 channel) |
| **Trained on** | ATLAS R2.1, 955 sessions, fold 0 of a stratified 5-fold split |
| **Schedule** | 250 epochs (reduced from default 1000), single fold, no TTA, no ensemble |
| **Dataset id** | `Dataset501_ATLASR21` under `work/nnunet/` |
| **Reports** | `baselines/reports/nnunet/` (soop_bench), `baselines/reports/nnunet_atlas_val/` (ATLAS held-out) |

## Pipeline (run in order)

| Script | Stage | Resources |
|---|---|---|
| `analysis_nnunet_01_prepare_dataset.sh` | build `.venv-nnunet/`, convert ATLAS→nnU-Net format (RAS canonical, re-binarize masks, SHA manifest), plan+preprocess, write stratified splits | CPU, ~15 min |
| `analysis_nnunet_02_train.sh` | train fold 0 / 3d_fullres / 250 ep (preempt-safe), then `find_best_configuration` | 1 GPU, ~2.5 h |
| `analysis_nnunet_03_predict_soop_bench.sh` | SynthStrip skull-strip → predict → ANTs rigid T1w→TRACE warp → `work/predictions_nnunet/` | 1 GPU array, 12 tasks |
| `analysis_nnunet_04_evaluate.sh` | `eval.run_eval` → `baselines/reports/nnunet/` | CPU |
| `analysis_nnunet_05a_compute_atlas_val_array.sh` | write fold-0 val list + print `--array=0-N` | login (fast) |
| `analysis_nnunet_05_predict_atlas_val.sh` | predict held-out ATLAS val (1 case/array task) → `work/predictions_atlas_val/` | 1 GPU array |
| `analysis_nnunet_06_evaluate_atlas_val.sh` | `run_atlas_eval.py` → `baselines/reports/nnunet_atlas_val/` | CPU |

Submit with dependencies, e.g.:
```bash
PREP=$(sbatch -p owners analysis_nnunet_01_prepare_dataset.sh | awk '{print $NF}')
TRAIN=$(sbatch -p owners --dependency=afterok:$PREP analysis_nnunet_02_train.sh | awk '{print $NF}')
sbatch -p owners --dependency=afterok:$TRAIN analysis_nnunet_03_predict_soop_bench.sh
# then 04; and 05a → 05 (--array=0-N) → 06 for the ATLAS-val track
```

`nnunet_helpers/`: `convert_atlas_r21_to_nnunet.py` (convert + stratified splits),
`nnUNetTrainer_250epochs.py` (trainer subclass), `install_trainer_variant.py`
(installs it into the venv), `run_atlas_eval.py` (ATLAS-val evaluator mirroring
`eval/run_eval.py`), `requirements_nnunet.txt` (pinned deps).

## Corrected center-grouped TopK10 run

The challenge candidate uses an independent immutable nnU-Net lineage,
`Dataset507_ATLASR30_CORRECTED`. It converts the corrected ATLAS R3.0 inventory
from source, preprocesses all 1,452 usable cases, and trains five
`nnUNetTrainerDiceTopK10Loss` folds. Validation centers are disjoint from each
fold's training centers. The allocator balances validation case count, stroke
phase, lesion-size bins, and lesion burden without splitting a center.

The frozen assignment and balance audit are in
`baselines/reports/nnunet_center_grouped_splits/`. SOOP is one indivisible
168-case center, so its unknown-phase cases necessarily occur together in one
validation fold; the report makes that constraint explicit.

| Script | Stage | Resources |
|---|---|---|
| `analysis_nnunet_42_prepare_center_grouped.sh` | fresh Dataset507 conversion, planning, preprocessing, split and lineage verification | CPU |
| `analysis_nnunet_43_train_center_topk10.sh` | five-fold corrected TopK10 training and validation | 1 GPU per fold |
| `analysis_nnunet_44_postprocess_center_topk10.sh` | consolidated cross-validation and postprocessing selection | CPU |
| `analysis_nnunet_45_eval_center_topk10_oof.sh` | rich raw and postprocessed out-of-fold evaluation | CPU |
| `analysis_nnunet_46_submit_center_topk10.sh` | submit the dependency chain above | login |

Submit the complete chain with:

```bash
bash baselines/models/nnunet_atlas_t1w/analysis_nnunet_46_submit_center_topk10.sh
```

Preparation refuses to reuse or overwrite another Dataset507 directory. Each
training task re-verifies the dataset certificate and refuses checkpoints whose
training lineage does not match the frozen data, split, trainer, and script.

## Results (2026-05-20)

| Eval set | n | Dice mean/median | Lesion F1 | Surface Dice 3mm | HD95 mm |
|---|---:|---:|---:|---:|---:|
| ATLAS R2.1 fold-0 held-out (in-distribution) | 194 | 0.640 / 0.725 | 0.654 | 0.756 | 18.8 |
| soop_bench (OOD, T1w-only) | 12 | 0.188 / 0.000 | 0.201 | 0.200 | 50.5 |

## How to read these numbers

**Strong in-distribution, collapses OOD — and the OOD collapse is structural,
not a model defect.** ATLAS Val 0.640 is ~MAPPING-level (the ATLAS R2 winner got
0.667 with a 5×1000-epoch ensemble; we used 1 fold / 250 ep). The soop_bench
0.188 is far below DeepISLES (0.540) because **soop_bench GT is DWI-defined acute
infarcts**, which are bright on DWI but subtle/invisible on T1w, whereas this
model learned **chronic** encephalomalacia appearance on T1w. soop_bench Dice is
monotonic with chronicity (chronic 0.53 > mixed 0.33 > acute 0.12) and the model
massively under-segments acute cases. See
`baselines/reports/nnunet/PIPELINE_STATUS.md` and
`docs/lessons/problems.md` Problems 3 & 8.

## Known caveats / next steps

- Reduced schedule (1 fold, 250 ep, no TTA, no ensemble) — the obvious upgrade is
  full 5-fold + TTA for a few more in-distribution Dice points.
- soop_bench predictions are warped T1w→TRACE per subject; eyeball
  `work/qc_t1_to_trace/*.png` before trusting individual soop cases.
- soop_bench T1w is SynthStrip-stripped at inference vs ATLAS curator
  brain-extraction at training — a secondary domain shift.
- Cluster-specific build/runtime gotchas (sdist deps, primus/torchvision,
  `nnUNet_compile=f`, trainer `__init__` signature, `nnUNetPlans.json`) are
  documented in `agents.md`.
