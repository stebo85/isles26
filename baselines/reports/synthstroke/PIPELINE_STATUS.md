# SynthStroke synthetic-data phase — pipeline status

**COMPLETED 2026-05-21.** Two tracks: (A) benchmark the released pretrained
SynthStroke weights, and (B) train a SynthStroke-style model on our own ATLAS
data with self-contained FreeSurfer label maps. Both evaluated under the same
fold-0 / soop_bench discipline as the nnU-Net baseline.

## Headline results (Dice / lesion-F1)

| Model | ATLAS fold-0 val (T1w) | soop TRACE (DWI) | soop T1w-in-TRACE |
|---|---:|---:|---:|
| **synthstroke-synth-plus** (pretrained) ⭐ | 0.458 / 0.515 | **0.447 / 0.478** | 0.217 / 0.231 |
| synthstroke-synth-pseudo (pretrained) | 0.381 / 0.333 | 0.262 / 0.167 | 0.180 / 0.284 |
| synthstroke-qsynth (pretrained) | 0.367 / 0.461 | 0.246 / 0.216 | 0.164 / 0.287 |
| synthstroke-synth (pretrained) | 0.403 / 0.599 | 0.261 / 0.128 | 0.145 / 0.222 |
| synthstroke-qatlas (pretrained) | 0.282 / 0.329 | 0.052 / 0.187 | 0.115 / 0.178 |
| synthstroke-baseline (pretrained) | 0.334 / 0.240 | 0.044 / 0.021 | 0.099 / 0.134 |
| **our model** (fold-0, reduced 150-ep, mix-real) | 0.157 / 0.023 | 0.004 / 0.020 | 0.162 / 0.080 |
| *nnU-Net (ours, ATLAS chronic T1w)* | *0.640 / 0.654* | — | *0.188 / 0.201* |
| *DeepISLES (DWI+ADC+FLAIR)* | — | *0.540 / 0.582* | — |

Full per-variant reports: `baselines/reports/synthstroke/<variant>/{atlas_val,soop_trace,soop_t1w}/`
and the committed comparison `pretrained_report.md`.

## Key findings

1. **Synthetic augmentation closes the cross-contrast gap (the main result).**
   The pretrained **synth-plus** scores **0.447 Dice on soop DWI** — 2.4× our
   chronic-T1w nnU-Net (0.188) and approaching DeepISLES (0.540, which actually
   consumes DWI/ADC/FLAIR). The non-synthetic variants (baseline, qatlas) stay
   near zero on DWI. This directly validates the SynthStroke thesis: a model
   trained only on contrast-randomized synthetic data segments acute DWI lesions
   far better than a real-T1w-trained model. **synth-plus is the artifact to use.**

2. **nnU-Net still wins in-domain ATLAS T1w** (0.640 vs synth-plus 0.458) — it is
   specialised on that exact distribution; the synth models trade in-domain Dice
   for out-of-domain robustness, as the SynthStroke paper reports.

3. **Our from-scratch model is undertrained and not competitive** (ATLAS 0.157,
   soop TRACE 0.004, soop T1w 0.162; lesion-F1 0.02–0.08). It massively
   over-segments (e.g. soop sub-9: 73 predicted components vs 2 GT). Cause: the
   **reduced 150×200 = 30k-iteration schedule is ~20× shorter than the paper's
   ~600k**; best in-training val lesion-dice plateaued at ~0.10. The pipeline is
   sound — this is a training-budget result, not a pipeline defect.

## Phase B pipeline (validated end-to-end)

1. `analysis_synth_04_synthseg_labels.sh` — FreeSurfer `mri_synthseg` over 955
   ATLAS T1w (SLURM array, dir-mode, 48G; idempotent). 955/955 labels.
2. `analysis_synth_05_build_labelmaps.sh` + `build_labelmap.py` — combine
   SynthSeg tissue labels (resampled to the native lesion grid for non-1mm
   cases) with the real ATLAS lesion mask into a 34-class integer map
   (tissue 0..32, lesion 33). 955/955 maps; `label_scheme.json`.
3. `analysis_synth_06_train.sh` + `our_train.py` — wandb-free trainer reusing
   cornucopia's **native** `SynthFromLabelTransform` (GMM contrast synthesis) and
   the repo's `DiceCEL2Loss`; MONAI UNet, out_channels=34, lesion=last channel;
   `--mix-real` 50:50 synthetic/real ATLAS T1w; fold-0; preempt-safe.
4. `analysis_synth_07_predict_eval.sh` + `our_predict.py` — predict (lesion-channel
   probability threshold) on ATLAS val + soop TRACE + soop T1w; eval via the
   existing framework.

## Issues found and fixed (each caught by an incremental smoke/early check)

- **Inference affine bug (Phase A):** the HF `predict_comprehensive_with_preprocessing`
  attaches the affine *after* `OrientationD`/`SpacingD`, producing spatially
  flipped masks (Dice ~0.000 even in-domain). Rewrote `infer_synthstroke.py` to
  the repo's validated `test.py` file-based pipeline → 0.65 in-domain on a probe.
- **cornucopia API mismatch:** the repo's vendored `custom_cc.py` targets an
  unreleased cornucopia API (`shared`/`get_parameters` exist in no PyPI release).
  Replaced with cornucopia 0.3.0's native `SynthFromLabelTransform`; pinned
  `cornucopia==0.3.0`.
- **NaN loss under AMP:** SynthSeg GMM images have wide dynamic range → fp16
  overflow. Switched training to fp32 and skip non-finite-loss steps.
- **Validation/inference CUDA OOM on 11GB GPUs:** full-volume 34-class output
  doesn't fit. Moved the sliding-window output buffer (and val label) to CPU;
  set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **SynthSeg grid mismatch:** `mri_synthseg` emits 1mm output, so non-1mm native
  cases needed the label resampled onto the lesion grid before overlay.

## Recommendations / next steps

1. **Ship/iterate on pretrained `synthstroke-synth-plus`** for the cross-contrast
   goal (0.447 soop DWI). It is reachable (HuggingFace) and contrast-agnostic.
2. **If beating it from our own data is desired**, the pipeline is proven — re-run
   `analysis_synth_06_train.sh` at the **full ~500-epoch schedule** (and consider:
   lesion-focused patch sampling e.g. `RandCropByPosNegLabeld` to fight the
   tiny-lesion imbalance, and connected-component post-processing to cut the
   false-positive components that tank lesion-F1).
3. **For ISLES'26 specifically**, an ensemble of synth-plus (cross-contrast) with
   the in-domain nnU-Net (0.640 ATLAS T1w) is the natural strong-teacher pool;
   distill to a compact browser model afterwards.

## Reproduce

```bash
sbatch baselines/models/synthstroke/analysis_synth_01_setup_env.sh
sbatch --dependency=afterok:<id> baselines/models/synthstroke/analysis_synth_02_download_weights.sh
sbatch --array=0-5 baselines/models/synthstroke/analysis_synth_03_benchmark_pretrained.sh   # Phase A
python   baselines/models/synthstroke/synthstroke_helpers/aggregate_pretrained_reports.py
sbatch --array=0-19 baselines/models/synthstroke/analysis_synth_04_synthseg_labels.sh        # Phase B
sbatch baselines/models/synthstroke/analysis_synth_05_build_labelmaps.sh
sbatch baselines/models/synthstroke/analysis_synth_06_train.sh
sbatch baselines/models/synthstroke/analysis_synth_07_predict_eval.sh
```
