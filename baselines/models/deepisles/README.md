# DeepISLES ensemble (SEALS + NVAUTO + SWAN)

Pretrained ISLES'22 winning ensemble, run as an Apptainer container. Reference
point for the **acute DWI/ADC/FLAIR** task — it consumes the modalities the
soop_bench GT is defined on, so it is the strong baseline on soop_bench.

| | |
|---|---|
| **Family** | ensemble (nnU-Net + MONAI Auto3DSeg + Factorizer) |
| **Input** | DWI (TRACE) + ADC + FLAIR |
| **Trained on** | ISLES'22 (acute multimodal) — pretrained, not retrained here |
| **Image** | `docker://isleschallenge/deepisles` → apptainer sandbox under `work/containers/` (gitignored, ~44 GB) |
| **Report** | `baselines/reports/deepisles/` (soop_bench) |

## Pipeline (run in order)

| Script | Stage | Resources |
|---|---|---|
| `analysis_01_pull_deepisles.sh` | build apptainer sandbox from pinned digest → `work/containers/deepisles_sandbox` | CPU, ~tens of min, 80 GB tmp |
| `analysis_02_run_deepisles.sh` | per-subject: stage TRACE/ADC/FLAIR (squeeze 4-D, resample ADC→TRACE), run ensemble → `work/predictions_deepisles/` | GPU array (CC ≤ 8.6; container's torch 1.11) |
| `analysis_03_evaluate.sh` | `eval.run_eval` → `baselines/reports/deepisles/` | CPU |

## Results (soop_bench, 12 subjects)

Dice 0.539 / 0.704 (mean/median), lesion-F1 0.582, surface-Dice 0.763, HD95 18.9 mm.
Best-in-class on soop_bench among real models because it segments on the
DWI/ADC/FLAIR the GT is drawn from. See `baselines/reports/deepisles/` for the
full report and `adversarial_review.md`.

## Notes

- GPU constraint excludes Hopper/Ada — the container's PyTorch 1.11 only supports
  up to sm_86.
- soop_bench TRACE/ADC are 4-D singleton and on different recon grids; the run
  script squeezes to 3-D and resamples ADC onto the TRACE grid before inference.
- This model is **not** retrained here; it is a fixed external reference.
