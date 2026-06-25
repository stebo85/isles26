# Evaluation framework (eval/)

Reproducible, sensitive evaluation of stroke lesion segmentation on
`data/soop_bench/`. Built from the lessons in
[problems_and_lessons.md](problems_and_lessons.md): Dice-only is misleading,
small lesions dominate detection failures, and per-center / per-chronicity
performance matters.

## Why these metrics

| Family | Metric | What it catches |
|---|---|---|
| Voxel overlap | Dice, IoU, sensitivity, specificity, precision, NPV | classic; bias-toward-large-lesions documented |
| Volume | absolute / signed / relative volume difference (mL) | systematic over- vs under-segmentation |
| Boundary | HD95, ASSD, MASD, surface Dice @ 3 mm | shape and contour quality |
| Lesion-wise (detection) | TP / FP / FN by 26-conn components, precision, recall, **F1**, absolute & signed count diff | direct clinical-detection proxy ATLAS uses |
| Small-lesion sensitivity | recall pooled in voxel-bins (0–10, 10–100, 100–1k, 1k–10k, ≥10k voxels) and mL-bins (<0.1, 0.1–1, 1–10, 10–50, ≥50 mL) | the documented ATLAS / ISLES'22 failure mode |

Each per-case result also reports per-component overlaps so we can audit "which
lesions did the model miss".

## Empty-mask conventions

- Dice / IoU / lesion-F1 on `(empty, empty)` → 1.0 (perfect background match).
- Dice / IoU / lesion-F1 on `(empty, non-empty)` or vice versa → 0.0.
- HD95 / ASSD on any empty side → NaN. Aggregation uses `nanmean` / `nanmedian`
  and reports the number of valid cases alongside each statistic.
- Voxel-wise sensitivity is NaN when GT is empty (denominator zero).
- Voxel-wise precision is NaN when prediction is empty.

These were chosen to avoid Dice gaming on empty predictions and to surface
empty-mask cases honestly in the report.

## Aggregation

- For overall metrics: per-case `nanmean`, `nanmedian`, `std`, `min`, `max`,
  and the count of valid cases.
- For size-binned recall: **pooled** counts (`Σ n_detected / Σ n_GT`). This
  avoids the "average of ratios" trap where a case with 1/1 small lesions
  detected has the same weight as a case with 50/100 detected.
- For per-group breakdowns (chronicity, center): per-case mean of each metric
  inside the group.

## Stratification

- **Chronicity** (`acute` / `chronic` / `mixed` / `unknown`): derived from
  whether the subject has a `desc-lesionAcute_mask`, `desc-lesionChronic_mask`,
  or both. soop_bench has no per-subject phase JSON, so this is a coarse
  approximation.
- **Center**: derived from `Manufacturer | ManufacturersModelName` in the JSON
  sidecar of the T1w scan (falling back to TRACE / FLAIR / ADC). For
  soop_bench all subjects are Philips, but on four different scanner models —
  small but real intra-vendor variation.
- **Lesion-size bin**: voxel-count and mL-volume bins, computed from the GT
  connected components.

## Layout

```
eval/
  metrics.py     # all metric implementations (pure numpy + scipy.ndimage)
  loader.py     # soop_bench discovery + nearest-neighbour resampling
  report.py     # aggregation + JSON + Markdown writers
  qc_viz.py     # per-subject PNG overlays
  run_eval.py    # CLI entry-point
  tests/test_metrics.py  # 24 synthetic-mask unit tests
analysis_01_pull_deepisles.sh  # apptainer pull DeepISLES SIF
analysis_02_run_deepisles.sh  # SLURM array, one subject per task
analysis_03_evaluate.sh       # run eval framework on predictions
```

## Tests

24 synthetic-mask tests cover:

- voxel-overlap perfect / disjoint / partial / all empty-mask combinations
- volume metrics with non-isotropic voxels
- boundary metrics with a one-voxel shift (HD95 ≤ 1.5 mm, surface Dice @ 3 mm
  still 1.0)
- lesion-wise: perfect detection, missed-small-lesion case, multi-FP case,
  empty-mask combinations
- size-binned recall: tiny lesion missed, big lesion detected
- end-to-end smoke test of `evaluate_case`

Run with:

```bash
module load devel python/3.12.1
export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}
source .venv-eval/bin/activate
PYTHONPATH=. python eval/tests/test_metrics.py
```

Output: `24/24 passed`.

## Reference model: DeepISLES

For the soop_bench benchmark we use **DeepISLES** (de la Rosa et al. 2025,
*Nat. Comm.*), the post-ISLES'22 ensemble of SEALS (nnU-Net), NVAUTO (MONAI
Auto3DSeg), and SWAN (Factorizer). Rationale:

- soop_bench ground truth is in DWI/TRACE space, so a DWI/ADC/FLAIR model is
  the direct match — registering a T1w model into TRACE space would introduce
  a confound on top of the segmentation quality.
- DeepISLES is the published state-of-the-art for DWI/ADC/FLAIR ischemic
  stroke segmentation and is the closest direct DWI counterpart of MAPPING in
  the literature review.
- It is released as a Docker image (`isleschallenge/deepisles`) with weights
  bundled, so the comparison is reproducible.

Inference is run via Apptainer on one GPU per subject (`gres=gpu:1`) in the
SLURM `owners` partition. Each prediction is resampled (nearest neighbour) onto
the GT TRACE grid before metrics are computed; in practice the DeepISLES
output is already in the input DWI grid so this is a no-op.
