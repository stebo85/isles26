# Adversarial Review — ATLAS R2.1 Training Data Characterization

Reviewer: Codex (gpt-5.x via `codex:rescue`).
Source artifacts reviewed (commit prior to fixes):
- [scripts/analysis/characterize_training_data.py](../../../scripts/analysis/characterize_training_data.py)
- [scripts/analysis/aggregate_training_data.py](../../../scripts/analysis/aggregate_training_data.py)
- [docs/data/atlas-r21-characterization.md](../../../docs/data/atlas-r21-characterization.md)
- This directory's `per_session.csv` and `summary.json`.

Priority order (highest to lowest severity):

1. **DEFECT 4** — DPS=0/-4 binned as acute in code but recommended as unknown in report (25 sessions, corrupts chronicity stratification).
2. **DEFECT 3** — 6-connectivity CC counting undocumented and non-standard (inflates fragmentation stats; eval/metrics.py uses 26-connectivity).
3. **DEFECT 1** — `np.percentile` instead of `np.nanpercentile` (silent NaN propagation).
4. **DEFECT 2** — `int()` truncation label collision for R039 (`lesion_unique_labels = "0,0"` for sessions with float-encoded `0.9999999776482582` masks).
5. **DEFECT 5** — `t1_p50 = 0` recorded but useless as a brain intensity metric (documentation/columnar gap).
6. **DEFECT 7** — Re-binarization missing from the preprocessing block in the doc (operational risk for nnU-Net `==1` helpers).
7. **DEFECT 6** — No post-resampling voxel-count floor for tiny lesions (1 R035 session at 0.06 mL could land near the resolution floor after 1 mm iso resampling).

Claims verification against per_session.csv / summary.json:

- "82% isotropic" — SUPPORTED (`isotropic_1mm_fraction = 0.8209`).
- "Median CC = 2" — SUPPORTED, but with caveat: 6-connectivity inflates counts (DEFECT 3).
- "126 unknown DPS" — SUPPORTED for missing-DPS-only; correct chronicity-unknown count is **151** (126 NaN + 24 DPS=0 + 1 DPS=-4) once DEFECT 4 is fixed.
- "T1w p99 range 26 → 2.5×10⁶" — SUPPORTED (exact match).

## Fixes applied in this commit

| Defect | Fix | File |
|---|---|---|
| 1 | `np.percentile → np.nanpercentile`, `t1f.mean()/.std() → np.nanmean/nanstd` | `scripts/analysis/characterize_training_data.py` |
| 2 | `int(u)` → `int(round(float(u)))` when stringifying unique labels | `scripts/analysis/characterize_training_data.py` |
| 3 | `ndi.label(lesion_bin, structure=_CC26)` with `_CC26 = ndi.generate_binary_structure(3, 3)` to match `eval/metrics.py:_CONNECTIVITY_26` | `scripts/analysis/characterize_training_data.py` |
| 4 | `chronicity = unknown if dps is NaN or dps <= 0 else (lt180d if dps < 180 else ge180d)` — applied in both the per-session script (`chronicity_lt180d` column) and the aggregator (`chronicity` derived column) | both analysis scripts |
| 5 | New CSV columns `t1_brain_p50`, `t1_brain_p99` computed over non-zero voxels; `t1_p50` retained for backwards comparison and documented as whole-volume | `scripts/analysis/characterize_training_data.py` |
| 6 | Preprocessing recommendation in the doc reordered so re-binarize (`> 0.5`) comes first; tiny-lesion post-resampling guard added (≥30 voxels) | `docs/data/atlas-r21-characterization.md` |
| 7 | Same fix as 6 (re-binarize first in the preprocessing block) | `docs/data/atlas-r21-characterization.md` |

After the re-run, `chronicity_counts` in `summary.json` is expected to shift from `{ge180d: 540, lt180d: 289, unknown: 126}` to `{ge180d: 540, lt180d: 264, unknown: 151}` (25 sessions reclassified from `lt180d → unknown`). CC distribution will tighten (26-conn merges previously-split edge-touching voxel groups).
