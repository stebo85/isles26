#!/bin/bash
#SBATCH --job-name=ood_ens_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# OOD-targeted ensemble, step B: blend the TRACE-space nn/sp probability maps
# from step 16, sweep probability threshold and blend weight, and report mean
# Dice on soop_bench using the eval framework's exact Dice + GT loader (so the
# numbers are directly comparable to the 0.255 nnU-Net soop baseline).
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_17_ood_ensemble_eval.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/baselines/reports/ood_ensemble"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
source "$REPO/.venv-eval/bin/activate"

PYTHONPATH="$REPO" python - <<'PY'
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from eval.loader import discover_soop_bench
from eval.metrics import overlap_metrics

import os
REPO = Path("/scratch/users/sciget/isles26challenge")
PROB_ROOT = REPO / os.environ.get("OOD_PROB_DIR", "work/predictions_ood_ens")
NN_DIR = PROB_ROOT / "nn_prob_trace"
SP_DIR = PROB_ROOT / "sp_prob_trace"
OUT = REPO / os.environ.get("OOD_REPORT_DIR", "baselines/reports/ood_ensemble")
OUT.mkdir(parents=True, exist_ok=True)

subjects = discover_soop_bench(REPO / "data/soop_bench")


def load(p, canonical=False):
    img = nib.load(str(p))
    if canonical:
        # The warped prob maps live on TRACE_REF3D = as_closest_canonical(TRACE)
        # (RAS), while the GT masks are stored in original TRACE orientation
        # (e.g. LAS). Canonicalizing the GT the same way aligns them voxel-wise
        # (pure axis flips, no interpolation), matching what eval.run_eval does.
        img = nib.as_closest_canonical(img)
    d = np.asanyarray(img.dataobj).astype(np.float32)
    while d.ndim > 3 and d.shape[-1] == 1:
        d = d[..., 0]
    return d


cases = []
for s in subjects:
    sid = s.subject_id
    nn_p = NN_DIR / f"{sid}.nii.gz"
    sp_p = SP_DIR / f"{sid}.nii.gz"
    if not (nn_p.is_file() and sp_p.is_file()):
        print(f"[warn] skipping {sid}: missing prob map(s)")
        continue
    gt = load(s.lesion_mask, canonical=True) > 0.5
    nn = load(nn_p)
    sp = load(sp_p)
    if nn.shape != gt.shape or sp.shape != gt.shape:
        print(f"[warn] {sid}: shape mismatch nn{nn.shape} sp{sp.shape} gt{gt.shape}; skipping")
        continue
    cases.append((sid, nn, sp, gt))

print(f"[info] evaluating {len(cases)} soop_bench cases\n")

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
# blend weight w applies to nnU-Net; (1-w) to synth-plus. w=1 -> nn only, w=0 -> sp only.
WEIGHTS = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]


def mean_dice(prob_w, thr):
    ds = []
    for sid, nn, sp, gt in cases:
        blended = prob_w * nn + (1.0 - prob_w) * sp
        ds.append(overlap_metrics(blended >= thr, gt).dice)
    return float(np.mean(ds)), float(np.median(ds))


grid = {}
best = None
for w in WEIGHTS:
    for thr in THRESHOLDS:
        m, md = mean_dice(w, thr)
        grid[f"w{w:.1f}_t{thr:.2f}"] = {"weight_nn": w, "threshold": thr, "mean_dice": m, "median_dice": md}
        tag = "nn-only" if w == 1.0 else ("sp-only" if w == 0.0 else f"blend w_nn={w:.1f}")
        if best is None or m > best["mean_dice"]:
            best = {"label": tag, "weight_nn": w, "threshold": thr, "mean_dice": m, "median_dice": md}

# Best per family
def best_of(pred):
    rows = [v for v in grid.values() if pred(v)]
    return max(rows, key=lambda r: r["mean_dice"]) if rows else None

best_nn = best_of(lambda r: r["weight_nn"] == 1.0)
best_sp = best_of(lambda r: r["weight_nn"] == 0.0)
best_blend = best_of(lambda r: 0.0 < r["weight_nn"] < 1.0)

summary = {
    "n_cases": len(cases),
    "baseline_nnunet_soop_meanDice_0p5_5fold_tta": 0.2546,
    "best_overall": best,
    "best_nn_only": best_nn,
    "best_sp_only": best_sp,
    "best_blend": best_blend,
    "grid": grid,
}
(OUT / "ood_ensemble_summary.json").write_text(json.dumps(summary, indent=2))

lines = [
    "# OOD ensemble: nnU-Net(T1w) + synth-plus(T1w) on soop_bench",
    "",
    f"Cases: {len(cases)}. Both models run on the skull-stripped T1w input, "
    "lesion probabilities warped to TRACE, blended, thresholded; Dice via the eval framework.",
    "",
    "| model | weight_nn | best threshold | mean Dice | median Dice |",
    "|---|---:|---:|---:|---:|",
]
for label, r in [("nnU-Net only", best_nn), ("synth-plus only", best_sp), ("blend", best_blend)]:
    if r:
        lines.append(f"| {label} | {r['weight_nn']:.1f} | {r['threshold']:.2f} | {r['mean_dice']:.4f} | {r['median_dice']:.4f} |")
lines += [
    "",
    f"**Best overall:** {best['label']} (w_nn={best['weight_nn']:.1f}, thr={best['threshold']:.2f}) "
    f"-> mean Dice {best['mean_dice']:.4f} (median {best['median_dice']:.4f}).",
    "",
    "Reference: nnU-Net 5-fold+TTA soop baseline (mask@0.5) = 0.2546 mean; "
    "synth-plus on the DWI/TRACE image directly = 0.447 (needs DWI, not available at T1w test time).",
    "",
    "## Full grid (mean Dice)",
    "",
    "| weight_nn \\ thr | " + " | ".join(f"{t:.2f}" for t in THRESHOLDS) + " |",
    "|---|" + "|".join(["---:"] * len(THRESHOLDS)) + "|",
]
for w in WEIGHTS:
    row = [f"{grid[f'w{w:.1f}_t{t:.2f}']['mean_dice']:.3f}" for t in THRESHOLDS]
    lines.append(f"| {w:.1f} | " + " | ".join(row) + " |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\n[done] wrote {OUT/'report.md'} and ood_ensemble_summary.json")
PY
