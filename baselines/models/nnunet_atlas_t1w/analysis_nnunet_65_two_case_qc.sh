#!/bin/bash
#SBATCH --job-name=two_case_qc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p normal
#
# Visual QC of the SHIPPED model on two named training subjects:
#   sub-r001s001 -> ATLAS_0001  (site R001, GT 89.06 mL, vlarge)
#   sub-soop0468 -> ATLAS_1316  (site SOOP, GT 23.50 mL, large)
#
# Both predictions are OUT OF FOLD: Dataset507 uses center-grouped folds and
# fold 4 held out both R001 and SOOP, so neither subject was seen in training
# by the weights that produced these probabilities. The probability maps are
# the ones staged by analysis_nnunet_48_stage_d507_probs.sh from that fold's
# validation softmax, i.e. the same nnUNetTrainerDiceTopK10Loss / 1000-epoch
# checkpoints that are baked into the submission container.
#
# Operating point applied here is the SHIPPED one: p >= 0.35, then drop
# connected components smaller than 20 voxels at 26-connectivity. A second mask
# is emitted at thr 0.30 / minCC 50 -- the point the no-TTA ablation
# (baselines/reports/tta_ablation/) picked as best-by-mean-rank -- so the two
# candidate operating points can be eyeballed on the same case.
#
# Caveat carried from the container README: these probabilities were computed
# WITH mirroring TTA (nnU-Net's validation default) while the container ships
# with TTA off for the CPU budget. The masks here are therefore slightly
# cleaner than what the container will emit.
#
# Outputs (work/qc_two_cases/): native-space NIfTIs that open directly in
# fsleyes, plus one PNG panel per case.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] missing .venv-eval" >&2
  exit 2
fi

RAW="$REPO/work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED"
PROB_DIR="$REPO/work/official_eval/_probs_d507/fold_4"
OUT="$REPO/work/qc_two_cases"
mkdir -p "$OUT"

for f in "$PROB_DIR/ATLAS_0001.nii.gz" "$PROB_DIR/ATLAS_1316.nii.gz" \
         "$RAW/imagesTr/ATLAS_0001_0000.nii.gz" "$RAW/labelsTr/ATLAS_0001.nii.gz" \
         "$RAW/imagesTr/ATLAS_1316_0000.nii.gz" "$RAW/labelsTr/ATLAS_1316.nii.gz"; do
  [[ -f "$f" ]] || { echo "[error] missing $f" >&2; exit 2; }
done

RAW="$RAW" PROB_DIR="$PROB_DIR" OUT="$OUT" "$PY" - <<'PYEOF'
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

RAW = os.environ["RAW"]
PROB_DIR = os.environ["PROB_DIR"]
OUT = os.environ["OUT"]

CASES = [
    ("ATLAS_0001", "sub-r001s001", "R001"),
    ("ATLAS_1316", "sub-soop0468", "SOOP"),
]

# Shipped operating point, and the no-TTA-optimal alternative.
POINTS = {"shipped_thr035_k20": (0.35, 20), "alt_thr030_k50": (0.30, 50)}

# 26-connectivity, matching the container's min_size_filter.
STRUCT = np.ones((3, 3, 3), dtype=bool)


def min_size_filter(mask, min_voxels):
    if min_voxels <= 0:
        return mask
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(sizes.shape, dtype=bool)
    keep[1:] = sizes[1:] >= min_voxels
    return keep[lab].astype(np.uint8)


def dice(a, b):
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * np.logical_and(a, b).sum() / d


def outline(mask2d):
    """Boundary voxels of a 2-D mask, for contour-style overlay."""
    if not mask2d.any():
        return mask2d
    eroded = ndimage.binary_erosion(mask2d, structure=np.ones((3, 3), bool))
    return mask2d & ~eroded


summary = {}

for case, subject, site in CASES:
    t1_img = nib.load(f"{RAW}/imagesTr/{case}_0000.nii.gz")
    gt_img = nib.load(f"{RAW}/labelsTr/{case}.nii.gz")
    pr_img = nib.load(f"{PROB_DIR}/{case}.nii.gz")

    t1 = np.asanyarray(t1_img.dataobj, dtype=np.float32)
    gt = (np.asanyarray(gt_img.dataobj) > 0.5).astype(np.uint8)
    prob = np.asanyarray(pr_img.dataobj, dtype=np.float32)

    if not (t1.shape == gt.shape == prob.shape):
        raise SystemExit(f"[error] {case} shape mismatch "
                         f"{t1.shape} / {gt.shape} / {prob.shape}")

    vox_ml = float(np.prod(nib.affines.voxel_sizes(t1_img.affine))) / 1000.0

    # Emit the viewable NIfTIs alongside each other, named by subject.
    shutil.copyfile(f"{RAW}/imagesTr/{case}_0000.nii.gz", f"{OUT}/{subject}_t1w.nii.gz")
    shutil.copyfile(f"{RAW}/labelsTr/{case}.nii.gz", f"{OUT}/{subject}_gt.nii.gz")
    shutil.copyfile(f"{PROB_DIR}/{case}.nii.gz", f"{OUT}/{subject}_pred_prob.nii.gz")

    masks = {}
    rows = {}
    for name, (thr, k) in POINTS.items():
        m = min_size_filter((prob >= thr).astype(np.uint8), k)
        masks[name] = m
        nib.save(nib.Nifti1Image(m, pr_img.affine, pr_img.header),
                 f"{OUT}/{subject}_pred_{name}.nii.gz")
        n_pred = int(ndimage.label(m, structure=STRUCT)[1])
        # int() first: summing uint8 arrays yields an unsigned type, and the
        # difference below underflows to ~1.8e19 instead of going negative.
        n_vox_pred, n_vox_gt = int(m.sum()), int(gt.sum())
        rows[name] = {
            "threshold": thr,
            "min_cc_voxels": k,
            "dice": round(float(dice(gt, m)), 4),
            "pred_ml": round(n_vox_pred * vox_ml, 3),
            "abs_volume_diff_ml": round(abs(n_vox_pred - n_vox_gt) * vox_ml, 3),
            "n_pred_components": n_pred,
        }

    n_gt = int(ndimage.label(gt, structure=STRUCT)[1])
    summary[subject] = {
        "case": case,
        "site": site,
        "shape": list(t1.shape),
        "voxel_ml": round(vox_ml, 6),
        "gt_ml": round(int(gt.sum()) * vox_ml, 3),
        "n_gt_components": n_gt,
        "out_of_fold": True,
        "fold_that_held_it_out": 4,
        "operating_points": rows,
    }

    # ---- panel -------------------------------------------------------------
    # Three axial slices through the lesion: the GT centroid slice and the two
    # slices carrying the most GT voxels, so the panel is driven by the label,
    # not by where the prediction happens to be.
    per_slice = gt.sum(axis=(0, 1))
    order = np.argsort(per_slice)[::-1]
    picks = sorted({int(order[0]), int(order[1]),
                    int(round(ndimage.center_of_mass(gt)[2]))})
    while len(picks) < 3:
        picks.append(min(picks[-1] + 8, t1.shape[2] - 1))
    picks = picks[:3]

    shipped = masks["shipped_thr035_k20"]
    lo, hi = np.percentile(t1[t1 > 0], [1, 99]) if (t1 > 0).any() else (0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6))
    for col, z in enumerate(picks):
        base = np.rot90(t1[:, :, z])
        g = np.rot90(gt[:, :, z]).astype(bool)
        p = np.rot90(shipped[:, :, z]).astype(bool)

        ax = axes[0, col]
        ax.imshow(base, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ov = np.zeros(base.shape + (4,), dtype=np.float32)
        go, po = outline(g), outline(p)
        ov[go] = (0.0, 1.0, 0.2, 1.0)   # GT   -> green
        ov[po] = (1.0, 0.2, 0.1, 1.0)   # pred -> red
        ov[go & po] = (1.0, 1.0, 0.0, 1.0)  # agreement -> yellow
        ax.imshow(ov, interpolation="nearest")
        ax.set_title(f"z={z}  GT(green) / pred(red)", fontsize=10)
        ax.axis("off")

        ax = axes[1, col]
        ax.imshow(base, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        pz = np.rot90(prob[:, :, z])
        hot = plt.get_cmap("inferno")(np.clip(pz, 0, 1))
        hot[..., 3] = np.clip(pz, 0, 1) * 0.85
        ax.imshow(hot, interpolation="nearest")
        cs = ax.contour(np.rot90(gt[:, :, z]).astype(float), levels=[0.5],
                        colors="#00ff33", linewidths=0.9)
        ax.set_title(f"z={z}  probability (GT outline)", fontsize=10)
        ax.axis("off")

    r = rows["shipped_thr035_k20"]
    fig.suptitle(
        f"{subject}  ({case}, site {site})  -- shipped model, OUT OF FOLD\n"
        f"GT {summary[subject]['gt_ml']:.2f} mL / {n_gt} comp   "
        f"pred {r['pred_ml']:.2f} mL / {r['n_pred_components']} comp   "
        f"Dice {r['dice']:.4f}   |dV| {r['abs_volume_diff_ml']:.2f} mL   "
        f"(p>=0.35, minCC 20)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/{subject}_qc.png", dpi=120)
    plt.close(fig)
    print(f"[ok] {subject}: {json.dumps(rows)}", flush=True)

with open(f"{OUT}/summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

echo "[done] outputs in $OUT"
ls -la "$OUT"
