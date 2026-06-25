#!/bin/bash
#SBATCH --job-name=nnunet_eval_dbl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Evaluate DBL/MSL+DBL fold-0 validation predictions after merging all
# foreground classes to binary, directly against the binary fold-0 baseline.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

DBL_SCHEME="${DBL_SCHEME:-dbl}"
case "$DBL_SCHEME" in
  dbl)
    DATASET_NAME=Dataset505_ATLASR30_DBL
    DEFAULT_REPORT=baselines/reports/dbl_fold0
    MODEL_LABEL="DBL"
    ;;
  msl_dbl)
    DATASET_NAME=Dataset506_ATLASR30_MSLDBL
    DEFAULT_REPORT=baselines/reports/msl_dbl_fold0
    MODEL_LABEL="MSL+DBL"
    ;;
  *)
    echo "[error] DBL_SCHEME must be 'dbl' or 'msl_dbl', got: $DBL_SCHEME" >&2
    exit 2
    ;;
esac
REPORT_DIR="${DBL_REPORT_DIR:-$DEFAULT_REPORT}"
mkdir -p "$REPO/$REPORT_DIR"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
source "$REPO/.venv-eval/bin/activate"

PYTHONPATH="$REPO" DATASET_NAME="$DATASET_NAME" REPORT_DIR="$REPORT_DIR" MODEL_LABEL="$MODEL_LABEL" python - <<'PY'
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from eval.metrics import overlap_metrics

REPO = Path("/scratch/users/sciget/isles26challenge")
RES = REPO / "work/nnunet/nnUNet_results"
GT_DIR = REPO / "work/nnunet/nnUNet_raw/Dataset502_ATLASR30/labelsTr"
SPLITS = json.loads((REPO / "work/nnunet/nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json").read_text())
VAL = SPLITS[0]["val"]
DATASET_NAME = os.environ["DATASET_NAME"]
MODEL_LABEL = os.environ["MODEL_LABEL"]
TRAINER = os.environ.get("DBL_TRAINER", "nnUNetTrainer_250epochs")
MC_VAL = RES / f"{DATASET_NAME}/{TRAINER}__nnUNetPlans__3d_fullres/fold_0/validation"
BIN_VAL = RES / "Dataset502_ATLASR30/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
C26 = np.ones((3, 3, 3), dtype=np.int64)
BINS = [("tiny_<10", 0, 10), ("small_10-100", 10, 100), ("med_100-1k", 100, 1000),
        ("large_1k-10k", 1000, 10000), ("huge_>=10k", 10000, np.inf)]


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def size_binned_recall(pred_bin, gt_bin):
    lab, n = ndi.label(gt_bin, structure=C26)
    det = {b[0]: 0 for b in BINS}
    tot = {b[0]: 0 for b in BINS}
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for cid, sz in enumerate(sizes, start=1):
            name = next(nm for nm, lo, hi in BINS if lo <= sz < hi)
            tot[name] += 1
            if np.logical_and(lab == cid, pred_bin).any():
                det[name] += 1
    return det, tot


def evaluate(val_dir, merge_multiclass):
    dices, det_all, tot_all = [], {b[0]: 0 for b in BINS}, {b[0]: 0 for b in BINS}
    used = 0
    for cid in VAL:
        pp = val_dir / f"{cid}.nii.gz"
        gp = GT_DIR / f"{cid}.nii.gz"
        if not (pp.is_file() and gp.is_file()):
            continue
        pred = load(pp)
        pred_bin = (pred > 0) if merge_multiclass else (pred > 0.5)
        gt_bin = load(gp) > 0.5
        dices.append(overlap_metrics(pred_bin, gt_bin).dice)
        det, tot = size_binned_recall(pred_bin, gt_bin)
        for b in BINS:
            det_all[b[0]] += det[b[0]]
            tot_all[b[0]] += tot[b[0]]
        used += 1
    recall = {k: (det_all[k] / tot_all[k] if tot_all[k] else None) for k in tot_all}
    return {
        "n": used,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "median_dice": float(np.median(dices)) if dices else None,
        "size_recall": recall,
        "size_detected": det_all,
        "size_total": tot_all,
    }


out = {}
if MC_VAL.is_dir() and any(MC_VAL.glob("*.nii.gz")):
    out["multiclass"] = evaluate(MC_VAL, merge_multiclass=True)
else:
    out["multiclass"] = {"error": f"no validation predictions at {MC_VAL}"}
out["binary_baseline"] = evaluate(BIN_VAL, merge_multiclass=False)

OUT = REPO / os.environ["REPORT_DIR"]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(out, indent=2))

lines = [f"# {MODEL_LABEL} vs binary baseline (fold-0 cross-validation, merged to binary)", ""]
m, b = out.get("multiclass", {}), out["binary_baseline"]
lines += [
    f"Cases: binary n={b['n']}" + (f", {MODEL_LABEL} n={m.get('n')}" if "n" in m else f" ({MODEL_LABEL}: {m.get('error')})"),
    "",
    "| model | mean Dice | median Dice |",
    "|---|---:|---:|",
    f"| binary baseline | {b['mean_dice']:.4f} | {b['median_dice']:.4f} |",
]
if "mean_dice" in m and m["mean_dice"] is not None:
    lines.append(f"| {MODEL_LABEL} (merged) | {m['mean_dice']:.4f} | {m['median_dice']:.4f} |")
lines += ["", "## Size-binned lesion recall (pooled, voxels)", "",
          f"| bin | binary recall | {MODEL_LABEL} recall |", "|---|---:|---:|"]
for nm, _, _ in BINS:
    br = b["size_recall"][nm]
    mr = m["size_recall"][nm] if "size_recall" in m else None
    bs = f"{br:.3f}" if br is not None else "-"
    ms = f"{mr:.3f}" if mr is not None else "-"
    lines.append(f"| {nm} | {bs} | {ms} |")
(OUT / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
