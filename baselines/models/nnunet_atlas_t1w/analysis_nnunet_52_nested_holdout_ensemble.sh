#!/bin/bash
#SBATCH --job-name=nested_holdout_ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --requeue
#
# The first honest measurement of the configuration we actually ship.
#
# The problem this closes: every number this project owns is a SINGLE-MODEL
# out-of-fold number -- nnU-Net predicts each case with the one fold that held it
# out. But the container ships a FIVE-FOLD ENSEMBLE, whose averaged softmax has a
# different probability distribution (averaging pulls mass off 0 and 1). So:
#   (a) the 5-fold test-time ensembling gain has never been measured at all, and
#       it is the entire reason for shipping five models instead of one;
#   (b) every operating point we own (threshold, min-component-size) was fitted
#       on single-model softmax and is not calibrated for the ensemble.
#
# Method: take Dataset507 fold 0's validation cases -- whose CENTERS are held out
# of folds... no. Fold 0's cases were held out of fold 0 only; folds 1-4 each
# trained on some of them. Dataset507's folds are center-grouped, so a case in
# fold 0's validation set belongs to a center that folds 1-4 DID train on.
# Therefore the clean holdout must come from Dataset502, whose folds partition
# cases (not centers): a Dataset502 fold-0 validation case is in the TRAINING set
# of folds 1-4... which is equally contaminated.
#
# There is no way to get an uncontaminated 5-fold ensemble estimate from a 5-fold
# split -- by construction every case trained 4 of the 5 members. What IS
# measurable, and what this job measures, is the LEAVE-ONE-MEMBER-OUT ensemble:
# predict each Dataset507 fold-k validation case with the four members that did
# NOT see it... which is impossible for the same reason.
#
# So the honest thing available: predict each fold-k validation case with member k
# ALONE (= the recorded out-of-fold baseline, re-derived through the deployment
# code path) and with ALL FIVE members, and report both. The five-member number is
# optimistically biased and is labelled as such; the value is the DIFFERENCE in
# calibration -- how far the ensemble's probability distribution moves the optimal
# threshold - which is a property of the averaging, not of the contamination.
#
# Scoring is by the OFFICIAL five metrics, on the deployment code path
# (submission/isles26_algorithm/inference.py), not by nnU-Net's internal
# validation -- because analysis_nnunet_51 measured a systematic -0.017 Dice gap
# between those two paths.

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
  module load py-pytorch/2.4.1_py312
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

PY="$REPO/.venv-nnunet/bin/python"
DATASET="${DATASET:-Dataset507_ATLASR30_CORRECTED}"
MODEL_DIR="${MODEL_DIR:-$REPO/work/submission_model/$DATASET}"
SPLITS="$nnUNet_preprocessed/$DATASET/splits_final.json"
RAW_DIR="$nnUNet_raw/$DATASET"
OUT_DIR="${OUT_DIR:-$REPO/work/nested_holdout}"
N_CASES="${N_CASES:-120}"      # per arm; 120 x 2 arms x ~30 s ~= 2 GPU-hours

for f in "$MODEL_DIR" "$SPLITS" "$RAW_DIR"; do
  [[ -e "$f" ]] || { echo "[error] missing $f" >&2; exit 2; }
done

mkdir -p "$OUT_DIR"

PYTHONPATH="$REPO/submission/isles26_algorithm:${PYTHONPATH:-}" "$PY" - \
  "$SPLITS" "$RAW_DIR" "$MODEL_DIR" "$OUT_DIR" "$N_CASES" <<'PYEOF'
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

splits_p, raw_dir, model_dir, out_dir, n_cases = sys.argv[1:6]
n_cases = int(n_cases)
REPO = Path("/scratch/users/sciget/isles26challenge")
raw_dir, out_dir = Path(raw_dir), Path(out_dir)

splits = json.loads(Path(splits_p).read_text())
# Balanced sample across folds so no single center dominates the estimate.
per_fold = max(1, n_cases // len(splits))
selected = [(k, cid) for k, s in enumerate(splits) for cid in sorted(s["val"])[:per_fold]]
print(f"[info] {len(selected)} cases, {per_fold} per fold", flush=True)

ARMS = {"single": None, "ensemble5": "0,1,2,3,4"}  # single -> the case's own held-out fold

def run(case_id, folds, tmp):
    in_dir = tmp / f"in_{folds}" / "images" / "t1w"
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz", in_dir / f"{case_id}.nii.gz")
    out_root = tmp / f"out_{folds}"
    env = dict(os.environ)
    env.update(
        ISLES26_INPUT_DIR=str(in_dir.parent.parent), ISLES26_OUTPUT_DIR=str(out_root),
        ISLES26_MODEL_DIR=str(model_dir), ISLES26_FOLDS=folds,
        ISLES26_THRESHOLD="0.5", ISLES26_MIN_CC_VOXELS="0", ISLES26_FLATTEN_EMPTY="0",
        PYTHONPATH=os.pathsep.join([str(REPO / "submission" / "isles26_algorithm"),
                                    os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    )
    p = subprocess.run([sys.executable, str(REPO / "submission" / "isles26_algorithm" / "inference.py")],
                       env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[warn] {case_id} folds={folds} failed\n{p.stderr[-1500:]}", flush=True)
        return None
    probs = sorted((out_root / "images").rglob("*.mha"))
    prob_p = next((m for m in probs if "probability" in m.parent.name), None)
    return prob_p

# Save the soft maps; the official-metric sweep is a separate, CPU-only step so
# the GPU is not held while thresholds are explored.
for arm in ARMS:
    (out_dir / arm).mkdir(parents=True, exist_ok=True)

import SimpleITK as sitk
done = 0
for fold, cid in selected:
    with tempfile.TemporaryDirectory(prefix=f"nh_{cid}_") as t:
        tmp = Path(t)
        for arm, folds in ARMS.items():
            target = out_dir / arm / f"{cid}.nii.gz"
            if target.exists():
                continue
            f = str(fold) if folds is None else folds
            prob_p = run(cid, f, tmp)
            if prob_p is None:
                continue
            img = sitk.ReadImage(str(prob_p))
            arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
            ref = nib.load(str(raw_dir / "imagesTr" / f"{cid}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header), str(target))
    done += 1
    if done % 10 == 0:
        print(f"[progress] {done}/{len(selected)}", flush=True)

manifest = {"n_selected": len(selected), "per_fold": per_fold,
            "cases": [{"fold": f, "case": c} for f, c in selected], "arms": list(ARMS)}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"[done] wrote soft maps for {len(ARMS)} arms under {out_dir}")
PYEOF

echo "[done] score both arms with:"
echo "  for ARM in single ensemble5; do"
echo "    .venv-official/bin/python eval/run_official_eval.py score \\"
echo "      --gt-dir $RAW_DIR/labelsTr --prob-dir $OUT_DIR/\$ARM \\"
echo "      --out-dir $OUT_DIR/scored_\$ARM --thresholds 0.20:0.70:0.05 --min-cc 0,5,10,20,35,50"
echo "  done"
