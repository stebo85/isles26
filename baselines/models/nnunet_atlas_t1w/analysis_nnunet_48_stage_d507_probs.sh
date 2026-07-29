#!/bin/bash
#SBATCH --job-name=stage_d507_probs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-4
#SBATCH --requeue
#
# Convert the Dataset507 (center-grouped) per-fold validation softmax NPZs into
# native-space float32 probability NIfTIs, so the official-metric sweep can score
# the HONEST evaluation surface.
#
# Why this surface matters more than the in-distribution one: Dataset502's folds
# are stratified-random, and 52 of 55 centers appear in both the train and the
# validation side of every fold. Dataset507's folds hold out whole CENTERS
# (verified invariants in baselines/reports/nnunet_center_grouped_splits/
# splits_summary.json), which is the situation we actually face -- the hidden
# ISLES'26 test set spans 60+ centers. Measured gap: Dice 0.6558 -> 0.6347 and,
# far worse, absolute volume difference 4.94 -> 6.28 mL (+27%). Since absolute
# volume difference is one of the five ranked metrics, tuning the operating point
# on the leaky surface would tune it against the wrong distribution.
#
# One array task per fold. Reuses npz_lesion_prob_to_nii.py, which already
# carries the hard-fail axis-permutation guards this project learned the hard
# way (MAX_MISMATCH_FRACTION and an explicit tie raise).

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
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] missing .venv-eval" >&2
  exit 2
fi

DATASET_NAME=Dataset507_ATLASR30_CORRECTED
TRAINER=nnUNetTrainerDiceTopK10Loss
RESULT_DIR="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__3d_fullres"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
VAL_DIR="$RESULT_DIR/fold_${FOLD}/validation"
OUT_DIR="$REPO/work/official_eval/_probs_d507/fold_${FOLD}"

if [[ ! -d "$VAL_DIR" ]]; then
  echo "[error] missing $VAL_DIR" >&2
  exit 2
fi

N_NPZ=$(find "$VAL_DIR" -maxdepth 1 -name '*.npz' | wc -l)
if [[ "$N_NPZ" -eq 0 ]]; then
  echo "[error] no validation NPZ files in $VAL_DIR" >&2
  exit 2
fi
echo "[info] fold $FOLD: $N_NPZ npz -> $OUT_DIR"

mkdir -p "$OUT_DIR"

# --ref-dir supplies the native-space affine/header for each case. The fold's own
# validation segmentations are the correct reference: they were written by
# nnU-Net on exactly the grid the probabilities belong to.
"$PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/npz_lesion_prob_to_nii.py" \
  --npz-dir "$VAL_DIR" \
  --seg-dir "$VAL_DIR" \
  --ref-dir "$VAL_DIR" \
  --out-dir "$OUT_DIR" \
  --lesion-channel 1

N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "$N_OUT" -ne "$N_NPZ" ]]; then
  echo "[error] converted $N_OUT of $N_NPZ cases for fold $FOLD" >&2
  exit 3
fi

echo "[done] fold $FOLD: $N_OUT probability volumes"
