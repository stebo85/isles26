#!/bin/bash
#SBATCH --job-name=synthplus_atlasval_prob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Phase 2 (ensemble) — generate native-space lesion PROBABILITY maps for the
# pretrained SynthStroke synth-plus variant on the ATLAS fold-0 val set (the same
# 194 cases our from-scratch synth models already have cached), so synth-plus can
# be combined with them via ensemble_complementarity.py.
#
# Output: work/synthstroke/pred/atlas_val_prob/synthplus/<case>.nii.gz
# (float32 lesion-channel softmax prob, native geometry == labelsTr/<case>.nii.gz).
#
# Submit (weights already downloaded by analysis_synth_02):
#   sbatch baselines/models/synthstroke/analysis_synth_09_synthplus_atlasval_prob.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1

VENV="$REPO/.venv-synthstroke"
INFER="$REPO/baselines/models/synthstroke/synthstroke_helpers/infer_synthstroke.py"
WEIGHTS="$REPO/work/synthstroke/weights/synthstroke-synth-plus"
IMAGES="$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr"
VAL_LIST="$REPO/work/predictions_atlas_val/val_cases.txt"
PROB_DIR="$REPO/work/synthstroke/pred/atlas_val_prob/synthplus"
MASK_DIR="$REPO/work/synthstroke/pred/atlas_val_mask/synthplus"   # required --out (not used downstream)

for p in "$VENV/bin/python" "$INFER" "$VAL_LIST"; do
  [[ -e "$p" ]] || { echo "[error] missing: $p" >&2; exit 2; }
done
[[ -d "$WEIGHTS" ]] || { echo "[error] missing weights: $WEIGHTS (run analysis_synth_02)" >&2; exit 2; }
mkdir -p "$PROB_DIR" "$MASK_DIR"

n=0; ok=0; skip=0
while IFS= read -r CASE; do
  CASE="${CASE//[$'\r\n\t ']/}"
  [[ -z "$CASE" ]] && continue
  n=$((n+1))
  IMG="$IMAGES/${CASE}_0000.nii.gz"
  PROB="$PROB_DIR/${CASE}.nii.gz"
  MASK="$MASK_DIR/${CASE}.nii.gz"
  if [[ ! -f "$IMG" ]]; then
    echo "[warn] missing image, skip: $IMG" >&2; skip=$((skip+1)); continue
  fi
  if [[ -f "$PROB" ]]; then
    ok=$((ok+1)); continue   # idempotent resume
  fi
  echo "[info] ($n) $CASE"
  "$VENV/bin/python" "$INFER" \
    --weights "$WEIGHTS" \
    --image "$IMG" \
    --out "$MASK" \
    --save-prob "$PROB" \
    --device auto
  ok=$((ok+1))
done < "$VAL_LIST"

echo "[done] synth-plus prob maps: total=$n saved/cached=$ok skipped=$skip -> $PROB_DIR"
