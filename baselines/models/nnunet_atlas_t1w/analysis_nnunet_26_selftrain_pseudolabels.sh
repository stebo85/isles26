#!/bin/bash
#SBATCH --job-name=nnunet_selftrain_pl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Self-training step A: run the current best TopK10 5-fold teacher on an
# explicitly staged unlabeled T1w folder in nnU-Net input format:
#
#   <SELFTRAIN_INPUT_DIR>/<case_id>_0000.nii.gz
#
# This script only generates pseudo-label candidates. Do not point it at hidden
# challenge test data or any evaluation set whose labels are used for reporting.
#
#   SELFTRAIN_INPUT_DIR=/path/to/unlabeled_nnunet_images \
#     sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_26_selftrain_pseudolabels.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if [[ -z "${SELFTRAIN_INPUT_DIR:-}" ]]; then
  echo "[error] set SELFTRAIN_INPUT_DIR to a folder of *_0000.nii.gz unlabeled T1w images" >&2
  exit 2
fi
if [[ ! -d "$SELFTRAIN_INPUT_DIR" ]]; then
  echo "[error] SELFTRAIN_INPUT_DIR is not a directory: $SELFTRAIN_INPUT_DIR" >&2
  exit 2
fi
if ! compgen -G "$SELFTRAIN_INPUT_DIR/*_0000.nii.gz" >/dev/null; then
  echo "[error] SELFTRAIN_INPUT_DIR must contain nnU-Net-style *_0000.nii.gz images" >&2
  exit 2
fi

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
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

SELFTRAIN_WORK="${SELFTRAIN_WORK:-$REPO/work/selftrain_topk10}"
PRED_DIR="$SELFTRAIN_WORK/predictions_topk10"
mkdir -p "$PRED_DIR"

nnUNetv2_predict \
  -d 502 \
  -c 3d_fullres \
  -tr nnUNetTrainerDiceTopK10Loss \
  -p nnUNetPlans \
  -f 0 1 2 3 4 \
  -i "$SELFTRAIN_INPUT_DIR" \
  -o "$PRED_DIR" \
  --save_probabilities

echo "[done] pseudo-label candidates written to $PRED_DIR"
