#!/bin/bash
#SBATCH --job-name=nnunet_prepare_st
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Self-training step B: merge ATLAS R3.0 labels with pseudo-label candidates
# from step 26 into Dataset504_ATLASR30_SELFTRAIN. Original ATLAS validation
# folds remain untouched; pseudo cases are appended to the train side of every
# fold, so CV metrics still measure real labels only.
#
#   SELFTRAIN_INPUT_DIR=/path/to/unlabeled_nnunet_images \
#     sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_27_prepare_selftrain_dataset.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if [[ -z "${SELFTRAIN_INPUT_DIR:-}" ]]; then
  echo "[error] set SELFTRAIN_INPUT_DIR to the same unlabeled image folder used in step 26" >&2
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

SELFTRAIN_WORK="${SELFTRAIN_WORK:-$REPO/work/selftrain_topk10}"
PSEUDO_LABELS="${PSEUDO_LABELS:-$SELFTRAIN_WORK/predictions_topk10}"
SRC="$nnUNet_raw/Dataset502_ATLASR30"
DST="$nnUNet_raw/Dataset504_ATLASR30_SELFTRAIN"
SRC_SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
ST_SPLITS="$SELFTRAIN_WORK/splits_final.json"

if [[ ! -d "$PSEUDO_LABELS" ]] || ! compgen -G "$PSEUDO_LABELS/*.nii.gz" >/dev/null; then
  echo "[error] missing pseudo-labels in $PSEUDO_LABELS; run step 26 first" >&2
  exit 2
fi
if [[ ! -s "$SRC_SPLITS" ]]; then
  echo "[error] missing source splits: $SRC_SPLITS" >&2
  exit 2
fi

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/prepare_selftrain_dataset.py" \
  --src-dir "$SRC" \
  --pseudo-images-dir "$SELFTRAIN_INPUT_DIR" \
  --pseudo-labels-dir "$PSEUDO_LABELS" \
  --dst-dir "$DST" \
  --src-splits "$SRC_SPLITS" \
  --dst-splits "$ST_SPLITS"

nnUNetv2_plan_and_preprocess -d 504 -c 3d_fullres --verify_dataset_integrity

cp -f "$ST_SPLITS" "$nnUNet_preprocessed/Dataset504_ATLASR30_SELFTRAIN/splits_final.json"
echo "[done] Dataset504_ATLASR30_SELFTRAIN prepared; pseudo cases are train-only"
