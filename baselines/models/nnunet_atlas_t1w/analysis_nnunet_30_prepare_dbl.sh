#!/bin/bash
#SBATCH --job-name=nnunet_prepare_dbl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Build a DBL or MSL+DBL multiclass dataset from Dataset502_ATLASR30, plan and
# preprocess it, then copy the original stratified 5-fold splits.
#
# Defaults:
#   DBL_SCHEME=dbl      -> Dataset505_ATLASR30_DBL
#   DBL_SCHEME=msl_dbl  -> Dataset506_ATLASR30_MSLDBL
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_30_prepare_dbl.sh
#   DBL_SCHEME=msl_dbl sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_30_prepare_dbl.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

DBL_SCHEME="${DBL_SCHEME:-dbl}"
case "$DBL_SCHEME" in
  dbl)
    DATASET_ID=505
    DATASET_NAME=Dataset505_ATLASR30_DBL
    ;;
  msl_dbl)
    DATASET_ID=506
    DATASET_NAME=Dataset506_ATLASR30_MSLDBL
    ;;
  *)
    echo "[error] DBL_SCHEME must be 'dbl' or 'msl_dbl', got: $DBL_SCHEME" >&2
    exit 2
    ;;
esac

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

SRC="$nnUNet_raw/Dataset502_ATLASR30"
DST="$nnUNet_raw/$DATASET_NAME"

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/make_msl_dataset.py" \
  --scheme "$DBL_SCHEME" \
  --src-dir "$SRC" \
  --dst-dir "$DST" \
  --boundary-radius "${BOUNDARY_RADIUS:-1}"

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 3d_fullres --verify_dataset_integrity

SRC_SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
DST_SPLITS="$nnUNet_preprocessed/$DATASET_NAME/splits_final.json"
if [[ ! -s "$SRC_SPLITS" ]]; then
  echo "[error] missing $SRC_SPLITS" >&2
  exit 2
fi
cp -f "$SRC_SPLITS" "$DST_SPLITS"
echo "[done] $DATASET_NAME prepared with Dataset502 splits"
