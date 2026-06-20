#!/bin/bash
#SBATCH --job-name=nnunet_prepare_msl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Build Dataset503_ATLASR30_MSL (size-stratified lesion classes) from
# Dataset502_ATLASR30, plan/preprocess 3d_fullres, and copy the Dataset502
# stratified 5-fold splits so MSL folds are directly comparable to the binary
# baseline. Merge classes back to lesion at inference (steps 19/20).
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_18_prepare_msl.sh

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
source "$VENV/bin/activate"
python -c "import nnunetv2, sys; assert '.venv-nnunet' in nnunetv2.__file__, nnunetv2.__file__"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

SRC="$nnUNet_raw/Dataset502_ATLASR30"
DST="$nnUNet_raw/Dataset503_ATLASR30_MSL"

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/make_msl_dataset.py" \
  --src-dir "$SRC" --dst-dir "$DST"

nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres --verify_dataset_integrity

# Reuse the exact Dataset502 stratified 5-fold splits (case ids are identical).
SRC_SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
DST_SPLITS="$nnUNet_preprocessed/Dataset503_ATLASR30_MSL/splits_final.json"
if [[ ! -s "$SRC_SPLITS" ]]; then
  echo "[error] missing $SRC_SPLITS" >&2; exit 2
fi
cp -f "$SRC_SPLITS" "$DST_SPLITS"
echo "[done] Dataset503_ATLASR30_MSL prepared; splits copied from Dataset502"
