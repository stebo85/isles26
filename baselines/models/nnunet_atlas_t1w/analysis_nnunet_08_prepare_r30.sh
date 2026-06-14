#!/bin/bash
#SBATCH --job-name=nnunet_prepare_r30
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Convert ATLAS R3.0 (1453 sessions: 1284 R-site superset of R2.1 + 169 SOOP) to
# nnU-Net v2 Dataset502_ATLASR30, run 3d_fullres planning/preprocessing, and
# install custom FRESH stratified 5-fold splits (seed 42) over all 1453.
# Reuses the R2.1 tooling (R3.0 is BIDS-identical). Larger corpus -> more
# time/mem/cpu than the R2.1 prepare.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/work/nnunet"

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

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first (builds the env)" >&2
  exit 2
fi
source "$VENV/bin/activate"

python -c "import nnunetv2, sys; assert '.venv-nnunet' in nnunetv2.__file__, nnunetv2.__file__"
NNUNET_TRAIN_BIN="$(command -v nnUNetv2_train || true)"
if [[ -z "$NNUNET_TRAIN_BIN" || "$NNUNET_TRAIN_BIN" != "$VENV/"* ]]; then
  echo "[error] nnUNetv2_train is not inside $VENV: ${NNUNET_TRAIN_BIN:-not found}" >&2
  exit 2
fi

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

DATASET_DIR="$nnUNet_raw/Dataset502_ATLASR30"
PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset502_ATLASR30"
INVENTORY="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"

if [[ ! -s "$INVENTORY" ]]; then
  echo "[error] missing R3.0 inventory: $INVENTORY (run analysis_01_characterize_r30.sh)" >&2
  exit 2
fi

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/convert_atlas_r21_to_nnunet.py" convert \
  --inventory "$INVENTORY" \
  --dataset-dir "$DATASET_DIR" \
  --repo-root "$REPO"

nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres --verify_dataset_integrity

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/convert_atlas_r21_to_nnunet.py" splits \
  --dataset-dir "$DATASET_DIR" \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --folds 5 \
  --seed 42

echo "[done] Dataset502_ATLASR30 prepared at $DATASET_DIR (1453 cases, fresh stratified 5-fold)"
