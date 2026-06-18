#!/bin/bash
#SBATCH --job-name=nnunet_plan_resenc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -p owners

# Generate the ResEnc-L (residual encoder U-Net) plans for Dataset502_ATLASR30
# and preprocess the 3d_fullres configuration under those plans. The custom
# stratified 5-fold splits_final.json already installed by step 08 is shared
# across plans (it lives at the dataset level), so ResEnc folds stay directly
# comparable to the default-plans folds. This is the MAPPING-style residual
# encoder upgrade -- the single biggest in-distribution accuracy lever.
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_12_plan_resenc.sh

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
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python -c "import nnunetv2, sys; assert '.venv-nnunet' in nnunetv2.__file__, nnunetv2.__file__"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset502_ATLASR30"
if [[ ! -s "$PREPROCESSED_DIR/splits_final.json" ]]; then
  echo "[error] missing $PREPROCESSED_DIR/splits_final.json; run step 08 first" >&2
  exit 2
fi

# Plan the ResEnc-L experiment (reads the existing dataset fingerprint).
nnUNetv2_plan_experiment -d 502 -pl nnUNetPlannerResEncL

# Preprocess only the 3d_fullres configuration under the ResEnc-L plans.
nnUNetv2_preprocess -d 502 -plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np "${SLURM_CPUS_PER_TASK:-16}"

echo "[done] ResEnc-L plans + 3d_fullres preprocessing ready for Dataset502_ATLASR30"
ls -la "$PREPROCESSED_DIR"/nnUNetResEncUNetLPlans* 2>/dev/null || true
