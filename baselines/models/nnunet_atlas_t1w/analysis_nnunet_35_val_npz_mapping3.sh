#!/bin/bash
#SBATCH --job-name=nnunet_val_npz_m3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-14
#SBATCH -p owners

# Save validation softmax (.npz) for the 3-family MAPPING ensemble measurement:
#   0-4:   Dice+TopK10, nnUNetPlans
#   5-9:   default nnUNetTrainer 1000ep, nnUNetPlans
#   10-14: ResEnc-L diversity, nnUNetResEncUNetLPlans
#
# Each task reruns validation only for one family/fold:
#   nnUNetv2_train 502 3d_fullres <FOLD> -tr <TRAINER> [-p <PLANS>] --val --npz
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_35_val_npz_mapping3.sh

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
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 14 )); then
  echo "[error] SLURM_ARRAY_TASK_ID must be 0-14, got: $TASK_ID" >&2
  exit 2
fi

FOLD=$(( TASK_ID % 5 ))
FAMILY=$(( TASK_ID / 5 ))
case "$FAMILY" in
  0)
    FAMILY_NAME=topk10
    TRAINER=nnUNetTrainerDiceTopK10Loss
    PLANS=nnUNetPlans
    ;;
  1)
    FAMILY_NAME=default1000
    TRAINER=nnUNetTrainer
    PLANS=nnUNetPlans
    ;;
  2)
    FAMILY_NAME=resenc_l
    TRAINER=nnUNetTrainer_250epochs
    PLANS=nnUNetResEncUNetLPlans
    ;;
  *)
    echo "[error] internal family index out of range: $FAMILY" >&2
    exit 2
    ;;
esac

RESULT_DIR="$nnUNet_results/Dataset502_ATLASR30/${TRAINER}__${PLANS}__3d_fullres"
VAL_DIR="$RESULT_DIR/fold_${FOLD}/validation"
CHECKPOINT="$RESULT_DIR/fold_${FOLD}/checkpoint_final.pth"
SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "[error] missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -s "$SPLITS" ]]; then
  echo "[error] missing splits: $SPLITS" >&2
  exit 2
fi

EXPECTED="$("$VENV/bin/python" - "$SPLITS" "$FOLD" <<'PY'
import json
import sys
splits = json.loads(open(sys.argv[1]).read())
print(len(splits[int(sys.argv[2])]["val"]))
PY
)"
EXISTING="$(find "$VAL_DIR" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | awk 'END {print NR}')"
if [[ "${VAL_NPZ_SKIP_EXISTING:-1}" == "1" && "$EXISTING" == "$EXPECTED" && "$EXPECTED" != "0" ]]; then
  echo "[info] $FAMILY_NAME fold $FOLD: found $EXISTING/$EXPECTED existing npz files; skipping"
  exit 0
fi

echo "[info] validating $FAMILY_NAME fold $FOLD -> $VAL_DIR (expected val cases=$EXPECTED)"
nnUNetv2_train 502 3d_fullres "$FOLD" -tr "$TRAINER" -p "$PLANS" --val --npz

ACTUAL="$(find "$VAL_DIR" -maxdepth 1 -name '*.npz' -type f | awk 'END {print NR}')"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "[error] $FAMILY_NAME fold $FOLD wrote $ACTUAL npz files, expected $EXPECTED" >&2
  exit 1
fi
echo "[done] $FAMILY_NAME fold $FOLD validation softmax ready ($ACTUAL npz)"
