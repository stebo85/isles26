#!/bin/bash
#SBATCH --job-name=nnunet_mapping_ens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Final MAPPING-style T1w ensemble inference:
#   TopK10 + default 1000ep + ResEnc-L diversity.
#
# Dataset504 self-training can be added with MAPPING_INCLUDE_SELFTRAIN=1, or
# with MAPPING_INCLUDE_SELFTRAIN=auto when all requested checkpoints exist.
#
# Each member is run with 5-fold TTA inference and --save_probabilities, then
# the helper averages softmax maps and thresholds the averaged lesion channel.
# Input must already be in nnU-Net format:
#
#   <MAPPING_INPUT_DIR>/<case_id>_0000.nii.gz
#
# Example:
#   MAPPING_INPUT_DIR=/path/to/challenge/imagesTs \
#     sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_34_mapping_ensemble_predict.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if [[ -z "${MAPPING_INPUT_DIR:-}" ]]; then
  echo "[error] set MAPPING_INPUT_DIR to a folder of *_0000.nii.gz T1w images" >&2
  exit 2
fi
if [[ ! -d "$MAPPING_INPUT_DIR" ]]; then
  echo "[error] MAPPING_INPUT_DIR is not a directory: $MAPPING_INPUT_DIR" >&2
  exit 2
fi
if ! compgen -G "$MAPPING_INPUT_DIR/*_0000.nii.gz" >/dev/null; then
  echo "[error] MAPPING_INPUT_DIR must contain nnU-Net-style *_0000.nii.gz images" >&2
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
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run the nnU-Net setup scripts first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"

read -r -a FOLDS <<< "${MAPPING_FOLDS:-0 1 2 3 4}"
if (( ${#FOLDS[@]} == 0 )); then
  echo "[error] MAPPING_FOLDS resolved to an empty fold list" >&2
  exit 2
fi

CHECKPOINT="${MAPPING_CHECKPOINT:-checkpoint_final.pth}"
OUT_ROOT="${MAPPING_OUT_DIR:-$REPO/work/predictions_mapping_ensemble}"
MEMBER_ROOT="$OUT_ROOT/members"
PRED_DIR="$OUT_ROOT/predictions"
mkdir -p "$MEMBER_ROOT" "$PRED_DIR"

case_count() {
  find "$1" -maxdepth 1 -name '*_0000.nii.gz' -type f | awk 'END {print NR}'
}

npz_count() {
  find "$1" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | awk 'END {print NR}'
}

require_checkpoints() {
  local dataset_dir="$1"
  local trainer="$2"
  local plans="$3"
  for fold in "${FOLDS[@]}"; do
    local checkpoint="$nnUNet_results/$dataset_dir/${trainer}__${plans}__3d_fullres/fold_${fold}/$CHECKPOINT"
    if [[ ! -s "$checkpoint" ]]; then
      echo "[error] missing checkpoint for $dataset_dir $trainer $plans fold $fold: $checkpoint" >&2
      exit 2
    fi
  done
}

check_checkpoints() {
  local dataset_dir="$1"
  local trainer="$2"
  local plans="$3"
  for fold in "${FOLDS[@]}"; do
    local checkpoint="$nnUNet_results/$dataset_dir/${trainer}__${plans}__3d_fullres/fold_${fold}/$CHECKPOINT"
    if [[ ! -s "$checkpoint" ]]; then
      return 1
    fi
  done
  return 0
}

run_member() {
  local name="$1"
  local dataset_id="$2"
  local dataset_dir="$3"
  local trainer="$4"
  local plans="$5"
  local out_dir="$MEMBER_ROOT/$name"

  require_checkpoints "$dataset_dir" "$trainer" "$plans"
  mkdir -p "$out_dir"

  local expected actual
  expected="$(case_count "$MAPPING_INPUT_DIR")"
  actual="$(npz_count "$out_dir")"
  if [[ "${MAPPING_SKIP_EXISTING:-1}" == "1" && "$actual" == "$expected" && "$expected" != "0" ]]; then
    echo "[info] $name: found $actual existing probability files; skipping inference"
    return
  fi

  echo "[info] $name: predicting $expected cases with dataset=$dataset_id trainer=$trainer plans=$plans folds=${FOLDS[*]}"
  nnUNetv2_predict \
    -i "$MAPPING_INPUT_DIR" \
    -o "$out_dir" \
    -d "$dataset_id" \
    -c 3d_fullres \
    -f "${FOLDS[@]}" \
    -tr "$trainer" \
    -p "$plans" \
    -chk "$CHECKPOINT" \
    --save_probabilities
}

AVG_MEMBER_ARGS=()
MEMBER_DESCRIPTIONS=()

add_average_member() {
  local name="$1"
  local description="$2"
  AVG_MEMBER_ARGS+=(--member "$name" "$MEMBER_ROOT/$name")
  MEMBER_DESCRIPTIONS+=("$description")
}

run_member topk10 502 Dataset502_ATLASR30 nnUNetTrainerDiceTopK10Loss nnUNetPlans
add_average_member topk10 "topk10: Dataset502_ATLASR30 / nnUNetTrainerDiceTopK10Loss / nnUNetPlans"
run_member default1000 502 Dataset502_ATLASR30 nnUNetTrainer nnUNetPlans
add_average_member default1000 "default1000: Dataset502_ATLASR30 / nnUNetTrainer / nnUNetPlans"
run_member resenc_l 502 Dataset502_ATLASR30 nnUNetTrainer_250epochs nnUNetResEncUNetLPlans
add_average_member resenc_l "resenc_l: Dataset502_ATLASR30 / nnUNetTrainer_250epochs / nnUNetResEncUNetLPlans"

SELFTRAIN_TRAINER="${SELFTRAIN_TRAINER:-nnUNetTrainerDiceTopK10Loss}"
case "${MAPPING_INCLUDE_SELFTRAIN:-0}" in
  1|true|TRUE|yes|YES)
    run_member selftrain_topk10 504 Dataset504_ATLASR30_SELFTRAIN "$SELFTRAIN_TRAINER" nnUNetPlans
    add_average_member selftrain_topk10 "selftrain_topk10: Dataset504_ATLASR30_SELFTRAIN / $SELFTRAIN_TRAINER / nnUNetPlans"
    ;;
  auto|AUTO)
    if check_checkpoints Dataset504_ATLASR30_SELFTRAIN "$SELFTRAIN_TRAINER" nnUNetPlans; then
      run_member selftrain_topk10 504 Dataset504_ATLASR30_SELFTRAIN "$SELFTRAIN_TRAINER" nnUNetPlans
      add_average_member selftrain_topk10 "selftrain_topk10: Dataset504_ATLASR30_SELFTRAIN / $SELFTRAIN_TRAINER / nnUNetPlans"
    else
      echo "[info] selftrain_topk10: Dataset504 checkpoints not complete; running 3-family ensemble"
    fi
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "[error] MAPPING_INCLUDE_SELFTRAIN must be 0, 1, or auto; got: ${MAPPING_INCLUDE_SELFTRAIN}" >&2
    exit 2
    ;;
esac

AVG_ARGS=(
  "${AVG_MEMBER_ARGS[@]}"
  --out-dir "$PRED_DIR"
  --threshold "${MAPPING_THRESHOLD:-0.5}"
  --decision "${MAPPING_DECISION:-threshold}"
  --save-prob-nifti
)

if [[ "${MAPPING_SAVE_SOFTMAX_NPZ:-0}" == "1" ]]; then
  AVG_ARGS+=(--save-softmax-npz)
fi
if [[ "${MAPPING_ALLOW_MISSING_MEMBERS:-0}" == "1" ]]; then
  AVG_ARGS+=(--allow-missing-members)
fi

python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/average_softmax_ensemble.py" "${AVG_ARGS[@]}"

{
  cat <<EOF
MAPPING-style nnU-Net ensemble predictions

Input: $MAPPING_INPUT_DIR
Output masks: $PRED_DIR
Member probabilities: $MEMBER_ROOT
Members:
EOF
  for member_description in "${MEMBER_DESCRIPTIONS[@]}"; do
    echo "  - $member_description"
  done
  cat <<EOF
Decision: ${MAPPING_DECISION:-threshold}, threshold=${MAPPING_THRESHOLD:-0.5}
Checkpoint: $CHECKPOINT
Folds: ${FOLDS[*]}
EOF
} > "$OUT_ROOT/README.txt"

echo "[done] MAPPING ensemble predictions written to $PRED_DIR"
