#!/bin/bash
#SBATCH --job-name=nnunet_train_resenc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-4
#SBATCH -p owners

# Train nnU-Net v2 ResEnc-L 3d_fullres on Dataset502_ATLASR30 (1450 cases,
# fresh stratified 5-fold). Residual-encoder U-Net (24GB VRAM target) is the
# MAPPING-style in-distribution accuracy upgrade over the default plans. One
# fold per array task, same 250-epoch schedule as the default-plans run so the
# folds are directly comparable. Preempt-safe (USR1/TERM -> requeue, --c resume).
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_13_train_resenc.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

child_pid=""
handle_signal() {
  echo "[warn] received preemption/termination signal; stopping active nnU-Net process"
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[warn] requesting Slurm requeue for job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID" || echo "[warn] requeue failed; exiting 0" >&2
  fi
  exit 0
}
trap handle_signal USR1 TERM

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
python - <<'PY'
import torch
print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this SLURM allocation")
PY

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$FOLD" =~ ^[0-4]$ ]]; then
  echo "[error] fold (SLURM_ARRAY_TASK_ID) must be 0-4, got: $FOLD" >&2
  exit 2
fi

PLANS=nnUNetResEncUNetLPlans
PREPROCESSED_DATASET="$nnUNet_preprocessed/Dataset502_ATLASR30"
for required in dataset.json "$PLANS.json" splits_final.json; do
  if [[ ! -s "$PREPROCESSED_DATASET/$required" ]]; then
    echo "[error] missing $PREPROCESSED_DATASET/$required; run analysis_nnunet_12_plan_resenc.sh" >&2
    exit 2
  fi
done

TRAIN_ARGS=(nnUNetv2_train 502 3d_fullres "$FOLD" -tr nnUNetTrainer_250epochs -p "$PLANS")
mapfile -t EXISTING < <(find "$nnUNet_results/Dataset502_ATLASR30" -path "*nnUNetTrainer_250epochs__${PLANS}__3d_fullres/fold_${FOLD}/checkpoint_*.pth" -type f 2>/dev/null | sort)
if (( ${#EXISTING[@]} > 0 )); then
  TRAIN_ARGS+=(--c)
  echo "[info] fold $FOLD: existing checkpoint found; continuing with --c"
else
  echo "[info] fold $FOLD: fresh 250-epoch ResEnc-L run"
fi

"${TRAIN_ARGS[@]}" &
child_pid=$!
wait "$child_pid"
child_pid=""

echo "[done] Dataset502_ATLASR30 ResEnc-L fold $FOLD training complete"
