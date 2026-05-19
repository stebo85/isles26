#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -p owners

# Train the canonical nnU-Net v2 3d_fullres configuration on Dataset501_ATLASR21,
# fold 0 only, with a reduced 250-epoch schedule for this time-bounded baseline.

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
    if ! scontrol requeue "$SLURM_JOB_ID"; then
      echo "[warn] scontrol requeue failed; exiting 0 so Slurm preemption handling can proceed" >&2
    fi
  else
    echo "[warn] SLURM_JOB_ID is unset; cannot request requeue" >&2
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
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"

python "$REPO/baselines/analyses/nnunet_helpers/install_trainer_variant.py"
python - <<'PY'
import torch
print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this SLURM allocation")
PY

PREPROCESSED_DATASET="$nnUNet_preprocessed/Dataset501_ATLASR21"
for required in dataset.json nnUNetPlans.json splits_final.json splits_summary.json; do
  if [[ ! -s "$PREPROCESSED_DATASET/$required" ]]; then
    echo "[error] Did you run analysis_nnunet_01? Missing $PREPROCESSED_DATASET/$required" >&2
    exit 2
  fi
done

python - "$PREPROCESSED_DATASET/dataset.json" "$PREPROCESSED_DATASET/splits_summary.json" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
dataset = json.loads(dataset_path.read_text())
summary = json.loads(summary_path.read_text())

errors = []
if summary.get("seed") != 42:
    errors.append(f"seed={summary.get('seed')!r}, expected 42")
if summary.get("num_folds") != 5:
    errors.append(f"num_folds={summary.get('num_folds')!r}, expected 5")
if summary.get("num_total_cases") != dataset.get("numTraining"):
    errors.append(
        f"num_total_cases={summary.get('num_total_cases')!r}, "
        f"dataset numTraining={dataset.get('numTraining')!r}"
    )

if errors:
    raise SystemExit("[error] invalid nnU-Net split summary: " + "; ".join(errors))

print(
    f"[ok] split preflight: seed={summary['seed']} "
    f"num_folds={summary['num_folds']} num_total_cases={summary['num_total_cases']}"
)
PY

TRAIN_ARGS=(nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_250epochs)
mapfile -t EXISTING_CHECKPOINTS < <(find "$nnUNet_results/Dataset501_ATLASR21" -path '*nnUNetTrainer_250epochs*3d_fullres/fold_0/checkpoint_*.pth' -type f 2>/dev/null | sort)
if (( ${#EXISTING_CHECKPOINTS[@]} > 0 )); then
  TRAIN_ARGS+=(--c)
  echo "[info] existing checkpoint found; continuing training with --c"
else
  echo "[info] no existing fold-0 checkpoint found; starting a fresh 250-epoch run"
fi

"${TRAIN_ARGS[@]}" &
child_pid=$!
wait "$child_pid"
child_pid=""

nnUNetv2_find_best_configuration \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -tr nnUNetTrainer_250epochs \
  --disable_ensembling

echo "[done] fold 0 training/postprocessing configuration complete"
