#!/bin/bash
#SBATCH --job-name=blob_split_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-19
#SBATCH --requeue
#
# Test whether splitting merged predicted blobs improves the two instance-based
# official metrics, on the center-held-out surface. See eval/blob_split_sweep.py
# for why this is a two-sided bet under one-to-one IoU>=0.25 matching.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
  module load devel; module load python/3.12.1
fi
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH="$REPO"

"$REPO/.venv-official/bin/python" "$REPO/eval/blob_split_sweep.py" \
  --prob-dir "$REPO/work/official_eval/_probs_d507/fold_*" \
  --gt-dir "$REPO/work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED/labelsTr" \
  --out-dir "$REPO/work/official_eval/blob_split_d507" \
  --threshold 0.35 --min-cc 20 --h-values 0,0.10,0.20,0.30,0.40 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --shard-count 20 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
