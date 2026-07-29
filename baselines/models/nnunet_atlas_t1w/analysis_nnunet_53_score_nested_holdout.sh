#!/bin/bash
#SBATCH --job-name=score_nested_holdout
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-11
#SBATCH --requeue
#
# Score the two nested-holdout arms (single held-out fold vs 5-fold ensemble)
# with the OFFICIAL five metrics, over the same threshold x min-component grid.
#
# The question this answers: the container ships a 5-fold ensemble, but every
# operating point this project owns was fitted on SINGLE-model softmax. Averaging
# five members pulls probability mass off 0 and 1, so the ensemble's optimal
# threshold is not the single model's optimal threshold. This measures the shift.
#
# Array is 12 tasks: 6 shards x 2 arms.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO"

PY="$REPO/.venv-official/bin/python"
DATASET="${DATASET:-Dataset507_ATLASR30_CORRECTED}"
GT_DIR="$REPO/work/nnunet/nnUNet_raw/$DATASET/labelsTr"
BASE="$REPO/work/nested_holdout"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
ARMS=(single ensemble5)
ARM="${ARMS[$((TASK / 6))]}"
SHARD=$((TASK % 6))

echo "[info] arm=$ARM shard=$SHARD/6"

"$PY" "$REPO/eval/run_official_eval.py" score \
  --gt-dir "$GT_DIR" \
  --prob-dir "$BASE/$ARM" \
  --out-dir "$BASE/scored_$ARM" \
  --thresholds 0.20:0.70:0.05 \
  --min-cc 0,5,10,20,35,50 \
  --connectivity 26 \
  --reference-threshold 0.50 \
  --shard-index "$SHARD" \
  --shard-count 6 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

echo "[done] arm=$ARM shard=$SHARD"
