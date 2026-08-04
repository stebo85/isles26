#!/bin/bash
#SBATCH --job-name=score_tta_ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-5
#SBATCH --requeue
#
# Score the no-TTA probability maps that analysis_nnunet_61 produced, with the
# official five metrics, on the same grid used everywhere else.
#
# Why this matters: the container ships with mirroring TTA OFF (5 folds + TTA
# costs 12.9 min/case on CPU against a ~10 min budget), but EVERY out-of-fold
# number this project quotes -- including the headline Dice 0.6283 -- was
# measured WITH TTA, and the operating point (p>=0.35, minCC 20) was fitted on
# TTA-smoothed probabilities. That makes the shipped configuration's real
# performance an unmeasured quantity. This closes it.
#
# The control is free and exactly paired: the same fold-0 cases, same weights,
# same grid, from baselines/reports/official_metrics_d507_topk10/per_case.csv.
# TTA is the only variable. Reduce and compare with analysis_nnunet_64.

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

DATASET=Dataset507_ATLASR30_CORRECTED
PROB_DIR="$REPO/work/official_eval/_probs_noTTA_fold0"
OUT_DIR="$REPO/work/official_eval/scored_noTTA_fold0"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

N_PROB=$(find "$PROB_DIR" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "$N_PROB" -ne 281 ]]; then
  echo "[error] expected 281 no-TTA probability maps in $PROB_DIR, found $N_PROB" >&2
  exit 2
fi

"$REPO/.venv-official/bin/python" "$REPO/eval/run_official_eval.py" score \
  --gt-dir "$REPO/work/nnunet/nnUNet_raw/$DATASET/labelsTr" \
  --prob-dir "$PROB_DIR" \
  --out-dir "$OUT_DIR" \
  --thresholds 0.20:0.60:0.05 --min-cc 0,5,10,20,35,50 \
  --connectivity 26 --reference-threshold 0.50 \
  --shard-index "$SHARD" --shard-count 6 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

echo "[done] no-TTA shard $SHARD"
