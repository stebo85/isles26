#!/bin/bash
#SBATCH --job-name=score_reduced_mirroring
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p normal,owners
#SBATCH --array=0-23
#SBATCH --requeue
#
# Score the reduced-mirroring probability maps from analysis_nnunet_68 with the
# official five metrics, on exactly the grid used for the no-TTA and full-TTA
# controls, so all six arms are directly comparable.
#
# 24 array tasks = 4 axis subsets x 6 shards. Variant = index / 6, shard = index % 6.

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
TAGS=("ax0" "ax1" "ax2" "ax01")

IDX="${SLURM_ARRAY_TASK_ID:-0}"
TAG="${TAGS[$((IDX / 6))]}"
SHARD="$((IDX % 6))"

PROB_DIR="$REPO/work/official_eval/_probs_mirror_${TAG}_fold0"
OUT_DIR="$REPO/work/official_eval/scored_mirror_${TAG}_fold0"

N_PROB=$(find "$PROB_DIR" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)
if [[ "$N_PROB" -ne 281 ]]; then
  echo "[error] expected 281 probability maps in $PROB_DIR, found $N_PROB" >&2
  exit 2
fi

echo "[info] scoring $TAG shard $SHARD"
"$REPO/.venv-official/bin/python" "$REPO/eval/run_official_eval.py" score \
  --gt-dir "$REPO/work/nnunet/nnUNet_raw/$DATASET/labelsTr" \
  --prob-dir "$PROB_DIR" \
  --out-dir "$OUT_DIR" \
  --thresholds 0.20:0.60:0.05 --min-cc 0,5,10,20,35,50 \
  --connectivity 26 --reference-threshold 0.50 \
  --shard-index "$SHARD" --shard-count 6 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

echo "[done] $TAG shard $SHARD"
