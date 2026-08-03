#!/bin/bash
#SBATCH --job-name=score_metric_aligned
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-11
#SBATCH --requeue
#
# Score the R1+R2 and R3+R4 fold-0 models with the OFFICIAL five metrics and
# apply the promotion gate that was written down BEFORE the runs started
# (analysis_nnunet_57 header):
#
#   promote to full 5-fold ONLY if, versus the existing fold-0 TopK10 model,
#     lesion-F1 (RQ)             >= +0.020   or
#     absolute volume difference <= -0.30 mL
#   and Dice does not drop by more than 0.005.
#
# The control is free and exactly schedule-matched: the existing Dataset507
# fold-0 TopK10 model at the same 1000 epochs, already scored per case in
# baselines/reports/official_metrics_d507_topk10/per_case.csv. Restricting that
# table to fold 0's 281 validation cases gives the baseline with no extra compute.
#
# Array: 6 shards x 2 arms.

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
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

DATASET=Dataset507_ATLASR30_CORRECTED
FOLD="${FOLD:-0}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
ARMS=(nnUNetTrainerBlobVolumeLoss nnUNetTrainerDA5TopK10)
TRAINER="${ARMS[$((TASK / 6))]}"
SHARD=$((TASK % 6))

VAL_DIR="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/validation"
PROB_DIR="$REPO/work/official_eval/_probs_${TRAINER}_fold${FOLD}"
OUT_DIR="$REPO/work/official_eval/scored_${TRAINER}_fold${FOLD}"

if [[ ! -d "$VAL_DIR" ]]; then
  echo "[error] $VAL_DIR not found -- has analysis_nnunet_57 finished for this arm?" >&2
  exit 2
fi

# Convert this arm's validation softmax to native-space probability NIfTIs once
# (shard 0 does it; the others wait for the marker).
if [[ "$SHARD" -eq 0 ]]; then
  mkdir -p "$PROB_DIR"
  "$REPO/.venv-eval/bin/python" \
    "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/npz_lesion_prob_to_nii.py" \
    --npz-dir "$VAL_DIR" --seg-dir "$VAL_DIR" --ref-dir "$VAL_DIR" \
    --out-dir "$PROB_DIR" --lesion-channel 1
  touch "$PROB_DIR/.staged"
else
  # Fold 4 has 317 cases and its staging took 3h45m; a 1-hour wait timed out
  # every other shard of that fold. Wait up to 6 hours, which is under the
  # job walltime and comfortably above the slowest observed staging.
  for _ in $(seq 1 720); do [[ -f "$PROB_DIR/.staged" ]] && break; sleep 30; done
  [[ -f "$PROB_DIR/.staged" ]] || { echo "[error] timed out waiting for probability staging" >&2; exit 3; }
fi

"$REPO/.venv-official/bin/python" "$REPO/eval/run_official_eval.py" score \
  --gt-dir "$REPO/work/nnunet/nnUNet_raw/$DATASET/labelsTr" \
  --prob-dir "$PROB_DIR" \
  --out-dir "$OUT_DIR" \
  --thresholds 0.20:0.60:0.05 --min-cc 0,5,10,20,35,50 \
  --connectivity 26 --reference-threshold 0.50 \
  --shard-index "$SHARD" --shard-count 6 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

echo "[done] $TRAINER shard $SHARD"
