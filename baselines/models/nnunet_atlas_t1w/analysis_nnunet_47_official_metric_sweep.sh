#!/bin/bash
#SBATCH --job-name=official_metric_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#SBATCH --array=0-59
#SBATCH --requeue
#
# Score our out-of-fold predictions with the ISLES'26 ORGANIZERS' OWN metric code,
# and sweep the two knobs we can still turn without retraining: the probability
# threshold and min-connected-component pruning.
#
# Why this job exists (audit findings 3, 7, 11):
#   - Every lesion-F1 in baselines/reports/ was computed with this repo's LENIENT
#     1-voxel any-overlap many-to-many matching. The challenge uses panoptica
#     Recognition Quality under ONE-TO-ONE IoU>=0.25 matching. The two are not
#     comparable, and the bias is model-dependent, so model selection to date was
#     made on the wrong quantity.
#   - PR-AUC is one of the five ranked metrics and has never been computed here.
#   - "Connected-component postprocessing: none chosen" was decided by
#     nnUNetv2_find_best_configuration, which only ever tests
#     remove_all_but_largest_component and only accepts it on a Dice gain.
#     MIN-SIZE pruning was never a candidate, yet two of the five official metrics
#     (absolute lesion count difference, and RQ) are directly sensitive to it.
#   - The shipped threshold 0.54 was selected in-sample on the lenient F1.
#
# Two evaluation surfaces, because they answer different questions:
#   d502_topk10  random stratified folds -> in-distribution, center-LEAKING
#                (52 of 55 centers appear in both train and val of every fold)
#   d507_topk10  center-grouped folds    -> whole centers held out; the honest
#                estimate of the hidden test set, which spans 60+ centers
#
# Zero retraining, zero GPU: consumes probability volumes already on disk.
#
# Sizing: 4 workers x 64 GB, not 8 x 24 GB. The official scorer casts both
# masks to int64 internally, so a 0.5 mm case (~80 M voxels, 50 such sessions
# in ATLAS R3.0) costs ~1.3 GB inside panoptica alone; 8 concurrent workers
# OOM-killed every task of the first full array.
#
# Usage:
#   sbatch analysis_nnunet_47_official_metric_sweep.sh                 # d502
#   SURFACE=d507 sbatch analysis_nnunet_47_official_metric_sweep.sh    # d507
#   SMOKE=1 sbatch --array=0-0 analysis_nnunet_47_official_metric_sweep.sh
# then reduce with analysis_nnunet_49_reduce_official_sweep.sh

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
export OMP_NUM_THREADS=1          # we parallelise over cases, not inside numpy
export PYTHONPATH="$REPO"

PY="$REPO/.venv-official/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] missing .venv-official -- run scripts/analysis_eval_01_setup_official_venv.sh first" >&2
  exit 2
fi

SURFACE="${SURFACE:-d502}"
THRESHOLDS="${THRESHOLDS:-0.20:0.60:0.05}"
# Min component size in VOXELS. Only 783/1453 ATLAS R3.0 sessions are 1 mm
# isotropic, so these are not all the same physical size; the reduce step also
# reports the mm^3 equivalent per case.
MIN_CC="${MIN_CC:-0,5,10,20,35,50}"

case "$SURFACE" in
  d502)
    GT_DIR="$REPO/work/nnunet/nnUNet_raw/Dataset502_ATLASR30/labelsTr"
    PROB_DIRS=("$REPO/work/nnunet_mapping3_oof_ensemble/folds/fold_*/probabilities")
    OUT_DIR="$REPO/work/official_eval/d502_topk10"
    ;;
  d507)
    GT_DIR="$REPO/work/nnunet/nnUNet_raw/Dataset507_ATLASR30_CORRECTED/labelsTr"
    PROB_DIRS=("$REPO/work/official_eval/_probs_d507/fold_*")
    OUT_DIR="$REPO/work/official_eval/d507_topk10"
    if [[ ! -d "$REPO/work/official_eval/_probs_d507" ]]; then
      echo "[error] Dataset507 probability NIfTIs not staged." >&2
      echo "        Run analysis_nnunet_48_stage_d507_probs.sh first." >&2
      exit 2
    fi
    ;;
  *)
    echo "[error] unknown SURFACE=$SURFACE (expected d502 or d507)" >&2
    exit 2
    ;;
esac

if [[ ! -d "$GT_DIR" ]]; then
  echo "[error] ground-truth directory not found: $GT_DIR" >&2
  exit 2
fi

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  # One shard of a 400-way split ~= 4 cases: enough to validate the harness and
  # to time a single case before committing the full array.
  SHARD_COUNT=400
  OUT_DIR="${OUT_DIR}_smoke"
  EXTRA+=(--overwrite)
fi

echo "[info] surface=$SURFACE shard=$SHARD_INDEX/$SHARD_COUNT out=$OUT_DIR"
echo "[info] thresholds=$THRESHOLDS min_cc=$MIN_CC"

# Called directly, not via srun: srun starts a fresh step whose environment does
# not carry the Lmod-set LD_LIBRARY_PATH, and the venv interpreter then cannot
# find libpython3.12.so.1.0. Every other analysis script in this repo does the
# same for the same reason.
"$PY" "$REPO/eval/run_official_eval.py" score \
  --gt-dir "$GT_DIR" \
  --prob-dir "${PROB_DIRS[@]}" \
  --out-dir "$OUT_DIR" \
  --thresholds "$THRESHOLDS" \
  --min-cc "$MIN_CC" \
  --connectivity 26 \
  --reference-threshold 0.50 \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  ${EXTRA[@]:+"${EXTRA[@]}"}
# The :+ guard is required: under `set -u` an empty bash array expands to an
# unbound-variable error, which killed the first full array submission.

echo "[done] shard $SHARD_INDEX"
