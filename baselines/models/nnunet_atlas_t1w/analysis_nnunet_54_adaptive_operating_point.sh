#!/bin/bash
#SBATCH --job-name=adaptive_operating_point
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH -p normal
#
# Fit and HONESTLY evaluate a per-case operating point (probability threshold +
# min-component size) on the center-held-out surface.
#
# Motivation, measured: an oracle per-case choice of (threshold, min-CC) scores
#   AVD             6.26 -> 4.50 mL   (-28%)
#   |lesion count|  1.86 -> 1.26      (-32%)
#   lesion-F1 (RQ)  0.586 -> 0.632
# versus the best single fixed configuration. And AVD and lesion-count are nearly
# uncorrelated with Dice (Spearman -0.081, +0.213), so a better model does not
# improve them -- only a better per-case operating point does.
#
# The oracle is an upper bound that cheats. This job measures how much of it a
# rule using only TEST-TIME-AVAILABLE information can actually capture, and it
# does so NESTED (fit on 4 center-grouped folds, apply to the held-out 5th) so
# the reported number is not the number that was optimised.
#
# Zero GPU. Consumes probability volumes already staged by step 48.

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
DATASET=Dataset507_ATLASR30_CORRECTED
GT_DIR="$REPO/work/nnunet/nnUNet_raw/$DATASET/labelsTr"
FEATURES="$REPO/work/official_eval/d507_operating_point_features.csv"

if [[ ! -f "$FEATURES" ]]; then
  echo "[step 1] extracting test-time-available features"
  "$PY" "$REPO/eval/extract_operating_point_features.py" \
    --prob-dir "$REPO/work/official_eval/_probs_d507/fold_*" \
    --gt-dir "$GT_DIR" \
    --out "$FEATURES" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
else
  echo "[skip] features already at $FEATURES"
fi

echo "[step 2] fitting and nested-evaluating the adaptive operating point"
"$PY" "$REPO/eval/fit_adaptive_operating_point.py" \
  --features "$FEATURES" \
  --per-case-metrics "$REPO/baselines/reports/official_metrics_d507_topk10/per_case.csv" \
  --splits "$REPO/work/nnunet/nnUNet_preprocessed/$DATASET/splits_final.json" \
  --manifest "$REPO/work/nnunet/nnUNet_raw/$DATASET/conversion_manifest.csv" \
  --baseline-config thr0.35_k20 \
  --out-dir "$REPO/baselines/reports/adaptive_operating_point_d507"

echo "[done]"
