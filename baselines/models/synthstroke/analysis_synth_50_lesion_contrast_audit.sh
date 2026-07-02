#!/bin/bash
#SBATCH --job-name=synth_contrast_audit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Step 50 (diagnostic): quantify how the cornucopia synth pipeline renders the
# LESION class. Tests the 2026-06-30 hypothesis that cc.SynthFromLabelTransform()
# gives the lesion an independent-random GMM intensity, so a large fraction of
# synthetic patches render the lesion ~invisible vs perilesional tissue
# (contradictory supervision). Falsifiable before spending a GPU run on a
# corrected lesion-contrast synthesis.
#
# Synthesis runs on GPU (as in training) — CPU synthesis of 128^3 patches is
# far too slow. Mirrors analysis_synth_06_train.sh environment.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

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
source "$REPO/.venv-synthstroke/bin/activate"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"

RC=0
python "$REPO/baselines/models/synthstroke/synthstroke_helpers/audit_lesion_contrast.py" \
  --labelmaps-dir "$REPO/work/synthstroke/labelmaps" \
  --lesion-idx 33 \
  --patch 128 \
  --n-cases 50 \
  --samples-per-case 6 \
  --device auto \
  --invisible-thr 0.5 \
  --out-dir "$REPO/baselines/reports/synthstroke/lesion_contrast_audit" || RC=$?
echo "[done] lesion-contrast audit exit=$RC"
exit $RC
