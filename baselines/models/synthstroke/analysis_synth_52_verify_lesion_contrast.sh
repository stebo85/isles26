#!/bin/bash
#SBATCH --job-name=synth_lc_verify
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Step 52 (verification): prove LesionContrastD fixes the invisible-lesion
# failure mode on REAL cornucopia synthesis. Runs the contrast audit twice with
# identical seed/cases: (A) raw synth (baseline) and (B) synth + LesionContrastD.
# Expected: frac_invisible 0.52 -> ~0, |CNR| distribution bounded into [0.75,2.0].

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
source "$REPO/.venv-synthstroke/bin/activate"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"

AUDIT="$REPO/baselines/models/synthstroke/synthstroke_helpers/audit_lesion_contrast.py"
COMMON="--labelmaps-dir $REPO/work/synthstroke/labelmaps --lesion-idx 33 --patch 128 \
--n-cases 50 --samples-per-case 6 --device auto --invisible-thr 0.5 --seed 0"

echo "=== (A) BASELINE: raw synthesis ==="
python "$AUDIT" $COMMON \
  --out-dir "$REPO/baselines/reports/synthstroke/lesion_contrast_audit_baseline"

echo "=== (B) FIXED: synthesis + LesionContrastD (prob=1, CNR in [0.75,2.0]) ==="
python "$AUDIT" $COMMON \
  --lesion-contrast-prob 1.0 --lesion-contrast-cnr-lo 0.75 --lesion-contrast-cnr-hi 2.0 \
  --out-dir "$REPO/baselines/reports/synthstroke/lesion_contrast_audit_fixed"

echo "[done] verify lesion-contrast"
