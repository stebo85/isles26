#!/bin/bash
#SBATCH --job-name=synth_lc_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -p normal

# Step 51 (verification): run the LesionContrastD unit tests (CPU, pure torch).
# Mirrors analysis_synth_06_train.sh environment.

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
export PYTHONNOUSERSITE=1
source "$REPO/.venv-synthstroke/bin/activate"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"

RC=0
python "$REPO/baselines/models/synthstroke/synthstroke_helpers/test_lesion_contrast.py" || RC=$?
echo "[done] lesion-contrast unit tests exit=$RC"
exit $RC
