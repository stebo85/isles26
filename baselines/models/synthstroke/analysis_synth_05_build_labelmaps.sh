#!/bin/bash
#SBATCH --job-name=synth_labelmaps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Phase B / Step 5: combine SynthSeg tissue labels with the real ATLAS lesion
# masks into single integer label maps that drive the cornucopia GMM image
# synthesis at training time. Light/CPU.
#   SCHEME=fs34     (default) -> 34 classes, lesion=33, out: work/synthstroke/labelmaps
#   SCHEME=compact6           -> 6 classes,  lesion=5,  out: work/synthstroke/labelmaps_compact6

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

SCHEME="${SCHEME:-fs34}"
python "$REPO/baselines/models/synthstroke/synthstroke_helpers/build_labelmap.py" --scheme "$SCHEME"

echo "[done] label maps built (scheme=$SCHEME)"
