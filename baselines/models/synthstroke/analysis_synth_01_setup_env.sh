#!/bin/bash
#SBATCH --job-name=synth_setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Phase A / Step 1: build .venv-synthstroke, vendor the SynthStroke repo at a
# pinned commit, install its deps (reusing the cluster torch), and verify the
# imports needed for generation + inference.
#
# Reuses the proven cluster module pattern from
# baselines/models/nnunet_atlas_t1w/analysis_nnunet_01_prepare_dataset.sh
# (python/3.12.1 + py-pytorch + --system-site-packages venv).

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/work/synthstroke"

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

# ---------------------------------------------------------------------------
# 1. Vendor the SynthStroke repo at a pinned commit.
# ---------------------------------------------------------------------------
SYNTH_PIN="f627b57b8f9291bfa870cb54cb21ddbeb4941d39"   # main @ 2025-11-27
REPO_DIR="$REPO/work/synthstroke/repo"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/liamchalcroft/synthstroke.git "$REPO_DIR"
fi
( cd "$REPO_DIR" && git fetch --all --tags )
( cd "$REPO_DIR" && git checkout -q "$SYNTH_PIN" )
echo "[info] synthstroke repo at $( cd "$REPO_DIR" && git rev-parse HEAD )"

# ---------------------------------------------------------------------------
# 2. Create the venv (reuse cluster torch via --system-site-packages).
# ---------------------------------------------------------------------------
VENV="$REPO/.venv-synthstroke"
REQ="$REPO/baselines/models/synthstroke/synthstroke_helpers/requirements_synthstroke.txt"
venv_created=0
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
  venv_created=1
fi
source "$VENV/bin/activate"

if (( venv_created )); then
  python -m pip install --upgrade pip setuptools wheel
fi

# Verify torch is visible from the cluster module before installing on top of it.
python - <<'PY'
import torch
print("[info] torch", torch.__version__, "cuda_build", torch.version.cuda)
PY

python -m pip install -r "$REQ"

# ---------------------------------------------------------------------------
# 3. Verify the imports the rest of the pipeline relies on.
# ---------------------------------------------------------------------------
export WANDB_MODE=offline
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" python - <<'PY'
import monai, cornucopia, huggingface_hub, safetensors, nibabel, skimage
from synthstroke_model import SynthStrokeModel
print("[ok] monai", monai.__version__)
print("[ok] cornucopia", getattr(cornucopia, "__version__", "n/a"))
print("[ok] huggingface_hub", huggingface_hub.__version__)
print("[ok] SynthStrokeModel imported")
PY

echo "[done] .venv-synthstroke ready; repo vendored at $REPO_DIR"
