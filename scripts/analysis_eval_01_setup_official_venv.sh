#!/bin/bash
#SBATCH --job-name=setup_official_venv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p owners
#
# Build `.venv-official`: the venv that can run the ISLES'26 organizers' OWN
# evaluation code (work/isles26_upstream/utils/eval_utils.py) verbatim.
#
# Why a separate venv from `.venv-eval`: the official metrics need `panoptica`
# (instance matching) and `scikit-learn` (precision_recall_curve), neither of
# which `.venv-eval` has. We deliberately do NOT add them to `.venv-eval` so the
# repo's own harness and the official harness stay independently reproducible.
#
# Cluster gotchas (documented in agents.md):
#   - build against the `python/3.12.1` module and export its lib dir on
#     LD_LIBRARY_PATH or the interpreter cannot find libpython at runtime
#   - numpy/scipy have no source-buildable path here (no OpenBLAS dev headers),
#     so they must come from wheels
#   - `panoptica` and its pure-python deps ship sdists, so a blanket
#     `--only-binary=:all:` breaks the install; pin the binary-only flag to the
#     compiled packages only.

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
export PIP_CACHE_DIR="${SCRATCH:-$REPO/work}/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"

VENV="$REPO/.venv-official"

if [[ -x "$VENV/bin/python" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "[skip] $VENV already exists (set FORCE_REBUILD=1 to rebuild)"
else
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip

# Compiled packages: wheels only. There are no OpenBLAS/libjpeg dev headers on
# this cluster, so anything with a C extension MUST come from a wheel. panoptica
# pulls scikit-image -> pillow, and pillow's sdist needs libjpeg, so pre-install
# that whole subtree binary-only before panoptica gets a chance to build it.
# panoptica 1.0.1 pins numpy<2, so let pip resolve numpy rather than pinning it.
"$PY" -m pip install --only-binary=:all: \
  "scipy<1.14" "scikit-learn==1.5.2" "nibabel==5.4.2" \
  "pandas" "connected-components-3d" \
  "pillow" "scikit-image>=0.22,<0.23" "imageio" "tifffile" "ruamel.yaml.clib"

# panoptica itself + its remaining pure-python deps: sdists are fine here.
"$PY" -m pip install "panoptica==1.0.1"

echo "=== installed ==="
"$PY" -m pip freeze | sort

# ---------------------------------------------------------------------------
# Verify the organizers' code actually imports and runs, and pin down the two
# semantics we had to assume during the audit:
#   (1) is NaiveThresholdMatching's threshold on IoU (not Dice)?
#   (2) what connectivity does ConnectedComponentsInstanceApproximator use --
#       26 (as eval/metrics.py assumes) or 6? Every lesion-count number in this
#       repo changes if it is 6.
# ---------------------------------------------------------------------------
echo "=== semantics probe ==="
PYTHONPATH="$REPO/work/isles26_upstream" "$PY" "$REPO/eval/probe_panoptica_semantics.py"

echo "[done] .venv-official ready at $VENV"
