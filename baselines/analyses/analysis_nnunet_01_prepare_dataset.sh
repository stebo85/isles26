#!/bin/bash
#SBATCH --job-name=nnunet_prepare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Convert ATLAS R2.1 to nnU-Net v2 Dataset501_ATLASR21, then run 3d_fullres
# planning/preprocessing and install the custom stratified splits.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/work/nnunet"

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

VENV="$REPO/.venv-nnunet"
REQ="$REPO/baselines/analyses/nnunet_helpers/requirements_nnunet.txt"
venv_created=0
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
  venv_created=1
fi
source "$VENV/bin/activate"

if (( venv_created )); then
  python -m pip install --upgrade pip setuptools wheel
fi
if ! python - <<'PY' >/dev/null 2>&1
import importlib.metadata
import nnunetv2

assert ".venv-nnunet" in nnunetv2.__file__, nnunetv2.__file__
assert importlib.metadata.version("nnunetv2") == "2.6.2"
assert importlib.metadata.version("scikit-learn") == "1.5.2"
PY
then
  # nnunetv2 + several MIC-DKFZ deps (acvl-utils, dynamic-network-architectures)
  # ship only sdist but are pure-Python. Don't pass --only-binary=:all:; pip will
  # still prefer wheels when available (numpy/scipy/scikit-image cp312 wheels
  # exist on PyPI for these pins, so they install as binary without the flag).
  python -m pip install --no-deps -r "$REQ"
  # dicom2nifti is omitted: it depends on python-gdcm whose PyPI sdist has a
  # broken CMake build, and nnunetv2 2.6.2 never imports dicom2nifti/gdcm/
  # pydicom (verified by `grep -r 'dicom2nifti\|import gdcm\|import pydicom'`).
  # Our inputs are NIfTI throughout.
  python -m pip install \
      "acvl-utils>=0.2.3,<0.3" "dynamic-network-architectures>=0.3.1,<0.5" \
      "batchgenerators==0.25.1" "batchgeneratorsv2==0.3.0" \
      "tqdm" "nibabel" "scikit-image>=0.19.3" "scipy<1.14" \
      "numpy<2.1" "matplotlib" "seaborn" "SimpleITK>=2.2.1" "pandas" \
      "pyyaml" "graphviz" "yacs" "blosc2>=3.0.0b1" \
      "einops"
fi

# nnU-Net's trainer discovery (find_class_by_name) imports every variant
# module to scan for the requested trainer class and does not catch
# ImportError. The Primus trainer's import chain
# (primus -> timm -> torchvision) cannot resolve against the cluster's
# custom torch 2.4.0a0+gitunknown build (no published torchvision wheel
# matches the dev-build ABI). Since we don't use Primus, neutralize its
# discovery module to a no-op so other trainer lookups succeed.
PRIMUS_VARIANT="$VENV/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/primus/primus_trainers.py"
if [[ -f "$PRIMUS_VARIANT" ]]; then
  primus_size=$(stat -c %s "$PRIMUS_VARIANT")
  if (( primus_size > 100 )); then
    cat > "$PRIMUS_VARIANT" <<'PY'
# Neutralized: Primus requires timm+torchvision compatible with the cluster's
# custom torch 2.4.0a0 dev build, which is not satisfiable. nnU-Net's
# find_class_by_name walker fails on any ImportError; this stub lets the
# walker skip past Primus and find the trainers we actually use.
PY
    echo "[info] neutralized $PRIMUS_VARIANT"
  fi
fi

python -c "import nnunetv2, sys; assert '.venv-nnunet' in nnunetv2.__file__, nnunetv2.__file__"
python - <<'PY'
import importlib.metadata

assert importlib.metadata.version("nnunetv2") == "2.6.2"
assert importlib.metadata.version("scikit-learn") == "1.5.2"
PY
NNUNET_TRAIN_BIN="$(command -v nnUNetv2_train || true)"
if [[ -z "$NNUNET_TRAIN_BIN" || "$NNUNET_TRAIN_BIN" != "$VENV/"* ]]; then
  echo "[error] nnUNetv2_train is not inside $VENV: ${NNUNET_TRAIN_BIN:-not found}" >&2
  exit 2
fi

python "$REPO/baselines/analyses/nnunet_helpers/install_trainer_variant.py"

export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

DATASET_DIR="$nnUNet_raw/Dataset501_ATLASR21"
PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset501_ATLASR21"
INVENTORY="$REPO/baselines/reports/training_data_characterization/per_session.csv"

python "$REPO/baselines/analyses/nnunet_helpers/convert_atlas_r21_to_nnunet.py" convert \
  --inventory "$INVENTORY" \
  --dataset-dir "$DATASET_DIR" \
  --repo-root "$REPO"

nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity

python "$REPO/baselines/analyses/nnunet_helpers/convert_atlas_r21_to_nnunet.py" splits \
  --dataset-dir "$DATASET_DIR" \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --folds 5 \
  --seed 42

echo "[done] Dataset501_ATLASR21 prepared at $DATASET_DIR"
