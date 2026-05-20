#!/bin/bash
#SBATCH --job-name=deepisles
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners
#SBATCH --array=0-11
# Note: the constraint excludes Hopper (CC 9.0) and Ada (CC 8.9) GPUs.
# PyTorch 1.11 in the DeepISLES container only supports up to sm_86.

# Run DeepISLES on one soop_bench subject per array task.
# Each task stages the required modalities (TRACE, ADC, FLAIR) into a per-
# subject working dir and runs inference inside the Apptainer container.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
SIF=$REPO/work/containers/deepisles_sandbox  # directory-based image (see analysis_01)
DATA=$REPO/data/soop_bench
WORK=$REPO/work/deepisles_runs
PREDS=$REPO/work/predictions_deepisles
mkdir -p "$WORK" "$PREDS" "$REPO/logs"

SUBJECTS=(sub-1 sub-2 sub-3 sub-4 sub-5 sub-7 sub-8 sub-9 sub-10 sub-11 sub-13 sub-14)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

STAGE=$WORK/$SUB
mkdir -p "$STAGE"

# Stage modalities under the names DeepISLES expects.
# soop_bench TRACE/ADC are 4D (X, Y, Z, 1) — DeepISLES asserts 3D, so squeeze.
module purge
module load python/3.12.1
source "$REPO/.venv-eval/bin/activate"
python - <<PY
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

def to_3d(img):
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return nib.Nifti1Image(data, img.affine, img.header)

# Load and squeeze each modality.
trace = to_3d(nib.load("$DATA/$SUB/dwi/${SUB}_rec-TRACE_dwi.nii.gz"))
adc   = to_3d(nib.load("$DATA/$SUB/dwi/${SUB}_rec-ADC_dwi.nii.gz"))
flair = to_3d(nib.load("$DATA/$SUB/anat/${SUB}_FLAIR.nii.gz"))

# DeepISLES requires DWI and ADC on the same grid. soop_bench TRACE and ADC are
# acquired with different reconstruction grids, so resample ADC -> TRACE.
if adc.shape != trace.shape or not np.allclose(adc.affine, trace.affine, atol=1e-3):
    print(f"  resampling ADC {adc.shape} -> TRACE grid {trace.shape}")
    adc = resample_from_to(adc, (trace.shape, trace.affine), order=1, mode="constant", cval=0)

for img, dst in [(trace, "$STAGE/dwi.nii.gz"),
                 (adc,   "$STAGE/adc.nii.gz"),
                 (flair, "$STAGE/flair.nii.gz")]:
    nib.save(img, dst)
    print(f"  {dst} -> shape={img.shape} dtype={img.get_data_dtype()}")
PY

# Idempotency: skip if the final prediction already exists.
FINAL_PRED=$PREDS/${SUB}.nii.gz
if [[ -f "$FINAL_PRED" ]]; then
  echo "[ok] $SUB already has prediction: $FINAL_PRED"
  exit 0
fi

echo "[info] $SUB: running DeepISLES (SEALS+NVAUTO+SWAN ensemble) ..."
apptainer run --nv \
  --pwd /app \
  --bind "$STAGE:/app/data" \
  "$SIF" \
  --dwi_file_name dwi.nii.gz \
  --adc_file_name adc.nii.gz \
  --flair_file_name flair.nii.gz \
  --skull_strip

# DeepISLES writes lesion_msk.nii.gz into /app/data/results/ by default.
RESULTS=$STAGE/results
echo "[info] $SUB: contents of results/:"
ls -la "$RESULTS" || true

# Move the final ensemble mask into the predictions dir.
if [[ -f "$RESULTS/lesion_msk.nii.gz" ]]; then
  cp "$RESULTS/lesion_msk.nii.gz" "$FINAL_PRED"
  echo "[done] $SUB -> $FINAL_PRED"
else
  echo "[error] $SUB: expected $RESULTS/lesion_msk.nii.gz not found" >&2
  exit 2
fi
