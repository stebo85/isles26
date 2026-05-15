#!/bin/bash
#SBATCH --job-name=deepisles
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p owners
#SBATCH --array=0-11

# Run DeepISLES on one soop_bench subject per array task.
# Each task stages the required modalities (TRACE, ADC, FLAIR) into a per-
# subject working dir and runs inference inside the Apptainer container.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
SIF=$REPO/work/containers/deepisles.sif
DATA=$REPO/data/soop_bench
WORK=$REPO/work/deepisles_runs
PREDS=$REPO/work/predictions_deepisles
mkdir -p "$WORK" "$PREDS" "$REPO/logs"

SUBJECTS=(sub-1 sub-2 sub-3 sub-4 sub-5 sub-7 sub-8 sub-9 sub-10 sub-11 sub-13 sub-14)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

STAGE=$WORK/$SUB
mkdir -p "$STAGE"

# Stage modalities under the names DeepISLES expects.
cp -L "$DATA/$SUB/dwi/${SUB}_rec-TRACE_dwi.nii.gz" "$STAGE/dwi.nii.gz"
cp -L "$DATA/$SUB/dwi/${SUB}_rec-ADC_dwi.nii.gz"   "$STAGE/adc.nii.gz"
cp -L "$DATA/$SUB/anat/${SUB}_FLAIR.nii.gz"        "$STAGE/flair.nii.gz"

# Idempotency: skip if the final prediction already exists.
FINAL_PRED=$PREDS/${SUB}.nii.gz
if [[ -f "$FINAL_PRED" ]]; then
  echo "[ok] $SUB already has prediction: $FINAL_PRED"
  exit 0
fi

echo "[info] $SUB: running DeepISLES (SEALS+NVAUTO+SWAN ensemble) ..."
apptainer run --nv \
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
