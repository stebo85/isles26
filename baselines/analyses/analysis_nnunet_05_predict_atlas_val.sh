#!/bin/bash
#SBATCH --job-name=nnunet_atlas_val_predict
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-0
#SBATCH -p owners

# Run nnU-Net prediction for one fold-0 ATLAS validation case per array task.
#
# First compute the validation list and array bound:
#   bash baselines/analyses/analysis_nnunet_05a_compute_atlas_val_array.sh
#
# That writes work/predictions_atlas_val/val_cases.txt and prints array=0-N.
# Submit this script with that bound, for example:
#   sbatch --array=0-N baselines/analyses/analysis_nnunet_05_predict_atlas_val.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

PREDICTION_VERSION="v1"

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
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/analyses/nnunet_helpers/install_trainer_variant.py"
NNUNET_PREDICT_BIN="$(command -v nnUNetv2_predict || true)"
if [[ -z "$NNUNET_PREDICT_BIN" || "$NNUNET_PREDICT_BIN" != "$VENV/"* ]]; then
  echo "[error] nnUNetv2_predict is not inside $VENV: ${NNUNET_PREDICT_BIN:-not found}" >&2
  exit 2
fi

EVAL_PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing $EVAL_PY; build the eval venv first" >&2
  exit 2
fi

RAW_IMAGES="$nnUNet_raw/Dataset501_ATLASR21/imagesTr"
PRED_DIR="$REPO/work/predictions_atlas_val"
VAL_LIST="$PRED_DIR/val_cases.txt"
TMP_BASE="$REPO/work/predictions_atlas_val_tmp"
mkdir -p "$PRED_DIR" "$TMP_BASE"

if [[ ! -d "$RAW_IMAGES" ]]; then
  echo "[error] missing nnU-Net raw imagesTr: $RAW_IMAGES" >&2
  exit 2
fi
if [[ ! -s "$VAL_LIST" ]]; then
  echo "[error] missing validation case list: $VAL_LIST" >&2
  echo "[error] run baselines/analyses/analysis_nnunet_05a_compute_atlas_val_array.sh first" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi
N_CASES="$(wc -l < "$VAL_LIST" | tr -d '[:space:]')"
if (( N_CASES <= 0 )); then
  echo "[error] validation case list is empty: $VAL_LIST" >&2
  exit 2
fi
ARRAY_MAX=$((N_CASES - 1))
if (( TASK_ID > ARRAY_MAX )); then
  echo "[error] SLURM_ARRAY_TASK_ID=$TASK_ID outside valid range 0-$ARRAY_MAX for $VAL_LIST" >&2
  exit 2
fi
LINE_NO=$((TASK_ID + 1))
CASE_ID="$(sed -n "${LINE_NO}p" "$VAL_LIST")"
if [[ -z "$CASE_ID" ]]; then
  echo "[error] empty case id at line $LINE_NO in $VAL_LIST" >&2
  exit 2
fi

RESULT_FOLD="$nnUNet_results/Dataset501_ATLASR21/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0"
if [[ -s "$RESULT_FOLD/checkpoint_final.pth" ]]; then
  CHECKPOINT="$RESULT_FOLD/checkpoint_final.pth"
elif [[ -s "$RESULT_FOLD/checkpoint_latest.pth" ]]; then
  CHECKPOINT="$RESULT_FOLD/checkpoint_latest.pth"
else
  echo "[error] no fold-0 checkpoint_final.pth or checkpoint_latest.pth found under $RESULT_FOLD" >&2
  exit 2
fi
MODEL_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

sidecar_matches() {
  "$EVAL_PY" - "$1" "$2" "$PREDICTION_VERSION" <<'PY'
import json
import sys
from pathlib import Path

sidecar = Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_version = sys.argv[3]
try:
    payload = json.loads(sidecar.read_text())
except Exception:
    raise SystemExit(1)
ok = (
    payload.get("model_checkpoint_sha256") == expected_sha
    and payload.get("prediction_version") == expected_version
)
raise SystemExit(0 if ok else 1)
PY
}

validate_integer_nifti() {
  "$EVAL_PY" - "$1" <<'PY'
import sys
import numpy as np
import nibabel as nib

path = sys.argv[1]
img = nib.load(path)
shape = img.shape
dtype = np.dtype(img.get_data_dtype())
if len(shape) < 3:
    raise SystemExit(f"{path} is not at least 3-D: shape={shape}")
if dtype.kind not in {"i", "u", "b"}:
    raise SystemExit(f"{path} does not have integer dtype: {dtype}")
print(f"[ok] validated integer NIfTI: {path} shape={shape} dtype={dtype}")
PY
}

RAW_IMAGE="$RAW_IMAGES/${CASE_ID}_0000.nii.gz"
FINAL_PRED="$PRED_DIR/${CASE_ID}.nii.gz"
SIDECAR="$PRED_DIR/${CASE_ID}.json"

if [[ -s "$FINAL_PRED" && -s "$SIDECAR" ]] && sidecar_matches "$SIDECAR" "$MODEL_SHA"; then
  echo "[ok] current ATLAS-val prediction exists for $CASE_ID: $FINAL_PRED"
  exit 0
fi

if [[ ! -s "$RAW_IMAGE" ]]; then
  echo "[error] missing raw validation image for $CASE_ID: $RAW_IMAGE" >&2
  exit 2
fi

if [[ -e "$FINAL_PRED" || -e "$SIDECAR" ]]; then
  echo "[stale] removing old prediction/sidecar for $CASE_ID before rerun"
  rm -f "$FINAL_PRED" "$SIDECAR"
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="$TMP_BASE/run_${JOB_TAG}_${TASK_ID}_${CASE_ID}"
RUN_INPUT="$RUN_DIR/input"
RUN_OUTPUT="$RUN_DIR/output"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_INPUT" "$RUN_OUTPUT"

STAGED_INPUT="$RUN_INPUT/${CASE_ID}_0000.nii.gz"
cp "$RAW_IMAGE" "$STAGED_INPUT.tmp"
mv "$STAGED_INPUT.tmp" "$STAGED_INPUT"

echo "[info] running nnU-Net prediction for ATLAS fold-0 val case $CASE_ID (task $TASK_ID/$ARRAY_MAX)"
nnUNetv2_predict \
  -i "$RUN_INPUT" \
  -o "$RUN_OUTPUT" \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -tr nnUNetTrainer_250epochs \
  --disable_tta

NNUNET_OUT="$RUN_OUTPUT/${CASE_ID}.nii.gz"
if [[ ! -s "$NNUNET_OUT" ]]; then
  echo "[error] nnU-Net did not create expected output for $CASE_ID: $NNUNET_OUT" >&2
  exit 2
fi
validate_integer_nifti "$NNUNET_OUT"

TMP_FINAL="$PRED_DIR/${CASE_ID}.nii.gz.tmp.${JOB_TAG}.${TASK_ID}"
cp "$NNUNET_OUT" "$TMP_FINAL"
mv "$TMP_FINAL" "$FINAL_PRED"

CREATED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
"$EVAL_PY" - "$SIDECAR" "$MODEL_SHA" "$PREDICTION_VERSION" "$CREATED_AT_UTC" "$CHECKPOINT" <<'PY'
import json
import os
import sys
from pathlib import Path

sidecar = Path(sys.argv[1])
payload = {
    "model_checkpoint_sha256": sys.argv[2],
    "synthstrip_model_info": "not_used_atlas_images_are_brain_extracted",
    "prediction_version": sys.argv[3],
    "created_at_utc": sys.argv[4],
    "model_checkpoint_path": sys.argv[5],
}
tmp = sidecar.with_name(sidecar.name + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, sidecar)
PY

echo "[done] $CASE_ID -> $FINAL_PRED"
