#!/bin/bash
#SBATCH --job-name=nnunet_atlas_val_probs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-0
#SBATCH -p owners

# Generate native-space lesion probability maps for ATLAS R2.1 fold-0 validation cases.

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
export nnUNet_compile=f  # cluster torch 2.4.0a0 has a broken _inductor/triton; skip torch.compile
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"
NNUNET_PREDICT_BIN="$(command -v nnUNetv2_predict || true)"
if [[ -z "$NNUNET_PREDICT_BIN" || "$NNUNET_PREDICT_BIN" != "$VENV/"* ]]; then
  echo "[error] nnUNetv2_predict is not inside $VENV: ${NNUNET_PREDICT_BIN:-not found}" >&2
  exit 2
fi

VAL_LIST="$REPO/work/predictions_atlas_val/val_cases.txt"
RAW_IMAGES="$nnUNet_raw/Dataset501_ATLASR21/imagesTr"
TMP_BASE="$REPO/work/predictions_atlas_val_prob_tmp"
TMP_IN="$TMP_BASE/in"
TMP_OUT="$TMP_BASE/out"
OUT_DIR="$REPO/work/synthstroke/pred/atlas_val_prob/nnunet"
mkdir -p "$TMP_BASE" "$OUT_DIR"

if [[ ! -d "$RAW_IMAGES" ]]; then
  echo "[error] missing nnU-Net raw imagesTr: $RAW_IMAGES" >&2
  exit 2
fi
if [[ ! -s "$VAL_LIST" ]]; then
  echo "[error] missing validation case list: $VAL_LIST" >&2
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
CHECKPOINT_BASENAME="$(basename "$CHECKPOINT")"

rm -rf "$TMP_IN" "$TMP_OUT"
mkdir -p "$TMP_IN" "$TMP_OUT"

N_CASES=0
while IFS= read -r CASE_ID || [[ -n "$CASE_ID" ]]; do
  if [[ -z "$CASE_ID" ]]; then
    continue
  fi
  RAW_IMAGE="$RAW_IMAGES/${CASE_ID}_0000.nii.gz"
  if [[ ! -s "$RAW_IMAGE" ]]; then
    echo "[error] missing raw validation image for $CASE_ID: $RAW_IMAGE" >&2
    exit 2
  fi
  ln -sf "$RAW_IMAGE" "$TMP_IN/${CASE_ID}_0000.nii.gz"
  N_CASES=$((N_CASES + 1))
done < "$VAL_LIST"

if (( N_CASES <= 0 )); then
  echo "[error] validation case list is empty: $VAL_LIST" >&2
  exit 2
fi

echo "[info] running nnU-Net fold-0 probability prediction for $N_CASES ATLAS validation cases"
nnUNetv2_predict \
  -i "$TMP_IN" \
  -o "$TMP_OUT" \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -tr nnUNetTrainer_250epochs \
  --disable_tta \
  --save_probabilities \
  -chk "$CHECKPOINT_BASENAME"

NPZ_COUNT="$(find "$TMP_OUT" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d '[:space:]')"
if (( NPZ_COUNT != N_CASES )); then
  echo "[error] expected $N_CASES probability NPZ files, found $NPZ_COUNT in $TMP_OUT" >&2
  exit 2
fi

set +e
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/npz_lesion_prob_to_nii.py" \
  --npz-dir "$TMP_OUT" \
  --seg-dir "$TMP_OUT" \
  --ref-dir "$REPO/work/predictions_atlas_val" \
  --out-dir "$OUT_DIR"
CONVERT_STATUS=$?
set -e

REPORT="$OUT_DIR/_convert_report.json"
if [[ ! -s "$REPORT" ]]; then
  echo "[error] missing conversion report: $REPORT" >&2
  exit 2
fi
python - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
print(
    f"[done] nnU-Net ATLAS-val probability maps: "
    f"ok={report['ok']} failed={len(report['failed'])} out_dir={Path(sys.argv[1]).parent}"
)
PY
exit "$CONVERT_STATUS"
