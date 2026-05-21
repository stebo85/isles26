#!/bin/bash
#SBATCH --job-name=synth_our_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Phase B / Step 7: predict + evaluate OUR self-trained SynthStroke model on
# ATLAS fold-0 val (T1w), soop_bench native TRACE, and soop_bench T1w-in-TRACE,
# using our_predict.py and the existing eval framework. Mirrors analysis_synth_03.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel python/3.12.1 py-pytorch/2.4.1_py312
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1

NAME="${SYNTH_TRAIN_NAME:-isles26_fs_synth_mixreal}"
CKPT="${SYNTH_CKPT:-$REPO/work/synthstroke/train/$NAME/checkpoint_best.pt}"
TAG="our_${NAME}"

INFER_PY="$REPO/.venv-synthstroke/bin/python"
EVAL_PY="$REPO/.venv-eval/bin/python"
PREDICT="$REPO/baselines/models/synthstroke/synthstroke_helpers/our_predict.py"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$CKPT" ]]; then echo "[error] missing checkpoint: $CKPT" >&2; exit 2; fi
echo "[info] checkpoint=$CKPT tag=$TAG"

predict_one() { # image out
  local image="$1" out="$2"
  [[ -s "$out" ]] && { echo "[skip] $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  "$INFER_PY" "$PREDICT" --checkpoint "$CKPT" --image "$image" --out "$out" --device auto
}

# A) ATLAS fold-0 val (T1w)
ATLAS_PRED="$REPO/work/synthstroke/pred/atlas_val/$TAG"
mapfile -t VAL_CASES < <("$EVAL_PY" - "$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); f=d if isinstance(d,list) else d["folds"]
[print(c) for c in f[0]["val"]]
PY
)
for c in "${VAL_CASES[@]}"; do
  predict_one "$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr/${c}_0000.nii.gz" "$ATLAS_PRED/${c}.nii.gz"
done
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" \
  baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py \
  --manifest work/nnunet/nnUNet_raw/Dataset501_ATLASR21/conversion_manifest.csv \
  --splits work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json \
  --pred-dir "$ATLAS_PRED" \
  --per-session-csv baselines/reports/training_data_characterization/per_session.csv \
  --out-dir "baselines/reports/synthstroke/$TAG/atlas_val" \
  --title "Our SynthStroke ($NAME) on ATLAS R2.1 fold-0 val"

# B) soop_bench native TRACE
SUBJECTS=$(find data/soop_bench -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -exec test -d '{}/dwi' \; -printf '%f\n' | sort -V)
TRACE_PRED="$REPO/work/synthstroke/pred/soop_trace/$TAG"
TRACE_TMP="$REPO/work/synthstroke/tmp/soop_trace_3d"
mkdir -p "$TRACE_TMP"
for sub in $SUBJECTS; do
  src="data/soop_bench/$sub/dwi/${sub}_rec-TRACE_dwi.nii.gz"
  three="$TRACE_TMP/${sub}_rec-TRACE_dwi_3d.nii.gz"
  if [[ ! -s "$three" ]]; then
    "$EVAL_PY" - "$src" "$three" <<'PY'
import sys,nibabel as nib,numpy as np
img=nib.load(sys.argv[1]); d=np.asanyarray(img.dataobj)
while d.ndim>3 and d.shape[-1]==1: d=d[...,0]
nib.save(nib.Nifti1Image(d,img.affine,img.header.copy()), sys.argv[2])
PY
  fi
  predict_one "$three" "$TRACE_PRED/${sub}.nii.gz"
done
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" -m eval.run_eval \
  --data-root data/soop_bench --pred-dir "$TRACE_PRED" \
  --out-dir "baselines/reports/synthstroke/$TAG/soop_trace" \
  --title "Our SynthStroke ($NAME) on soop_bench (TRACE)"

# C) soop_bench T1w-in-TRACE (reuse nnU-Net warped T1w)
T1W_PRED="$REPO/work/synthstroke/pred/soop_t1w/$TAG"
for sub in $SUBJECTS; do
  t1="$REPO/work/predictions_nnunet_tmp/$sub/${sub}_t1_in_trace.nii.gz"
  [[ -s "$t1" ]] || { echo "[warn] missing $t1; skip"; continue; }
  predict_one "$t1" "$T1W_PRED/${sub}.nii.gz"
done
PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" -m eval.run_eval \
  --data-root data/soop_bench --pred-dir "$T1W_PRED" \
  --out-dir "baselines/reports/synthstroke/$TAG/soop_t1w" \
  --title "Our SynthStroke ($NAME) on soop_bench (T1w in TRACE)"

echo "[done] reports under baselines/reports/synthstroke/$TAG/"
