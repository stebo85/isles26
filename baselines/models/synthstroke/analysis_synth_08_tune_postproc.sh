#!/bin/bash
#SBATCH --job-name=synth_tune_postproc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH -p owners

# Phase B / Step 8: dump ATLAS fold-0 val lesion probability maps (GPU, Step 1)
# then sweep (prob-threshold x min-cc-voxels) on CPU (Step 2) and write the
# chosen operating point to baselines/reports/synthstroke/$TAG/postproc_tuning/.

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
N_CLASSES="${N_CLASSES:-34}"   # must match the trained checkpoint (Run D compact6 = 6)
TAG="our_${NAME}"

INFER_PY="$REPO/.venv-synthstroke/bin/python"
EVAL_PY="$REPO/.venv-eval/bin/python"
PREDICT="$REPO/baselines/models/synthstroke/synthstroke_helpers/our_predict.py"
TUNE="$REPO/baselines/models/synthstroke/synthstroke_helpers/tune_postproc.py"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$CKPT" ]]; then echo "[error] missing checkpoint: $CKPT" >&2; exit 2; fi
echo "[info] checkpoint=$CKPT tag=$TAG"

# Extract fold-0 val cases (same logic as analysis_synth_07)
mapfile -t VAL_CASES < <("$EVAL_PY" - "$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); f=d if isinstance(d,list) else d["folds"]
[print(c) for c in f[0]["val"]]
PY
)

# ---------------------------------------------------------------------------
# Step 1 (GPU): dump lesion probability maps for each ATLAS fold-0 val case.
# Skip if already exists (idempotent). Use prob-threshold=0.5, min-cc=0
# so the raw probability map is dumped (tuning happens in Step 2).
# ---------------------------------------------------------------------------
PROB_DIR="$REPO/work/synthstroke/pred/atlas_val_prob/$TAG"
mkdir -p "$PROB_DIR"

echo "[step1] dumping lesion prob maps for ${#VAL_CASES[@]} cases -> $PROB_DIR"
for c in "${VAL_CASES[@]}"; do
  OUT_PROB="$PROB_DIR/${c}.nii.gz"
  if [[ -s "$OUT_PROB" ]]; then
    echo "[skip] $OUT_PROB"
    continue
  fi
  IMG="$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr/${c}_0000.nii.gz"
  if [[ ! -s "$IMG" ]]; then
    echo "[error] missing image for $c: $IMG" >&2
    exit 2
  fi
  # --out is a throwaway binary mask (not used in tuning); prob goes to --save-prob
  TMP_OUT="$PROB_DIR/${c}_mask_tmp.nii.gz"
  if ! "$INFER_PY" "$PREDICT" \
    --checkpoint "$CKPT" \
    --image "$IMG" \
    --out "$TMP_OUT" \
    --n-classes "$N_CLASSES" \
    --prob-threshold 0.5 \
    --min-cc-voxels 0 \
    --save-prob "$OUT_PROB" \
    --device auto; then
    echo "[error] predict failed for $c" >&2
    exit 1
  fi
  # Clean up the throwaway mask to save disk
  rm -f "$TMP_OUT"
done
echo "[step1] done"

# ---------------------------------------------------------------------------
# Step 2 (CPU): sweep (thr x cc) grid on the saved prob maps.
# ---------------------------------------------------------------------------
OUT_TUNE="$REPO/baselines/reports/synthstroke/$TAG/postproc_tuning"
mkdir -p "$OUT_TUNE"

echo "[step2] running postproc sweep -> $OUT_TUNE"
# tune_postproc fails closed (SystemExit) if any fold-0 val prob map is missing;
# propagate that as a hard job failure instead of printing a success line.
if ! PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" "$TUNE" \
  --prob-dir "$PROB_DIR" \
  --splits "$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json" \
  --labels-dir "$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/labelsTr" \
  --out-dir "$OUT_TUNE" \
  --thresholds "0.3,0.4,0.5,0.6" \
  --cc-voxels "0,5,10,20,50" \
  --small-recall-tolerance 0.95; then
  echo "[error] tune_postproc failed (likely missing prob maps)" >&2
  exit 1
fi

echo "[step2] done"
echo ""
echo "=== Chosen operating point ==="
cat "$OUT_TUNE/postproc_operating_point.json"
echo ""
echo "[done] reports under $OUT_TUNE"
