#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -p owners

# Phase B / Step 6: train a SynthStroke-style model on our self-contained ATLAS
# label maps (34-class: tissue 0..32 + lesion 33), fold 0, extended schedule
# (500 epochs x 200 iters), with configurable synthetic:real mixing.
# --synth-prob 0.33 = 33% synthetic / 67% real (ATLAS-first default).
# Preempt-safe: SIGUSR1 triggers checkpoint + scontrol requeue.
# --time 24:00:00 covers ~21h; SLURM requeue handles overflow automatically.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

# Preempt handling: forward USR1 to the python process (it saves + requeues).
PY_PID=""
on_usr1() {
  echo "[warn] caught USR1; forwarding to trainer"
  if [[ -n "$PY_PID" ]]; then
    kill -USR1 "$PY_PID" 2>/dev/null || true
    wait "$PY_PID" || true
  fi
}
trap on_usr1 USR1

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
source "$REPO/.venv-synthstroke/bin/activate"
export PYTHONPATH="$REPO/work/synthstroke/repo:$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Scheme/recipe knobs (defaults preserve the original 34-class behavior).
#   Run D recipe: LABELMAPS_DIR=$REPO/work/synthstroke/labelmaps_compact6 N_CLASSES=6 FADE=1 SYNTH_PROB=0.33
#   REAL_AUG=1 enables real-image intensity augmentation (bias field + noise + gamma + intensity shift); default off.
#   DWI_PROB=0.5 enables DWI-aware lesion-hyperintensity synth augmentation; default unset/off.
# Validation checkpoint-selection knobs default to the deployment defaults used
# by analysis_synth_07_predict_eval.sh when no tuned operating point exists.
NAME="${SYNTH_TRAIN_NAME:-isles26_fs_synth_mixreal}"
LABELMAPS_DIR="${LABELMAPS_DIR:-$REPO/work/synthstroke/labelmaps}"
N_CLASSES="${N_CLASSES:-34}"
LOSS_FLAG=""
TVERSKY_FLAG=""
TVERSKY_ALPHA="${TVERSKY_ALPHA:-0.7}"
TVERSKY_BETA="${TVERSKY_BETA:-0.3}"
if [[ -n "${LOSS:-}" ]]; then
  LOSS_FLAG="--loss $LOSS"
  TVERSKY_FLAG="--tversky-alpha $TVERSKY_ALPHA --tversky-beta $TVERSKY_BETA"
fi
FADE_FLAG=""
case "${FADE:-}" in 1|true|TRUE|yes|YES) FADE_FLAG="--fade";; esac
DWI_FLAG=""
if [[ -n "${DWI_PROB:-}" ]]; then DWI_FLAG="--dwi-prob $DWI_PROB"; fi
REAL_AUG_FLAG=""
case "${REAL_AUG:-}" in 1|true|TRUE|yes|YES) REAL_AUG_FLAG="--real-aug";; esac
VAL_BINARIZE="${VAL_BINARIZE:-threshold}"
VAL_THRESHOLD="${VAL_THRESHOLD:-${PROB_THRESHOLD:-0.5}}"
VAL_MIN_CC_VOXELS="${VAL_MIN_CC_VOXELS:-${MIN_CC_VOXELS:-0}}"
VAL_BRAIN_MASK="${VAL_BRAIN_MASK:-${BRAIN_MASK:-1}}"
VAL_BRAIN_MASK_FLAG=""
case "$VAL_BRAIN_MASK" in 0|false|FALSE|no|NO) VAL_BRAIN_MASK_FLAG="--no-val-brain-mask";; esac

python "$REPO/baselines/models/synthstroke/synthstroke_helpers/our_train.py" \
  --name "$NAME" \
  --labelmaps-dir "$LABELMAPS_DIR" \
  --n-classes "$N_CLASSES" \
  --epochs 500 \
  --epoch-length 200 \
  --val-interval 5 \
  --patch 128 \
  --lr 0.001 \
  --l2 50 \
  --lesion-weight 2 \
  $LOSS_FLAG $TVERSKY_FLAG \
  --mix-real \
  --synth-prob "${SYNTH_PROB:-0.33}" \
  --val-binarize "$VAL_BINARIZE" \
  --val-threshold "$VAL_THRESHOLD" \
  --val-min-cc-voxels "$VAL_MIN_CC_VOXELS" \
  $FADE_FLAG $DWI_FLAG $REAL_AUG_FLAG \
  $VAL_BRAIN_MASK_FLAG \
  --device auto &
PY_PID=$!
RC=0
wait "$PY_PID" || RC=$?
if [[ "$RC" -ne 0 ]]; then
  echo "[error] trainer exited with code $RC for $NAME" >&2
  exit "$RC"
fi
echo "[done] training finished for $NAME"
