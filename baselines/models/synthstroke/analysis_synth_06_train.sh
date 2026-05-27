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

set -uo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

# Preempt handling: forward USR1 to the python process (it saves + requeues).
trap 'echo "[warn] caught USR1; forwarding to trainer"; kill -USR1 "$PY_PID" 2>/dev/null; wait "$PY_PID"' USR1

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
NAME="${SYNTH_TRAIN_NAME:-isles26_fs_synth_mixreal}"
LABELMAPS_DIR="${LABELMAPS_DIR:-$REPO/work/synthstroke/labelmaps}"
N_CLASSES="${N_CLASSES:-34}"
FADE_FLAG=""
case "${FADE:-}" in 1|true|TRUE|yes|YES) FADE_FLAG="--fade";; esac

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
  --mix-real \
  --synth-prob "${SYNTH_PROB:-0.33}" \
  $FADE_FLAG \
  --device auto &
PY_PID=$!
wait "$PY_PID"
echo "[done] training finished for $NAME"
