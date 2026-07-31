#!/bin/bash
#SBATCH --job-name=train_metric_aligned
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Two fold-0 hypothesis tests, on the CENTER-GROUPED Dataset507 split, both at
# 1000 epochs so they are directly comparable to the model we already have.
#
#   task 0  R1+R2  nnUNetTrainerBlobVolumeLoss   instance-balanced + volume-aware
#   task 1  R3+R4  nnUNetTrainerDA5TopK10        heavy domain/resolution augmentation
#
# WHY 1000 EPOCHS AND NOT A CHEAP 250. The existing Dataset507 fold-0 model is
# nnUNetTrainerDiceTopK10Loss at 1000 epochs, and its per-case official metrics
# are already tabulated in baselines/reports/official_metrics_d507_topk10/. Using
# the same schedule makes that the exact schedule-matched control for free, and
# removes the cross-schedule comparison trap the MSL/DBL results fell into. A
# 250-epoch run would need a third training job just to build its own baseline.
#
# PROMOTION GATE, stated before the runs so it cannot be moved afterwards.
# Score fold 0's 281 validation cases with eval/run_official_eval.py and promote
# to a full 5-fold run ONLY if, versus the existing fold-0 TopK10 model:
#     lesion-F1 (RQ)               >= +0.020      (the primary target)
#  or absolute volume difference   <= -0.30 mL    (the neglected metric)
# and in either case Dice does not drop by more than 0.005.
# Anything smaller is inside the noise this project has repeatedly mistaken for
# a result.
#
# HARD SCHEDULE GATE: any model not submitted by 2026-08-11 does not ship.
# Measured 1000-epoch fold time on this cluster is 5h47m to 17h02m.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

# Preemption safety: requeue cleanly so a preempted run resumes from its last
# checkpoint instead of restarting a multi-hour job from scratch.
_requeue() { echo "[signal] USR1 -> requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap _requeue USR1

module purge
[[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# The cluster torch is a dev build with a broken inductor/triton; torch.compile
# crashes with "must be called with a dataclass type". Documented in agents.md.
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

PY="$REPO/.venv-nnunet/bin/python"
HELPERS="$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers"
DATASET=Dataset507_ATLASR30_CORRECTED
FOLD="${FOLD:-0}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TRAINER=nnUNetTrainerBlobVolumeLoss; SRC="$HELPERS/nnUNetTrainerBlobVolumeLoss.py"; SUBDIR=loss ;;
  1) TRAINER=nnUNetTrainerDA5TopK10;      SRC="$HELPERS/nnUNetTrainerDA5TopK10.py";      SUBDIR=data_augmentation ;;
  *) echo "[error] unknown array task" >&2; exit 2 ;;
esac

# Install the trainer into the venv's nnU-Net package so its class discovery can
# find it. nnU-Net imports EVERY module under variants/ during discovery and does
# not catch ImportError, so a broken file here breaks all training, not just ours.
PKG=$("$PY" -c "import nnunetv2,pathlib;print(pathlib.Path(nnunetv2.__file__).parent)")
TARGET_DIR="$PKG/training/nnUNetTrainer/variants/$SUBDIR"
mkdir -p "$TARGET_DIR"
cp -f "$SRC" "$TARGET_DIR/"
echo "[install] $TRAINER -> $TARGET_DIR/$(basename "$SRC")"

# Fail fast and loudly if discovery cannot see the class, rather than letting
# nnUNetv2_train fail 20 minutes in with an opaque message.
"$PY" - "$TRAINER" <<'PYEOF'
import sys
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2, pathlib
name = sys.argv[1]
cls = recursive_find_python_class(
    str(pathlib.Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer"),
    name, "nnunetv2.training.nnUNetTrainer")
if cls is None:
    raise SystemExit(f"[error] nnU-Net cannot discover trainer {name}")
print(f"[ok] discovered {cls}")
PYEOF

# ALWAYS resume when a checkpoint exists. The USR1 trap requeues this job on
# preemption, but SLURM re-runs the script from the top -- so without --c the
# requeued job silently starts again from epoch 0 and throws away the work.
# That is exactly what happened to the DA5 arm on 2026-07-30: ~14 hours lost,
# and it only showed up as a second training_log_* file in the fold directory.
CKPT="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_latest.pth"
CONTINUE_FLAG=()
if [[ -f "$CKPT" ]]; then
  echo "[resume] found $CKPT -- continuing training rather than restarting"
  CONTINUE_FLAG=(--c)
fi

echo "[train] $TRAINER on $DATASET fold $FOLD (1000 epochs)"
"$REPO/.venv-nnunet/bin/nnUNetv2_train" "$DATASET" 3d_fullres "$FOLD" \
  -tr "$TRAINER" \
  --npz \
  ${CONTINUE_FLAG[@]:+"${CONTINUE_FLAG[@]}"} &
wait $!

echo "[done] $TRAINER fold $FOLD"
echo "Next: score fold-$FOLD validation with the OFFICIAL metrics:"
echo "  analysis_nnunet_58_score_metric_aligned.sh"
