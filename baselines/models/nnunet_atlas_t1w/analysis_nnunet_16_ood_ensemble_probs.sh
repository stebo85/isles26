#!/bin/bash
#SBATCH --job-name=ood_ens_probs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-11
#SBATCH -p owners

# OOD-targeted ensemble, step A: produce TRACE-space lesion PROBABILITY maps for
# two T1w-input models on every soop_bench subject, so step 17 can blend them.
#
# Both models run on the SAME challenge-realistic input: the skull-stripped T1w
# (work/soop_bench_t1w_brain, cached from step 10). We then warp each native
# (T1w-grid) lesion-probability volume into TRACE/DWI space with the cached
# rigid T1->TRACE affine (Linear interpolation, so probabilities stay smooth):
#
#   * nn  = ATLAS R3.0 nnU-Net 5-fold ensemble (Dataset502, --save_probabilities)
#   * sp  = pretrained synthstroke-synth-plus (6-class, lesion channel 5)
#
# Outputs (float32 probability NIfTI in TRACE space):
#   work/predictions_ood_ens/nn_prob_trace/<sub>.nii.gz
#   work/predictions_ood_ens/sp_prob_trace/<sub>.nii.gz
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_16_ood_ensemble_probs.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

MASK_ROOT="$REPO/data/soop_bench/derivatives/lesion_masks"
mapfile -t SUBJECTS < <(ls -1 "$MASK_ROOT" | grep '^sub-' | sort -V)
ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-11}"
if (( ${#SUBJECTS[@]} != ARRAY_MAX + 1 )); then
  echo "[error] found ${#SUBJECTS[@]} soop_bench subjects, expected $((ARRAY_MAX + 1))" >&2
  exit 2
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUB="${SUBJECTS[$TASK_ID]}"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312
module load ants/2.6.0

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

NN_VENV="$REPO/.venv-nnunet"
SP_VENV="$REPO/.venv-synthstroke"
EVAL_PY="$REPO/.venv-eval/bin/python"
SYNTH_REPO="$REPO/work/synthstroke/repo"
SP_WEIGHTS="$REPO/work/synthstroke/weights/synthstroke-synth-plus"
INFER_HELPER="$REPO/baselines/models/synthstroke/synthstroke_helpers/infer_synthstroke.py"
CONVERTER="$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/npz_lesion_prob_to_nii.py"

BRAIN_T1="$REPO/work/soop_bench_t1w_brain/${SUB}_T1w_brain.nii.gz"
REG_DIR="$REPO/work/predictions_nnunet_r30_tmp/$SUB"
AFFINE_MAT="$REG_DIR/${SUB}_t1_to_trace_ref3d_0GenericAffine.mat"
TRACE_REF3D="$REG_DIR/${SUB}_TRACE_ref3d.nii.gz"
for f in "$BRAIN_T1" "$AFFINE_MAT" "$TRACE_REF3D"; do
  if [[ ! -s "$f" ]]; then
    echo "[error] missing cached artifact (run step 10 first): $f" >&2
    exit 2
  fi
done

WORK="$REPO/work/predictions_ood_ens${OOD_TAG:+_$OOD_TAG}"
NN_PROB_TRACE="$WORK/nn_prob_trace"; SP_PROB_TRACE="$WORK/sp_prob_trace"
TMP="$WORK/tmp/$SUB"
mkdir -p "$NN_PROB_TRACE" "$SP_PROB_TRACE" "$TMP"

# ---- nn: nnU-Net 5-fold ensemble probabilities on the brain T1w ----------------
NN_IN="$TMP/nn_in"; NN_OUT="$TMP/nn_out"; NN_PROB_T1="$TMP/nn_prob_t1w"
rm -rf "$NN_IN" "$NN_OUT" "$NN_PROB_T1"; mkdir -p "$NN_IN" "$NN_OUT" "$NN_PROB_T1"
cp "$BRAIN_T1" "$NN_IN/${SUB}_0000.nii.gz"

source "$NN_VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"
NN_TRAINER="${NN_TRAINER:-nnUNetTrainer_250epochs}"
nnUNetv2_predict -i "$NN_IN" -o "$NN_OUT" -d 502 -c 3d_fullres -f 0 1 2 3 4 \
  -tr "$NN_TRAINER" --save_probabilities
if [[ ! -s "$NN_OUT/${SUB}.npz" ]]; then
  echo "[error] nnU-Net did not write $NN_OUT/${SUB}.npz" >&2; exit 2
fi
# npz (class-first, lesion channel 1) -> native T1w-grid probability NIfTI.
python "$CONVERTER" --npz-dir "$NN_OUT" --seg-dir "$NN_OUT" --ref-dir "$NN_OUT" \
  --out-dir "$NN_PROB_T1" --lesion-channel 1
deactivate || true

# ---- sp: synthstroke-synth-plus probabilities on the SAME brain T1w ------------
source "$SP_VENV/bin/activate"
export SYNTHSTROKE_REPO="$SYNTH_REPO"
export PYTHONPATH="$SYNTH_REPO:$REPO:${PYTHONPATH:-}"
SP_PROB_T1="$TMP/sp_prob_t1w.nii.gz"
"$SP_VENV/bin/python" "$INFER_HELPER" \
  --weights "$SP_WEIGHTS" --image "$BRAIN_T1" \
  --out "$TMP/sp_seg_t1w.nii.gz" --save-prob "$SP_PROB_T1" --patch 128
deactivate || true
if [[ ! -s "$SP_PROB_T1" ]]; then
  echo "[error] synth-plus did not write $SP_PROB_T1" >&2; exit 2
fi

# ---- warp both native probabilities into TRACE space (Linear) ------------------
antsApplyTransforms -d 3 -i "$NN_PROB_T1/${SUB}.nii.gz" -r "$TRACE_REF3D" \
  -o "$NN_PROB_TRACE/${SUB}.nii.gz" -n Linear -t "$AFFINE_MAT"
antsApplyTransforms -d 3 -i "$SP_PROB_T1" -r "$TRACE_REF3D" \
  -o "$SP_PROB_TRACE/${SUB}.nii.gz" -n Linear -t "$AFFINE_MAT"

echo "[done] $SUB -> nn/sp TRACE probabilities written"
