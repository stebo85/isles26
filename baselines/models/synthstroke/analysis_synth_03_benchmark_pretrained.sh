#!/bin/bash
#SBATCH --job-name=synth_pretrained_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6|GPU_CC:8.0|GPU_CC:7.5|GPU_CC:7.0"
#SBATCH --array=0-5
#SBATCH -p owners

# Benchmark one pretrained SynthStroke variant per SLURM array task on:
#   A) ATLAS R2.1 fold-0 validation T1w images
#   B) soop_bench native TRACE/DWI images
#   C) soop_bench T1w images already warped into TRACE space by the nnU-Net run
#
# Submit after setup + weight download have completed:
#   sbatch baselines/models/synthstroke/analysis_synth_03_benchmark_pretrained.sh

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/work/synthstroke/pred" "$REPO/baselines/reports/synthstroke"

VARIANTS=(
  synthstroke-baseline
  synthstroke-synth
  synthstroke-synth-pseudo
  synthstroke-synth-plus
  synthstroke-qatlas
  synthstroke-qsynth
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be a non-negative integer, got: $TASK_ID" >&2
  exit 2
fi
if (( TASK_ID >= ${#VARIANTS[@]} )); then
  echo "[error] SLURM_ARRAY_TASK_ID=$TASK_ID outside valid range 0-$((${#VARIANTS[@]} - 1))" >&2
  exit 2
fi
VARIANT="${VARIANTS[$TASK_ID]}"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load devel
module load python/3.12.1
module load py-pytorch/2.4.1_py312
module load freesurfer/8.1.0
module load ants/2.6.0

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline

SYNTH_REPO="$REPO/work/synthstroke/repo"
WEIGHTS="$REPO/work/synthstroke/weights/$VARIANT"
VENV="$REPO/.venv-synthstroke"
INFER_HELPER="$REPO/baselines/models/synthstroke/synthstroke_helpers/infer_synthstroke.py"
EVAL_PY="$REPO/.venv-eval/bin/python"

if [[ ! -d "$SYNTH_REPO" ]]; then
  echo "[error] missing vendored SynthStroke repo: $SYNTH_REPO" >&2
  echo "[error] run baselines/models/synthstroke/analysis_synth_01_setup_env.sh first" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run baselines/models/synthstroke/analysis_synth_01_setup_env.sh first" >&2
  exit 2
fi
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing $EVAL_PY; build the eval venv first" >&2
  exit 2
fi
if [[ ! -s "$WEIGHTS/model.safetensors" || ! -s "$WEIGHTS/config.json" ]]; then
  echo "[error] missing local weights for $VARIANT under $WEIGHTS" >&2
  echo "[error] expected model.safetensors and config.json; run analysis_synth_02_download_weights.sh first" >&2
  exit 2
fi
if [[ ! -s "$INFER_HELPER" ]]; then
  echo "[error] missing inference helper: $INFER_HELPER" >&2
  exit 2
fi

source "$VENV/bin/activate"
INFER_PY="$VENV/bin/python"
export SYNTHSTROKE_REPO="$SYNTH_REPO"
export PYTHONPATH="$SYNTH_REPO:$REPO:${PYTHONPATH:-}"

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_ID}"
INFER_RESULT="unknown"

section() {
  printf '\n[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

tmp_nifti_path() {
  local out="$1"
  if [[ "$out" == *.nii.gz ]]; then
    printf '%s.tmp.%s.%s.nii.gz' "${out%.nii.gz}" "$JOB_TAG" "$$"
  elif [[ "$out" == *.nii ]]; then
    printf '%s.tmp.%s.%s.nii' "${out%.nii}" "$JOB_TAG" "$$"
  else
    printf '%s.tmp.%s.%s.nii.gz' "$out" "$JOB_TAG" "$$"
  fi
}

infer_one() {
  local image="$1"
  local out="$2"
  local label="$3"

  INFER_RESULT="unknown"
  if [[ -s "$out" ]]; then
    echo "[skip] $label already has a non-empty prediction: $out"
    INFER_RESULT="skipped"
    return 0
  fi
  if [[ ! -s "$image" ]]; then
    echo "[error] missing input for $label: $image" >&2
    exit 2
  fi

  mkdir -p "$(dirname "$out")"
  local tmp
  tmp="$(tmp_nifti_path "$out")"
  rm -f "$tmp"
  "$INFER_PY" "$INFER_HELPER" \
    --weights "$WEIGHTS" \
    --image "$image" \
    --out "$tmp" \
    --patch 128 \
    --device auto \
    --repo "$SYNTH_REPO"
  if [[ ! -s "$tmp" ]]; then
    echo "[error] SynthStroke did not create a non-empty output for $label: $tmp" >&2
    exit 2
  fi
  mv "$tmp" "$out"
  echo "[done] $label -> $out"
  INFER_RESULT="ran"
}

prepare_trace_3d() {
  local src="$1"
  local dst="$2"
  local subject="$3"

  if [[ -s "$dst" ]]; then
    echo "[ok] 3-D TRACE input exists for $subject: $dst"
    return 0
  fi
  if [[ ! -s "$src" ]]; then
    echo "[error] missing TRACE input for $subject: $src" >&2
    exit 2
  fi

  mkdir -p "$(dirname "$dst")"
  "$EVAL_PY" - "$src" "$dst" "$subject" <<'PY'
from pathlib import Path
import os
import sys

import nibabel as nib
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
subject = sys.argv[3]

img = nib.load(str(src))
data = np.asanyarray(img.dataobj)
original_shape = tuple(data.shape)
while data.ndim > 3 and data.shape[-1] == 1:
    data = data[..., 0]
if data.ndim != 3:
    raise SystemExit(f"{src} is not 3-D after trailing singleton squeeze: shape={original_shape}")

# Keep the native TRACE affine/grid. This is only a 4-D singleton squeeze, not a
# registration or reorientation step.
out = nib.Nifti1Image(data, img.affine, img.header.copy())
out.set_data_dtype(data.dtype)
tmp = dst.with_name(dst.name.replace(".nii.gz", f".tmp.{os.getpid()}.nii.gz"))
nib.save(out, str(tmp))
if not tmp.exists() or tmp.stat().st_size == 0:
    raise SystemExit(f"failed to write non-empty 3-D TRACE image: {tmp}")
os.replace(tmp, dst)
print(f"[done] {subject}: TRACE {original_shape} -> {tuple(data.shape)} at {dst}")
PY
}

evaluate_atlas_val() {
  local pred_dir="$1"
  local out_dir="$2"

  mkdir -p "$out_dir"
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" \
    baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py \
    --manifest work/nnunet/nnUNet_raw/Dataset501_ATLASR21/conversion_manifest.csv \
    --splits work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json \
    --pred-dir "$pred_dir" \
    --per-session-csv baselines/reports/training_data_characterization/per_session.csv \
    --out-dir "$out_dir" \
    --title "SynthStroke $VARIANT on ATLAS R2.1 fold-0 val"
}

evaluate_soop() {
  local pred_dir="$1"
  local out_dir="$2"
  local title="$3"

  mkdir -p "$out_dir"
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$EVAL_PY" -m eval.run_eval \
    --data-root data/soop_bench \
    --pred-dir "$pred_dir" \
    --out-dir "$out_dir" \
    --title "$title"
}

section "SynthStroke pretrained benchmark: variant=$VARIANT task=$TASK_ID"
echo "[info] weights=$WEIGHTS"
echo "[info] synthstroke_repo=$SYNTH_REPO"

# ---------------------------------------------------------------------------
# A. ATLAS fold-0 validation set, T1w input.
# ---------------------------------------------------------------------------
section "A) ATLAS R2.1 fold-0 validation"
SPLITS="$REPO/work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json"
ATLAS_IMAGES="$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr"
ATLAS_PRED_DIR="$REPO/work/synthstroke/pred/atlas_val/$VARIANT"
ATLAS_REPORT_DIR="baselines/reports/synthstroke/$VARIANT/atlas_val"

if [[ ! -s "$SPLITS" ]]; then
  echo "[error] missing ATLAS fold splits: $SPLITS" >&2
  exit 2
fi
if [[ ! -d "$ATLAS_IMAGES" ]]; then
  echo "[error] missing ATLAS imagesTr directory: $ATLAS_IMAGES" >&2
  exit 2
fi
mkdir -p "$ATLAS_PRED_DIR"

mapfile -t VAL_CASES < <("$EVAL_PY" - "$SPLITS" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
folds = data.get("folds", data) if isinstance(data, dict) else data
try:
    val_cases = list(folds[0]["val"])
except Exception as exc:
    raise SystemExit(f"could not read fold 0 val cases from {sys.argv[1]}: {exc}")
if not val_cases:
    raise SystemExit(f"fold 0 validation list is empty in {sys.argv[1]}")
for case_id in val_cases:
    if not isinstance(case_id, str) or not case_id:
        raise SystemExit(f"invalid case id in fold 0 validation list: {case_id!r}")
    print(case_id)
PY
)

atlas_ran=0
atlas_skipped=0
for case_id in "${VAL_CASES[@]}"; do
  infer_one \
    "$ATLAS_IMAGES/${case_id}_0000.nii.gz" \
    "$ATLAS_PRED_DIR/${case_id}.nii.gz" \
    "ATLAS $case_id"
  if [[ "$INFER_RESULT" == "ran" ]]; then
    ((atlas_ran += 1))
  else
    ((atlas_skipped += 1))
  fi
done
echo "[summary] ATLAS predictions for $VARIANT: ran=$atlas_ran skipped=$atlas_skipped total=${#VAL_CASES[@]}"
evaluate_atlas_val "$ATLAS_PRED_DIR" "$ATLAS_REPORT_DIR"
echo "[summary] ATLAS evaluation written to $ATLAS_REPORT_DIR"

# ---------------------------------------------------------------------------
# B. soop_bench native TRACE input. Predictions stay in TRACE physical space.
# ---------------------------------------------------------------------------
section "B) soop_bench native TRACE"
SOOP_ROOT="$REPO/data/soop_bench"
TRACE_TMP_DIR="$REPO/work/synthstroke/tmp/soop_trace_3d"
TRACE_PRED_DIR="$REPO/work/synthstroke/pred/soop_trace/$VARIANT"
TRACE_REPORT_DIR="baselines/reports/synthstroke/$VARIANT/soop_trace"
mkdir -p "$TRACE_TMP_DIR" "$TRACE_PRED_DIR"

mapfile -t SUBJECTS < <(
  find "$SOOP_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print \
    | while read -r sub_dir; do
        if [[ -d "$sub_dir/dwi" ]]; then
          basename "$sub_dir"
        fi
      done \
    | sort -V
)
if (( ${#SUBJECTS[@]} == 0 )); then
  echo "[error] found no soop_bench sub-* directories with dwi/ under $SOOP_ROOT" >&2
  exit 2
fi
echo "[info] found ${#SUBJECTS[@]} soop_bench subject(s): ${SUBJECTS[*]}"

trace_ran=0
trace_skipped=0
for sub in "${SUBJECTS[@]}"; do
  trace_src="$SOOP_ROOT/$sub/dwi/${sub}_rec-TRACE_dwi.nii.gz"
  trace_3d="$TRACE_TMP_DIR/${sub}_rec-TRACE_dwi_3d.nii.gz"
  prepare_trace_3d "$trace_src" "$trace_3d" "$sub"
  infer_one \
    "$trace_3d" \
    "$TRACE_PRED_DIR/${sub}.nii.gz" \
    "soop TRACE $sub"
  if [[ "$INFER_RESULT" == "ran" ]]; then
    ((trace_ran += 1))
  else
    ((trace_skipped += 1))
  fi
done
echo "[summary] soop TRACE predictions for $VARIANT: ran=$trace_ran skipped=$trace_skipped total=${#SUBJECTS[@]}"
evaluate_soop "$TRACE_PRED_DIR" "$TRACE_REPORT_DIR" "SynthStroke $VARIANT on soop_bench (TRACE)"
echo "[summary] soop TRACE evaluation written to $TRACE_REPORT_DIR"

# ---------------------------------------------------------------------------
# C. soop_bench T1w already warped into TRACE space by the nnU-Net pipeline.
# ---------------------------------------------------------------------------
section "C) soop_bench T1w already warped into TRACE"
T1W_PRED_DIR="$REPO/work/synthstroke/pred/soop_t1w/$VARIANT"
T1W_REPORT_DIR="baselines/reports/synthstroke/$VARIANT/soop_t1w"
mkdir -p "$T1W_PRED_DIR"

t1w_ran=0
t1w_skipped=0
t1w_missing=0
for sub in "${SUBJECTS[@]}"; do
  t1_in_trace="$REPO/work/predictions_nnunet_tmp/$sub/${sub}_t1_in_trace.nii.gz"
  if [[ ! -s "$t1_in_trace" ]]; then
    echo "[warn] skipping soop T1w path for $sub; missing $t1_in_trace"
    ((t1w_missing += 1))
    continue
  fi
  infer_one \
    "$t1_in_trace" \
    "$T1W_PRED_DIR/${sub}.nii.gz" \
    "soop T1w-in-TRACE $sub"
  if [[ "$INFER_RESULT" == "ran" ]]; then
    ((t1w_ran += 1))
  else
    ((t1w_skipped += 1))
  fi
done
echo "[summary] soop T1w-in-TRACE predictions for $VARIANT: ran=$t1w_ran skipped=$t1w_skipped missing_inputs=$t1w_missing total=${#SUBJECTS[@]}"
evaluate_soop "$T1W_PRED_DIR" "$T1W_REPORT_DIR" "SynthStroke $VARIANT on soop_bench (T1w in TRACE)"
echo "[summary] soop T1w-in-TRACE evaluation written to $T1W_REPORT_DIR"

section "Done: variant=$VARIANT"
