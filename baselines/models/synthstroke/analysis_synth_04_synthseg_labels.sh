#!/bin/bash
#SBATCH --job-name=synth_synthseg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-19
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -p owners

# Phase B / Step 4: generate FreeSurfer SynthSeg anatomical tissue labels for the
# 955 ATLAS R2.1 T1w images. These feed the SynthStroke-style synthetic generator
# (the lesion class is overlaid later in step 05). mri_synthseg is contrast-
# agnostic and CPU-friendly; we run it in batch mode (model loads once) over a
# round-robin chunk of cases per SLURM array task.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

module purge
if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
  module use "$GROUP_HOME/modules"
fi
module load freesurfer/8.1.0

NCHUNKS=20
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

IN_DIR="$REPO/work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr"
OUT_DIR="$REPO/work/synthstroke/synthseg_labels"
mkdir -p "$OUT_DIR"

if [[ ! -d "$IN_DIR" ]]; then
  echo "[error] missing ATLAS imagesTr dir: $IN_DIR" >&2
  exit 2
fi

# Deterministic sorted list of all CASE ids (strip the _0000.nii.gz channel suffix).
mapfile -t ALL_CASES < <(
  find "$IN_DIR" -maxdepth 1 -name '*_0000.nii.gz' -printf '%f\n' \
    | sed 's/_0000\.nii\.gz$//' | sort
)
NTOTAL=${#ALL_CASES[@]}
if (( NTOTAL == 0 )); then
  echo "[error] no *_0000.nii.gz found in $IN_DIR" >&2
  exit 2
fi
echo "[info] task $TASK_ID/$((NCHUNKS-1)): $NTOTAL total ATLAS cases"

# mri_synthseg's --i/--o take a single image OR a directory (NOT a textfile list).
# We stage this chunk's not-yet-done cases as symlinks named <CASE>.nii.gz in a
# temp input dir, run SynthSeg once (single model load) dir->dir, then move the
# outputs into OUT_DIR.
STAGE_IN="$(mktemp -d "${TMPDIR:-/tmp}/synthseg_in_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.XXXXXX")"
STAGE_OUT="$(mktemp -d "${TMPDIR:-/tmp}/synthseg_out_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.XXXXXX")"
trap 'rm -rf "$STAGE_IN" "$STAGE_OUT"' EXIT

n_chunk=0
n_todo=0
n_skip=0
TODO_CASES=()
for i in "${!ALL_CASES[@]}"; do
  if (( i % NCHUNKS != TASK_ID )); then
    continue
  fi
  n_chunk=$((n_chunk + 1))
  case_id="${ALL_CASES[$i]}"
  if [[ -s "$OUT_DIR/${case_id}.nii.gz" ]]; then
    n_skip=$((n_skip + 1))
    continue
  fi
  ln -sf "$IN_DIR/${case_id}_0000.nii.gz" "$STAGE_IN/${case_id}.nii.gz"
  TODO_CASES+=("$case_id")
  n_todo=$((n_todo + 1))
done

echo "[info] chunk size=$n_chunk already_done=$n_skip to_segment=$n_todo"
if (( n_todo == 0 )); then
  echo "[done] task $TASK_ID has nothing to do"
  exit 0
fi

# Batch SynthSeg over the staged directory (loads the model once).
mri_synthseg --i "$STAGE_IN" --o "$STAGE_OUT" --threads "${SLURM_CPUS_PER_TASK:-4}"

# Move outputs into OUT_DIR. SynthSeg writes one output per input; match by the
# CASE id prefix to tolerate any filename suffix it may add.
missing=0
for case_id in "${TODO_CASES[@]}"; do
  produced="$(find "$STAGE_OUT" -maxdepth 1 -type f -name "${case_id}*.nii.gz" | head -1)"
  if [[ -z "$produced" || ! -s "$produced" ]]; then
    echo "[error] synthseg produced no output for $case_id" >&2
    missing=$((missing + 1))
    continue
  fi
  mv -f "$produced" "$OUT_DIR/${case_id}.nii.gz"
done
if (( missing > 0 )); then
  echo "[error] task $TASK_ID: $missing missing outputs" >&2
  exit 3
fi

echo "[done] task $TASK_ID: segmented=$n_todo skipped=$n_skip chunk=$n_chunk"
