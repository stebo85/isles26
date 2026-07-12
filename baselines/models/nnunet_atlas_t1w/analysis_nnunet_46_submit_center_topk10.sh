#!/bin/bash

# Submit the corrected center-grouped TopK experiment as one afterok pipeline.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
MODEL_DIR="$REPO/baselines/models/nnunet_atlas_t1w"
REPORT_DIR="$REPO/baselines/reports/nnunet_center_grouped_splits"
PREP_SCRIPT="$MODEL_DIR/analysis_nnunet_42_prepare_center_grouped.sh"
TRAIN_SCRIPT="$MODEL_DIR/analysis_nnunet_43_train_center_topk10.sh"
POSTPROCESS_SCRIPT="$MODEL_DIR/analysis_nnunet_44_postprocess_center_topk10.sh"
EVAL_SCRIPT="$MODEL_DIR/analysis_nnunet_45_eval_center_topk10_oof.sh"

cd "$REPO"
mkdir -p "$REPO/logs"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[error] sbatch is unavailable; run this wrapper on a SLURM login node" >&2
  exit 2
fi

for required in "$PREP_SCRIPT" "$TRAIN_SCRIPT" "$POSTPROCESS_SCRIPT" "$EVAL_SCRIPT" \
  "$REPORT_DIR/splits_final.json" "$REPORT_DIR/splits_summary.json"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing pipeline input: $required" >&2
    exit 2
  fi
done

for script in "$PREP_SCRIPT" "$TRAIN_SCRIPT" "$POSTPROCESS_SCRIPT" "$EVAL_SCRIPT"; do
  bash -n "$script"
done

SUBMITTED=()
PIPELINE_COMPLETE=0
rollback_submissions() {
  local status=$?
  trap - EXIT
  if (( status != 0 || PIPELINE_COMPLETE == 0 )) && (( ${#SUBMITTED[@]} > 0 )); then
    echo "[error] pipeline submission failed; cancelling jobs submitted by this invocation: ${SUBMITTED[*]}" >&2
    scancel "${SUBMITTED[@]}" || \
      echo "[warn] automatic cancellation failed; inspect jobs: ${SUBMITTED[*]}" >&2
  fi
  exit "$status"
}
trap rollback_submissions EXIT

PREP_RAW="$(sbatch --parsable "$PREP_SCRIPT")"
PREP_JOB="${PREP_RAW%%;*}"
if ! [[ "$PREP_JOB" =~ ^[0-9]+$ ]]; then
  echo "[error] sbatch returned an invalid preparation job ID: $PREP_RAW" >&2
  exit 2
fi
SUBMITTED+=("$PREP_JOB")

TRAIN_RAW="$(sbatch --parsable --dependency="afterok:$PREP_JOB" "$TRAIN_SCRIPT")"
TRAIN_JOB="${TRAIN_RAW%%;*}"
if ! [[ "$TRAIN_JOB" =~ ^[0-9]+$ ]]; then
  echo "[error] sbatch returned an invalid training job ID: $TRAIN_RAW" >&2
  exit 2
fi
SUBMITTED+=("$TRAIN_JOB")

POSTPROCESS_RAW="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" "$POSTPROCESS_SCRIPT")"
POSTPROCESS_JOB="${POSTPROCESS_RAW%%;*}"
if ! [[ "$POSTPROCESS_JOB" =~ ^[0-9]+$ ]]; then
  echo "[error] sbatch returned an invalid postprocessing job ID: $POSTPROCESS_RAW" >&2
  exit 2
fi
SUBMITTED+=("$POSTPROCESS_JOB")

EVAL_RAW="$(sbatch --parsable --dependency="afterok:$POSTPROCESS_JOB" "$EVAL_SCRIPT")"
EVAL_JOB="${EVAL_RAW%%;*}"
if ! [[ "$EVAL_JOB" =~ ^[0-9]+$ ]]; then
  echo "[error] sbatch returned an invalid evaluation job ID: $EVAL_RAW" >&2
  exit 2
fi
SUBMITTED+=("$EVAL_JOB")

PIPELINE_COMPLETE=1
printf '[submitted] prepare:    %s\n' "$PREP_JOB"
printf '[submitted] train 0-4:  %s\n' "$TRAIN_JOB"
printf '[submitted] postprocess: %s\n' "$POSTPROCESS_JOB"
printf '[submitted] evaluate:    %s\n' "$EVAL_JOB"
printf '[pipeline] %s -> %s_[0-4] -> %s -> %s\n' \
  "$PREP_JOB" "$TRAIN_JOB" "$POSTPROCESS_JOB" "$EVAL_JOB"
