#!/bin/bash
#SBATCH --job-name=nnunet_map3_thr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Sweep threshold over the saved MAPPING3 out-of-fold lesion-probability
# NIfTIs from step 36. This does not rerun nnU-Net inference or recompute the
# softmax ensemble; it thresholds existing fold_*/probabilities/*.nii.gz files.
#
#   bash baselines/models/nnunet_atlas_t1w/analysis_nnunet_39_sweep_mapping3_oof_threshold.sh
#
# Tunables:
#   MAPPING3_THRESHOLDS=0.35:0.55:0.01
#   MAPPING3_SWEEP_PRIMARY_SCORE=lesion_f1_mean
#     choices: lesion_f1_mean, lesion_f1_pooled, dice_lesion_f1_hmean

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONNOUSERSITE=1
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"

EVAL_PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing .venv-eval; run eval environment setup first" >&2
  exit 2
fi

SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
MANIFEST="$nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv"
PER_SESSION="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
for required in "$SPLITS" "$MANIFEST" "$PER_SESSION"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] missing required input: $required" >&2
    exit 2
  fi
done

MAPPING3_WORK="${MAPPING3_WORK:-$REPO/work/nnunet_mapping3_oof_ensemble}"
PROB_DIRS=(
  "$MAPPING3_WORK/folds/fold_0/probabilities"
  "$MAPPING3_WORK/folds/fold_1/probabilities"
  "$MAPPING3_WORK/folds/fold_2/probabilities"
  "$MAPPING3_WORK/folds/fold_3/probabilities"
  "$MAPPING3_WORK/folds/fold_4/probabilities"
)
for prob_dir in "${PROB_DIRS[@]}"; do
  if [[ ! -d "$prob_dir" ]]; then
    echo "[error] missing probability directory: $prob_dir" >&2
    echo "[hint] run step 36 with --save-prob-nifti first" >&2
    exit 2
  fi
done

expected="$("$EVAL_PY" - "$SPLITS" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1]).read())
folds = payload.get("folds", payload) if isinstance(payload, dict) else payload
print(sum(len(fold["val"]) for fold in folds))
PY
)"
actual=0
for prob_dir in "${PROB_DIRS[@]}"; do
  n="$(find "$prob_dir" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
  actual=$((actual + n))
done
if [[ "$actual" != "$expected" ]]; then
  echo "[error] probability NIfTI count is $actual, expected $expected under $MAPPING3_WORK/folds" >&2
  exit 1
fi

THRESHOLDS="${MAPPING3_THRESHOLDS:-0.35:0.55:0.01}"
PRIMARY_SCORE="${MAPPING3_SWEEP_PRIMARY_SCORE:-lesion_f1_mean}"
SWEEP_WORK="${MAPPING3_SWEEP_WORK:-$REPO/work/nnunet_mapping3_oof_threshold_sweep}"
REPORT_DIR="${MAPPING3_SWEEP_REPORT_DIR:-$REPO/baselines/reports/nnunet_mapping3_oof_ensemble_threshold_tuned}"
SWEEP_WORKERS="${SWEEP_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

rm -rf "$SWEEP_WORK"
mkdir -p "$SWEEP_WORK" "$REPORT_DIR"

prob_args=()
for prob_dir in "${PROB_DIRS[@]}"; do
  prob_args+=(--prob-dir "$prob_dir")
done

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/sweep_prob_thresholds.py" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --folds all \
  "${prob_args[@]}" \
  --out-dir "$SWEEP_WORK" \
  --thresholds "$THRESHOLDS" \
  --primary-score "$PRIMARY_SCORE" \
  --workers "$SWEEP_WORKERS" \
  --write-best-masks \
  --best-mask-dir "$SWEEP_WORK/best_masks"

BEST_THRESHOLD="$(tr -d '[:space:]' < "$SWEEP_WORK/best_threshold.txt")"

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --folds all \
  --pred-dir "$SWEEP_WORK/best_masks" \
  --per-session-csv "$PER_SESSION" \
  --out-dir "$REPORT_DIR" \
  --title "MAPPING 3-family softmax ensemble OOF CV on ATLAS R3.0 (threshold $BEST_THRESHOLD)" \
  --workers "$EVAL_WORKERS"

cp -f "$SWEEP_WORK"/sweep_summary.{csv,json,md} "$REPORT_DIR"/
cp -f "$SWEEP_WORK/best_threshold.txt" "$REPORT_DIR"/

"$EVAL_PY" - "$REPORT_DIR/meta.json" "$BEST_THRESHOLD" "$THRESHOLDS" "$PRIMARY_SCORE" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
best_threshold = sys.argv[2]
thresholds = sys.argv[3]
primary_score = sys.argv[4]
payload = {
    "method": "nnunet_mapping3_oof_ensemble_threshold_tuned",
    "display_name": f"MAPPING 3-family softmax ensemble OOF CV, threshold {best_threshold}",
    "model_family": "ensemble",
    "eval_set": "atlas_r30_5fold_cv",
    "input_modalities": ["T1w"],
    "trained_on": "ATLAS R3.0 / Dataset502_ATLASR30; OOF average of TopK10 1000ep, default 1000ep, and ResEnc-L 250ep validation softmax",
    "status": "complete",
    "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
    "notes": (
        "Threshold selected by sweeping saved OOF probability NIfTIs only; "
        f"thresholds={thresholds}; primary_score={primary_score}; no re-inference"
    ),
}
out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"

"$EVAL_PY" - "$REPORT_DIR/report.json" "$REPORT_DIR/sweep_summary.json" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
sweep = json.loads(open(sys.argv[2]).read())
metrics = {row["metric"]: row for row in report["overall"]}
print("[done] MAPPING3 OOF threshold sweep")
print(f"selected_threshold: {sweep['selected_threshold']}")
for metric in ("dice", "lesion_f1", "lesion_recall", "lesion_precision", "surface_dice_3mm", "hd95_mm"):
    row = metrics.get(metric)
    if row:
        print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
