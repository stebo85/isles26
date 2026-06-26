#!/bin/bash
#SBATCH --job-name=nnunet_eval_topk10_oof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Evaluate the Dice+TopK10 ATLAS R3.0 5-fold out-of-fold masks with this repo's
# full segmentation harness. The nnU-Net summary only reports overlap/confusion
# metrics; this adds lesion-wise F1, HD95/ASSD, surface Dice, volume errors,
# stratified sections, per-case JSON, and QC images.
#
# Requires step 25 to have produced:
#   nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres/
#     crossval_results_folds_0_1_2_3_4[/postprocessed]/*.nii.gz
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_37_eval_topk10_oof.sh

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
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
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

RESULT_DIR="${TOPK10_RESULT_DIR:-$nnUNet_results/Dataset502_ATLASR30/nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres}"
CV_DIR="$RESULT_DIR/crossval_results_folds_0_1_2_3_4"
if [[ -n "${TOPK10_PRED_DIR:-}" ]]; then
  PRED_DIR="$TOPK10_PRED_DIR"
elif [[ -d "$CV_DIR/postprocessed" ]]; then
  PRED_DIR="$CV_DIR/postprocessed"
else
  PRED_DIR="$CV_DIR"
fi

SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
MANIFEST="$nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv"
PER_SESSION="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
REPORT_DIR="$REPO/baselines/reports/nnunet_topk10_crossval"

for required in "$SPLITS" "$MANIFEST" "$PER_SESSION" "$PRED_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing required input: $required" >&2
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
actual="$(find "$PRED_DIR" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
if [[ "$actual" != "$expected" ]]; then
  echo "[error] TopK10 OOF mask count is $actual, expected $expected in $PRED_DIR" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"
cp -f "$CV_DIR/summary.json" "$REPORT_DIR/crossval_summary.json" 2>/dev/null || true
cp -f "$CV_DIR/postprocessed/summary.json" "$REPORT_DIR/crossval_postprocessed_summary.json" 2>/dev/null || true
cp -f "$nnUNet_results/Dataset502_ATLASR30/inference_instructions.txt" "$REPORT_DIR/" 2>/dev/null || true

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --folds all \
  --pred-dir "$PRED_DIR" \
  --per-session-csv "$PER_SESSION" \
  --out-dir "$REPORT_DIR" \
  --title "nnU-Net R3.0 Dice+TopK10 1000ep OOF CV on ATLAS R3.0" \
  --workers "$EVAL_WORKERS"

cat > "$REPORT_DIR/meta.json" <<'EOF'
{
  "method": "nnunet_r30_topk10_cv",
  "display_name": "nnU-Net R3.0 Dice+TopK10 1000ep, 5-fold CV",
  "model_family": "nnunet",
  "eval_set": "atlas_r30_5fold_cv",
  "input_modalities": ["T1w"],
  "trained_on": "ATLAS R3.0 / ISLES'26 public training release, Dataset502_ATLASR30",
  "status": "complete",
  "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
  "notes": "current best single nnU-Net in-distribution model; rich OOF eval via repo harness"
}
EOF

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"

"$EVAL_PY" - "$REPORT_DIR/report.json" <<'PY'
import json
import sys
report = json.loads(open(sys.argv[1]).read())
metrics = {row["metric"]: row for row in report["overall"]}
print("[done] TopK10 OOF rich evaluation")
for metric in ("dice", "lesion_f1", "surface_dice_3mm", "hd95_mm", "abs_volume_diff_ml"):
    row = metrics.get(metric)
    if row:
        print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
