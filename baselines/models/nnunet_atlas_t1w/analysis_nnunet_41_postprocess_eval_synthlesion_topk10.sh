#!/bin/bash
#SBATCH --job-name=nnunet_eval_synthlesion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Postprocess and evaluate the Dice+TopK10 synthetic-lesion nnU-Net run.
# Requires analysis_nnunet_40_train_synthlesion_topk10.sh to have completed all
# 5 folds for the selected trainer.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

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
export nnUNet_compile=f
export nnUNet_raw="$REPO/work/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/work/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/work/nnunet/nnUNet_results"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TRAINER="${SYNTHLESION_TRAINER:-nnUNetTrainerDiceTopK10SynthLesion}"
RESULT_DIR="$nnUNet_results/Dataset502_ATLASR30/${TRAINER}__nnUNetPlans__3d_fullres"
CV_DIR="$RESULT_DIR/crossval_results_folds_0_1_2_3_4"
REPORT_DIR="$REPO/baselines/reports/nnunet_synthlesion_topk10_crossval"
EVAL_PY="$REPO/.venv-eval/bin/python"

if [[ ! -x "$EVAL_PY" ]]; then
  echo "[error] missing .venv-eval; run eval environment setup first" >&2
  exit 2
fi

VENV="$REPO/.venv-nnunet"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] missing $VENV; run analysis_nnunet_01_prepare_dataset.sh first" >&2
  exit 2
fi
source "$VENV/bin/activate"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py"
python "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/install_trainer_variant.py" \
  --source "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/nnUNetTrainerSyntheticLesion.py"

nnUNetv2_find_best_configuration 502 \
  -c 3d_fullres \
  -tr "$TRAINER" \
  -p nnUNetPlans \
  --disable_ensembling \
  -np "${SLURM_CPUS_PER_TASK:-8}"

if [[ -d "$CV_DIR/postprocessed" ]]; then
  PRED_DIR="$CV_DIR/postprocessed"
else
  PRED_DIR="$CV_DIR"
fi

SPLITS="$nnUNet_preprocessed/Dataset502_ATLASR30/splits_final.json"
MANIFEST="$nnUNet_raw/Dataset502_ATLASR30/conversion_manifest.csv"
PER_SESSION="$REPO/baselines/reports/training_data_characterization_r30/per_session.csv"
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
  echo "[error] synthetic-lesion OOF mask count is $actual, expected $expected in $PRED_DIR" >&2
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
  --title "nn-U-Net R3.0 Dice+TopK10 + synthetic lesion insertion OOF CV on ATLAS R3.0" \
  --workers "${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

cat > "$REPORT_DIR/meta.json" <<EOF
{
  "method": "nnunet_r30_topk10_synthlesion_cv",
  "display_name": "nnU-Net R3.0 Dice+TopK10 + synthetic lesion insertion, 5-fold CV",
  "model_family": "nnunet",
  "eval_set": "atlas_r30_5fold_cv",
  "input_modalities": ["T1w"],
  "trained_on": "ATLAS R3.0 / ISLES'26 public training release, Dataset502_ATLASR30",
  "status": "complete",
  "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
  "notes": "Dice+TopK10 baseline with size-rebalanced synthetic lesion insertion augmentation targeting tiny/small-lesion recall"
}
EOF

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"

"$EVAL_PY" - "$REPORT_DIR/report.json" <<'PY'
import json
import sys
report = json.loads(open(sys.argv[1]).read())
metrics = {row["metric"]: row for row in report["overall"]}
print("[done] synthetic-lesion TopK10 OOF rich evaluation")
for metric in ("dice", "lesion_f1", "surface_dice_3mm", "hd95_mm", "abs_volume_diff_ml"):
    row = metrics.get(metric)
    if row:
        print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
