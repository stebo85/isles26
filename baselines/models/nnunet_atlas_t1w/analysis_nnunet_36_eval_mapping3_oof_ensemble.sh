#!/bin/bash
#SBATCH --job-name=nnunet_eval_map3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Average the 3 MAPPING-family validation softmax sets fold-by-fold and evaluate
# the concatenated 1450-case out-of-fold masks on ATLAS R3.0.
#
# Requires step 35 to have produced per-fold validation .npz files for:
#   TopK10 + default 1000ep + ResEnc-L.
#
#   sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_36_eval_mapping3_oof_ensemble.sh

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

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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

TOPK_DIR="$nnUNet_results/Dataset502_ATLASR30/nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres"
DEFAULT_DIR="$nnUNet_results/Dataset502_ATLASR30/nnUNetTrainer__nnUNetPlans__3d_fullres"
RESENC_DIR="$nnUNet_results/Dataset502_ATLASR30/nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans__3d_fullres"

WORK="${MAPPING3_WORK:-$REPO/work/nnunet_mapping3_oof_ensemble}"
OOF_DIR="$WORK/oof_masks"
FOLD_WORK="$WORK/folds"
REPORT_DIR="$REPO/baselines/reports/nnunet_mapping3_oof_ensemble"
rm -rf "$OOF_DIR" "$FOLD_WORK"
mkdir -p "$OOF_DIR" "$FOLD_WORK" "$REPORT_DIR"

expected_for_fold() {
  "$EVAL_PY" - "$SPLITS" "$1" <<'PY'
import json
import sys
splits = json.loads(open(sys.argv[1]).read())
print(len(splits[int(sys.argv[2])]["val"]))
PY
}

check_member_npz() {
  local name="$1"
  local dir="$2"
  local fold="$3"
  local expected="$4"
  local val_dir="$dir/fold_${fold}/validation"
  local actual
  actual="$(find "$val_dir" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | awk 'END {print NR}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[error] $name fold $fold has $actual npz files, expected $expected in $val_dir" >&2
    echo "[hint] run: sbatch baselines/models/nnunet_atlas_t1w/analysis_nnunet_35_val_npz_mapping3.sh" >&2
    exit 2
  fi
}

for fold in 0 1 2 3 4; do
  expected="$(expected_for_fold "$fold")"
  check_member_npz topk10 "$TOPK_DIR" "$fold" "$expected"
  check_member_npz default1000 "$DEFAULT_DIR" "$fold" "$expected"
  check_member_npz resenc_l "$RESENC_DIR" "$fold" "$expected"

  fold_out="$FOLD_WORK/fold_${fold}"
  "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/average_softmax_ensemble.py" \
    --member topk10 "$TOPK_DIR/fold_${fold}/validation" \
    --member default1000 "$DEFAULT_DIR/fold_${fold}/validation" \
    --member resenc_l "$RESENC_DIR/fold_${fold}/validation" \
    --out-dir "$fold_out" \
    --threshold "${MAPPING3_THRESHOLD:-0.5}" \
    --decision "${MAPPING3_DECISION:-threshold}" \
    --save-prob-nifti

  copied="$(find "$fold_out" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
  if [[ "$copied" != "$expected" ]]; then
    echo "[error] fold $fold ensemble wrote $copied masks, expected $expected" >&2
    exit 1
  fi
  cp "$fold_out"/ATLAS_*.nii.gz "$OOF_DIR"/
done

total="$(find "$OOF_DIR" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
if [[ "$total" != "1450" ]]; then
  echo "[error] OOF mask count is $total, expected 1450" >&2
  exit 1
fi

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py" \
  --manifest "$MANIFEST" \
  --splits "$SPLITS" \
  --folds all \
  --pred-dir "$OOF_DIR" \
  --per-session-csv "$PER_SESSION" \
  --out-dir "$REPORT_DIR" \
  --title "MAPPING 3-family softmax ensemble OOF CV on ATLAS R3.0"

cat > "$REPORT_DIR/meta.json" <<'EOF'
{
  "method": "nnunet_mapping3_oof_ensemble",
  "display_name": "MAPPING 3-family softmax ensemble OOF CV",
  "model_family": "ensemble",
  "eval_set": "atlas_r30_5fold_cv",
  "input_modalities": ["T1w"],
  "trained_on": "ATLAS R3.0 / Dataset502_ATLASR30; OOF average of TopK10 1000ep, default 1000ep, and ResEnc-L 250ep validation softmax",
  "status": "complete",
  "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
  "notes": "MAPPING-style 3-family per-fold validation softmax average; threshold 0.5; no Dataset504/self-training member"
}
EOF

cp -f "$WORK"/folds/fold_*/ensemble_manifest.csv "$REPORT_DIR"/ 2>/dev/null || true
PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"

"$EVAL_PY" - "$REPORT_DIR/report.json" <<'PY'
import json
import sys
report = json.loads(open(sys.argv[1]).read())
metrics = {row["metric"]: row for row in report["overall"]}
print("[done] mapping3 OOF ensemble")
for metric in ("dice", "lesion_f1", "hd95_mm"):
    row = metrics.get(metric)
    if row:
        print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
