#!/bin/bash
#SBATCH --job-name=nnunet_eval_map3_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p owners

# Evaluate the single-family OOF argmax masks that correspond exactly to the
# step-35 validation softmax inputs used by the 3-family MAPPING ensemble:
#   TopK10, default 1000ep, and ResEnc-L.
#
# This is CPU-only. For each family/fold it verifies the validation .npz
# softmax files exist, then uses the corresponding validation argmax NIfTIs
# produced by the same nnU-Net validation run. Set MAPPING3_SINGLE_REARGMAX_NPZ=1
# to regenerate those masks from .npz with average_softmax_ensemble.py
# --decision argmax.
#
#   bash baselines/models/nnunet_atlas_t1w/analysis_nnunet_38_eval_mapping3_single_oof.sh

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

WORK="${MAPPING3_SINGLE_WORK:-$REPO/work/nnunet_mapping3_single_family_oof}"
EVAL_WORKERS="${EVAL_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

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

check_member_masks() {
  local name="$1"
  local dir="$2"
  local fold="$3"
  local expected="$4"
  local val_dir="$dir/fold_${fold}/validation"
  local actual
  actual="$(find "$val_dir" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f 2>/dev/null | awk 'END {print NR}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[error] $name fold $fold has $actual argmax masks, expected $expected in $val_dir" >&2
    exit 2
  fi
}

evaluate_family() {
  local name="$1"
  local result_dir="$2"
  local report_dir="$3"
  local method="$4"
  local display_name="$5"
  local notes="$6"

  local oof_dir="$WORK/$name/oof_masks"
  local fold_work="$WORK/$name/folds"
  rm -rf "$oof_dir" "$fold_work"
  mkdir -p "$oof_dir" "$fold_work" "$report_dir"

  for fold in 0 1 2 3 4; do
    local expected fold_out copied
    expected="$(expected_for_fold "$fold")"
    check_member_npz "$name" "$result_dir" "$fold" "$expected"
    check_member_masks "$name" "$result_dir" "$fold" "$expected"

    fold_out="$fold_work/fold_${fold}"
    if [[ "${MAPPING3_SINGLE_REARGMAX_NPZ:-0}" == "1" ]]; then
      "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/average_softmax_ensemble.py" \
        --member "$name" "$result_dir/fold_${fold}/validation" \
        --out-dir "$fold_out" \
        --decision argmax
    else
      mkdir -p "$fold_out"
      cp "$result_dir/fold_${fold}/validation"/ATLAS_*.nii.gz "$fold_out"/
    fi

    copied="$(find "$fold_out" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
    if [[ "$copied" != "$expected" ]]; then
      echo "[error] $name fold $fold wrote $copied masks, expected $expected" >&2
      exit 1
    fi
    cp "$fold_out"/ATLAS_*.nii.gz "$oof_dir"/
  done

  local total
  total="$(find "$oof_dir" -maxdepth 1 -name 'ATLAS_*.nii.gz' -type f | awk 'END {print NR}')"
  if [[ "$total" != "1450" ]]; then
    echo "[error] $name OOF mask count is $total, expected 1450" >&2
    exit 1
  fi

  PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/models/nnunet_atlas_t1w/nnunet_helpers/run_atlas_eval.py" \
    --manifest "$MANIFEST" \
    --splits "$SPLITS" \
    --folds all \
    --pred-dir "$oof_dir" \
    --per-session-csv "$PER_SESSION" \
    --out-dir "$report_dir" \
    --title "$display_name on ATLAS R3.0" \
    --workers "$EVAL_WORKERS"

  cat > "$report_dir/meta.json" <<EOF
{
  "method": "$method",
  "display_name": "$display_name",
  "model_family": "nnunet",
  "eval_set": "atlas_r30_5fold_cv",
  "input_modalities": ["T1w"],
  "trained_on": "ATLAS R3.0 / Dataset502_ATLASR30",
  "status": "complete",
  "pipeline_dir": "baselines/models/nnunet_atlas_t1w",
  "notes": "$notes"
}
EOF

  "$EVAL_PY" - "$report_dir/report.json" "$name" <<'PY'
import json
import sys
report = json.loads(open(sys.argv[1]).read())
metrics = {row["metric"]: row for row in report["overall"]}
print(f"[done] {sys.argv[2]} single-family OOF argmax")
for metric in ("dice", "lesion_f1", "surface_dice_3mm", "hd95_mm", "abs_volume_diff_ml"):
    row = metrics.get(metric)
    if row:
        print(f"{metric}: mean={row.get('mean'):.4f} median={row.get('median'):.4f}")
PY
}

BASE="$nnUNet_results/Dataset502_ATLASR30"
evaluate_family \
  topk10 \
  "$BASE/nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres" \
  "$REPO/baselines/reports/nnunet_mapping3_single_topk10_oof_argmax" \
  nnunet_mapping3_single_topk10_oof_argmax \
  "MAPPING input single TopK10 OOF argmax" \
  "Single-family argmax from step-35 validation softmax; exact TopK10 input source for 3-family ensemble comparison"

evaluate_family \
  default1000 \
  "$BASE/nnUNetTrainer__nnUNetPlans__3d_fullres" \
  "$REPO/baselines/reports/nnunet_mapping3_single_default1000_oof_argmax" \
  nnunet_mapping3_single_default1000_oof_argmax \
  "MAPPING input single default 1000ep OOF argmax" \
  "Single-family argmax from step-35 validation softmax; exact default 1000ep input source for 3-family ensemble comparison"

evaluate_family \
  resenc_l \
  "$BASE/nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans__3d_fullres" \
  "$REPO/baselines/reports/nnunet_mapping3_single_resenc_l_oof_argmax" \
  nnunet_mapping3_single_resenc_l_oof_argmax \
  "MAPPING input single ResEnc-L OOF argmax" \
  "Single-family argmax from step-35 validation softmax; exact ResEnc-L input source for 3-family ensemble comparison"

PYTHONPATH="$REPO" "$EVAL_PY" "$REPO/baselines/compare/compare_models.py"
